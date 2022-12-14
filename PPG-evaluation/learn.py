
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from collections import defaultdict
import sys

from torch.utils.data import Dataset, DataLoader
from Mmetrics import *

import LTR
import datautil
import permutationgraph
import DTR
import EEL
import PPG
import PPG_single
import PL

def df2ds(df_path):
    with open(df_path, 'rb') as f:
        df = pickle.load(f)
    ds = df.to_dict(orient='list')
    for k in ds:
        ds[k] = np.array(ds[k])  # converts dataset columns to arrays

    # Indicates where new queries start in flattened dataset
    ds['dlr'] = np.concatenate([np.zeros(1),  # concat([0], where( diff(qid)==1 )[0] + 1, n_qids ))
                                np.where(
                                    np.diff(
                                        ds['qid']
                                        )==1  # checks for successive query ids
                                    )[0]+1,
                                    np.array([ds['qid'].shape[0]])
                                ]).astype(int)
    return type('ltr', (object,), ds)


def dict2ds(df_path):
    with open(df_path, 'rb') as f:
        ds = pickle.load(f)
    return type('ltr', (object,), ds)

ds2019 = df2ds('nLTR2019.df')
ds2020 = df2ds('LTR2020.df')
sds2019 = dict2ds('s_LTR2019.df')
sds2020 = dict2ds('s_LTR2020.df')


epochs = 50


alg = 'PPG'
if len(sys.argv) > 1:
    alg = sys.argv[1]

metric = 'EEL'
if len(sys.argv) > 2:
    metric = sys.argv[2]
    

intra = False
suffix = 'nointra_'
if len(sys.argv) > 3:
    if sys.argv[3] == 'intra':
        intra = True
        suffix='intra_'

    
sessions_cnt=20
if len(sys.argv) > 4:
    sessions_cnt = eval(sys.argv[4])
    suffix += f'{alg}_{sessions_cnt}_{sys.argv[5]}'
else:
    suffix += f'{alg}_{sessions_cnt}'



exposure2020 = np.array([1./np.log2(2+i) for i in range(1,np.diff(ds2020.dlr).max()+2)])  ################# ?
exposure2019 = np.array([1./np.log2(2+i) for i in range(1,np.diff(ds2019.dlr).max()+2)])




def learn_one_PPG_single(metric, qid, verbose, y_pred, g, dlr, epochs, lr, exposure, grade_levels, samples_cnt, sessions_cnt):
    s, e = dlr[qid:qid+2]  # s(tart), e(nd)
    print(s, e, g[s:e])  # qid starts at doc s, ends at doc next_qid exclusive, docs[dlr[qid] : docs[dlr[qid+1]]
    print(y_pred)  # 'normalised' y_pred (example non-normalised = 0.6849...)
    
    metric = ''
    
    if metric == 'EEL':
        objective_ins = EEL.EEL(y_pred = y_pred[s:e], g = g[s:e], dlr = np.array([0,e-s]), exposure=exposure, grade_levels = grade_levels)
    else:
        objective_ins = DTR.DTR(y_pred = y_pred[s:e], g = g[s:e], dlr = np.array([0,e-s]), exposure=exposure)
        
        
    learner = PPG_single.Learner(  PPG_mat=None, samples_cnt=samples_cnt, 
                                objective_ins=objective_ins,
                                sorted_docs = y_pred[s:e].argsort()[::-1], # idx array from highest y_pred -> smallest y_pred
                                intra = g[s:e] if intra else np.arange(g[s:e].shape[0]),
                                sessions_cnt=sessions_cnt)
    print(learner.PPG)
    print()
    print(PPG_single._PPG_sample(learner.PPG))
    vals = learner.fit(epochs, lr, verbose=verbose)
    print()
    print(learner.PPG)
    print()
    print(PPG_single._PPG_sample(learner.PPG))
    quit()
    return vals


########################################################################
def learn_all_PPG_single(metric, y_pred, g, dlr, epochs, lr, exposure, grade_levels, samples_cnt, sessions_cnt):
    sorted_docs = []
#    print(len(dlr))
#    print(len(exposure))
#    quit()
    
#     for qid in trange(dlr.shape[0] - 1, leave=False):
    for qid in range(dlr.shape[0] - 1):
        min_b = learn_one_PPG_single(metric, qid, 0, y_pred, g, dlr, epochs, lr, exposure, grade_levels, samples_cnt, sessions_cnt)
        sorted_docs.append(min_b)
        

    # print(ndcg_dtr(exposure, lv, np.concatenate(y_rerank), dlr, g, query_counts))
    return sorted_docs


def learn_one_PPG(metric, qid, verbose, y_pred, g, dlr, epochs, lr, exposure, grade_levels, samples_cnt, sessions_cnt):
    s, e = dlr[qid:qid+2]
    y_pred_s, g_s, sorted_docs_s, dlr_s = \
        EEL.copy_sessions(y=y_pred[s:e], g=g[s:e], sorted_docs=y_pred[s:e].argsort()[::-1], sessions=sessions_cnt)
    
    if metric == 'EEL':
        objective_ins = EEL.EEL(y_pred = y_pred_s, g = g_s, dlr = dlr_s, exposure=exposure, grade_levels = grade_levels)
    else:
        objective_ins = DTR.DTR(y_pred = y_pred_s, g = g_s, dlr = dlr_s, exposure=exposure)
        
    learner = PPG.Learner(  PPG_mat=None, samples_cnt=samples_cnt, 
                                objective_ins=objective_ins, 
                                sorted_docs = sorted_docs_s, 
                                dlr = dlr_s,
                                intra = g_s if intra else np.arange(g_s.shape[0]),
                                inter = np.repeat(dlr_s[:-1], np.diff(dlr_s)))
    vals = learner.fit(epochs, lr, verbose=verbose)

    return vals


#
#
#
#
#
def learn_all_PPG(metric, y_pred, g, dlr, epochs, lr, exposure, grade_levels, samples_cnt, sessions_cnt):
    sorted_docs = []
    
#     for qid in trange(dlr.shape[0] - 1, leave=False):
    print('learn_all_PPG()')
    for qid in range(dlr.shape[0] - 1):
        print(' qid:{}'.format(qid))
        if qid > 2:
            break
        min_b = learn_one_PPG(metric, qid, 1, y_pred, g, dlr, epochs, lr, exposure, grade_levels, samples_cnt, sessions_cnt)
        sorted_docs.append(min_b)
        

    # print(ndcg_dtr(exposure, lv, np.concatenate(y_rerank), dlr, g, query_counts))
    return sorted_docs


def learn_one_PL(metric, qid, verbose, y_pred, g, dlr, epochs, lr, exposure, grade_levels, samples_cnt, sessions_cnt):
    s, e = dlr[qid:qid+2]
    
    if metric == 'EEL':
        objective_ins = EEL.EEL(y_pred = y_pred[s:e], g = g[s:e], dlr = np.array([0,e-s]), exposure=exposure, grade_levels = grade_levels)
    else:
        objective_ins = DTR.DTR(y_pred = y_pred[s:e], g = g[s:e], dlr = np.array([0,e-s]), exposure=exposure)
        
    learner = PL.Learner(logits=y_pred[s:e], samples_cnt=samples_cnt, 
                        objective_ins=objective_ins, sessions_cnt=sessions_cnt)
    vals = learner.fit(epochs, lr, verbose=verbose)
    return vals

def learn_all_PL(metric, y_pred, g, dlr, epochs, lr, exposure, grade_levels, samples_cnt, sessions_cnt):
    sorted_docs = []
    
#     for qid in trange(dlr.shape[0] - 1, leave=False):
    for qid in range(dlr.shape[0] - 1):
        min_b = learn_one_PL(metric, qid, 0, y_pred, g, dlr, epochs, lr, exposure, grade_levels, samples_cnt, sessions_cnt)
        sorted_docs.append(min_b)
        

    # print(ndcg_dtr(exposure, lv, np.concatenate(y_rerank), dlr, g, query_counts))
    return sorted_docs



def estimated_evaluate_one(metric, qid, y_pred, g, dlr, output_permutation, exposure, sessions_cnt):
    s, e = dlr[qid:qid+2]
    permutation = output_permutation[qid]
    lv_s, g_s, sorted_docs_s, dlr_s = \
        EEL.copy_sessions(y=y_pred[s:e], g=g[s:e], sorted_docs=y_pred[s:e].argsort()[::-1], sessions=sessions_cnt)
    
    
    if metric == 'EEL':
        objective_ins = EEL.EEL(y_pred = lv_s, g = g_s, dlr = dlr_s, exposure=exposure, grade_levels = 2)
    else:
        objective_ins = DTR.DTR(y_pred = lv_s, g = g_s, dlr = dlr_s, exposure=exposure)
        
    return objective_ins.eval(permutation)
 
def estimated_evaluate_all(metric, y_pred, g, dlr, output_permutation, exposure, sessions_cnt):
    eel_res = []
    for qid in range(dlr.shape[0] - 1):
        s,e = dlr[qid:qid+2]
        if len(np.unique(g[s:e])) == 1:
            continue
        eel = estimated_evaluate_one(metric, qid, y_pred, g, dlr, output_permutation, exposure, sessions_cnt)
        eel_res.append(eel)
        
    return np.array(eel_res).mean()


learn_fn = eval(f'learn_all_{alg}')
learn_fn = learn_all_PPG_single  ###########################################

res = {}

def find_best(y_pred, sds, exposure, sessions_cnt):
    best_lr = 0
    best_samples_cnt = 0
    best_fairness = np.inf

    for learning_rate in ['0.01', '0.1']:
        for samples_cnt in [4,8,16,32]:
            output_permutation = learn_fn(metric, y_pred, sds.g, sds.dlr, 
                                          epochs, eval(learning_rate), exposure=exposure,
                                          grade_levels=5, samples_cnt=samples_cnt, sessions_cnt=sessions_cnt)
            fairness = estimated_evaluate_all(metric, y_pred, sds.g, sds.dlr, 
                                              output_permutation, exposure, sessions_cnt)
            if fairness < best_fairness:
                best_lr = learning_rate
                best_samples_cnt = samples_cnt
                best_fairness = fairness
        return best_lr, best_samples_cnt

print('hi')
for run_i in range(2):
#     learning_rate, samples_cnt = find_best(sds2020.y_pred, sds2020, exposure2020, sessions_cnt)
    learning_rate, samples_cnt = '0.4', 16
    res[f'2020_{learning_rate}_{samples_cnt}_{run_i}'] = \
        learn_fn(metric, ds2020.y_pred, ds2020.g, ds2020.dlr, epochs, eval(learning_rate), exposure=exposure2020,
        grade_levels=5, samples_cnt=samples_cnt, sessions_cnt=sessions_cnt)
    print('1')
    with open(f'n/2/{suffix}_{metric}_results.pkl', 'wb') as f:
        pickle.dump(res, f)


    print('2')
    learning_rate, samples_cnt = '0.4', 16
#     learning_rate, samples_cnt = find_best(sds2019.y_pred, sds2019, exposure2019, sessions_cnt)
    res[f'2019_{learning_rate}_{samples_cnt}_{run_i}'] = \
        learn_fn(metric, ds2019.y_pred, ds2019.g, ds2019.dlr, epochs, eval(learning_rate), exposure=exposure2019,
        grade_levels=5, samples_cnt=samples_cnt, sessions_cnt=sessions_cnt)
    print('3')
    
    with open(f'n/2/{suffix}_{metric}_results.pkl', 'wb') as f:
        pickle.dump(res, f)

    print('4')


#     learning_rate, samples_cnt = '0.1', 16
# #     learning_rate, samples_cnt = find_best(sds2020.lv, sds2020, exposure2020, sessions_cnt)
#     res[f'lv_2020_{learning_rate}_{samples_cnt}_{run_i}'] = \
#         learn_fn(metric, ds2020.lv, ds2020.g, ds2020.dlr, epochs, eval(learning_rate), exposure=exposure2020,
#         grade_levels=5, samples_cnt=samples_cnt, sessions_cnt=sessions_cnt)

#     with open(f'/ivi/ilps/personal/avardas/_data/PPG/32/{suffix}_{metric}_results.pkl', 'wb') as f:
#         pickle.dump(res, f)

        
#     learning_rate, samples_cnt = '0.1', 16
# #     learning_rate, samples_cnt = find_best(sds2019.lv, sds2019, exposure2019, sessions_cnt)
#     res[f'lv_2019_{learning_rate}_{samples_cnt}_{run_i}'] = \
#         learn_fn(metric, ds2019.lv, ds2019.g, ds2019.dlr, epochs, eval(learning_rate), exposure=exposure2019,
#         grade_levels=5, samples_cnt=samples_cnt, sessions_cnt=sessions_cnt)

#     with open(f'/ivi/ilps/personal/avardas/_data/PPG/32/{suffix}_{metric}_results.pkl', 'wb') as f:
#         pickle.dump(res, f)
