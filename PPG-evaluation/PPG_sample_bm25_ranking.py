import DTR
import PPG_single
import numpy as np
import json

import bm25

np.random.seed(42)

# Random set of ranking scores and groups for testing
# n_docs = 10
# y_pred = np.random.random(n_docs)
# groups = np.char.array([np.random.choice(['H', 'L']) for i in range(n_docs)])


def author_citations():
    d = {}
    author_file = '..\corpus-subset-for-queries.authors.csv'
    header = True
    with open(author_file, 'r', encoding='utf-8') as f:  # d[author_id] = n_citations
        for line in f:
            if header:
                header = False
                continue
            line = line.split(',')
            author_id = int(line[0])
            n_citations = int(line[-2])  # Change for experiments
            # n_papers = int(line[-3])
            d[author_id] = n_citations

    return d




def load_data():

    doc_file = '..\corpus-subset-for-queries.jsonl.txt'
    train_file = '..\TREC-Fair-Ranking-training-sample.json.txt'

    with open(doc_file, 'r', encoding='utf-8') as f:
        doc_data = json.loads('[{}]'.format(','.join(f)))

    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.loads('[{}]'.format(','.join(f)))


    doc_id_idx = {x['id']:i for i, x in enumerate(doc_data)}

    labels = ['title', 'paperAbstract', 'fieldsOfStudy']

    d = {}
    q_docs = {}
    for query in train_data:
        q = query['query']
        d[q] = {}
        q_docs[q] = query['documents']
        for doc in query['documents']:
            try:
                doc_id = doc['doc_id']
                s = []
                data = doc_data[doc_id_idx[doc_id]]
                s.append(data[labels[0]])
                s.append(data[labels[1]])
                if len(data[labels[2]]) > 0:
                    s.append(' '.join(data[labels[2]]))
                d[q][doc_id] = ' '.join(s)
            except KeyError:
                continue

    return doc_data, doc_id_idx, d, q_docs




# doc_data: docs info and features
# doc_id_idx: doc_id to doc_data idx
# query_dataset: bm25 input for each query
# query_docs: data per query in train set (qid, freq, documents, relevance labels)
def generate_pre_ranking(author_data, doc_data, doc_id_idx, query_dataset, query_docs, query='tik tok'):

    # Rank the documents that belong to a query
    ranking = bm25.rank_bm25(query, query_dataset[query])

    if not ranking or sum(y for _, y in ranking) == 0.0:
        return None

    docs_rel = {x['doc_id']:x['relevance'] for x in query_docs[query]}


    # Define the 'H' threshold, 'L' if n_citations < threshold
    protected_threshold = 10


    # Construct a dict that contains all the info required for reranking
    post_rank_dataset = {'doc_id':[], 'rel':[], 'y_pred':[], 'groups':[]}
    for doc, score in ranking:
        doc_info = doc_data[doc_id_idx[doc]]
        try:
            author_id = int(doc_info['authors'][0]['ids'][0])

        # What to do with documents with no authors?
        except IndexError as e:
            print('Missing authors for doc:')
            print(query, doc_info['title'], sep='\n')
            print()
            author_id = 0
        
        h_index = author_data.get(author_id, 0)

        doc_rel = docs_rel[doc]
        group = 'L' if h_index < protected_threshold else 'H'  # Change for experiments

        post_rank_dataset['doc_id'].append(doc)
        post_rank_dataset['rel'].append(doc_rel)
        post_rank_dataset['y_pred'].append(score)
        post_rank_dataset['groups'].append(group)

    return post_rank_dataset




def rerank_PPG(pre_ranking, k=None):
    
    y_pred = np.array(pre_ranking['y_pred'])
    groups = np.char.array(pre_ranking['groups'])
    n_docs = min(len(y_pred), k) if k else len(y_pred)

    exposure = np.array([1. / np.log2(2 + i) for i in range(1, n_docs+2)])
    dlr_dtr = np.array([0, n_docs])


    obj_ins = DTR.DTR(y_pred, groups, dlr_dtr, exposure)

    learner = PPG_single.Learner(PPG_mat = None, samples_cnt = 16, objective_ins = obj_ins, sorted_docs = y_pred.argsort()[::-1],
                                intra = np.arange(n_docs), sessions_cnt = 20)

    # print('Starting PPG:\n', learner.PPG)
    # print()

    vals = learner.fit(50, 0.4, 0)  # Epochs, Learnrate hyperparams
    rerank_idx = PPG_single._PPG_sample(learner.PPG)


    # print('=============')
    # print('Learned PPG:\n', learner.PPG)
    # print('=============')
    # print('Sampling PPG:\n')
    # print(rerank_idx)
    # print()

    return rerank_idx




def calculate_DTR_exposure_utility(rel_rerank, y_pred_rerank, g_rerank, k=None):
    if not k:
        k = len(y_pred_rerank)
    # print('Exposure and utility:')
    exp_L, exp_H, util_L, util_H, *_ = DTR.calculateExposureAndUtility(rel_rerank, y_pred_rerank, g_rerank, 10)
    # print(exp_L, exp_H, util_L, util_H, sep=', ')


    # print()
    # print('DTR:')
    dtr = DTR.calculatedTR(rel_rerank, y_pred_rerank, g_rerank, np.array([0, k]))

    return dtr, exp_L, exp_H, util_L, util_H




def dcg_at_k(sorted_labels, k):
  if k > 0:
    k = min(sorted_labels.shape[0], k)
  else:
    k = sorted_labels.shape[0]
  denom = 1./np.log2(np.arange(k)+2.)
  nom = 2**sorted_labels-1.
  dcg = np.sum(nom[:k]*denom)
  return dcg

def ndcg10(scores, labels):
  sort_ind = np.argsort(scores)[::-1]
  sorted_labels = labels[sort_ind]
  ideal_labels = np.sort(labels)[::-1]
  return dcg_at_k(sorted_labels, 10) / dcg_at_k(ideal_labels, 10)



query = 'tik tok'
k = 10
i = 0
doc_data, doc_id_idx, query_dataset, query_docs = load_data()
author_data = author_citations()

failed_queries = []
for query in query_dataset:
    # i += 1
    if i > 20:
        break
    pre_ranking = generate_pre_ranking(author_data, doc_data, doc_id_idx, query_dataset, query_docs, query)
    if not pre_ranking:
        failed_queries.append(query)
        continue
    continue
    rerank_idx = rerank_PPG(pre_ranking)

    rel_rerank = np.take(pre_ranking['rel'], rerank_idx)
    y_pred_rerank = np.take(pre_ranking['y_pred'], rerank_idx)
    g_rerank = np.take(pre_ranking['groups'], rerank_idx)

    dtr, exp_L, exp_H, util_L, util_H = calculate_DTR_exposure_utility(rel_rerank, y_pred_rerank, g_rerank)

    ndcg_10_prerank = ndcg10(np.array(pre_ranking['y_pred']), pre_ranking['rel'])
    ndcg_10_rerank = ndcg10(y_pred_rerank, pre_ranking['rel'])

print(failed_queries)

