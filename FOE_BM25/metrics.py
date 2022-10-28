import numpy as np
import DTR

def calculate_DTR(rel_rerank, y_pred_rerank, g_rerank):
    dlr = np.array([0, len(y_pred_rerank)])
    dtr = DTR.calculatedTR(rel_rerank, y_pred_rerank, g_rerank, dlr)
    return dtr

def dcg_at_k(sorted_labels, k):
    if k > 0:
        k = min(sorted_labels.shape[0], k)
    else:
        k = sorted_labels.shape[0]
    denom = 1./np.log2(np.arange(k)+2.)
    nom = 2**sorted_labels-1.
    dcg = np.sum(nom[:k]*denom)
    return dcg

def ndcg_k(scores, labels, k=10):
    sort_ind = np.argsort(scores)[::-1]
    sorted_labels = labels[sort_ind]
    ideal_labels = np.sort(labels)[::-1]
    return dcg_at_k(sorted_labels, k) / dcg_at_k(ideal_labels, k)