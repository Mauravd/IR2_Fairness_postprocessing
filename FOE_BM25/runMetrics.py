from bm25 import load_data
from FOE import runFOEIR
from metrics import *
from randomize import shuffle_ranking
import numpy as np
import json
from bm25 import author_citations

# with open('./bm25_preranking.json.txt', 'r', encoding='utf-8') as f:
#         query_rankings = json.loads('[{}]'.format(','.join(f)))
with open('./bm25_preranking.json', 'r') as f:
    query_rankings = json.loads(f.read())


def generate_post_ranking(query, protected_threshold=10):
    author_data = author_citations()
    doc_data, doc_id_idx, query_dataset, query_docs = load_data()
    ranking = query_rankings[query][:10]
    # Define the 'H' threshold, 'L' if n_citations < threshold
    # protected_threshold = 10

    # Construct a dict that contains all the info required for reranking
    docs_rel = {x['doc_id']:x['relevance'] for x in query_docs[query]}
    post_rank_dataset = {'query':query, 'doc_id':[], 'rel':[], 'y_pred':[], 'groups':[]}
    for doc, score in ranking:
        doc_info = doc_data[doc_id_idx[doc]]
        try:
            author_id = int(doc_info['authors'][0]['ids'][0])

        except IndexError as e:
            author_id = 0
            
        protected_var = author_data.get(author_id, 0)

        doc_rel = docs_rel[doc]
        group = 'L' if protected_var < protected_threshold else 'H'  # Change for experiments

        post_rank_dataset['doc_id'].append(doc)
        post_rank_dataset['rel'].append(doc_rel)
        post_rank_dataset['y_pred'].append(score)
        post_rank_dataset['groups'].append(group)
    
    return post_rank_dataset


def run_metrics_bm25(queries):
    ndcg_5_bm25 = []
    ndcg_10_bm25 = []
    dtr_bm25 = []

    ndcg_5_bm25_FOE = []
    ndcg_10_bm25_FOE = []
    dtr_bm25_FOE = []

    ndcg_5_bm25_random = []
    ndcg_10_bm25_random = []
    dtr_bm25_random = []

    for query in queries:
        ranking = generate_post_ranking(query, 10)
        # ranking = ranking
        random_ranking = shuffle_ranking(ranking)
        foe_ranking, _, _ = runFOEIR(ranking, 'trec', 'FOEIR-DTC', k=10, query_rep=1)

        # NDCG@5
        ndcg_5_bm25.append(ndcg_k(ranking['y_pred'], ranking['rel'], 5))
        ndcg_5_bm25_FOE.append(ndcg_k(foe_ranking['y_pred'], foe_ranking['rel'], 10))
        ndcg_5_bm25_random.append(ndcg_k(random_ranking['y_pred'], random_ranking['rel'], 10))

        # NDCG@10
        ndcg_10_bm25.append(ndcg_k(ranking['y_pred'], ranking['rel'], 10))
        ndcg_10_bm25_FOE.append(ndcg_k(foe_ranking['y_pred'], foe_ranking['rel'], 10))
        ndcg_10_bm25_random.append(ndcg_k(random_ranking['y_pred'], random_ranking['rel'], 10))

        # DTR
        dtr_bm25.append(calculate_DTR(ranking['rel'], ranking['y_pred'], ranking['groups'], k=None))
        dtr_bm25_FOE.append(calculate_DTR(foe_ranking['rel'], foe_ranking['y_pred'], foe_ranking['groups'], k=None))
        dtr_bm25_random.append(calculate_DTR(random_ranking['rel'], random_ranking['y_pred'], random_ranking['groups'], k=None))
        

    bm25_metrics = [np.mean(ndcg_5_bm25), np.mean(ndcg_10_bm25), np.mean(dtr_bm25)]
    bm25_foe_metrics = [np.mean(ndcg_5_bm25_FOE), np.mean(ndcg_10_bm25_FOE), np.mean(dtr_bm25_FOE)]
    bm25_random_metrics = [np.mean(ndcg_5_bm25_random), np.mean(ndcg_10_bm25_random), np.mean(dtr_bm25_random)]
    return bm25_metrics, bm25_foe_metrics, bm25_random_metrics

doc_data, doc_id_idx, query_dataset, query_docs = load_data()
queries = query_docs.keys()
# print(queries)
print(run_metrics_bm25(queries))