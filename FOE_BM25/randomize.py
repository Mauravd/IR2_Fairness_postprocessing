from regex import E
from bm25 import generate_bm25_ranking
import random

ranking = generate_bm25_ranking(query='cloud computing')
random_ranking = ranking
def shuffle_ranking(ranking):
    to_shuffle = list(zip(ranking['doc_id'], ranking['rel'], ranking['y_pred'], ranking['groups']))
    random.shuffle(to_shuffle)
    res = list(zip(*to_shuffle))
    random_ranking['doc_id'] = list(res[0])
    random_ranking['rel'] = list(res[1])
    random_ranking['y_pred'] = list(res[2])
    random_ranking['groups'] = list(res[3])
    
    return ranking
