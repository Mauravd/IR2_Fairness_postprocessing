#!/usr/bin/env python
# coding: utf-8

# In[45]:


from collections import defaultdict, Counter
import numpy as np
import nltk
from nltk.corpus import stopwords


# In[46]:


# stop_words = set(stopwords.words('english'))
# nltk.download()


# In[ ]:


## Load data

import json

def load_data():

    doc_file = './corpus-subset-for-queries.jsonl.txt'
    train_file = './TREC-Fair-Ranking-training-sample.json.txt'

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


def tokenize(text):
    """
        Tokenizes the input text. Use the WordPunctTokenizer
        Input: text - a string
        Output: a list of tokens
    """
    tk = nltk.tokenize.WordPunctTokenizer()   
    return tk.tokenize(text)

def stem_token(token):
    """
        Stems the given token using the PorterStemmer from the nltk library
        Input: a single token
        Output: the stem of the token
    """
    stemmer = nltk.stem.PorterStemmer()
    return stemmer.stem(token)

#### Putting it all together
def process_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = []
    for token in tokenize(text):
        token = token.lower()
        if token in stop_words:
            continue
        token = stem_token(token)
        tokens.append(token)

    return tokens

def process_docs(documents):
    processed_docs = []
    for doc_id in documents:
        processed_docs.append((doc_id, process_text(documents[doc_id])))
    return processed_docs

def compute_df(documents):
    """
        Compute the document frequency of all terms in the vocabulary
        Input: A list of documents
        Output: A dictionary with {token: document frequency (int)}
    """

    df = defaultdict(int)

    for tokens in documents:
        used_tokens = set()

        for token in tokens:
            if token not in used_tokens:
                df[token] += 1
                used_tokens.add(token)
    
    return df  

def build_tf_index(documents):
    """
        Build an inverted index (with counts). The output is a dictionary which takes in a token
        and returns a list of (doc_id, count) where 'count' is the count of the 'token' in 'doc_id'
        Input: a list of documents - (doc_id, tokens) 
        Output: An inverted index implemented within a pyhton dictionary: [token] -> [(doc_id, token_count)]
    """
    
    tf_index = defaultdict(list)
    
    token_set = set()

    for doc_id, tokens in documents:
        for token in tokens:
            token_set.add(token)

    for doc_id, tokens in documents:
        used_tokens = set()
        counter = Counter(tokens)

        for token in tokens:
            if token not in used_tokens:
                tf_index[token].append((doc_id, counter[token]))
                used_tokens.add(token)
        
        for token in token_set:
            if token not in used_tokens:
                tf_index[token].append((doc_id, 0))
                used_tokens.add(token)

    return tf_index


#### Document length for normalization
def doc_lengths(documents):
    lengths = {doc_id : len(doc) for (doc_id, doc) in documents}
    return lengths

def bm25_search(query, documents):
    """
        Perform a search over all documents with the given query using BM25. Use k_1 = 1.5 and b = 0.75
        Note #1: You have to use the `get_index` (and `get_doc_lengths`) function created in the previous cells
        Note #2: You might have to create some variables beforehand and use them in this function
        
        Input: 
            query - a (unprocessed) query
            index_set - the index to use
        Output: a list of (document_id, score), sorted in descending relevance to the given query 
    """
    
    documents = process_docs(documents)
    processed_query = process_text(query)

    index = build_tf_index(documents)
    df = compute_df([d[1] for d in documents])
    doc_lens = doc_lengths(documents)
    
    k_1 = 1.5
    b = 0.75

    N = len(doc_lens)
    avgdl = sum(doc_lens.values()) / N


    # List of [doc_id, score] pairs
    scores = []
    
    # Keeps track of idx of document rank in the scores list
    doc_id_pos = dict()

    for token in processed_query:
        for doc_id, count in index[token]:
            count = float(count)

            idf = np.log(N / df[token])
            tf = count

            score = idf * ((k_1 + 1) * tf) / (k_1 * ((1 - b) + b * doc_lens[doc_id] / avgdl) + tf)

            if doc_id not in doc_id_pos:
                scores.append([doc_id, score])
                doc_id_pos[doc_id] = len(scores) - 1
            else:
                scores[doc_id_pos[doc_id]][1] += score
            
    
    # convert to list of tuples
    scores = [(doc_id, score) for doc_id, score in scores]

    return sorted(scores, key=lambda x: x[1], reverse=True)



def rank_bm25(query, docs):
##    for query in d.keys():
##    docs = d[query]
    ranking = bm25_search(query, docs)
    # print(ranking)
    if not ranking:
        print(query)

    return ranking


def author_citations():
    d = {} 
    author_file = './corpus-subset-for-queries.authors.csv'
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


def docs_relevance(query_data):
    return {x['doc_id']:x['relevance'] for x in query_data}


def generate_pre_ranking(query='tik tok'):
    author_data = author_citations()
    doc_data, doc_id_idx, query_dataset, query_docs = load_data()
    ranking = rank_bm25(query, query_dataset[query])
    docs_rel = docs_relevance(query_docs[query])

    # Define the 'H' threshold, 'L' if n_citations < threshold
    protected_threshold = 10

    # Construct a dict that contains all the info required for reranking
    n_authors_not_found = 0
    post_rank_dataset = {'query_id':query, 'doc_id':[], 'rel':[], 'y_pred':[], 'groups':[]}
    for doc, score in ranking[:10]:
        doc_info = doc_data[doc_id_idx[doc]]
        author_id = int(doc_info['authors'][0]['ids'][0])
        n_citations = author_data.get(author_id, 0)
        if not author_id in author_data:
            n_authors_not_found += 1
        doc_rel = docs_rel[doc]
        protect = 'L' if n_citations < protected_threshold else 'H'  # Change for experiments

        post_rank_dataset['doc_id'].append(doc)
        post_rank_dataset['rel'].append(doc_rel)
        post_rank_dataset['y_pred'].append(score)
        post_rank_dataset['groups'].append(protect)

    return post_rank_dataset

# Testing the output used for the post-processing ranking function
result = generate_pre_ranking()
# print(result)
