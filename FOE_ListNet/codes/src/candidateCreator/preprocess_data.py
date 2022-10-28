import json
import pandas as pd


def load_data():

    doc_file = '././data/TREC2020/features/fold/corpus-subset-for-queries.jsonl.txt'
    train_file = '././data/TREC2020/features/fold/TREC-Fair-Ranking-training-sample.json.txt'
    
    # train_file = '././data/TREC2020/features/fold/TREC-Fair-Ranking-eval-sample.json.txt'
    with open(doc_file, 'r', encoding='utf-8') as f:
        doc_data = json.loads('[{}]'.format(','.join(f)))

    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.loads('[{}]'.format(','.join(f)))


    doc_id_idx = {x['id']:i for i, x in enumerate(doc_data)}

    labels = ['title', 'paperAbstract', 'fieldsOfStudy']

    d = {}
    q_docs = {}
    for query in train_data:
        q = query['qid']
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

def author_citations():
    d = {} 
    author_file = '././data/TREC2020/features/fold/corpus-subset-for-queries.authors.csv'
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


author_data = author_citations()
doc_data, doc_id_idx, query_dataset, query_docs = load_data()

# Define the 'H' threshold, 'L' if n_citations < threshold
# protected_threshold = 10
def col_to_csv(protected_threshold, filename):
    data = pd.read_csv(filename)
    queries = list(data['session'].unique())

    new_col = []
    for query in queries:
        docs = query_docs[query]
        doc_len = len(docs)
        doc_len_data = len(data[data['session'] == query])
        if doc_len != doc_len_data:
            print(query, doc_len, doc_len_data)

        for doc in docs:
            doc_id = doc['doc_id']
            try:
                doc_info = doc_data[doc_id_idx[doc_id]]
                author_id = int(doc_info['authors'][0]['ids'][0])
                n_citations = author_data.get(author_id, 0)
            except:
                # n_citations = 0
                continue

            group = 'L' if n_citations < protected_threshold else 'H'  # Change for experiments
            
            new_col.append(group)
    return new_col


def new_row_2csv(col, header, filename):
    data_new = pd.read_csv(filename, index_col=0)
    data_new[header] = col
    data_new.to_csv(filename)

file_to_write = '././data/TREC2020/features/fold/new_train.csv'
n_10 = col_to_csv(10, file_to_write)
n_30 = col_to_csv(30, file_to_write)
print(len(n_10), len(pd.read_csv(file_to_write)))
# new_col_2csv(n_10, 'n_citations=10', file_to_write)
# new_col_2csv(n_30, 'n_citations=30', file_to_write)


# h_5 = col_to_csv(5, file)
# h_20 = col_to_csv(20, file)
# new_col_2csv(h_5, 'h_index_5')
# new_col_2csv(h_20, 'h_index_20')