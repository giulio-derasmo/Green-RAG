import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


import sys
sys.path.append(".")

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from memmap_interface import MemmapCorpusEncoding, MemmapQueriesEncoding
import argparse
from tqdm.autonotebook import tqdm

def search_faiss(args, collection_path, base_collection, k=1000):
    # read the queries
    query_reader_params = {'sep': "\t", 'names': ["query_id", "text"], 'header': None, 'dtype': {"query_id": str}}
    queries = pd.read_csv(f"{args.data_dir}/queries/{collection_path}/queries.tsv", **query_reader_params)

    memmapsdir = f"{args.data_dir}/memmaps/{args.retrieval_model}/{collection_path}"
    qrys_encoder = MemmapQueriesEncoding(f"{memmapsdir}/queries.dat",
                                        f"{memmapsdir}/queries_mapping.tsv")


    
    # load faiss index
    faiss_path = f"{args.data_dir}/vectordb/{args.retrieval_model}/corpora/{base_collection}/"
    index_name = "index_db.faiss"
    index = faiss.read_index(faiss_path + index_name)
    
    # mapper
    corpora_memmapsdir = f"{args.data_dir}/memmaps/{args.retrieval_model}/corpora/{base_collection}"
    mapping = pd.read_csv(f"{corpora_memmapsdir}/corpus_mapping.csv", dtype={'did': str})
    mapper = mapping.set_index('offset').did.to_list()
    
    
    qembs = qrys_encoder.get_encoding(queries.query_id.to_list())
    print('Search in faiss index')
    ip, idx = index.search(qembs, k)
    nqueries = len(ip)
    out = []
    for i in tqdm(range(nqueries)):
        run = pd.DataFrame(list(zip([queries.iloc[i]['query_id']] * len(ip[i]), idx[i], ip[i])), columns=["query_id", "did", "score"])
        run.sort_values("score", ascending=False, inplace=True)
        run['did'] = run['did'].apply(lambda x: mapper[x])
        run['rank'] = np.arange(len(ip[i]))
        out.append(run)
    out = pd.concat(out)
    out["Q0"] = "Q0"
    out["run"] = args.retrieval_model.replace('_', '-')
    out = out[["query_id", "Q0", "did", "rank", "score", "run"]]

    return out


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--collection", type=str, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--retrieval_model", type=str, required=True)

    args = parser.parse_args()

    # Store the original collection name
    base_collection = args.collection
    
    # Modify collection path only when split is provided
    if args.split is not None:
        collection_with_split = f'{args.collection}/{args.split}'
    else:
        collection_with_split = args.collection
        
    # Modify the search_faiss function to accept both paths
    out = search_faiss(args, collection_with_split, base_collection, k=1000)
    out.to_csv(f"{args.data_dir}/runs/{collection_with_split}/{args.retrieval_model}.tsv", header=None, index=None, sep="\t")