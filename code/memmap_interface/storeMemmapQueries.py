import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
sys.path.append('.') 

import pandas as pd
import numpy as np
import argparse
from tqdm.autonotebook import tqdm

from normalize_text import normalize
import torch
from sentence_transformers import SentenceTransformer

m2hf = {
        "tasb": 'sentence-transformers/msmarco-distilbert-base-tas-b',
        "contriever": "facebook/contriever-msmarco",
        }

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_dir", default="../data")
        parser.add_argument("--collection", type=str, required=True)
        parser.add_argument("--split", type=str, default=None)
        parser.add_argument("--retrieval_model", type=str, required=True)
        args = parser.parse_args()

        queries = []
        
        if args.split is not None:
                args.collection = f'{args.collection}/{args.split}'
        qpath = f"{args.data_dir}/queries/{args.collection}/queries.tsv"
        queries = pd.read_csv(qpath, sep="\t", header=None, names=["qid", "text"], dtype={"qid": str})
        queries["offset"] = np.arange(len(queries.index))
        model = SentenceTransformer(m2hf[args.retrieval_model])

        repr = np.array(queries.text.apply(model.encode).to_list())
        fp = np.memmap(f"{args.data_dir}/memmaps/{args.retrieval_model}/{args.collection}/queries.dat",
                        dtype='float32', mode='w+', shape=repr.shape)
        fp[:] = repr[:]
        fp.flush()

        queries[['qid', 'offset']].to_csv(f"{args.data_dir}/memmaps/{args.retrieval_model}/{args.collection}/queries_mapping.tsv", sep="\t", index=False)