import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import sys
sys.path.append('.') 

import pandas as pd
import numpy as np
import argparse
from tqdm.autonotebook import tqdm

from normalize_text import normalize
from datasets import load_dataset

import torch
from sentence_transformers import SentenceTransformer


m2hf = {
        "tasb": 'sentence-transformers/msmarco-distilbert-base-tas-b',
        "contriever": "facebook/contriever-msmarco",
        }

coll2corpus = {
            "nq": "giulioderasmo/nq_wikidump2018",
            }

def normalize_batch(batch):
    batch["normalized_text"] = [normalize(t) for t in batch['formatted_text']]
    return batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--data_dir", default="../data")
    parser.add_argument("-c", "--collection", type=str, required=True)
    parser.add_argument("-r", "--retrieval_model", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Loading collection...')
    collection = load_dataset(coll2corpus[args.collection], cache_dir='/home/filgiu/projects/cache_dir')

    if 'train' in collection:
        collection = collection['train']

    print('Normalizing collection...')
    collection = collection.map(normalize_batch, batched=True)
    docs = collection.to_pandas()[['id', 'normalized_text']].rename(columns={'id': 'did', 'normalized_text': 'text'})
    docs['text'] = docs['text'].astype(str) # conver to str
    docs = docs.sort_values("did")  # sort and create and offset column for the matrix notation and recovering mapping
    docs["offset"] = np.arange(len(docs.index))

    print('Loading retrieval model...')
    model = SentenceTransformer(m2hf[args.retrieval_model]).to(device)

    fp = np.memmap(f"{args.data_dir}/memmaps/{args.retrieval_model}/corpora/{args.collection}/corpus.dat", dtype='float32', mode='w+', shape=(len(docs), 768))
    step = 10_000
    for i in tqdm(range(0, len(docs.text.to_list()), step)):
        # Compute the embeddings and populate the memmap
        fp[i:i + step] = model.encode(docs.text.to_list()[i:i + step])

    docs[["did", "offset"]].to_csv(f"{args.data_dir}/memmaps/{args.retrieval_model}/corpora/{args.collection}/corpus_mapping.csv", index=False)
