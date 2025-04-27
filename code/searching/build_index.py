import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append(".")

import faiss
from memmap_interface import MemmapCorpusEncoding
import argparse
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--data_dir", default="../data")
    parser.add_argument("-c", "--collection", type=str, required=True)
    parser.add_argument("-r", "--retrieval_model", type=str, required=True)
    args = parser.parse_args()

    print('Loading collection...')
    # load memmap for the corpus
    corpora_memmapsdir = f"{args.data_dir}/memmaps/{args.retrieval_model}/corpora/{args.collection}"
    docs_encoder = MemmapCorpusEncoding(f"{corpora_memmapsdir}/corpus.dat",
                                        f"{corpora_memmapsdir}/corpus_mapping.csv")
    d, d_model = docs_encoder.get_shape()
    index = faiss.IndexFlatIP(d_model)

    # Add to index
    data = docs_encoder.get_data()
    print('Add corpus to FAISS index')
    for s in range(0, d, 1024):
        e = min(s + 1024, d)
        keys = data[s:e]
        index.add(keys)
        print(s)

    faiss_path = f"{args.data_dir}/vectordb/{args.retrieval_model}/corpora/{args.collection}/index_db.faiss"
    print('Save to: ', faiss_path)
    faiss.write_index(index, faiss_path)
