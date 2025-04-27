import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import pandas as pd
import numpy as np
import json
import pickle
import json
import os
from tqdm.autonotebook import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import re
import logging
import warnings
import transformers
from tqdm import tqdm
transformers.logging.set_verbosity_error()
import pickle
import argparse

from codecarbon import EmissionsTracker
import deepspeed
from deepspeed.profiling.flops_profiler import FlopsProfiler

from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from ast import literal_eval
from normalize_answer import is_answer_in_text


import sys
sys.setrecursionlimit(100000)

def extract_info_from_file(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Define a dictionary to store the extracted values
    extracted_info = {}

    # Define a helper function to safely extract the match
    def extract_value(pattern, content):
        match = re.search(pattern, content)
        return match.group(1) if match else None

    # Use regex to find the values for each field, safely extracting them
    extracted_info['params_per_GPU'] = extract_value(r'params per GPU:\s+([0-9\.]+ [A-Za-z]+)', content)
    extracted_info['params_of_model'] = extract_value(r'params of model = params per GPU \* mp_size:\s+([0-9\.]+ [A-Za-z]+)', content)
    extracted_info['fwd_MACs_per_GPU'] = extract_value(r'fwd MACs per GPU:\s+([0-9\.]+ [A-Za-z]+)', content)
    extracted_info['fwd_flops_per_GPU'] = extract_value(r'fwd flops per GPU:\s+([0-9\.]+ [A-Za-z]+)', content)
    extracted_info['fwd_flops_of_model'] = extract_value(r'fwd flops of model = fwd flops per GPU \* mp_size:\s+([0-9\.]+ [A-Za-z]+)', content)
    extracted_info['fwd_latency'] = extract_value(r'fwd latency:\s+([0-9\.]+ ms)', content)
    extracted_info['fwd_FLOPS_per_GPU'] = extract_value(r'fwd FLOPS per GPU = fwd flops per GPU / fwd latency:\s+([0-9\.]+ [A-Za-z]+)', content)

    return extracted_info


def load_model_and_tokenizer(READER_MODEL_NAME, load_in_8bit=True, load_in_4bit=False):
    
    if load_in_4bit:
        #  4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif load_in_8bit:
        #  8-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else: 
        print('No quantization config provided, loading model in full precision')
        quantization_config = None

    ## cache dir cosi teniamo tutto li!
    model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, 
                                                quantization_config=quantization_config,
                                                cache_dir='/home/filgiu/projects/cache_dir',
                                                device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

    READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500,
    )

    return READER_LLM, tokenizer, model


models2name = {
                'llama1b': "meta-llama/Llama-3.2-1B-Instruct",
                'llama8b': "meta-llama/Llama-3.1-8B-Instruct", 
               'llama70b': "meta-llama/Llama-3.1-70B-Instruct",
            } 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--collection", type=str, default="nq")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--retrieval_model", type=str, default="contriever")
    parser.add_argument("--llm", type=str, required=True)
    #parser.add_argument("--load_in_8bit", type=int)  # any non-zero value will be treated as True
    #parser.add_argument("--load_in_4bit", type=int)  # any non-zero value will be treated as True
 
    args = parser.parse_args()
    ###################################
    #### FORCE LOAD IN 8bit ###
    args.load_in_8bit = True
    args.load_in_4bit = False
    
    #if args.load_in_8bit: 
    #    args.load_in_8bit = True
    #if args.load_in_4bit:
    #    args.load_in_4bit = True
    ####################################


    # Modify collection path only when split is provided
    if args.split is not None:
        collection_with_split = f'{args.collection}/{args.split}'
    else:
        collection_with_split = args.collection

    run = pd.read_csv(f'{args.data_dir}/runs/{collection_with_split}/contriever.csv')
    qrels = pd.read_csv(f'{args.data_dir}/qrels/{collection_with_split}/qrels.csv')

    # Prepare the dataset
    def prepare_context(batch):
        context = ''
        for doc_id, text in zip(batch['doc_id'], batch['text']):
            context += f"Document [{doc_id}]: {text}.\n"
        return batch['query_text'].unique()[0], context
    
    ## select only the top-5 docs
    dataset = run.groupby('qid').apply(lambda group: prepare_context(group), include_groups=False)
    dataset = pd.DataFrame(dataset.tolist(), index=dataset.index,  columns=['query_text', 'context']).reset_index()
    dataset = pd.merge(
        dataset,
        qrels[['qid', 'answers', 'text', 'idx_gold_in_corpus']].rename(columns={'answers': 'golden_answer', 'text': 'golden_text'}),
        on='qid',
        how='left'
    )


    # Load the Large Language Model (LLM) and tokenizer
    model_name = models2name[args.llm]
    READER_LLM, tokenizer, model = load_model_and_tokenizer(READER_MODEL_NAME = model_name,
                                                            load_in_8bit=args.load_in_8bit, 
                                                            load_in_4bit=args.load_in_4bit)
    
    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """Directly answer the question based on the context passage, no explanation is needed. If the context does not contain any evidence, output 'I cannot answer based on the given context.'""",
        },
        {
            "role": "user",
            "content": """Context:\n{context}\n\nQuestion: {question}""",
        },
        ]
    
    RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )



    outputs = []

    #################################### MODIFIED SOMETHING BY PIPPO #########################################################
    #datas =  dataset.iloc[960:].reset_index(drop=True)
    #####################################################################################################

    EMISSION_OUTPUT_PATH = f'{args.data_dir}/emission/{args.llm}/{args.split}'
    for row_id, sample in tqdm(dataset.iterrows(), desc='Processing samples', total=len(dataset)):

        profiler = FlopsProfiler(model)
        
        query_id = sample['qid']
        question = sample['query_text']
        context = sample['context']
        true_answer = sample['golden_answer']

        
        ## prompt
        final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)
        
        EMISSION_QUERY_PATH = os.path.join(EMISSION_OUTPUT_PATH, f'query_{query_id}')
        TRAIN_FLOPS_OUTPUT_PATH = os.path.join(EMISSION_QUERY_PATH, "train_flops.txt")

        if not os.path.isdir(EMISSION_QUERY_PATH):
            # Create model folder if it doesn't exist
            if not os.path.isdir(EMISSION_OUTPUT_PATH):
                os.mkdir(EMISSION_OUTPUT_PATH) 
            # Create experiment folder
            os.mkdir(EMISSION_QUERY_PATH)  

        total_tracker = EmissionsTracker(
                tracking_mode="process",
                log_level="critical",
                output_dir=EMISSION_QUERY_PATH,
                measure_power_secs=30,
                api_call_interval=4,
                experiment_id=EMISSION_QUERY_PATH,
            )
        profiler.output_dir = EMISSION_QUERY_PATH

        total_tracker.start()
        profiler.start_profile()
        answer = READER_LLM(final_prompt)[0]["generated_text"]
        total_emissions = total_tracker.stop()
        profiler.print_model_profile(
        output_file=TRAIN_FLOPS_OUTPUT_PATH,
        )
        profiler.stop_profile()
        del profiler


        deepseed_info1 = extract_info_from_file(TRAIN_FLOPS_OUTPUT_PATH)
        deepseed_info2 = pd.read_csv(EMISSION_QUERY_PATH + '/emissions.csv', 
                                     usecols=["duration", "emissions", "energy_consumed"]).to_dict('index')[0]

        outputs.append({
            'query_id': query_id,
            'answer': answer,
            'golden_answer': literal_eval(true_answer),
            'params_per_GPU': deepseed_info1['params_per_GPU'],
            'fwd_latency': deepseed_info1['fwd_latency'],
            **deepseed_info2, 
            'EM': is_answer_in_text(answer, literal_eval(true_answer))
        })
        
        if row_id % 100 == 0:
            print(outputs[-1])

    with open(f'{args.data_dir}/rag_emission_data/experiment_{args.llm}.pickle', 'wb') as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)