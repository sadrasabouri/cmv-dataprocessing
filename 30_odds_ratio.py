"""Odds ratio based delta arability score"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
import torch.nn.functional as F
import argparse
import scipy
from torch.nn.utils.rnn import pad_sequence
import os
import tqdm
import json
import sys
from utils.params import AGREEMENT_TERMS, DISAGREEMENT_TERMS
from utils.functions import post_text_cleaning


CACHE_DIR = os.environ.get("MY_HF_CACHE", '.cache')
MODEL_NAME = "gpt2"

def compute_set_log_sum_exp(model, inputs_batch):
    with torch.no_grad():
        outputs = model(**inputs_batch, labels=inputs_batch["input_ids"])
        log_probs = -outputs.loss  # Negative loss gives log probabilities
    return torch.logsumexp(log_probs, dim=0).item()


def main():
    parser = argparse.ArgumentParser(description="Process a file")
    parser.add_argument('data_path', type=str, help='path to the test dataset')
    parser.add_argument('output', type=str, help='name of inference output file')
    
    args = parser.parse_args()
    
    data_path = args.data_path
    output_file = args.output
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", cache_dir=CACHE_DIR)
    model.eval()


    results = []
    with open(data_path, 'r') as f:
        for line in tqdm(f, desc="Loading delta ..."):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            post_title = data['post_title']
            post_text = post_text_cleaning(data['conversation'][0])
            prompt = f"{post_title}\n\n{post_text}"
            responses = '\n'.join(data['conversation'][1:])
            authors = data['conversation_authors']
            input_messages = [
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": a},
                ]
            full_input = tokenizer.apply_chat_template(input_messages, return_tensors="pt", padding=True, add_generation_prompt=False).to(model.device)


if __name__ == "__main__":
    main()
