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
from tqdm import tqdm
import json
from utils.params import AGREEMENT_TERMS, DISAGREEMENT_TERMS
from utils.functions import post_text_cleaning


CACHE_DIR = os.environ.get("MY_HF_CACHE", '.cache')
MODEL_NAME = "gpt2"
DELIM = "\n" + '-' * 10

def format_text(row):
    author_map = {row['post_author']: 'OP'}
    counter = ord('A')
    
    text = f"{row['post_title']}" + DELIM
    for author, message in zip(row['conversation_authors'], row['conversation']):
        message = post_text_cleaning(message)
        if author not in author_map:
            author_map[author] = chr(counter)
            counter += 1
        text += f"{author_map[author]}:\n{message}" + DELIM
    return text


def _log_sum_exp(model, inputs_batch):
    with torch.no_grad():
        outputs = model(**inputs_batch, labels=inputs_batch["input_ids"])
        log_probs = -outputs.loss
    return torch.logsumexp(log_probs, dim=0).item()


def compute_score(prompt, postfixes, model, tokenizer, n_samples=100):
    input_batches = []
    for pf in postfixes:
        input_prompt = prompt + f"OP:\n{pf}"
        inputs = tokenizer(input_prompt, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
        input_batches.append(inputs)
    input_ids = pad_sequence([inp['input_ids'].squeeze(0) for inp in input_batches], batch_first=True)
    attention_mask = pad_sequence([inp['attention_mask'].squeeze(0) for inp in input_batches], batch_first=True)
    inputs_batch = {'input_ids': input_ids, 'attention_mask': attention_mask}
    
    results = []
    for _ in range(n_samples):
        results.append(_log_sum_exp(inputs_batch))

    return np.mean(results), scipy.stats.sem(results)


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
        for line in tqdm(f, desc="Inference ..."):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            prompt = format_text(data)
            a_score, a_std = compute_score(prompt, AGREEMENT_TERMS, model, tokenizer)
            d_score, d_std = compute_score(prompt, DISAGREEMENT_TERMS, model, tokenizer)
            lratio = a_score - d_score
            std = np.sqrt(a_std**2 + d_std**2)
            predicted = lratio > 0
            results.append([prompt, lratio, std, predicted, data['is_op_delta']])
    results = pd.DataFrame.from_records(results,
                                        columns=['prompt', 'log_ratio', 'std_log_ratio',
                                                 'predicted', 'actual'])
    # Save results csv
    results.to_csv(output_file, index=False)

    # report accuracy/ other metrics
    accuracy = (results['predicted'] == results['actual']).mean()
    print(f"Accuracy: {accuracy:.4f}")
    true_positives = ((results['predicted'] == True) & (results['actual'] == True)).sum()
    false_positives = ((results['predicted'] == True) & (results['actual'] == False)).sum()
    false_negatives = ((results['predicted'] == False) & (results['actual'] == True)).sum()
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    main()
