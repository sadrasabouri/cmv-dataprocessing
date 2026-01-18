"""Odds ratio based delta arguablity score"""

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


def compute_batch_log_likelihood(prompt, terms, model, tokenizer):
    # We pad to the left or right; for Causal LM loss, right padding is standard
    tokenizer.pad_token = tokenizer.eos_token
    full_texts = [prompt + f"OP:\n{term}" for term in terms]
    
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True).to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_enc = tokenizer(prompt + "OP:\n", add_special_tokens=False)
    prompt_len = len(prompt_enc.input_ids)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # Shape: [batch_size, sequence_length, vocab_size]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size()) # Shape: [batch_size, seq_len - 1]

    mask = torch.ones_like(shift_labels)
    mask[:, :prompt_len-1] = 0 # Mask prompt
    mask[attention_mask[:, 1:] == 0] = 0 # Mask padding
    
    per_sequence_log_prob = -(loss * mask).sum(dim=1)
    return per_sequence_log_prob.mean().item(), per_sequence_log_prob.std().item()


def main():
    parser = argparse.ArgumentParser(description="Process a file")
    parser.add_argument('data_path', type=str, help='path to the test dataset')
    parser.add_argument('output', type=str, help='name of inference output file')
    
    args = parser.parse_args()
    
    # Initialize the Table object with columns
    columns = ['prompt', 'log_ratio', 'std_log_ratio', 'predicted', 'actual']

    data_path = args.data_path
    output_file = args.output
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", cache_dir=CACHE_DIR)
    model.eval()

    results_list = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(tqdm(f, desc="Inference ...")):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            prompt = format_text(data)
            
            a_score, a_std = compute_batch_log_likelihood(prompt, AGREEMENT_TERMS, model, tokenizer)
            d_score, d_std = compute_batch_log_likelihood(prompt, DISAGREEMENT_TERMS, model, tokenizer)
            
            lratio = a_score - d_score
            std = np.sqrt(a_std**2 + d_std**2)
            predicted = lratio > 0
            actual = data['is_op_delta']

            row_data = [prompt, lratio, std, predicted, actual]
            results_list.append(row_data)
            print(lratio, std, predicted, actual, flush=True)


    # Summary metrics calculation
    results_df = pd.DataFrame.from_records(results_list, columns=columns)
    results_df.to_csv(output_file, index=False)

    accuracy = (results_df['predicted'] == results_df['actual']).mean()
    true_positives = ((results_df['predicted'] == True) & (results_df['actual'] == True)).sum()
    false_positives = ((results_df['predicted'] == True) & (results_df['actual'] == False)).sum()
    false_negatives = ((results_df['predicted'] == False) & (results_df['actual'] == True)).sum()
    
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    main()
