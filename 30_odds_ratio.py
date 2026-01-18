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
from utils.functions import post_text_cleaning, has_non_in_conv
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


CACHE_DIR = os.environ.get("MY_HF_CACHE", '.cache')
# MODEL_NAME = "gpt2"
# MODEL_NAME = "EleutherAI/pythia-160m" 
# MODEL_NAME = "meta-llama/Llama-3.2-1B"
DELIM = "\n" + '-' * 10

def format_text(row):
    if has_non_in_conv(row):
        return None
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


def compute_batch_log_likelihood(prompt, terms, model, tokenizer, n_samples=10):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_length = getattr(model.config, "max_position_embeddings", 1024)

    max_prompt_len = max_length - 100
    prompt_ids = tokenizer.encode(prompt + "OP:\n", add_special_tokens=False)
    print(f"Initial prompt len: {len(prompt_ids)} --> cut down to {max_prompt_len}")
    prompt_ids = prompt_ids[-max_prompt_len:] # Truncate old context
    prompt_len = len(prompt_ids)
    
    all_input_ids = []
    for term in terms:
        term_ids = tokenizer.encode(term, add_special_tokens=False)
        combined = (prompt_ids + term_ids)[:max_length]
        all_input_ids.append(torch.tensor(combined))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        all_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    ).to(model.device)
    
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(model.device)

    sample_means = []    
    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size())

            mask = torch.zeros_like(shift_labels)
            mask[:, (prompt_len - 1):] = 1
            mask[shift_labels == tokenizer.pad_token_id] = 0    
            # Log prob for each sequence in the batch for THIS sample pass
            per_sequence_log_prob = -(loss * mask).sum(dim=1)
            sample_means.append(per_sequence_log_prob.mean().item())
    return np.mean(sample_means), np.std(sample_means)


def main():
    parser = argparse.ArgumentParser(description="Process a file")
    parser.add_argument('data_path', type=str, help='path to the test dataset')
    parser.add_argument('output', type=str, help='name of inference output file')
    parser.add_argument('--model_name', type=str, help='the model used for inference', default="EleutherAI/pythia-160m")
    
    args = parser.parse_args()
    
    # Initialize the Table object with columns
    columns = ['prompt', 'log_ratio', 'std_log_ratio', 'predicted', 'actual', 'prompt_len']

    data_path = args.data_path
    output_file = args.output
    model_name = args.model_name
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=CACHE_DIR)
    model.train() # for having dropout random effect

    results_list = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(tqdm(f, desc="Inference ...")):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            prompt = format_text(data)
            if prompt is None:
                continue
            a_score, a_std = compute_batch_log_likelihood(prompt, DISAGREEMENT_TERMS, model, tokenizer)
            d_score, d_std = compute_batch_log_likelihood(prompt, AGREEMENT_TERMS, model, tokenizer)
            
            lratio = a_score - d_score
            std = np.sqrt(a_std**2 + d_std**2)
            predicted = lratio > 0
            actual = data['is_op_delta']
            prompt_ntoken = len(tokenizer.encode(prompt, add_special_tokens=False))

            row_data = [prompt, lratio, std, predicted, actual, prompt_ntoken]
            results_list.append(row_data)
            print(lratio, std, predicted, actual, flush=True)


    # Summary metrics calculation
    results_df = pd.DataFrame.from_records(results_list, columns=columns)
    results_df.drop(['prompt']).to_csv(output_file, index=False)

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
