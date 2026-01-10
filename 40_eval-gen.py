"""Evaluation module"""

from typing import List, Dict
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import os
import argparse 
import wandb
import sys
import random

WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)
CACHE_DIR = os.environ.get("MY_HF_CACHE", '.cache')
BATCH_SIZE = 1
wandb.login(key=WANDB_API_KEY)

#############################

def model_response_gen(model, tokenizer, prompts, batch_size: int = BATCH_SIZE) -> List[str]:
    results = []
    model.eval()

    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    for i in tqdm(range(0, len(prompts), batch_size), desc="Running Inference", unit="prompt"):
        batch_prompts = prompts[i:i + batch_size]
        messages_batch = [
            [{"role": "user", "content": p}]
            for p in batch_prompts
        ]
        input_ids = tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True
        ).to(model.device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
        for j in range(len(batch_prompts)):
            prompt_len = attention_mask[j].sum()
            response_ids = outputs[j][prompt_len:]
            model_response = tokenizer.decode(
                response_ids,
                skip_special_tokens=True
            )
            results.append(model_response)
            if j == 0:
                print(f"> Prompt: {batch_prompts[j][:200]}\n>Model Response: {model_response[:200]}\n"f"{'-'*40}", flush=True)
    return results


def model_loss(model, tokenizer, prompts, chosen_list, rejected_list, batch_size: int = BATCH_SIZE) -> List[Dict]:
    results = []
    model.eval()
    ignore_index = -100  # Standard ignore index for CrossEntropyLoss

    for i in tqdm(range(0, len(prompts), batch_size), desc="Calculating Losses", unit="prompt"):
        batch_prompts = prompts[i:i + batch_size]
        batch_chosen = chosen_list[i:i + batch_size]
        batch_rejected = rejected_list[i:i + batch_size]
        batch_losses = {}

        for label, answers in [("chosen", batch_chosen),
                               ("rejected", batch_rejected)]:
            messages_batch = [
                [
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": a},
                ]
                for p, a in zip(batch_prompts, answers)
            ]
            full_input = tokenizer.apply_chat_template(messages_batch, return_tensors="pt", padding=True, add_generation_prompt=False).to(model.device)
            prompt_only = tokenizer.apply_chat_template([[{"role": "user", "content": p}] for p in batch_prompts], return_tensors="pt", padding=True, add_generation_prompt=True).to(model.device)
            prompt_lens = prompt_only.ne(tokenizer.pad_token_id).sum(dim=1)

            labels = full_input.clone()
            for idx, plen in enumerate(prompt_lens):
                labels[idx, :plen] = ignore_index

            with torch.no_grad():
                outputs = model(full_input)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(shift_labels.size(0), -1)
                loss_per_sample = loss.sum(dim=1) / (shift_labels != ignore_index).sum(dim=1).float()
                loss_per_sample = loss_per_sample.detach().cpu()

            batch_losses[label] = loss_per_sample

        for j in range(len(batch_prompts)):
            chosen_loss = batch_losses["chosen"][j].item()
            rejected_loss = batch_losses["rejected"][j].item()

            print('=' * 20, flush=True)
            print(f"Chosen Loss: {chosen_loss}, Rejected Loss: {rejected_loss}\n", flush=True)
            results.append({
                "chosen_loss": chosen_loss,
                "rejected_loss": rejected_loss,
                "loss_margin": rejected_loss - chosen_loss,
                "correct": rejected_loss > chosen_loss
            })

    return results


def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Process a file")
    parser.add_argument('model_name', type=str, help='name of model output file')
    parser.add_argument('output', type=str, help='name of inference output file')
    parser.add_argument('--test-dataset', type=str, help='path to the test dataset.', default='data/dpo_data_test.jsonl')
    parser.add_argument('--base-model', type=str, help="the base model to compare against.", default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    
    # Parse the arguments
    args = parser.parse_args()
    output_file = args.output
    model_name = args.model_name
    test_dataset = pd.read_json(args.test_dataset, lines=True, orient='records')

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Model loaded on %s" % device)
    test_dataset["model_response"] = model_response_gen(model, tokenizer, [prompt for prompt in test_dataset["prompt"]])
    loss_df = pd.DataFrame(model_loss(model,
                                      tokenizer,
                                      prompts=[prompt for prompt in test_dataset["prompt"]],
                                      chosen_list=[prompt for prompt in test_dataset["chosen"]],
                                      rejected_list=[prompt for prompt in test_dataset["rejected"]]))
    loss_df.rename({x: f'model_{x}' for x in loss_df.columns})
    test_dataset = pd.concat([test_dataset, loss_df], axis=1)


    if args.base_model:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=CACHE_DIR)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.base_model, cache_dir=CACHE_DIR)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print("Model loaded on %s" % device)
        test_dataset["base_response"] = model_response_gen(model, tokenizer, [prompt for prompt in test_dataset["prompt"]])
        loss_df = pd.DataFrame(model_loss(model,
                                        tokenizer,
                                        prompts=[prompt for prompt in test_dataset["prompt"]],
                                        chosen_list=[prompt for prompt in test_dataset["chosen"]],
                                        rejected_list=[prompt for prompt in test_dataset["rejected"]]))
        loss_df.rename({x: f'base_{x}' for x in loss_df.columns})
        test_dataset = pd.concat([test_dataset, loss_df], axis=1)

    test_dataset.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
