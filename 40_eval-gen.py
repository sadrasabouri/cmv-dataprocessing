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

WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)
CACHE_DIR = os.environ.get("MY_HF_CACHE", '.cache')

# load model
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LEARNING_RATE = 5e-5
BATCH_SIZE = 1
NUM_EPOCHS = 3
wandb.login(key=WANDB_API_KEY)

#############################

def model_response_gen(model, tokenizer, prompts) -> List[str]:
    results = []
    for prompt in tqdm(prompts, desc="Running Inference", unit="prompt"):
        messages = [{"role": "user", "content": prompt},]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        model_response = tokenizer.decode(response, skip_special_tokens=True)
        print(f"Prompt: {prompt}\nModel Response: {model_response}\n{'-'*40}", flush=True)
        results.append(model_response)
    return results


def model_loss(model, tokenizer, prompts, chosen_list, rejected_list) -> List[Dict]:
    results = []
    model.eval()

    for prompt, chosen, rejected in tqdm(zip(prompts, chosen_list, rejected_list), desc="Calculating Losses", total=len(prompts)):
        losses = {}
        for label, answer in [("chosen", chosen), ("rejected", rejected)]:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer}
            ]
            full_input = tokenizer.apply_chat_template(messages,
                                                       return_tensors="pt",
                                                       add_generation_prompt=False).to(model.device)
            prompt_messages = [{"role": "user", "content": prompt}]
            prompt_input = tokenizer.apply_chat_template(prompt_messages,
                                                         return_tensors="pt",
                                                         add_generation_prompt=True).to(model.device)
            prompt_len = prompt_input.shape[1]
            labels = full_input.clone()
            labels[:, :prompt_len] = nn.CrossEntropyLoss().ignore_index  
            
            with torch.no_grad():
                outputs = model(full_input, labels=labels)
                loss = outputs.loss.item()
                losses[f"{label}_loss"] = loss
        
        print('='*20, flush=True)
        print(f"Chosen Loss: {losses['chosen_loss']}, Rejected Loss: {losses['rejected_loss']}\n{'-'*40}", flush=True)
        results.append({
            "chosen_loss": losses["chosen_loss"],
            "rejected_loss": losses["rejected_loss"],
            "loss_margin": losses["rejected_loss"] - losses["chosen_loss"],
            "correct": losses["rejected_loss"] > losses["chosen_loss"] # Positive margin means model prefers 'chosen'
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
