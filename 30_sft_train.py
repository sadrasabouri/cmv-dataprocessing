import torch
import json
import os
from tqdm import tqdm
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import wandb

WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)
CACHE_DIR = os.environ.get("MY_HF_CACHE", '.cache')

# --- Configuration ---
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LEARNING_RATE = 2e-5  # SFT usually uses lower LR than DPO
BATCH_SIZE = 4        # Can be higher for SFT than DPO
MAX_SEQ_LENGTH = 2048
wandb.login(key=WANDB_API_KEY)

def load_sft_data(path_to_data: str) -> Dataset:
    """
    Loads data and returns a dataset with 'prompt' and 'completion' columns.
    SFTTrainer will automatically apply the chat template to these.
    """
    data_list = []
    with open(path_to_data, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # SFT only needs the prompt and the "gold" answer
                data_list.append({
                    "prompt": item['prompt'],
                    "completion": item['chosen'] # Treat 'chosen' as the target
                })
    return Dataset.from_list(data_list)

def main():
    parser = argparse.ArgumentParser(description="Process a file")
    
    parser.add_argument('train_data_path', type=str, help='the preference data (training)')
    parser.add_argument('dev_data_path', type=str, help='the preference data (dev)')
    parser.add_argument('model_name', type=str, help='name of model output file')

    args = parser.parse_args()
    
    train_data_path = args.train_data_path
    dev_data_path = args.dev_data_path
    model_name = args.model_name

    # 1. Quantization for memory efficiency
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    # )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    # Llama 3.1 specific: Ensure padding is on the right for SFT
    tokenizer.padding_side = "right" 

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        # quantization_config=bnb_config,
        device_map="auto",
        # torch_dtype=torch.bfloat16,
    )

    # 2. Configure SFT specific arguments
    sft_config = SFTConfig(
        output_dir=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        completion_only_loss=True,
        num_train_epochs=3,
        logging_steps=10,
        bf16=True,
        packing=True, # Packs multiple samples into one sequence for 2x speedup
        # Important: Don't train on the prompt, only the assistant's response
        dataset_kwargs={
            "add_special_tokens": False,  # Managed by template
            "skip_prepare_dataset": False,
        }
    )

    # 3. LoRA Configuration
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=load_sft_data(train_data_path),
        eval_dataset=load_sft_data(dev_data_path),
        peft_config=peft_config,
    )

    with wandb.init(project="cmv-sft") as run:
        trainer.train()
    trainer.save_model(model_name+'final')

if __name__ == "__main__":
    main()
