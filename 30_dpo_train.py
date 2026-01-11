import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
import argparse 
import wandb
import json
import os
from tqdm import tqdm

WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)
CACHE_DIR = os.environ.get("MY_HF_CACHE", '.cache')

# load model
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LEARNING_RATE = 5e-5
BATCH_SIZE = 1
NUM_EPOCHS = 3
wandb.login(key=WANDB_API_KEY)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model loaded on %s" % device)


def load_data(path_to_data: str) -> Dataset:
    """
    Load the given data into a dataset object.

    :param path_to_data: path to the data
    """
    prompt_list = []
    chosen_list = []
    rejected_list = []
    with open(path_to_data, 'r') as f:
        for line in tqdm(f, desc="Loading delta ..."):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            prompt_list.append(data['prompt'])
            chosen_list.append(data['chosen'])
            rejected_list.append(data['rejected'])

    dataset = Dataset.from_dict({
        'prompt': prompt_list,
        'chosen': chosen_list,
        'rejected': rejected_list})
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Process a file")
    
    parser.add_argument('train_data_path', type=str, help='the preference data (training)')
    parser.add_argument('dev_data_path', type=str, help='the preference data (dev)')
    parser.add_argument('model_name', type=str, help='name of model output file')

    args = parser.parse_args()
    
    train_data_path = args.train_data_path
    dev_data_path = args.dev_data_path
    model_name = args.model_name

    # TODO: play around with config
    training_args = DPOConfig(
        output_dir="llama",
        logging_steps=10,
        per_device_train_batch_size=1,
        eval_steps=50,
        save_only_model=True,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        gradient_checkpointing=True,
        report_to="wandb",
        project="cmv-rlhf",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        run_name=model_name,
    )

    # TODO: play around with config
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=load_data(train_data_path),
        eval_dataset=load_data(dev_data_path),
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    dpo_trainer.train()
    dpo_trainer.save_model(model_name)


if __name__ == "__main__":
    main()
