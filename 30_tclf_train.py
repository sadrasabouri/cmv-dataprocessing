"""Transformer-base Delta Classifier"""
import pandas as pd
import torch
import json
import argparse
import wandb
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoProcessor
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType
from utils.functions import post_text_cleaning
from utils.callbacks import StopOnZeroLossCallback, SampleLoggingCallback


def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def format_text(row):
    author_map = {row['post_author']: 'OP'}
    counter = ord('A')
    
    text = f"Title: {row['post_title']}\n\n"
    
    for author, message in zip(row['conversation_authors'], row['conversation']):
        message = post_text_cleaning(message)
        if author not in author_map:
            author_map[author] = chr(counter)
            counter += 1
        text += f"{author_map[author]}: {message}\n"
    
    return text

class DeltaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=2048):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False, # Collator will handle padding
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision = np.sum((preds == 1) & (labels == 1)) / (np.sum(preds == 1) + 1e-8)
    recall = np.sum((preds == 1) & (labels == 1)) / (np.sum(labels == 1) + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return {"accuracy": (preds == labels).mean(), "f1": f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('output_model', type=str)
    parser.add_argument('--model_name', default="meta-llama/Llama-3.2-1B")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()

    df = load_data(args.data_path)
    df['text'] = df.apply(format_text, axis=1)
    df['label'] = df['is_op_delta'].astype(int)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.1, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=2, 
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False, 
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset = DeltaDataset(train_texts, train_labels, tokenizer)
    val_dataset = DeltaDataset(val_texts, val_labels, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_model,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4, # Simulates a larger batch size
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        logging_strategy="epoch",
        logging_steps=0.01,
        logging_first_step=True,
        eval_strategy="epoch",
        eval_steps=0.1,
        save_strategy="epoch",
        save_steps=0.5,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        bf16=True,
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[StopOnZeroLossCallback,
                   SampleLoggingCallback("clf", eval_max_new_tokens=100)]
    )

    with wandb.init(project="cmv-tclf") as run:
        trainer.train()
    model.save_pretrained(args.output_model)

if __name__ == "__main__":
    main()