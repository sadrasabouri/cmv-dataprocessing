"""BERT Delta Classifier"""
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import json
import argparse
import wandb
import numpy as np
from utils.functions import post_text_cleaning
from utils.callbacks import StopOnZeroLossCallback


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
    def __init__(self, texts, labels, tokenizer, max_length=512):
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
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision = np.sum((preds == 1) & (labels == 1)) / (np.sum(preds == 1) + 1e-8)
    recall = np.sum((preds == 1) & (labels == 1)) / (np.sum(labels == 1) + 1e-8)
    return {
        "accuracy": (preds == labels).mean(),
        "precision": precision,
        "recall": recall,
        "f1": 2 * (precision * recall) / (precision + recall + 1e-8),
    }


def main():
    parser = argparse.ArgumentParser(description="Train a classifier for predicting delta")

    parser.add_argument('data_path', type=str, help='the cmv_delta.jsonl file')
    parser.add_argument('output_model', type=str, help='name of model output file')
    parser.add_argument('--model_name', type=str, help='name of base model', default="distilbert-base-uncased")
    parser.add_argument('--seed', type=int, help='random generation seed', default=42)
    parser.add_argument('--batch_size', type=int, help="batch size", default=32)
    parser.add_argument('--epochs', type=int, help="batch size", default=3)

    args = parser.parse_args()

    data_path = args.data_path
    output_model = args.output_model
    base_model = args.model_name

    df = load_data(data_path)
    df['text'] = df.apply(format_text, axis=1)
    df['label'] = df['is_op_delta'].astype(int)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )
    print(f"[size] Train: {len(train_labels)}, Test: {len(val_labels)}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2, device_map="auto")
    max_length = model.config.max_position_embeddings
    print(f"Model loaded with max length: {max_length}")
    if max_length >= 1024: # 2048 for better coverage
        max_length = 1024

    train_dataset = DeltaDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = DeltaDataset(val_texts, val_labels, tokenizer, max_length)

    training_args = TrainingArguments(
        output_dir=output_model,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
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
        report_to="wandb",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[StopOnZeroLossCallback]
    )

    with wandb.init(project="cmv-clf") as run:
        trainer.train()
    trainer.save_model(output_model)
    tokenizer.save_pretrained(output_model)


if __name__ == "__main__":
    main()
