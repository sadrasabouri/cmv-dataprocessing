"""BERT Delta Classifier"""
# TODO: if memory issue persist we do relational dataset
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import json
import argparse
import wandb
import numpy as np


BASE_MODEL = 'bert-base-uncased'
BATCH_SIZE = 16
EPOCHS = 3


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
        # TODO: clean the message
        if author not in author_map:
            author_map[author] = chr(counter)
            counter += 1
        text += f"{author_map[author]}: {message}\n"
    
    return text


# class DeltaDataset(Dataset):
#     def __init__(self, df, tokenizer, max_length=512):
#         self.labels = df['labels']
#         self.pid = df['post_id']
#         self.cid = df['comment_id']
#         self.conversation_ids = df['conversation_ids']
#         self.tokenizer = tokenizer
#         self.max_length = max_length
    
#     def __len__(self):
#         return len(self.labels)
    
#     def __getitem__(self, idx):
#         format_text()
#         encoding = self.tokenizer(
#             self.texts[idx],
#             truncation=True,
#             padding='max_length',
#             max_length=self.max_length,
#         )
#         item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
#         item["labels"] = torch.tensor(self.labels[idx])
#         return item
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'labels': torch.tensor(self.labels[idx], dtype=torch.long)
#         }

# # NEW; should merge
# class DeltaDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_length=512):
#         self.encodings = tokenizer(
#             texts,
#             truncation=True,
#             padding=True,
#             max_length=max_length,
#         )
#         self.labels = labels

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
#         item["labels"] = torch.tensor(self.labels[idx])
#         return item

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

    parser.add_argument('data_path', type=str, help='the cmv_delta-rel.jsonl file')
    parser.add_argument('model_name', type=str, help='name of model output file')
    parser.add_argument('--seed', type=int, help='random generation seed', default=42)
    parser.add_argument('--use-cache', action="store_true", help='flag indicating use of cache')

    args = parser.parse_args()

    data_path = args.data_path
    model_name = args.model_name

    df = load_data(data_path)
    df['text'] = df.apply(format_text, axis=1)
    df['label'] = df['is_op_delta'].astype(int)

    # train_df, val_df = train_test_split(df, test_size=0.2, random_state=args.seed)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    model = BertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2, device_map="auto")

    # train_dataset = DeltaDataset(train_df, tokenizer)
    # val_dataset = DeltaDataset(val_df, tokenizer)
    train_dataset = DeltaDataset(train_texts, train_labels, tokenizer)
    val_dataset = DeltaDataset(val_texts, val_labels, tokenizer)

    training_args = TrainingArguments(
        output_dir=model_name,
        # --- training ---
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        weight_decay=0.01,
        # --- logging (LOSS!) ---
        logging_strategy="steps",
        logging_steps=50,
        logging_first_step=True,
        # --- evaluation ---
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        # --- UX / infra ---
        report_to="wandb",
        project="cmv-clf",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(model_name)
    tokenizer.save_pretrained(model_name)


if __name__ == "__main__":
    main()
