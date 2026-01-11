"""BERT Delta Classifier"""
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import json
import argparse


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


def main():
    parser = argparse.ArgumentParser(description="Process a file")

    parser.add_argument('data_path', type=str, help='the cmv_delta.jsonl file')
    parser.add_argument('model_name', type=str, help='name of model output file')

    args = parser.parse_args()

    data_path = args.data_path
    model_name = args.model_name

    df = load_data(data_path)

    df['text'] = df.apply(format_text, axis=1)
    df['label'] = df['is_op_delta'].astype(int)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    train_dataset = DeltaDataset(train_texts, train_labels, tokenizer)
    val_dataset = DeltaDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Accuracy={correct/total:.4f}')

    model.save_pretrained(model_name)
    tokenizer.save_pretrained(model_name)


if __name__ == "__main__":
    main()
