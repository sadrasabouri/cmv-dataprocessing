"""BERT Delta Classifier"""

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
import wandb
import argparse


def load_data(file_path):
    """Load JSONL data"""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def format_text(example):
    """Format post title and conversation into single text"""
    # TODO: fix the effect of author name
    conv = " ".join([
        f"{author}: {msg}"
        for msg, author in zip(example['conversation'], example['conversation_authors'])
    ])
    return f"Title: {example['post_title']}\nConversation: {conv}"


def preprocess_data(data, tokenizer, max_length=512):
    """Convert data to HuggingFace Dataset format"""
    formatted_data = {
        'text': [format_text(ex) for ex in data],
        'label': [1 if ex['is_op_delta'] else 0 for ex in data]
    }
    
    dataset = Dataset.from_dict(formatted_data)
    
    def tokenize(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding=False  # Dynamic padding handled by DataCollator
        )
    
    return dataset.map(tokenize, batched=True, remove_columns=['text'])


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    auc = roc_auc_score(labels, probs)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def compute_class_weights(labels):
    """Compute class weights for imbalanced dataset"""
    pos = sum(labels)
    neg = len(labels) - pos
    return torch.tensor([len(labels)/(2*neg), len(labels)/(2*pos)])


class WeightedTrainer(Trainer):
    """Custom Trainer with class weights"""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


def train_model(
    train_file,
    dev_file,
    output_dir,
    model_name='bert-base-uncased',
    epochs=3,
    batch_size=8,
    learning_rate=2e-5,
    max_length=512,
    use_class_weights=True,
    wandb_project='cmv-delta-prediction',
    seed=42
):
    """
    Train BERT classifier
    
    Args:
        train_file: Path to JSONL data
        model_name: HuggingFace model name
        output_dir: Directory to save model
        val_split: Validation split ratio
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_length: Max sequence length
        use_class_weights: Use weighted loss
        wandb_project: W&B project name
        seed: Random seed
    """
    
    # Initialize W&B
    wandb.init(
        project=wandb_project,
        config={
            'model': model_name,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'max_length': max_length,
            'use_class_weights': use_class_weights
        }
    )
    
    # Load data
    print("Loading data...")
    train_data = load_data(train_file)
    val_data = load_data(train_file)
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Positive rate - Train: {sum(d['is_op_delta'] for d in train_data)/len(train_data):.2%}, "
          f"Val: {sum(d['is_op_delta'] for d in val_data)/len(val_data):.2%}")
    
    # Load tokenizer and prepare datasets
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = preprocess_data(train_data, tokenizer, max_length)
    val_dataset = preprocess_data(val_data, tokenizer, max_length)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    
    # Compute class weights
    class_weights = None
    if use_class_weights:
        labels = [ex['label'] for ex in train_dataset]
        class_weights = compute_class_weights(labels).to(model.device)
        print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        report_to='wandb',
        seed=seed,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    )
    
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Initialize trainer
    trainer_cls = WeightedTrainer if use_class_weights else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights if use_class_weights else None
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Final evaluation
    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    print(metrics)
    
    # Save model
    trainer.save_model(f"{output_dir}/best_model")
    tokenizer.save_pretrained(f"{output_dir}/best_model")
    
    wandb.finish()
    
    return model, tokenizer


def predict(model_path, texts, batch_size=8):
    """
    Make predictions on new texts
    
    Args:
        model_path: Path to saved model
        texts: List of formatted text strings
        batch_size: Batch size for inference
    
    Returns:
        List of predictions with probabilities
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        
        for pred, prob in zip(predictions, probs):
            results.append({
                'prediction': bool(pred.item()),
                'prob_no_delta': prob[0].item(),
                'prob_delta': prob[1].item(),
                'confidence': prob[pred].item()
            })
    
    return results


def predict_single(model_path, post_title, conversation, conversation_authors):
    """Predict for a single example"""
    text = f"Post: {post_title} Conversation: " + " ".join([
        f"{author}: {msg}" 
        for msg, author in zip(conversation, conversation_authors)
    ])
    return predict(model_path, [text])[0]


def main():
    parser = argparse.ArgumentParser(description="Process a file")
    
    parser.add_argument('train_data_path', type=str, help='the preference data (training)')
    parser.add_argument('dev_data_path', type=str, help='the preference data (dev)')
    parser.add_argument('model_name', type=str, help='name of model output file')

    args = parser.parse_args()

    train_data_path = args.train_data_path
    dev_data_path = args.dev_data_path
    output_model_name = args.model_name

    # Train
    model, tokenizer = train_model(
        train_file=train_data_path,
        dev_file=dev_data_path,
        model_name='bert-base-uncased',
        output_dir=output_model_name,
        epochs=3,
        batch_size=8,
        learning_rate=2e-5,
        wandb_project='cmv-delta-prediction'
    )
    
    # Example prediction
    result = predict_single(
        model_path=f'{output_model_name}/best_model',
        post_title="I believe nihilism is a fundamentally flawed philosophy. CMV",
        conversation=[
            "I believe that the idea that life has no meaning is entirely ridiculous.",
            "Actually all nihilists need to do is reject the idea that happiness is meaningful."
        ],
        conversation_authors=["rubywoundz", "Eh_Priori"]
    )
    
    print(f"\nPrediction: {'DELTA' if result['prediction'] else 'NO DELTA'}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"P(Delta): {result['prob_delta']:.2%}")


if __name__ == '__main__':
    main()
