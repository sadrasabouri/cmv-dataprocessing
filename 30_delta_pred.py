"""BERT Delta Classifier (Streaming Version)"""

import json
import torch
import argparse
import numpy as np
import wandb

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def count_samples(file_path):
    """Cheap line count (no JSON parsing)."""
    with open(file_path) as f:
        return sum(1 for _ in f)


def format_text(example):
    """Format post title and conversation into single text."""
    conv = " ".join(
        f"{author}: {msg}"
        for msg, author in zip(
            example["conversation"],
            example["conversation_authors"],
        )
    )
    return {
        "text": f"Title: {example['post_title']}\nConversation: {conv}",
        "label": 1 if example["is_op_delta"] else 0,
    }


def tokenize_fn(tokenizer, max_length):
    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    return _tokenize


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    auc = roc_auc_score(labels, probs)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }


# --------------------------------------------------
# Class weights
# --------------------------------------------------

def compute_class_weights_streaming(dataset, n):
    pos = 0
    for i, ex in enumerate(dataset):
        pos += ex["label"]
        if i + 1 >= n:
            break

    neg = n - pos
    return torch.tensor([
        n / (2 * neg),
        n / (2 * pos),
    ])


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


def train_model(
    train_file,
    output_dir,
    model_name="bert-base-uncased",
    val_split=0.2,
    epochs=3,
    batch_size=8,
    learning_rate=2e-5,
    max_length=512,
    use_class_weights=True,
    wandb_project="cmv-delta-prediction",
    seed=42,
):
    wandb.init(
        project=wandb_project,
        config=locals(),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Count once (cheap)
    total_samples = count_samples(train_file)
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size

    # Streaming dataset
    raw_ds = load_dataset(
        "json",
        data_files=train_file,
        split="train",
        streaming=True,
    )

    raw_ds = raw_ds.shuffle(seed=seed, buffer_size=10_000)

    ds = raw_ds.map(format_text)

    val_ds = ds.take(val_size)
    train_ds = ds.skip(val_size)

    train_ds = train_ds.map(
        tokenize_fn(tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )

    val_ds = val_ds.map(
        tokenize_fn(tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights_streaming(
            train_ds, train_size
        ).to(model.device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="wandb",
        seed=seed,
        fp16=torch.cuda.is_available(),
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer_cls = WeightedTrainer if use_class_weights else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    trainer.save_model(f"{output_dir}/best_model")
    tokenizer.save_pretrained(f"{output_dir}/best_model")

    wandb.finish()
    return model, tokenizer


def predict(model_path, texts, batch_size=8):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = []
    for i in range(0, len(texts), batch_size):
        inputs = tokenizer(
            texts[i:i + batch_size],
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        for p, prob in zip(preds, probs):
            results.append({
                "prediction": bool(p.item()),
                "prob_no_delta": prob[0].item(),
                "prob_delta": prob[1].item(),
                "confidence": prob[p].item(),
            })

    return results


def predict_single(model_path, post_title, conversation, conversation_authors):
    text = f"Title: {post_title}\nConversation: " + " ".join(
        f"{a}: {m}" for m, a in zip(conversation, conversation_authors)
    )
    return predict(model_path, [text])[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_path")
    parser.add_argument("model_name")
    args = parser.parse_args()

    train_model(
        train_file=args.train_data_path,
        output_dir=args.model_name,
    )


if __name__ == "__main__":
    main()
