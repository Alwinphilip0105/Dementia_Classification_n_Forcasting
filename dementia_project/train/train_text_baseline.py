"""Text-only baseline: This is a distilled Bert Model"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from dementia_project.data.io import load_metadata, load_splits
from dementia_project.features.text_features import build_text_dataframe
from dementia_project.viz.metrics import save_confusion_matrix_png


def validate_text_input(text: str, max_length: int) -> bool:
    """Validate text input before processing."""
    if not isinstance(text, str):
        return False
    if len(text.strip()) == 0:
        return False
    if len(text) > max_length * 10:  # Reasonable upper bound
        return False
    return True


def sanitize_for_json(obj):
    """Helper function to recursively sanitize an object for JSON serialization.

    Replaces NaN and infinite float values with None, and recursively
    processes nested dictionaries and lists.

    Args:
        obj: Object to sanitize (can be any type).

    Returns:
        Sanitized object with NaN/inf values replaced by None.
    """
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


class TextDataset(Dataset):
    """Dataset for text classification."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class TextClassifier(nn.Module):
    """Transformer-based text classifier."""

    def __init__(self, model_name: str, num_classes: int = 2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def train_epoch(model, loader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def eval_model(model, loader, device):
    """Evaluate model on a dataset."""
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()

            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    y_true = np.array(all_labels, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)
    y_prob = np.array(all_probs, dtype=np.float32)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")

    metrics["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).tolist()

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True, type=Path)
    parser.add_argument("--splits_csv", required=True, type=Path)
    parser.add_argument("--asr_manifest_csv", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import random

    torch.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1337)

    metadata_df = load_metadata(args.metadata_csv)
    splits_df = load_splits(args.splits_csv)
    asr_manifest_df = pd.read_csv(args.asr_manifest_csv)

    # df has columns (audio_path, label, text) where text is the the transcript as a string
    text_df = build_text_dataframe(metadata_df, asr_manifest_df)

    # split this into train, test, validate df's
    df = text_df.merge(splits_df[["audio_path", "split"]], on="audio_path", how="inner")

    if args.limit is not None and args.limit > 0 and args.limit < len(df):
        df = df.sample(n=args.limit, random_state=1337).reset_index(drop=True)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    valid_df = df[df["split"] == "valid"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Create datasets and dataloaders
    train_dataset = TextDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        args.max_length,
    )
    valid_dataset = TextDataset(
        valid_df["text"].tolist(),
        valid_df["label"].tolist(),
        tokenizer,
        args.max_length,
    )
    test_dataset = TextDataset(
        test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, args.max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextClassifier(args.model_name, num_classes=2).to(device)

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Train loss: {train_loss:.4f}")

        # Evaluate on validation set
        valid_metrics = eval_model(model, valid_loader, device)
        print(
            f"Valid - Acc: {valid_metrics['accuracy']:.4f}, F1: {valid_metrics['f1']:.4f}"
        )

    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Save model checkpoint
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": args.model_name,
            "num_classes": 2,
        },
        args.out_dir / "model.pth",
    )

    # Save config
    config_dict = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

    # Final evaluation
    metrics = {
        "train": eval_model(model, train_loader, device),
        "valid": eval_model(model, valid_loader, device),
        "test": eval_model(model, test_loader, device),
        "n": int(len(df)),
    }

    # Save results
    (args.out_dir / "metrics.json").write_text(
        json.dumps(sanitize_for_json(metrics), indent=2, allow_nan=False)
    )

    # Save confusion matrix
    cm = np.array(metrics["test"]["confusion_matrix"], dtype=np.int64)
    save_confusion_matrix_png(
        cm=cm,
        labels=["no_dementia", "dementia"],
        out_path=args.out_dir / "confusion_matrix_test.png",
        title="Text-only baseline (test)",
    )

    print(f"\nWrote metrics to: {args.out_dir / 'metrics.json'}")
    print(f"Test accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"Test F1: {metrics['test']['f1']:.4f}")


if __name__ == "__main__":
    main()
