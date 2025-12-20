"""Train multimodal fusion model with frozen encoders.

UPDATED VERSION: Uses pre-computed word-level audio embeddings for efficiency.

This script:
1. Loads pre-computed word-level audio embeddings from align_audio.py
2. Uses frozen DistilBERT for text encoding
3. Trains cross-attention fusion layer + classifier
4. Evaluates on train/valid/test splits
5. Saves model, metrics, and config
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoTokenizer

from dementia_project.data.io import load_metadata, load_splits
from dementia_project.models.fusion_model import FusionClassifier
from dementia_project.train.fusion_dataset import build_fusion_dataloaders
from dementia_project.viz.metrics import save_confusion_matrix_png


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


def train_epoch(model, loader, optimizer, loss_fn, device):
    """Train for one epoch.

    Args:
        model: Fusion model
        loader: Training dataloader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to train on

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training"):
        text_input_ids = batch["text_input_ids"].to(device)
        text_attention_mask = batch["text_attention_mask"].to(device)
        audio_embeddings = batch["audio_embeddings"].to(device)
        audio_mask = batch["audio_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            audio_embeddings=audio_embeddings,
            audio_mask=audio_mask,
        )

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def eval_model(model, loader, device):
    """Evaluate model on a dataset.

    Args:
        model: Fusion model
        loader: Evaluation dataloader
        device: Device to evaluate on

    Returns:
        Dictionary with metrics: accuracy, f1, roc_auc, confusion_matrix
    """
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            text_input_ids = batch["text_input_ids"].to(device)
            text_attention_mask = batch["text_attention_mask"].to(device)
            audio_embeddings = batch["audio_embeddings"].to(device)
            audio_mask = batch["audio_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                audio_embeddings=audio_embeddings,
                audio_mask=audio_mask,
            )

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

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train multimodal fusion model with frozen encoders"
    )
    parser.add_argument("--metadata_csv", required=True, type=Path)
    parser.add_argument("--splits_csv", required=True, type=Path)
    parser.add_argument("--asr_manifest_csv", required=True, type=Path)
    parser.add_argument("--word_embed_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--text_model_name", default="distilbert-base-uncased")
    parser.add_argument("--max_text_length", type=int, default=512)
    parser.add_argument("--max_audio_words", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze_text_encoder", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Set random seeds
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1337)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load metadata
    print("Loading metadata...")
    metadata_df = load_metadata(args.metadata_csv)
    splits_df = load_splits(args.splits_csv)
    asr_manifest_df = pd.read_csv(args.asr_manifest_csv)

    # Load tokenizer
    print(f"Loading tokenizer: {args.text_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)

    # Build dataloaders
    print("Building fusion dataloaders...")
    dataloaders = build_fusion_dataloaders(
        metadata_df=metadata_df,
        splits_df=splits_df,
        asr_manifest_df=asr_manifest_df,
        word_embed_dir=args.word_embed_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_text_length=args.max_text_length,
        max_audio_words=args.max_audio_words,
    )

    train_loader = dataloaders["train"]
    valid_loader = dataloaders["valid"]
    test_loader = dataloaders["test"]

    print(f"Train: {dataloaders['train_size']} samples")
    print(f"Valid: {dataloaders['valid_size']} samples")
    print(f"Test: {dataloaders['test_size']} samples")

    # Initialize model
    print("\nInitializing fusion model...")
    model = FusionClassifier(
        text_model_name=args.text_model_name,
        text_dim=768,
        audio_dim=768,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_classes=2,
        dropout=args.dropout,
        freeze_text_encoder=args.freeze_text_encoder,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    # Training setup
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_valid_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Train loss: {train_loss:.4f}")

        # Evaluate on validation set
        valid_metrics = eval_model(model, valid_loader, device)
        print(
            f"Valid - Acc: {valid_metrics['accuracy']:.4f}, "
            f"F1: {valid_metrics['f1']:.4f}, "
            f"ROC-AUC: {valid_metrics.get('roc_auc', 'N/A')}"
        )

        # Save best model
        if valid_metrics["accuracy"] > best_valid_acc:
            best_valid_acc = valid_metrics["accuracy"]
            print(f"New best validation accuracy: {best_valid_acc:.4f}")

    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Save model checkpoint
    print("\nSaving model checkpoint...")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "text_model_name": args.text_model_name,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "freeze_text_encoder": args.freeze_text_encoder,
        },
        args.out_dir / "model.pth",
    )

    # Save config
    config_dict = {
        "text_model_name": args.text_model_name,
        "max_text_length": args.max_text_length,
        "max_audio_words": args.max_audio_words,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "freeze_text_encoder": args.freeze_text_encoder,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

    # Final evaluation on all splits
    print("\nFinal evaluation...")
    metrics = {
        "train": eval_model(model, train_loader, device),
        "valid": eval_model(model, valid_loader, device),
        "test": eval_model(model, test_loader, device),
        "n": len(metadata_df),
    }

    # Print final metrics
    for split in ["train", "valid", "test"]:
        m = metrics[split]
        print(f"\n{split.upper()}:")
        print(f"  Accuracy: {m['accuracy']:.4f}")
        print(f"  F1: {m['f1']:.4f}")
        print(f"  ROC-AUC: {m.get('roc_auc', 'N/A')}")

    # Save metrics
    (args.out_dir / "metrics.json").write_text(
        json.dumps(sanitize_for_json(metrics), indent=2, allow_nan=False)
    )

    # Save confusion matrices
    for split in ["train", "valid", "test"]:
        cm = np.array(metrics[split]["confusion_matrix"], dtype=np.int64)
        save_confusion_matrix_png(
            cm=cm,
            labels=["no_dementia", "dementia"],
            out_path=args.out_dir / f"confusion_matrix_{split}.png",
            title=f"Fusion Model ({split})",
        )

    print(f"\n✓ Training complete!")
    print(f"  Output directory: {args.out_dir}")
    print(f"  Test accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"  Test F1: {metrics['test']['f1']:.4f}")


if __name__ == "__main__":
    main()
