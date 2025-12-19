"""Training script for multimodal fusion model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dementia_project.data.io import load_metadata, load_splits
from dementia_project.features.text_features import (
    TextEmbedConfig,
    load_text_model,
)
from dementia_project.features.wav2vec2_embed import (
    Wav2Vec2EmbedConfig,
    load_wav2vec2,
)
from dementia_project.models.fusion_model import MultimodalFusionClassifier
from dementia_project.train.fusion_dataset import MultimodalFusionDataset
from dementia_project.viz.metrics import save_confusion_matrix_png


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    word_audio_list, text_emb_list, labels = zip(*batch)

    # Pad word-level audio embeddings to same length
    max_words = max(a.shape[0] for a in word_audio_list)
    audio_dim = word_audio_list[0].shape[1]

    padded_audio = []
    for audio_seq in word_audio_list:
        pad_length = max_words - audio_seq.shape[0]
        if pad_length > 0:
            padding = torch.zeros(pad_length, audio_dim)
            padded = torch.cat([audio_seq, padding], dim=0)
        else:
            padded = audio_seq
        padded_audio.append(padded)

    # Stack
    word_audio_batch = torch.stack(padded_audio)  # [batch, max_words, audio_dim]
    text_emb_batch = torch.stack(text_emb_list)  # [batch, 1, text_dim]
    labels_batch = torch.stack(labels)  # [batch]

    return word_audio_batch, text_emb_batch, labels_batch


def eval_model(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float | list]:
    """Evaluate model on a data loader."""
    model.eval()
    all_probs: list[float] = []
    all_preds: list[int] = []
    all_y: list[int] = []

    with torch.no_grad():
        for word_audio, text_emb, y in loader:
            word_audio = word_audio.to(device)
            text_emb = text_emb.to(device)
            y = y.to(device)

            logits = model(word_audio, text_emb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()

            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_y.extend(y.detach().cpu().numpy().tolist())

    y_true = np.array(all_y, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)
    y_prob = np.array(all_probs, dtype=np.float32)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["roc_auc"] = None
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return out


def sanitize_for_json(obj):
    """Convert NaN/Inf to None for JSON serialization."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True, type=Path)
    parser.add_argument("--splits_csv", required=True, type=Path)
    parser.add_argument("--word_segments_csv", required=True, type=Path)
    parser.add_argument("--asr_manifest_csv", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--text_model", default="roberta-base")
    parser.add_argument("--audio_model", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--max_words_per_sample", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Load data
    metadata_df = load_metadata(args.metadata_csv)
    splits_df = load_splits(args.splits_csv)
    word_segments_df = pd.read_csv(args.word_segments_csv)
    asr_manifest_df = pd.read_csv(args.asr_manifest_csv)

    # Merge to get labels and splits for word segments
    word_segments_df = word_segments_df.merge(
        metadata_df[["audio_path", "label"]], on="audio_path", how="left"
    )
    word_segments_df = word_segments_df.merge(
        splits_df[["audio_path", "split"]], on="audio_path", how="left"
    )

    # Filter by split
    train_words = word_segments_df[word_segments_df["split"] == "train"]
    valid_words = word_segments_df[word_segments_df["split"] == "valid"]
    test_words = word_segments_df[word_segments_df["split"] == "test"]

    if args.limit is not None:
        # Limit by unique audio files
        train_audio = train_words["audio_path"].unique()[: args.limit]
        train_words = train_words[train_words["audio_path"].isin(train_audio)]

    print(f"Train: {train_words['audio_path'].nunique()} audio files")
    print(f"Valid: {valid_words['audio_path'].nunique()} audio files")
    print(f"Test: {test_words['audio_path'].nunique()} audio files")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load encoders
    text_cfg = TextEmbedConfig(model_name=args.text_model, max_length=512)
    audio_cfg = Wav2Vec2EmbedConfig(model_name=args.audio_model, max_audio_sec=2.0)

    print("Loading text encoder...")
    text_model, text_tokenizer = load_text_model(text_cfg, device)
    print("Loading audio encoder...")
    audio_model, audio_feature_extractor = load_wav2vec2(audio_cfg, device)

    # Create datasets
    train_dataset = MultimodalFusionDataset(
        train_words,
        asr_manifest_df,
        text_cfg,
        audio_cfg,
        text_model,
        text_tokenizer,
        audio_model,
        audio_feature_extractor,
        device,
        max_words_per_sample=int(args.max_words_per_sample),
    )
    valid_dataset = MultimodalFusionDataset(
        valid_words,
        asr_manifest_df,
        text_cfg,
        audio_cfg,
        text_model,
        text_tokenizer,
        audio_model,
        audio_feature_extractor,
        device,
        max_words_per_sample=int(args.max_words_per_sample),
    )
    test_dataset = MultimodalFusionDataset(
        test_words,
        asr_manifest_df,
        text_cfg,
        audio_cfg,
        text_model,
        text_tokenizer,
        audio_model,
        audio_feature_extractor,
        device,
        max_words_per_sample=int(args.max_words_per_sample),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Create model
    model = MultimodalFusionClassifier(
        text_encoder_dim=768, audio_encoder_dim=768, hidden_dim=256, num_heads=4
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    # Training loop
    print("Starting training...")
    for epoch in range(int(args.epochs)):
        model.train()
        train_loss = 0.0
        for word_audio, text_emb, y in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"
        ):
            word_audio = word_audio.to(device)
            text_emb = text_emb.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(word_audio, text_emb)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}")

        # Validate
        valid_metrics = eval_model(model, valid_loader, device)
        print(
            f"Valid: acc={valid_metrics['accuracy']:.4f}, "
            f"f1={valid_metrics['f1']:.4f}, "
            f"roc_auc={valid_metrics.get('roc_auc', 'N/A')}"
        )

    # Final evaluation
    print("Evaluating on test set...")
    test_metrics = eval_model(model, test_loader, device)
    train_metrics = eval_model(model, train_loader, device)
    valid_metrics = eval_model(model, valid_loader, device)

    # Save results
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "train": train_metrics,
        "valid": valid_metrics,
        "test": test_metrics,
        "n": len(test_dataset),
    }
    metrics_sanitized = sanitize_for_json(metrics)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics_sanitized, indent=2))

    # Save confusion matrix
    if len(test_metrics["confusion_matrix"]) > 0:
        cm_test = np.array(test_metrics["confusion_matrix"])
        save_confusion_matrix_png(
            cm=cm_test,
            labels=["No Dementia", "Dementia"],
            out_path=args.out_dir / "confusion_matrix_test.png",
            title="Fusion Model (Test Set)",
        )

    print(f"Wrote metrics to: {args.out_dir / 'metrics.json'}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test ROC-AUC: {test_metrics.get('roc_auc', 'N/A')}")


if __name__ == "__main__":
    main()
