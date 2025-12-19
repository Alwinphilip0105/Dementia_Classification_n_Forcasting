"""Text-only baseline: Transformer embeddings -> sklearn classifier.

This extracts text embeddings from ASR transcripts using a pretrained Transformer
(e.g., RoBERTa) and trains a Logistic Regression classifier on top.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from dementia_project.data.io import load_metadata, load_splits
from dementia_project.features.text_features import (
    TextEmbedConfig,
    embed_file_mean_pool,
    load_text_model,
)
from dementia_project.viz.metrics import save_confusion_matrix_png


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
    parser.add_argument("--asr_manifest_csv", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Load data
    metadata_df = load_metadata(args.metadata_csv)
    splits_df = load_splits(args.splits_csv)
    asr_manifest_df = pd.read_csv(args.asr_manifest_csv)

    # Merge all dataframes
    df = metadata_df.merge(
        splits_df[["audio_path", "split"]], on="audio_path", how="inner"
    )
    df = df.merge(
        asr_manifest_df[["audio_path", "transcript_json"]],
        on="audio_path",
        how="inner",
    )

    # Filter out rows without transcripts
    df = df[df["transcript_json"].notna()].copy()
    df["transcript_json"] = df["transcript_json"].astype(str)

    if args.limit is not None and args.limit > 0 and args.limit < len(df):
        df = df.sample(n=args.limit, random_state=1337).reset_index(drop=True)

    print(f"Processing {len(df)} samples with transcripts")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TextEmbedConfig(
        model_name=args.model_name, max_length=int(args.max_length), device=str(device)
    )
    model, tokenizer = load_text_model(cfg, device)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    split_list: list[str] = []

    for row in tqdm(df.to_dict(orient="records"), desc="Extracting text embeddings"):
        transcript_path = Path(str(row["transcript_json"]))
        if not transcript_path.exists():
            print(f"Warning: transcript not found: {transcript_path}")
            continue

        try:
            emb = embed_file_mean_pool(
                transcript_json=transcript_path,
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
            X_list.append(emb)
            y_list.append(int(row["label"]))
            split_list.append(str(row["split"]))
        except Exception as e:
            print(f"Error processing {transcript_path}: {e}")
            continue

    if not X_list:
        raise ValueError("No valid embeddings extracted")

    X = np.vstack(X_list)
    y = np.array(y_list)
    splits = np.array(split_list)

    print(f"Extracted embeddings: shape={X.shape}, dtype={X.dtype}")

    # Train classifier
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000, random_state=1337)),
        ]
    )

    train_mask = splits == "train"
    valid_mask = splits == "valid"
    test_mask = splits == "test"

    X_train, y_train = X[train_mask], y[train_mask]
    X_valid, y_valid = X[valid_mask], y[valid_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")

    clf.fit(X_train, y_train)

    # Evaluate
    metrics: dict[str, dict[str, float | list]] = {}

    for split_name, X_split, y_split in [
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
        ("test", X_test, y_test),
    ]:
        if len(X_split) == 0:
            metrics[split_name] = {
                "accuracy": None,
                "f1": None,
                "roc_auc": None,
            }
            continue

        y_pred = clf.predict(X_split)
        y_proba = clf.predict_proba(X_split)[:, 1]

        acc = accuracy_score(y_split, y_pred)
        f1 = f1_score(y_split, y_pred, zero_division=0)
        try:
            roc_auc = roc_auc_score(y_split, y_proba)
        except ValueError:
            roc_auc = None

        cm = confusion_matrix(y_split, y_pred).tolist()

        metrics[split_name] = {
            "accuracy": float(acc),
            "f1": float(f1),
            "roc_auc": float(roc_auc) if roc_auc is not None else None,
            "confusion_matrix": cm,
        }

    metrics["n"] = len(X)

    # Save outputs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_sanitized = sanitize_for_json(metrics)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics_sanitized, indent=2))

    # Save confusion matrix plot for test set
    if len(X_test) > 0:
        cm_test = np.array(metrics["test"]["confusion_matrix"])
        save_confusion_matrix_png(
            cm=cm_test,
            labels=["No Dementia", "Dementia"],
            out_path=args.out_dir / "confusion_matrix_test.png",
            title="Text Baseline (Test Set)",
        )

    print(f"Wrote metrics to: {args.out_dir / 'metrics.json'}")
    print(f"Test accuracy: {metrics['test'].get('accuracy', 'N/A')}")
    print(f"Test F1: {metrics['test'].get('f1', 'N/A')}")
    print(f"Test ROC-AUC: {metrics['test'].get('roc_auc', 'N/A')}")


if __name__ == "__main__":
    main()
