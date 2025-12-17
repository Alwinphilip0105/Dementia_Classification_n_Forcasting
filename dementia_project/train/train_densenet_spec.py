"""DenseNet baseline on log-mel spectrograms."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import densenet121
from tqdm import tqdm

from dementia_project.data.io import load_metadata, load_splits
from dementia_project.features.spectrograms import MelSpecConfig, load_mono_resampled, log_mel_spectrogram
from dementia_project.viz.metrics import save_confusion_matrix_png


def sanitize_for_json(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    epochs: int = 3
    lr: float = 1e-4


class SpecDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: MelSpecConfig):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav = load_mono_resampled(str(row["audio_path"]), self.cfg.sample_rate_hz)
        spec = log_mel_spectrogram(wav, self.cfg)  # (M, T)
        # Normalize per-sample
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        # Make 3-channel image-like tensor: (3, M, T)
        x = spec.unsqueeze(0).repeat(3, 1, 1)
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, y


def eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_probs: list[float] = []
    all_preds: list[int] = []
    all_y: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
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
        "f1": float(f1_score(y_true, y_pred)),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True, type=Path)
    parser.add_argument("--splits_csv", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_audio_sec", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    metadata_df = load_metadata(args.metadata_csv)
    splits_df = load_splits(args.splits_csv)
    df = metadata_df.merge(splits_df[["audio_path", "split"]], on="audio_path", how="inner")
    if args.limit is not None and args.limit > 0 and args.limit < len(df):
        df = df.sample(n=args.limit, random_state=1337).reset_index(drop=True)

    cfg = MelSpecConfig(max_audio_sec=float(args.max_audio_sec))
    train_cfg = TrainConfig(batch_size=int(args.batch_size), epochs=int(args.epochs))

    train_df = df[df["split"] == "train"]
    valid_df = df[df["split"] == "valid"]
    test_df = df[df["split"] == "test"]

    train_loader = DataLoader(SpecDataset(train_df, cfg), batch_size=train_cfg.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(SpecDataset(valid_df, cfg), batch_size=train_cfg.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(SpecDataset(test_df, cfg), batch_size=train_cfg.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=train_cfg.lr)

    for epoch in range(train_cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.detach().cpu().item()))

    metrics = {
        "train": eval_model(model, train_loader, device),
        "valid": eval_model(model, valid_loader, device),
        "test": eval_model(model, test_loader, device),
        "n": int(len(df)),
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(sanitize_for_json(metrics), indent=2, allow_nan=False))

    cm = np.array(metrics["test"]["confusion_matrix"], dtype=np.int64)
    save_confusion_matrix_png(
        cm=cm,
        labels=["no_dementia", "dementia"],
        out_path=out_dir / "confusion_matrix_test.png",
        title="DenseNet spectrogram baseline (test)",
    )

    print(f"Wrote metrics to: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()


