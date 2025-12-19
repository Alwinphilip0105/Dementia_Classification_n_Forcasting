"""Audio-only baseline: Wav2Vec2 embeddings -> sklearn classifier.

This avoids end-to-end fine-tuning (faster, simpler), while still leveraging a
strong pretrained speech representation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from dementia_project.data.io import load_metadata, load_splits
from dementia_project.features.wav2vec2_embed import (
    Wav2Vec2EmbedConfig,
    embed_file_mean_pool,
    load_wav2vec2,
)
from dementia_project.viz.metrics import save_confusion_matrix_png


def sanitize_for_json(obj):
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
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--model_name", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--max_audio_sec", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    metadata_df = load_metadata(args.metadata_csv)
    splits_df = load_splits(args.splits_csv)
    df = metadata_df.merge(
        splits_df[["audio_path", "split"]], on="audio_path", how="inner"
    )
    if args.limit is not None and args.limit > 0 and args.limit < len(df):
        df = df.sample(n=args.limit, random_state=1337).reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Wav2Vec2EmbedConfig(
        model_name=args.model_name, max_audio_sec=float(args.max_audio_sec)
    )
    model, feature_extractor = load_wav2vec2(cfg, device)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    split_list: list[str] = []

    for row in tqdm(
        df.to_dict(orient="records"), desc="Extracting Wav2Vec2 embeddings"
    ):
        emb = embed_file_mean_pool(
            path=Path(str(row["audio_path"])),
            cfg=cfg,
            model=model,
            feature_extractor=feature_extractor,
            device=device,
        )
        X_list.append(emb)
        y_list.append(int(row["label"]))
        split_list.append(str(row["split"]))

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    train_mask = np.array([s == "train" for s in split_list])
    valid_mask = np.array([s == "valid" for s in split_list])
    test_mask = np.array([s == "test" for s in split_list])

    clf: Pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=5000, class_weight="balanced", random_state=1337
                ),
            ),
        ]
    )
    clf.fit(X[train_mask], y[train_mask])

    def eval_split(mask: np.ndarray) -> dict[str, float]:
        probs = clf.predict_proba(X[mask])[:, 1]
        preds = (probs >= 0.5).astype(int)
        out = {
            "accuracy": float(accuracy_score(y[mask], preds)),
            "f1": float(f1_score(y[mask], preds)),
        }
        if len(np.unique(y[mask])) == 2:
            out["roc_auc"] = float(roc_auc_score(y[mask], probs))
        else:
            out["roc_auc"] = float("nan")
        return out

    metrics = {
        "train": eval_split(train_mask),
        "valid": eval_split(valid_mask),
        "test": eval_split(test_mask),
        "n": int(len(df)),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(
        json.dumps(sanitize_for_json(metrics), indent=2, allow_nan=False)
    )

    test_probs = clf.predict_proba(X[test_mask])[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)
    cm = confusion_matrix(y[test_mask], test_preds, labels=[0, 1])
    save_confusion_matrix_png(
        cm=cm,
        labels=["no_dementia", "dementia"],
        out_path=args.out_dir / "confusion_matrix_test.png",
        title="Wav2Vec2 embeddings baseline (test)",
    )

    print(f"Wrote metrics to: {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
