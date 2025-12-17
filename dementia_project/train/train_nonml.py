"""Train/eval a non-ML baseline on hand-crafted audio features.

Run:
  poetry run python -m dementia_project.train.train_nonml ^
    --metadata_csv data/processed/metadata.csv ^
    --splits_csv data/processed/splits.csv ^
    --out_dir runs/nonml_baseline
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from dementia_project.data.io import load_metadata, load_splits
from dementia_project.features.audio_features import MfccConfig, extract_mfcc_pause_features
from dementia_project.viz.metrics import save_confusion_matrix_png


def _build_feature_table(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cfg = MfccConfig()
    feature_rows: list[dict[str, float]] = []
    labels: list[int] = []
    splits: list[str] = []

    for row in tqdm(df.to_dict(orient="records"), desc="Extracting MFCC features"):
        feats = extract_mfcc_pause_features(Path(row["audio_path"]), cfg)
        feature_rows.append(feats)
        labels.append(int(row["label"]))
        splits.append(str(row["split"]))

    X_df = pd.DataFrame(feature_rows).fillna(0.0)
    X = X_df.to_numpy(dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y, splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True, type=Path)
    parser.add_argument("--splits_csv", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of audio files to use (for smoke tests).",
    )
    args = parser.parse_args()

    metadata_df = load_metadata(args.metadata_csv)
    splits_df = load_splits(args.splits_csv)
    df = metadata_df.merge(splits_df[["audio_path", "split"]], on="audio_path", how="inner")

    if args.limit is not None and args.limit > 0 and args.limit < len(df):
        df = df.sample(n=args.limit, random_state=1337).reset_index(drop=True)

    X, y, split_list = _build_feature_table(df)

    train_mask = np.array([s == "train" for s in split_list])
    valid_mask = np.array([s == "valid" for s in split_list])
    test_mask = np.array([s == "test" for s in split_list])

    def sanitize_for_json(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize_for_json(v) for v in obj]
        return obj

    model: Pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=1337,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(X[train_mask], y[train_mask])

    def eval_split(mask: np.ndarray) -> dict[str, float]:
        probs = model.predict_proba(X[mask])[:, 1]
        preds = (probs >= 0.5).astype(int)
        out = {
            "accuracy": float(accuracy_score(y[mask], preds)),
            "f1": float(f1_score(y[mask], preds)),
        }
        # roc_auc requires both classes present
        if len(np.unique(y[mask])) == 2:
            out["roc_auc"] = float(roc_auc_score(y[mask], probs))
        else:
            out["roc_auc"] = float("nan")
        return out

    metrics = {
        "valid": eval_split(valid_mask),
        "test": eval_split(test_mask),
        "train": eval_split(train_mask),
        "n": int(len(df)),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(
        json.dumps(sanitize_for_json(metrics), indent=2, allow_nan=False)
    )

    # Confusion matrix on test
    test_probs = model.predict_proba(X[test_mask])[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)
    cm = confusion_matrix(y[test_mask], test_preds, labels=[0, 1])
    save_confusion_matrix_png(
        cm=cm,
        labels=["no_dementia", "dementia"],
        out_path=args.out_dir / "confusion_matrix_test.png",
        title="Non-ML baseline (test)",
    )

    print(f"Wrote metrics to: {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()


