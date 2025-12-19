"""I/O helpers for metadata and split tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_metadata(metadata_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    if "audio_path" not in df.columns:
        raise ValueError("metadata.csv must contain 'audio_path'")
    if "label" not in df.columns:
        raise ValueError("metadata.csv must contain 'label'")
    return df


def load_splits(splits_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(splits_csv)
    if "audio_path" not in df.columns:
        raise ValueError("splits.csv must contain 'audio_path'")
    if "split" not in df.columns:
        raise ValueError("splits.csv must contain 'split'")
    return df
