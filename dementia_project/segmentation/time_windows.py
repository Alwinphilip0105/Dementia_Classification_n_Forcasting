"""Time-window segmentation manifest generation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WindowConfig:
    window_sec: float = 2.0
    hop_sec: float = 0.5


def build_time_window_manifest(
    metadata_df: pd.DataFrame, splits_df: pd.DataFrame, cfg: WindowConfig
) -> pd.DataFrame:
    """Build a per-window segment manifest.

    Expects:
    - metadata_df columns: audio_path, label, duration_sec
    - splits_df columns: audio_path, split
    """
    df = metadata_df.merge(
        splits_df[["audio_path", "split"]], on="audio_path", how="left"
    )

    segments: list[dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        dur = row.get("duration_sec")
        if dur is None:
            continue
        duration_sec = float(dur)
        start = 0.0
        while start < duration_sec:
            end = min(start + cfg.window_sec, duration_sec)
            if end - start <= 0:
                break
            segments.append(
                {
                    "audio_path": row["audio_path"],
                    "label": int(row["label"]),
                    "split": row.get("split"),
                    "start_sec": float(start),
                    "end_sec": float(end),
                }
            )
            start += cfg.hop_sec
    return pd.DataFrame(segments)
