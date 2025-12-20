"""Word-level segmentation manifest generation from ASR word timestamps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class WordSegmentConfig:
    """Configuration for word-level segmentation."""

    min_word_duration_sec: float = 0.05  # Filter very short words
    max_word_duration_sec: float = 2.0  # Filter very long words (likely ASR errors)


def load_words_json(words_json: Path) -> list[dict[str, float | str]]:
    """Load word timestamps from a words.json file.

    Args:
        words_json: Path to words.json file.

    Returns:
        List of word dicts with keys: word, start, end.
    """
    data = json.loads(words_json.read_text(encoding="utf-8"))
    words = data.get("words", [])
    return words


def build_word_segment_manifest(
    metadata_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    asr_manifest_df: pd.DataFrame,
    cfg: WordSegmentConfig,
) -> pd.DataFrame:
    """Build a per-word segment manifest from ASR word timestamps.

    Expects:
    - metadata_df columns: audio_path, label, person_name (optional)
    - splits_df columns: audio_path, split
    - asr_manifest_df columns: audio_path, words_json

    Args:
        metadata_df: Metadata with audio paths and labels.
        splits_df: Split assignments.
        asr_manifest_df: ASR manifest with words.json paths.
        cfg: Configuration for filtering words.

    Returns:
        DataFrame with columns: audio_path, word, word_index, start_sec, end_sec,
        label, split, person_name (if available).
    """
    df = metadata_df.merge(
        splits_df[["audio_path", "split"]], on="audio_path", how="left"
    )
    df = df.merge(
        asr_manifest_df[["audio_path", "words_json"]],
        on="audio_path",
        how="inner",
    )

    # Filter out rows without words.json
    df = df[df["words_json"].notna()].copy()
    df["words_json"] = df["words_json"].astype(str)

    segments: list[dict[str, object]] = []

    for row in df.to_dict(orient="records"):
        words_json_path = Path(str(row["words_json"]))
        if not words_json_path.exists():
            continue

        try:
            words = load_words_json(words_json_path)
        except Exception:
            continue

        for word_idx, word_dict in enumerate(words):
            word_text = str(word_dict.get("word", "")).strip()
            start = word_dict.get("start")
            end = word_dict.get("end")

            if not word_text or start is None or end is None:
                continue

            start_sec = float(start)
            end_sec = float(end)
            duration = end_sec - start_sec

            # Filter by duration
            if duration < cfg.min_word_duration_sec:
                continue
            if duration > cfg.max_word_duration_sec:
                continue

            segment = {
                "audio_path": row["audio_path"],
                "word": word_text,
                "word_index": int(word_idx),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "label": int(row["label"]),
                "split": row.get("split"),
            }

            # Add person_name if available
            if "person_name" in row and row["person_name"] is not None:
                segment["person_name"] = str(row["person_name"])

            segments.append(segment)

    return pd.DataFrame(segments)
