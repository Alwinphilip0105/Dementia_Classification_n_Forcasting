"""Text feature extraction from ASR transcripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TextConfig:
    """Configuration for text feature extraction."""

    max_length: int = 512
    model_name: str = "distilbert-base-uncased"


def load_transcript(transcript_json_path: Path) -> str:
    """Load transcript text from ASR output.

    Args:
        transcript_json_path: Path to transcript.json file

    Returns:
        Full transcript text as a string
    """
    with open(transcript_json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("text", "")


def build_text_dataframe(
    metadata_df: pd.DataFrame, asr_manifest_df: pd.DataFrame
) -> pd.DataFrame:
    """Build a dataframe with audio_path, label, split, and transcript text.

    Args:
        metadata_df: Metadata with audio_path, label
        asr_manifest_df: ASR manifest with audio_path, transcript_json

    Returns:
        DataFrame with columns: audio_path, label, text
    """
    # Merge metadata with ASR manifest
    df = metadata_df.merge(
        asr_manifest_df[["audio_path", "transcript_json"]], on="audio_path", how="inner"
    )

    # Load transcript text for each file
    texts = []
    for _, row in df.iterrows():
        try:
            text = load_transcript(Path(row["transcript_json"]))
            texts.append(text)
        except Exception as e:
            print(f"Warning: Failed to load transcript for {row['audio_path']}: {e}")
            texts.append("")

    df["text"] = texts

    # Filter out empty transcripts
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    return df[["audio_path", "label", "text"]]
