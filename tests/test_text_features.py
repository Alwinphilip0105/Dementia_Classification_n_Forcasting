"""Tests for text feature extraction."""

import json
import tempfile
from pathlib import Path

import pandas as pd

from dementia_project.features.text_features import (
    build_text_dataframe,
    load_transcript,
)


def test_load_transcript():
    """Test loading transcript from JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"text": "Hello world"}, f)
        temp_path = Path(f.name)

    text = load_transcript(temp_path)
    assert text == "Hello world"
    temp_path.unlink()


def test_build_text_dataframe():
    """Test building text dataframe."""
    # Create test data
    metadata_df = pd.DataFrame(
        {"audio_path": ["audio1.wav", "audio2.wav"], "label": [0, 1]}
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create transcript files
        transcript1 = Path(tmpdir) / "transcript1.json"
        transcript2 = Path(tmpdir) / "transcript2.json"
        json.dump({"text": "Hello"}, transcript1.open("w"))
        json.dump({"text": "World"}, transcript2.open("w"))

        asr_df = pd.DataFrame(
            {
                "audio_path": ["audio1.wav", "audio2.wav"],
                "transcript_json": [str(transcript1), str(transcript2)],
            }
        )

        result = build_text_dataframe(metadata_df, asr_df)
        assert len(result) == 2
        assert "text" in result.columns


def test_validate_text_input():
    """Test input validation function."""
    from dementia_project.features.text_features import validate_text_input

    assert validate_text_input("Hello", 512)
    assert not validate_text_input("", 512)
    assert not validate_text_input("   ", 512)
    assert not validate_text_input(123, 512)  # Not a string
    assert not validate_text_input("x" * 10000, 512)  # Too long
