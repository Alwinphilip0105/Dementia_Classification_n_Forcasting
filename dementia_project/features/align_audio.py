"""Word-level audio alignment: Map Whisper timestamps to Wav2Vec2 embeddings.

This script:
1. Loads full audio through Wav2Vec2 to get hidden states [Time, 768]
2. Loads Whisper word-level timestamps from words.json
3. Maps each word to corresponding Wav2Vec2 frames
4. Average-pools frames for each word
5. Saves as .pt dictionary {word_idx: tensor[768], ...}

Key insight: We slice EMBEDDINGS not audio, preserving acoustic context.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoFeatureExtractor, Wav2Vec2Model


@dataclass(frozen=True)
class AlignConfig:
    """Configuration for word-level audio alignment."""

    model_name: str = "facebook/wav2vec2-base-960h"
    target_sample_rate: int = 16_000
    wav2vec2_frame_rate: int = 50  # Wav2Vec2 outputs at 50Hz (320x downsampling)


def load_mono_resampled(audio_path: Path, target_sr: int) -> torch.Tensor:
    """Load audio file as mono and resample to target sample rate.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate in Hz

    Returns:
        Audio waveform tensor [samples]
    """
    wav, sr = torchaudio.load(str(audio_path))

    # Convert to mono if stereo
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)

    # Resample if needed
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)

    return wav


def load_word_timestamps(words_json_path: Path) -> list[dict]:
    """Load word-level timestamps from Whisper ASR output.

    Args:
        words_json_path: Path to words.json file

    Returns:
        List of dicts with keys: word, start, end
    """
    with open(words_json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("words", [])


@torch.no_grad()
def extract_wav2vec2_hidden_states(
    audio_path: Path,
    cfg: AlignConfig,
    model: Wav2Vec2Model,
    feature_extractor,
    device: torch.device,
) -> torch.Tensor:
    """Extract Wav2Vec2 hidden states for full audio file.

    Args:
        audio_path: Path to audio file
        cfg: Alignment configuration
        model: Pre-loaded Wav2Vec2 model
        feature_extractor: Pre-loaded feature extractor
        device: Device to run inference on

    Returns:
        Hidden states tensor [num_frames, 768]
    """
    # Load audio
    wav = load_mono_resampled(audio_path, cfg.target_sample_rate)
    wav = wav.to(device)

    # Extract features and get hidden states
    inputs = feature_extractor(
        wav.cpu().numpy(), sampling_rate=cfg.target_sample_rate, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get last hidden state from Wav2Vec2
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.last_hidden_state.squeeze(0)  # [num_frames, 768]

    return hidden_states


def align_word_to_frames(
    word_info: dict, hidden_states: torch.Tensor, frame_rate: int
) -> torch.Tensor | None:
    """Align a single word to Wav2Vec2 frames and average-pool.

    Args:
        word_info: Dict with keys: word, start, end (timestamps in seconds)
        hidden_states: Wav2Vec2 hidden states [num_frames, 768]
        frame_rate: Frame rate of hidden states (Hz)

    Returns:
        Average-pooled embedding [768] or None if no valid frames
    """
    start_sec = word_info["start"]
    end_sec = word_info["end"]

    # Convert timestamps to frame indices
    start_frame = int(start_sec * frame_rate)
    end_frame = int(end_sec * frame_rate)

    # Clamp to valid range
    num_frames = hidden_states.shape[0]
    start_frame = max(0, min(start_frame, num_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, num_frames))

    # Slice and average
    word_frames = hidden_states[start_frame:end_frame, :]  # [n_frames, 768]

    if word_frames.shape[0] == 0:
        return None

    word_embedding = word_frames.mean(dim=0)  # [768]

    return word_embedding


def align_audio_file(
    audio_path: Path,
    words_json_path: Path,
    cfg: AlignConfig,
    model: Wav2Vec2Model,
    feature_extractor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Process one audio file and align all words to embeddings.

    Args:
        audio_path: Path to audio file
        words_json_path: Path to words.json with timestamps
        cfg: Alignment configuration
        model: Pre-loaded Wav2Vec2 model
        feature_extractor: Pre-loaded feature extractor
        device: Device to run inference on

    Returns:
        Dictionary mapping word_idx -> embedding tensor [768]
        Also includes metadata keys: 'audio_path', 'num_words'
    """
    # Extract full audio hidden states
    hidden_states = extract_wav2vec2_hidden_states(
        audio_path, cfg, model, feature_extractor, device
    )

    # Load word timestamps
    words = load_word_timestamps(words_json_path)

    # Align each word
    word_embeddings = {}

    for idx, word_info in enumerate(words):
        emb = align_word_to_frames(word_info, hidden_states, cfg.wav2vec2_frame_rate)

        if emb is not None:
            word_embeddings[f"word_{idx}"] = emb.cpu()

    # Add metadata
    word_embeddings["audio_path"] = str(audio_path)
    word_embeddings["num_words"] = len(words)

    return word_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Align Whisper word timestamps to Wav2Vec2 embeddings"
    )
    parser.add_argument(
        "--asr_manifest_csv",
        required=True,
        type=Path,
        help="Path to ASR manifest CSV with audio_path, words_json columns",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=Path,
        help="Output directory for aligned embeddings (.pt files)",
    )
    parser.add_argument(
        "--model_name",
        default="facebook/wav2vec2-base-960h",
        help="Wav2Vec2 model name",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of files for testing"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and feature extractor
    print(f"Loading model: {args.model_name}...")
    model = Wav2Vec2Model.from_pretrained(args.model_name).to(device)
    model.eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)

    cfg = AlignConfig(model_name=args.model_name)

    # Load manifest
    print(f"Loading ASR manifest from {args.asr_manifest_csv}...")
    manifest_df = pd.read_csv(args.asr_manifest_csv)

    if args.limit is not None and args.limit > 0:
        manifest_df = manifest_df.head(args.limit)

    print(f"Processing {len(manifest_df)} audio files...")

    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Process each file
    successful = 0
    failed = 0

    for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Aligning"):
        audio_path = Path(row["audio_path"])
        words_json_path = Path(row["words_json"])

        # Check files exist
        if not audio_path.exists():
            print(f"Warning: Audio file not found: {audio_path}")
            failed += 1
            continue

        if not words_json_path.exists():
            print(f"Warning: Words JSON not found: {words_json_path}")
            failed += 1
            continue

        try:
            # Align words to embeddings
            word_embeddings = align_audio_file(
                audio_path, words_json_path, cfg, model, feature_extractor, device
            )

            # Save to .pt file (use audio_path as unique identifier)
            # Create safe filename from audio_path
            safe_name = str(audio_path).replace("/", "__").replace(" ", "_")
            out_path = args.out_dir / f"{safe_name}.pt"

            torch.save(word_embeddings, out_path)
            successful += 1

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            failed += 1

    print(f"\nAlignment complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
