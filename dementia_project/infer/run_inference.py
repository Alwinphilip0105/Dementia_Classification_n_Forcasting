"""CLI for running inference on audio files."""

from __future__ import annotations

import argparse
from pathlib import Path

from dementia_project.features.spectrograms import MelSpecConfig
from dementia_project.infer.predict import batch_predict, save_predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference on audio files using trained model"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--audio_paths",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to audio files to predict",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        required=True,
        help="Path to save predictions JSON",
    )
    parser.add_argument(
        "--max_audio_sec",
        type=float,
        default=10.0,
        help="Maximum audio duration in seconds",
    )
    args = parser.parse_args()

    cfg = MelSpecConfig(max_audio_sec=args.max_audio_sec)

    print(f"Loading model from: {args.checkpoint_path}")
    print(f"Processing {len(args.audio_paths)} audio files...")

    predictions = batch_predict(
        audio_paths=args.audio_paths,
        checkpoint_path=args.checkpoint_path,
        cfg=cfg,
    )

    save_predictions(predictions, args.output_json)

    # Print summary
    dementia_count = sum(1 for p in predictions if p["predicted_class"] == 1)
    print(f"\nSummary:")
    print(f"  Total files: {len(predictions)}")
    print(f"  Predicted dementia: {dementia_count}")
    print(f"  Predicted control: {len(predictions) - dementia_count}")


if __name__ == "__main__":
    main()
