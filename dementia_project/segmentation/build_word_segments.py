"""CLI to build word-level segmentation manifest from ASR word timestamps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dementia_project.segmentation.word_segments import (
    WordSegmentConfig,
    build_word_segment_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True, type=Path)
    parser.add_argument("--splits_csv", required=True, type=Path)
    parser.add_argument("--asr_manifest_csv", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--min_word_duration_sec", type=float, default=0.05)
    parser.add_argument("--max_word_duration_sec", type=float, default=2.0)
    args = parser.parse_args()

    metadata_df = pd.read_csv(args.metadata_csv)
    splits_df = pd.read_csv(args.splits_csv)
    asr_manifest_df = pd.read_csv(args.asr_manifest_csv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = WordSegmentConfig(
        min_word_duration_sec=float(args.min_word_duration_sec),
        max_word_duration_sec=float(args.max_word_duration_sec),
    )
    word_df = build_word_segment_manifest(metadata_df, splits_df, asr_manifest_df, cfg)
    out_path = args.out_dir / "word_segments.csv"
    word_df.to_csv(out_path, index=False)

    # Generate report
    report = {
        "total_segments": int(len(word_df)),
        "unique_audio_files": int(word_df["audio_path"].nunique()),
        "unique_words": int(word_df["word"].nunique()),
        "avg_segments_per_audio": (
            float(len(word_df) / word_df["audio_path"].nunique())
            if len(word_df) > 0
            else 0.0
        ),
        "min_word_duration_sec": float(args.min_word_duration_sec),
        "max_word_duration_sec": float(args.max_word_duration_sec),
    }
    report_path = args.out_dir / "word_segments_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"Wrote: {out_path} ({len(word_df)} word segments)")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
