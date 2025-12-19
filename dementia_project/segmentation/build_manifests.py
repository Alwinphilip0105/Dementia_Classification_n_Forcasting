"""CLI to build segmentation manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dementia_project.segmentation.time_windows import (
    WindowConfig,
    build_time_window_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True, type=Path)
    parser.add_argument("--splits_csv", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--window_sec", type=float, default=2.0)
    parser.add_argument("--hop_sec", type=float, default=0.5)
    args = parser.parse_args()

    metadata_df = pd.read_csv(args.metadata_csv)
    splits_df = pd.read_csv(args.splits_csv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = WindowConfig(window_sec=args.window_sec, hop_sec=args.hop_sec)
    time_df = build_time_window_manifest(metadata_df, splits_df, cfg)
    out_path = args.out_dir / "time_segments.csv"
    time_df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} ({len(time_df)} segments)")


if __name__ == "__main__":
    main()
