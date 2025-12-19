"""Build `data/processed/splits.csv` from `metadata.csv` with hybrid subject-level logic.

Run:
    python -m dementia_project.data.build_splits --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dementia_project.data.splitting import build_hybrid_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)
    df2 = build_hybrid_splits(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits_path = args.out_dir / "splits.csv"
    df2[["audio_path", "person_name_norm", "label", "split"]].to_csv(
        splits_path, index=False
    )

    report = {
        "rows": int(len(df2)),
        "split_counts": df2["split"].value_counts().to_dict(),
        "label_counts": df2["label"].value_counts().to_dict(),
        "split_by_label": (
            df2.groupby(["split", "label"])
            .size()
            .rename("n")
            .reset_index()
            .to_dict("records")
        ),
        "num_subjects": int(df2["person_name_norm"].nunique()),
    }
    (args.out_dir / "splits_report.json").write_text(json.dumps(report, indent=2))

    print(f"Wrote: {splits_path}")


if __name__ == "__main__":
    main()
