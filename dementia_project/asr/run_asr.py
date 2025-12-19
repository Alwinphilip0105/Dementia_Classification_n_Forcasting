"""CLI to run ASR for a set of audio files defined by metadata/splits.

This runs Whisper via transformers and writes cached outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from dementia_project.asr.transcribe import (
    save_asr_result,
    transcribe_with_whisper_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="openai/whisper-tiny")
    parser.add_argument(
        "--device", type=str, default=None, help="cuda or cpu (default: auto)"
    )
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--task", type=str, default="transcribe")
    parser.add_argument("--chunk_length_s", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)
    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit)

    manifest_rows: list[dict[str, str]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="ASR files"):
        audio_path = Path(str(row["audio_path"]))
        audio_id = audio_path.as_posix().replace("/", "__").replace(":", "")
        out = args.out_dir / audio_id
        transcript_path = out / "transcript.json"
        words_path = out / "words.json"

        if transcript_path.exists() and not args.overwrite:
            manifest_rows.append(
                {
                    "audio_path": str(audio_path.as_posix()),
                    "asr_dir": str(out.as_posix()),
                    "transcript_json": str(transcript_path.as_posix()),
                    "words_json": str(words_path.as_posix())
                    if words_path.exists()
                    else "",
                }
            )
            continue

        result = transcribe_with_whisper_pipeline(
            audio_path=audio_path,
            model_name=args.model_name,
            device=args.device,
            language=args.language,
            task=args.task,
            chunk_length_s=int(args.chunk_length_s),
        )
        save_asr_result(out, result)

        manifest_rows.append(
            {
                "audio_path": str(audio_path.as_posix()),
                "asr_dir": str(out.as_posix()),
                "transcript_json": str(transcript_path.as_posix()),
                "words_json": str(words_path.as_posix()) if words_path.exists() else "",
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "asr_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(
        manifest_path, index=False, quoting=csv.QUOTE_MINIMAL
    )
    report = {
        "rows": int(len(manifest_rows)),
        "model_name": args.model_name,
        "language": args.language,
        "task": args.task,
        "chunk_length_s": int(args.chunk_length_s),
    }
    (args.out_dir / "asr_report.json").write_text(json.dumps(report, indent=2))

    print(f"Wrote ASR outputs to: {args.out_dir}")
    print(f"Wrote ASR manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
