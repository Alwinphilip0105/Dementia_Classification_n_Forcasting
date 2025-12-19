"""Build `data/processed/metadata.csv` by joining audio files to the dementia CSV.

This module is designed to be run as:

    python -m dementia_project.data.build_metadata --help
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torchaudio

from dementia_project.data.name_normalize import normalize_person_name


@dataclass(frozen=True)
class AudioRow:
    audio_path: str
    label: int
    filename: str
    person_dir_name: str


def _iter_wavs(root_dir: Path, label: int) -> Iterable[AudioRow]:
    for path in sorted(root_dir.rglob("*.wav")):
        yield AudioRow(
            audio_path=str(path.as_posix()),
            label=label,
            filename=path.name,
            person_dir_name=path.parent.name,
        )


def _safe_audio_info(path: str) -> dict[str, Any]:
    """Best-effort audio header info. Never throws; records errors instead."""
    try:
        info = torchaudio.info(path)
        num_frames = int(info.num_frames)
        sample_rate_hz = int(info.sample_rate)
        duration_sec = (
            float(num_frames) / float(sample_rate_hz) if sample_rate_hz else None
        )
        return {
            "sample_rate_hz": sample_rate_hz,
            "num_frames": num_frames,
            "duration_sec": duration_sec,
            "audio_info_error": None,
        }
    except Exception as exc:  # noqa: BLE001 (we want robustness here)
        return {
            "sample_rate_hz": None,
            "num_frames": None,
            "duration_sec": None,
            "audio_info_error": f"{type(exc).__name__}: {exc}",
        }


def _load_dementia_csv(dementia_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dementia_csv_path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df["name_norm"] = df["name"].astype(str).map(normalize_person_name)
    return df


def _best_name_match(
    filename_norm: str, name_norm_candidates: list[str]
) -> tuple[str | None, float, str | None]:
    """Return (matched_name_norm, confidence, note)."""
    matches = [n for n in name_norm_candidates if n and n in filename_norm]
    if not matches:
        return None, 0.0, "no_match"
    # Prefer the longest match (reduces collisions like 'john' vs 'johnmackey')
    matches.sort(key=len, reverse=True)
    best = matches[0]
    confidence = 1.0 if len(matches) == 1 else 0.7
    note = None if len(matches) == 1 else f"multiple_matches:{len(matches)}"
    return best, confidence, note


def build_metadata(
    dementia_dir: Path,
    control_dir: Path,
    dementia_csv_path: Path,
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    dementia_df = _load_dementia_csv(dementia_csv_path)
    name_norm_to_row = {
        r["name_norm"]: r for r in dementia_df.to_dict(orient="records")
    }
    name_norm_candidates = sorted(name_norm_to_row.keys())

    audio_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []

    for audio in list(_iter_wavs(dementia_dir, label=1)) + list(
        _iter_wavs(control_dir, label=0)
    ):
        info = _safe_audio_info(audio.audio_path)
        # Prefer matching via the person folder name (more reliable than filename suffixes).
        folder_norm = normalize_person_name(audio.person_dir_name)
        matched_norm = folder_norm if folder_norm in name_norm_to_row else None
        conf = 1.0 if matched_norm else 0.0
        note = None if matched_norm else "folder_name_not_in_csv"

        # Fall back to filename-based matching if folder match fails.
        if matched_norm is None:
            filename_norm = normalize_person_name(audio.filename)
            matched_norm, conf2, note2 = _best_name_match(
                filename_norm, name_norm_candidates
            )
            if matched_norm is not None:
                conf = conf2
                note = note2

        # Join metadata if present (mostly dementia subjects).
        meta = name_norm_to_row.get(matched_norm, {})

        # Always populate a subject identifier, even if not present in the dementia CSV.
        person_name = meta.get("name") or audio.person_dir_name
        person_name_norm = matched_norm or normalize_person_name(person_name)

        row: dict[str, Any] = {
            "audio_path": audio.audio_path,
            "label": audio.label,
            "filename": audio.filename,
            "person_name": person_name,
            "person_name_norm": person_name_norm,
            "join_confidence": conf,
            "join_notes": note,
            "dementia_type": meta.get("dementia type"),
            "gender": meta.get("gender"),
            "ethnicity": meta.get("ethnicity"),
            "language": meta.get("language"),
            "datasplit_csv": meta.get("datasplit"),
            **info,
        }

        # Guardrails: flag obviously bad files (but don't drop automatically yet).
        if info["duration_sec"] is not None and info["duration_sec"] < 1.0:
            row["guardrail_flag"] = "too_short_lt_1s"
            dropped_rows.append(row)
        else:
            row["guardrail_flag"] = None
            audio_rows.append(row)

    metadata_path = out_dir / "metadata.csv"
    dropped_path = out_dir / "dropped.csv"

    pd.DataFrame(audio_rows).to_csv(
        metadata_path, index=False, quoting=csv.QUOTE_MINIMAL
    )
    pd.DataFrame(dropped_rows).to_csv(
        dropped_path, index=False, quoting=csv.QUOTE_MINIMAL
    )

    report = {
        "counts": {
            "kept": len(audio_rows),
            "dropped": len(dropped_rows),
        }
    }
    (out_dir / "metadata_report.json").write_text(json.dumps(report, indent=2))

    return metadata_path, dropped_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dementia_dir", required=True, type=Path)
    parser.add_argument("--control_dir", required=True, type=Path)
    parser.add_argument("--dementia_csv", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()

    metadata_path, dropped_path = build_metadata(
        dementia_dir=args.dementia_dir,
        control_dir=args.control_dir,
        dementia_csv_path=args.dementia_csv,
        out_dir=args.out_dir,
    )
    print(f"Wrote: {metadata_path}")
    print(f"Wrote: {dropped_path}")


if __name__ == "__main__":
    main()
