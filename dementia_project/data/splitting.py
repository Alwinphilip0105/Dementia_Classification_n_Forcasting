"""Train/val/test split utilities (subject-level, hybrid with CSV datasplit)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.7
    valid: float = 0.15
    test: float = 0.15


def _stable_hash_float(text: str) -> float:
    """Map a string to a stable float in [0, 1)."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    # Use first 16 hex digits -> 64-bit int
    value = int(digest[:16], 16)
    return value / float(2**64)


def assign_split_fallback(person_name_norm: str, ratios: SplitRatios) -> str:
    """Deterministic split assignment when CSV datasplit is missing."""
    x = _stable_hash_float(person_name_norm or "unknown")
    if x < ratios.train:
        return "train"
    if x < ratios.train + ratios.valid:
        return "valid"
    return "test"


def build_hybrid_splits(
    metadata_df: pd.DataFrame,
    ratios: SplitRatios = SplitRatios(),
) -> pd.DataFrame:
    """Create subject-level splits.

    Rules:
    - If `datasplit_csv` is present for a subject, prefer it.
    - Otherwise assign deterministically using hashing.
    - Enforce: each `person_name_norm` appears in exactly one split.
    """
    df = metadata_df.copy()
    if "person_name_norm" not in df.columns:
        raise ValueError("metadata_df must include person_name_norm")

    def initial_split(row: pd.Series) -> str:
        split = str(row.get("datasplit_csv") or "").strip().casefold()
        if split in {"train", "valid", "test"}:
            return split
        return assign_split_fallback(str(row.get("person_name_norm") or ""), ratios)

    df["split_initial"] = df.apply(initial_split, axis=1)

    # Enforce subject-level split by choosing the majority split for that subject.
    grouped = (
        df.groupby("person_name_norm")["split_initial"]
        .value_counts()
        .rename("count")
        .reset_index()
    )
    grouped = grouped.sort_values(
        ["person_name_norm", "count"], ascending=[True, False]
    )
    chosen_df = grouped.drop_duplicates(subset=["person_name_norm"], keep="first")
    chosen: dict[str, str] = chosen_df.set_index("person_name_norm")[
        "split_initial"
    ].to_dict()

    df["split"] = df["person_name_norm"].map(chosen)  # type: ignore[call-overload]
    return df
