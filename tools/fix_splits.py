"""Fix data splits using subject-level stratified splitting.

Ensures:
- No subject leakage (all files from one subject in same split)
- Balanced dementia/control in all splits
- Target: 70% train, 15% valid, 15% test (by subject count)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def get_subject_labels(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Get one label per subject (majority vote if mixed).

    Args:
        metadata_df: DataFrame with person_name_norm and label columns

    Returns:
        DataFrame with person_name_norm and label (one row per subject)
    """
    # Group by subject and take majority label
    subject_labels = (
        metadata_df.groupby("person_name_norm")["label"]
        .agg(lambda x: x.mode()[0])  # Majority vote
        .reset_index()
    )

    return subject_labels


def stratified_subject_split(
    subject_labels_df: pd.DataFrame,
    train_size: float = 0.7,
    valid_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> dict[str, str]:
    """Perform stratified split on subjects.

    Args:
        subject_labels_df: DataFrame with person_name_norm and label
        train_size: Fraction of subjects for training
        valid_size: Fraction of subjects for validation
        test_size: Fraction of subjects for testing
        random_state: Random seed for reproducibility

    Returns:
        Dictionary mapping person_name_norm -> split
    """
    assert abs(train_size + valid_size + test_size - 1.0) < 1e-6, "Split sizes must sum to 1.0"

    subjects = subject_labels_df["person_name_norm"].values
    labels = subject_labels_df["label"].values

    # First split: train vs (valid + test)
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=(valid_size + test_size), random_state=random_state
    )
    train_idx, temp_idx = next(sss1.split(subjects, labels))

    # Second split: valid vs test
    temp_subjects = subjects[temp_idx]
    temp_labels = labels[temp_idx]

    # Adjust test_size relative to temp set
    relative_test_size = test_size / (valid_size + test_size)

    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=relative_test_size, random_state=random_state
    )
    valid_idx_temp, test_idx_temp = next(sss2.split(temp_subjects, temp_labels))

    # Map back to original indices
    valid_idx = temp_idx[valid_idx_temp]
    test_idx = temp_idx[test_idx_temp]

    # Create split assignments
    split_map = {}
    for idx in train_idx:
        split_map[subjects[idx]] = "train"
    for idx in valid_idx:
        split_map[subjects[idx]] = "valid"
    for idx in test_idx:
        split_map[subjects[idx]] = "test"

    return split_map


def apply_splits_to_files(
    metadata_df: pd.DataFrame, split_map: dict[str, str]
) -> pd.DataFrame:
    """Apply subject-level splits to all files.

    Args:
        metadata_df: DataFrame with all files
        split_map: Dictionary mapping person_name_norm -> split

    Returns:
        DataFrame with audio_path, person_name_norm, label, split
    """
    df = metadata_df.copy()
    df["split"] = df["person_name_norm"].map(split_map)

    # Validate no missing splits
    assert df["split"].notna().all(), "Some subjects missing split assignment"

    return df[["audio_path", "person_name_norm", "label", "split"]]


def print_split_summary(splits_df: pd.DataFrame):
    """Print detailed summary of split distribution."""
    print("=" * 70)
    print("NEW SPLIT DISTRIBUTION")
    print("=" * 70)

    for split in ["train", "valid", "test"]:
        split_df = splits_df[splits_df["split"] == split]
        n_files = len(split_df)
        n_subjects = split_df["person_name_norm"].nunique()

        n_dementia_files = (split_df["label"] == 1).sum()
        n_control_files = (split_df["label"] == 0).sum()

        dementia_subjects = split_df[split_df["label"] == 1]["person_name_norm"].nunique()
        control_subjects = split_df[split_df["label"] == 0]["person_name_norm"].nunique()

        print(f"\n{split.upper()}:")
        print(f"  Files:    {n_files:3d} total")
        print(
            f"    Dementia: {n_dementia_files:3d} ({n_dementia_files/n_files*100:5.1f}%)"
        )
        print(
            f"    Control:  {n_control_files:3d} ({n_control_files/n_files*100:5.1f}%)"
        )
        print(f"  Subjects: {n_subjects:3d} total")
        print(
            f"    Dementia: {dementia_subjects:3d} ({dementia_subjects/n_subjects*100:5.1f}%)"
        )
        print(
            f"    Control:  {control_subjects:3d} ({control_subjects/n_subjects*100:5.1f}%)"
        )

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Fix data splits using subject-level stratified splitting"
    )
    parser.add_argument(
        "--metadata_csv",
        type=Path,
        default=Path("data/processed/metadata.csv"),
        help="Path to metadata.csv",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("data/processed/splits.csv"),
        help="Output path for splits.csv",
    )
    parser.add_argument(
        "--train_size", type=float, default=0.7, help="Fraction of subjects for training"
    )
    parser.add_argument(
        "--valid_size", type=float, default=0.15, help="Fraction of subjects for validation"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.15, help="Fraction of subjects for testing"
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Load metadata
    print(f"Loading metadata from {args.metadata_csv}...")
    metadata_df = pd.read_csv(args.metadata_csv)

    print(f"Total files: {len(metadata_df)}")
    print(f"Total subjects: {metadata_df['person_name_norm'].nunique()}")

    # Get subject-level labels
    subject_labels_df = get_subject_labels(metadata_df)
    n_dementia_subjects = (subject_labels_df["label"] == 1).sum()
    n_control_subjects = (subject_labels_df["label"] == 0).sum()

    print(f"Dementia subjects: {n_dementia_subjects}")
    print(f"Control subjects: {n_control_subjects}")

    # Perform stratified split on subjects
    print("\nPerforming subject-level stratified split...")
    split_map = stratified_subject_split(
        subject_labels_df,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Apply splits to all files
    splits_df = apply_splits_to_files(metadata_df, split_map)

    # Print summary
    print_split_summary(splits_df)

    # Save to CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    splits_df.to_csv(args.out_csv, index=False)
    print(f"\n✓ Saved new splits to {args.out_csv}")

    # Validate no subject leakage
    subject_split_counts = (
        splits_df.groupby("person_name_norm")["split"].nunique()
    )
    if (subject_split_counts > 1).any():
        print("\n⚠️  WARNING: Subject leakage detected!")
        print(subject_split_counts[subject_split_counts > 1])
    else:
        print("\n✓ No subject leakage - all subjects in exactly one split")


if __name__ == "__main__":
    main()
