"""Quick helper scipt to rebuild ASR manifest from actual directories on disk (FIXED, help from Claude)."""

from pathlib import Path

import pandas as pd

asr_dir = Path("data/processed/asr_whisper")

# Find all actual ASR output directories
asr_subdirs = [
    d for d in asr_dir.iterdir() if d.is_dir() and (d / "transcript.json").exists()
]

print(f"Found {len(asr_subdirs)} ASR directories")

rows = []
for subdir in asr_subdirs:
    # Parse the original audio_path from directory name
    # Format: <old_dataset>__<class>__<person>__<file>.wav
    dir_name = subdir.name
    parts = dir_name.split("__")

    if len(parts) >= 4:
        old_dataset = parts[0]  # e.g., dementia-20251217T041331Z-1-001
        class_name = parts[1]  # e.g., dementia or nodementia
        person = parts[2]  # e.g., "Abe Burrows"
        filename = "__".join(parts[3:])  # e.g., "AbeBurrows_5.wav"

        # Map old dataset names to new ones
        if "dementia-2025121" in old_dataset and class_name == "dementia":
            new_dataset = "dementia-20251218T112637Z-3-001"
        elif "nodementia-2025121" in old_dataset and class_name == "nodementia":
            new_dataset = "nodementia-20251218T113334Z-3-001"
        else:
            print(f"Warning: Unknown dataset pattern: {old_dataset} / {class_name}")
            new_dataset = old_dataset  # Fallback

        # Reconstruct audio_path with NEW dataset name
        audio_path = f"{new_dataset}/{class_name}/{person}/{filename}"

        rows.append(
            {
                "audio_path": audio_path,
                "asr_dir": str(subdir),
                "transcript_json": str(subdir / "transcript.json"),
                "words_json": str(subdir / "words.json"),
            }
        )

df = pd.DataFrame(rows)

# Sort for consistency
df = df.sort_values("audio_path").reset_index(drop=True)

# Save
output_path = asr_dir / "asr_manifest.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ Rebuilt manifest with {len(df)} entries")
print("\nClass breakdown:")
print(df["audio_path"].str.contains("/dementia/").value_counts())

print("\nFirst dementia entry:")
dementia_row = df[df["audio_path"].str.contains("/dementia/")].iloc[0]
print(f"  audio_path: {dementia_row['audio_path']}")
print(f"  asr_dir: {dementia_row['asr_dir']}")

print("\nFirst control entry:")
control_row = df[df["audio_path"].str.contains("/nodementia/")].iloc[0]
print(f"  audio_path: {control_row['audio_path']}")
print(f"  asr_dir: {control_row['asr_dir']}")
