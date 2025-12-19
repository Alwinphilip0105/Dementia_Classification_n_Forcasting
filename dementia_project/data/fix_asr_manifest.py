"""Quick helper script to rename audio_path column in asr_manifest.csv with new directory names."""

import pandas as pd

# ASR Manifest is formatted with these columns audio_path,asr_dir,transcript_json,words_json
manifest_path = "data/processed/asr_whisper/asr_manifest.csv"
df = pd.read_csv(manifest_path)

# Replace old directory names with new ones, you can obviously change this as needed for your own downloads
old_dementia = "dementia-20251217T041331Z-1-001"
new_dementia = "dementia-20251218T112637Z-3-001"

old_control = "nodementia-20251217T041501Z-1-001"
new_control = "nodementia-20251218T113334Z-3-001"

df["audio_path"] = df["audio_path"].str.replace(old_dementia, new_dementia, regex=False)
df["audio_path"] = df["audio_path"].str.replace(old_control, new_control, regex=False)

# Save back
df.to_csv(manifest_path, index=False)

print(f"✅ Updated {len(df)} entries in {manifest_path}")
print(f"   Old dementia dir: {old_dementia}")
print(f"   New dementia dir: {new_dementia}")
print(f"   Old control dir: {old_control}")
print(f"   New control dir: {new_control}")
print("\nFirst 3 rows after update:")
print(df.head(3)["audio_path"].tolist())
