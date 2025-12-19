## DATA_CARD: Dementia Audio Classification Dataset

### Source
- **Dementia class audio**: `dementia-20251217T041331Z-1-001/` (WAV files)
- **No-dementia class audio**: `nodementia-20251217T041501Z-1-001/` (WAV files)
- **Metadata**: `DementiaNet - dementia.csv`

### Purpose
Binary classification: **dementia vs no dementia** from speech audio (and ASR-derived text).

### Ownership / License
- See `LICENSE` for this repository’s code license.
- Dataset licensing may differ; document the dataset’s original license here once confirmed.

### Contents
- **Audio files**: 355 WAV files total
  - 224 control (no-dementia) samples
  - 131 dementia samples
- **Metadata fields** (from CSV): name, dementia type, birth/death, first symptoms year, URLs, gender, ethnicity, datasplit, language
- **Audio characteristics**: Variable duration, sample rates; processed to 16kHz mono for consistency

### Labeling
- Primary label: folder-based (`dementia-*` vs `nodementia-*`) mapped to {1, 0}.
- Additional metadata label (optional): `dementia type` (for subgroup analysis).

### Preprocessing (completed)
- ✅ Validated all 355 audio files; computed duration and sample rate
- ✅ Joined audio filenames to metadata CSV using name normalization
- ✅ Generated ASR transcripts using Whisper (openai/whisper-tiny)
- ✅ Extracted word-level timestamps for 51,144 word segments
- ✅ Created subject-level train/valid/test splits (256/48/51 samples)
- ✅ No files dropped (all 355 passed validation)

### Known limitations
- **Class imbalance**: Test set has 48 controls vs 3 dementia cases, affecting evaluation metrics
- **Metadata coverage**: CSV metadata primarily covers dementia subjects; controls have limited metadata
- **Recording variability**: Audio includes variability in recording conditions (noise, channel, microphone)
- **Dataset size**: 355 samples is relatively small for deep learning; may limit generalization


