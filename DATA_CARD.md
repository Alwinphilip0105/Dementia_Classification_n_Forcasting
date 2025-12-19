## DATA_CARD: Dementia Audio Classification Dataset

We got this dataset from the Dementia Net project by Shreyas Gite https://github.com/shreyasgite/dementianet

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
- Audio files: WAV format (counts will be re-computed by the pipeline).
- Metadata fields (from CSV): name, dementia type, birth/death, first symptoms year, URLs, gender, ethnicity, datasplit, language.

### Labeling
- Primary label: folder-based (`dementia-*` vs `nodementia-*`) mapped to {1, 0}.
- Additional metadata label (optional): `dementia type` (for subgroup analysis).

### Preprocessing (planned)
- Validate readable audio, compute duration + sample rate.
- Remove/flag low-quality samples (too short, near-silence, corrupt headers).
- Generate ASR transcripts and word-level timestamps for multimodal modeling.

### Known limitations (initial)
- Metadata CSV appears to cover dementia subjects; controls may have limited metadata.
- Audio may include variability in recording conditions (noise, channel, microphone).


