## PracticalAI Final Project Execution Plan (Dementia Multimodal Classification)

This plan follows the required order and repo structure from `INSTRUCTIONS.md` while implementing your multimodal (audio + text) dementia detection pipeline using **audio-only** data + **ASR transcripts with word timestamps**.

### Confirmed inputs in your workspace
- **Audio**: two zip files at the repo root
  - `dementia-20251217T041331Z-1-001.zip`
  - `nodementia-20251217T041501Z-1-001.zip`
- **Metadata**: `DementiaNet - dementia.csv` (columns include `name`, `dementia type`, `gender`, `ethnicity`, `datasplit`, `language`, plus URLs)
- **Join rule**: audio filenames include the person `name` (user confirmed).
- **Split rule**: hybrid — prefer CSV `datasplit` but enforce subject-level separation.
- **Stack**: PyTorch + HuggingFace on GPU (user confirmed).

### Required deliverables (what we will produce)
- `notebooks/final_report.ipynb` (professional report; minimal code; imports modules)
- `dementia_project/` (Python package with reusable modules; no long notebook code)
- `README.md` (with required sections)
- `DATA_CARD.md`
- `MODEL_CARD.md`
- `pyproject.toml` + `poetry.lock` (Poetry-managed deps)
- `LICENSE` (already present)

### Important grading constraints we will satisfy (from `INSTRUCTIONS.md`)
- Notebook imports modules (minimal notebook code)
- Separate training vs inference entrypoints
- YAML/JSON configs (no local paths)
- Export final model to ONNX + conformance test
- Use `black` formatting
- Include explainability + baseline comparisons

---

## Phase 0 — Repo scaffold (match course structure)
Create the course-style project skeleton:
- `dementia_project/` (package)
  - `__init__.py`
  - `config/` (yaml parsing)
  - `data/` (loading, manifests, validation)
  - `asr/` (WhisperX transcription + alignment)
  - `segmentation/` (time-window + word-level)
  - `features/` (Wav2Vec2, spectrogram, text)
  - `models/` (sklearn baseline, DenseNet, Wav2Vec2 baseline, text baseline, fusion)
  - `train/` (training entrypoints)
  - `infer/` (inference entrypoints)
  - `export/` (ONNX export + conformance test)
  - `viz/` (plots)
- `notebooks/` (final notebook + small helper notebook if needed)
- `tests/` (lightweight tests: name-join, split leakage, ONNX conformance)
- `configs/` (YAML configs for experiments)
- `.gitignore` (ensure data/audio outputs ignored)
- Add `DATA_CARD.md` and `MODEL_CARD.md` templates
- Update `README.md` to required sections

Implementation notes:
- Format with `black` and keep lines within ~88 chars.
- No hardcoded local paths; use relative paths and config files.

---

## Phase 1 — Data extraction + metadata join (CSV-driven)
1. **Unzip audio** into:
   - `data/raw/dementia/`
   - `data/raw/nodementia/`
2. **Inventory** audio:
   - list extensions, sample rates, durations, missing/broken files
3. **Normalize names for joining**:
   - create a deterministic `normalize_name()` function that maps:
     - casefold
     - strip punctuation
     - treat `_` and whitespace as equivalent
4. **Build `data/processed/metadata.csv`** with at least:
   - `audio_path` (relative)
   - `label` (0 control, 1 dementia)
   - `person_name` (CSV `name`)
   - `dementia_type` (CSV `dementia type`, optional analysis field)
   - `datasplit_csv` (CSV `datasplit`)
   - `gender`, `ethnicity`, `language` (as available)
   - `duration_sec`, `sample_rate_hz`
   - `join_confidence` / `join_notes` for ambiguous matches
5. **Data guardrails (mandatory)**:
   - drop/flag files with duration < N sec, very low RMS (silence), unsupported formats
   - record all drops in `data/processed/dropped.csv`

---

## Phase 2 — Hybrid split (respect CSV + prevent leakage)
Goal: Never allow the same `person_name` in multiple splits.
1. Start from CSV `datasplit` (train/valid/test).
2. Validate:
   - any person appears in >1 split? fix by moving all that person’s files to one split
   - class balance by split
3. Save:
   - `data/processed/splits.csv` (audio_path → split)
   - `data/processed/split_report.json` (counts, fixes applied)

---

## Phase 3 — ASR (audio → transcript + word timestamps)
Because you have **audio only**, generate transcripts:
1. Use **WhisperX**:
   - transcript text
   - word-level timestamps (start/end per word)
2. Cache outputs under:
   - `data/processed/asr/<audio_id>/{transcript.json,words.json,segments.json}`
3. Build `data/processed/asr_manifest.csv` linking `audio_path` → ASR artifacts.

---

## Phase 4 — Two segmentation strategies (manifests first)
We will generate manifests rather than writing millions of tiny wav files.

### A) Time-window segmentation
- windows: 2–3 seconds (config), hop 0.5–1.0s (config)
- save `data/processed/time_segments.csv` with:
  - `audio_path,start_sec,end_sec,label,split,person_name`

### B) Word-level segmentation
- based on WhisperX word timestamps
- filter too-short/too-long words (config)
- save `data/processed/word_segments.csv` with:
  - `audio_path,word,word_index,start_sec,end_sec,label,split,person_name`

---

## Phase 5 — Baselines (benchmark first, as required best practice)
Train/evaluate on identical splits; log metrics + confusion matrix.

1. **Non-ML baseline** (counts toward “non-ML baseline” requirement):
   - MFCC summary stats + pause features (from energy / ASR gaps)
   - model: Logistic Regression or Linear SVM (sklearn)
2. **DenseNet baseline**:
   - log-mel spectrogram “images”
   - model: DenseNet (torchvision)
3. **Audio-only Wav2Vec2 baseline**:
   - features: pretrained Wav2Vec2 encoder
   - pooling: mean or attention pooling
   - classifier: MLP
4. **Text-only baseline**:
   - model: transformer encoder (e.g., `roberta-base`) on Whisper transcript
   - classifier head: linear/MLP

Artifacts:
- `runs/<exp_name>/metrics.json`
- `runs/<exp_name>/confusion_matrix.png`

---

## Phase 6 — Multimodal fusion model (cross-attention)
Your target architecture:
- Text stream: transformer hidden states over transcript
- Audio stream: **word-level** audio embeddings (Wav2Vec2 per word snippet)
- Fusion: cross-attention block(s) to align text↔audio
- Output: dementia probability

Training recipe (fast + reliable):
- Start with encoders frozen to validate the pipeline
- Unfreeze top layers if GPU/time allows

---

## Phase 7 — Explainability + robustness (rubric-aligned)
1. Explainability:
   - Captum Integrated Gradients on the classifier head
   - plus attention visualization for a few examples
2. Robustness:
   - additive noise test (multiple SNR levels) and performance vs SNR plot
   - simple time-shift test
3. Drift discussion (write-up requirement):
   - propose monitoring audio duration/SNR/transcript perplexity and retraining triggers

---

## Phase 8 — ONNX export + conformance test (mandatory)
1. Export the best model to ONNX:
   - `artifacts/model.onnx`
2. Conformance test:
   - compare PyTorch vs ONNX logits on a small fixed batch
   - assert max absolute difference < epsilon

---

## Phase 9 — Final notebook + documentation (deliverables)
1. `notebooks/final_report.ipynb` follows the outline from `INSTRUCTIONS.md` and includes:
   - Abstract paragraph (required)
   - EDA plots
   - baseline comparison table
   - fusion model results
   - explainability + robustness results
   - at least 3 actionable insights
2. `README.md` updated to include the required sections:
   - Purpose, Usage, Known Issues, Roadmap, Contributing, License, Contact
3. `DATA_CARD.md` and `MODEL_CARD.md` filled out.

---

## Minimal questions we’ll resolve during implementation (not blockers)
- Exact audio filename pattern (to implement robust `name` matching; we’ll infer from extracted files).
- Whether any audio is not English (to set ASR language if needed).


