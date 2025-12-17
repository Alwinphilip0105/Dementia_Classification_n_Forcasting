# Dementia Classification (Audio + ASR Text)

## Purpose
This repository implements a dementia vs no-dementia classifier from speech using:
- Audio-only baselines (hand-crafted features, spectrogram CNN, Wav2Vec2 embeddings)
- Text-only baselines (Transformer on ASR transcripts)
- A multimodal fusion model (cross-attention over word-level audio + text)

## Usage Instructions

### 1) Install
This project uses Poetry.

```bash
python -m pip install --upgrade pip
python -m pip install poetry
poetry install
```

### 2) Prepare metadata
Build metadata by joining WAV filenames to `DementiaNet - dementia.csv`.

```bash
poetry run python -m dementia_project.data.build_metadata ^
  --dementia_dir "dementia-20251217T041331Z-1-001" ^
  --control_dir "nodementia-20251217T041501Z-1-001" ^
  --dementia_csv "DementiaNet - dementia.csv" ^
  --out_dir "data/processed"
```

### 3) Train (example)
Training entrypoints will live under `dementia_project.train`.

## Known Issues
- WhisperX installation can be difficult on Windows; the ASR module will support a fallback ASR mode if WhisperX is unavailable.
- Some audio may have inconsistent gain/noise; guardrails will flag suspect samples.

## Feature Roadmap
- ASR + word timestamp pipeline (WhisperX) with caching
- Baselines (sklearn, DenseNet, Wav2Vec2, text Transformer)
- Cross-attention fusion model
- ONNX export + conformance tests
- Captum explainability and robustness tests (SNR curves)

## Contributing
PRs welcome. Keep changes modular (modules in `dementia_project/`, minimal code in notebooks) and format with `black`.

## License
MIT — see `LICENSE`.

## Contact
Primary contact: add your email here.
