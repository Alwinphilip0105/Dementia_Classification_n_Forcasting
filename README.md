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

### 3) Build splits
Create subject-level train/valid/test splits:

```bash
poetry run python -m dementia_project.data.build_splits ^
  --metadata_csv "data/processed/metadata.csv" ^
  --out_dir "data/processed"
```

### 4) Run ASR (generate transcripts)
Generate ASR transcripts and word timestamps:

```bash
poetry run python -m dementia_project.asr.run_asr ^
  --metadata_csv "data/processed/metadata.csv" ^
  --out_dir "data/processed/asr_whisper" ^
  --model_name "openai/whisper-tiny" ^
  --language en ^
  --task transcribe
```

### 5) Train baselines
Train and evaluate baseline models:

**Non-ML baseline (MFCC + Logistic Regression):**
```bash
poetry run python -m dementia_project.train.train_nonml ^
  --metadata_csv "data/processed/metadata.csv" ^
  --splits_csv "data/processed/splits.csv" ^
  --out_dir "runs/nonml_baseline_scaled"
```

**DenseNet on spectrograms (best performing):**
```bash
poetry run python -m dementia_project.train.train_densenet_spec ^
  --metadata_csv "data/processed/metadata.csv" ^
  --splits_csv "data/processed/splits.csv" ^
  --out_dir "runs/densenet_spec_full_cuda" ^
  --epochs 5 --batch_size 16
```

**Text-only baseline (RoBERTa):**
```bash
poetry run python -m dementia_project.train.train_text_baseline ^
  --metadata_csv "data/processed/metadata.csv" ^
  --splits_csv "data/processed/splits.csv" ^
  --asr_manifest_csv "data/processed/asr_whisper/asr_manifest.csv" ^
  --out_dir "runs/text_baseline_roberta"
```

### 6) Export to ONNX
Export best model to ONNX format:

```bash
poetry run python -m dementia_project.export.run_onnx_export ^
  --model_type densenet ^
  --out_dir artifacts ^
  --test
```

### 7) Run explainability
Generate attribution visualizations:

```bash
poetry run python -m dementia_project.viz.run_explainability ^
  --model_type densenet ^
  --metadata_csv "data/processed/metadata.csv" ^
  --splits_csv "data/processed/splits.csv" ^
  --out_dir "runs/explainability" ^
  --num_samples 5
```

## Results Summary

**Best Model**: DenseNet on spectrograms
- Test Accuracy: 90.2%
- Test F1: 0.29 (limited by class imbalance)
- Test ROC-AUC: 0.72

**All Baselines**:
- Non-ML (MFCC): 68.6% accuracy
- Wav2Vec2 audio-only: 58.8% accuracy
- DenseNet spectrogram: 90.2% accuracy ⭐
- Text-only (RoBERTa): 62.7% accuracy

## Known Issues
- Test set has severe class imbalance (48 controls vs 3 dementia), affecting F1 scores
- ASR errors may occur in noisy audio; transcripts are cached for reproducibility
- Fusion model training is computationally intensive; performance optimizations recommended

## Completed Features
- ✅ ASR + word timestamp pipeline (Whisper via transformers)
- ✅ All baselines (sklearn, DenseNet, Wav2Vec2, text Transformer)
- ✅ Cross-attention fusion model architecture
- ✅ ONNX export + conformance tests
- ✅ Captum explainability and robustness tests

## Contributing
PRs welcome. Keep changes modular (modules in `dementia_project/`, minimal code in notebooks) and format with `black`.

## License
MIT — see `LICENSE`.

## Dependency Audit

All non-standard dependencies have been investigated according to professional software engineering best practices. See `DEPENDENCY_AUDIT.md` for detailed analysis of:
- **captum** (explainability) - Meta AI Research
- **transformers** (ASR, text embeddings) - HuggingFace
- **librosa** (audio processing) - NYU Music Information Retrieval Lab
- **onnx/onnxruntime** (model export) - Microsoft

All dependencies are:
- ✅ From reputable sources with active maintenance
- ✅ Properly licensed (BSD, Apache 2.0, MIT - all permissive)
- ✅ Peer-reviewed or industry-standard
- ✅ Production-ready with no cost implications

## Contact
- ap2823@scarletmail.rutgers.edu
- ll1006@scarletmail.rutgers.edu
- dom22@scarletmail.rutgers.edu
