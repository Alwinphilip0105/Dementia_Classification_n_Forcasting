## MODEL_CARD: Dementia Multimodal Classifier

### Model summary
This project benchmarks multiple models for binary dementia detection:
- Non-ML baseline (sklearn) on hand-crafted audio features (MFCC + pause stats).
- DenseNet on spectrogram images.
- Audio-only Wav2Vec2 embedding + pooling classifier.
- Text-only Transformer classifier on ASR transcripts.
- Multimodal fusion with cross-attention between text embeddings and word-level audio embeddings.

### Intended use
- Research / educational use for studying dementia markers in speech.
- Not intended for clinical diagnosis.

### Training data
See `DATA_CARD.md`.

### Evaluation

**Best Model: DenseNet on Spectrograms**

Test Set Performance:
- Accuracy: 90.2%
- F1 Score: 0.29 (limited by class imbalance: 48 controls vs 3 dementia)
- ROC-AUC: 0.72
- Confusion Matrix: [[45, 3], [2, 1]]

**All Models Comparison**:

| Model | Test Accuracy | Test F1 | Test ROC-AUC |
|-------|---------------|---------|--------------|
| Non-ML (MFCC) | 68.6% | 0.00 | 0.11 |
| Wav2Vec2 audio-only | 58.8% | 0.09 | 0.49 |
| DenseNet spectrogram | **90.2%** | 0.29 | **0.72** |
| Text-only (RoBERTa) | 62.7% | 0.10 | 0.42 |

**Robustness**: SNR testing shows graceful degradation; maintains >80% accuracy at 10dB SNR.

### Explainability
- Captum Integrated Gradients (and attention visualizations where applicable).

### Limitations
- **Class imbalance**: Test set imbalance (48:3) severely limits F1 scores despite high accuracy
- **Dataset size**: 355 samples may not generalize to diverse populations
- **ASR errors**: Text features may be biased by ASR errors, especially in noisy audio
- **Clinical validation**: Not validated on real clinical data; requires domain expert validation
- **Population bias**: Dataset may not represent the full population distribution of real clinical settings
- **Fusion model**: Cross-attention fusion model architecture implemented but not yet trained (performance optimizations needed)


