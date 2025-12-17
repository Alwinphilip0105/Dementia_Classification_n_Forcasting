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
Metrics to report:
- Accuracy, F1, ROC-AUC
- Confusion matrix
- Robustness curves vs SNR (noise)

### Explainability
- Captum Integrated Gradients (and attention visualizations where applicable).

### Limitations
- ASR errors may bias text features, especially in noisy audio.
- Dataset may not represent the population distribution of real clinical settings.


