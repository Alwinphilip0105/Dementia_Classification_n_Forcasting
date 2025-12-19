# Model Card: Text-Only Dementia Classification Baseline

## Model Details

**Developed by:** Alwin Philip, Lucas Liona, David Majomi

**Model date:** December 2025

**Model version:** 1.0 (text baseline)

**Model type:** Binary text classification

**License:** MIT

### Architecture

- **Base model:** To start we used DistilBERT for its small size (`distilbert-base-uncased`)
- **Task:** Binary classification (dementia vs. control from speech transcripts)
- **Input:** ASR-transcribed speech text (Whisper)
- **Output:** Binary prediction + confidence score
- **Parameters:** ~67M (DistilBERT encoder + classification head)
- **Training:** Fine-tuned on dementia classification task

## Intended Use

### Primary intended uses
- Linguistic baseline for multimodal fusion model
- Research and educational purposes for dementia detection from speech patterns
- Demonstration of NLP-based biomarker extraction from speech

### Out-of-scope uses
- **NOT for clinical diagnosis** - research prototype only
- NOT for real-time screening without expert review
- NOT suitable for production deployment without validation

## Training Data

**Source:** DementiaNet dataset (speech transcriptions)
**Preprocessing:** OpenAI Whisper ASR (tiny model)
**Size:** 341 transcripts after ASR processing

**Split distribution:**
- Train: 249 samples (141 control, 108 dementia) - 43.4% dementia
- Valid: 46 samples (26 control, 20 dementia) - 43.5% dementia
- Test: 46 samples (43 control, 3 dementia) - 6.5% dementia

**Known data limitations:**
- Class imbalance in test set (only 3 dementia cases, no imbalance in train set)
- Subject-level split (no speaker leakage between splits)
- All speech in English
- Interview/narrative speech only (not conversational)
- ASR errors propagate to model

## Training Procedure

**Hyperparameters:**
- Optimizer: AdamW
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3
- Max sequence length: 512 tokens
- Dropout: 0.1

**Training time:** ~5 minutes (3 epochs on CPU)

**Reproducibility:**
- Random seed: 1337
- Configuration saved in `runs/text_baseline/config.json`

## Performance

### Metrics (Training Set)

| Metric | Value |
|--------|-------|
| Accuracy | 0.888 |
| F1 Score | 0.861 |
| ROC AUC | 0.964 |
| Sensitivity | 0.805 (87/108) |
| Specificity | 0.950 (134/141) |

**Confusion Matrix (Train):**
        Predicted
        Control  Dementia
Actual Control    134       7
        Dementia    21      87

### Metrics (Validation Set - Primary Evaluation)

| Metric | Value |
|--------|-------|
| Accuracy | 0.609 |
| F1 Score | 0.357 |
| ROC AUC | 0.665 |
| Sensitivity | 0.250 (5/20) |
| Specificity | 0.885 (23/26) |

**Confusion Matrix (Valid):**
        Predicted
        Control  Dementia
Actual Control    23       3
        Dementia    15       5

### Metrics (Test Set - Use with Caution)

 **We noticed the Test Set is severely imbalanced (only 3 dementia cases)**

| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | 0.826 | Inflated by imbalance |
| F1 Score | 0.200 | Poor (1/3 detected) |
| ROC AUC | 0.620 | Unreliable with 3 samples |
| Sensitivity | 0.333 (1/3) | Missed 2/3 dementia cases |
| Specificity | 0.860 (37/43) | High (mostly controls) |

Therefore we will use validation metrics for evaluation as a precaution

## Performance Analysis

**Strengths:**
- Good training performance (88.8% accuracy)
- High specificity (low false alarm rate)
- Fast inference (~10ms per sample)

**Weaknesses:**
- Poor validation performance (60.9% accuracy)
- Low sensitivity (misses many dementia cases)
- **Significant overfitting** (train 88.8% vs valid 60.9%)
- Validation F1 of 0.357 indicates struggle with dementia class

**Likely causes:**
1. Small dataset (249 training samples)
2. Limited linguistic diversity in transcripts
3. ASR errors introducing noise
4. Model learning spurious correlations (see Explainability)

## Explainability

**Method:** Captum LayerIntegratedGradients
**Analysis:** 5 validation samples analyzed

### Key Findings

1. **Spurious reliance on [SEP] token**: Model heavily weights the `[SEP]` special token (end-of-sequence marker), suggesting it learned positional artifacts rather than meaningful linguistic features. This is a **critical limitation**.

2. **Punctuation artifacts**: High attribution scores on commas and periods, indicating shallow pattern matching rather than semantic understanding.

3. **Meaningful patterns (limited)**:
    - Model does attend to hesitation markers ("you know", "well")
    - Some attention on repetitive phrases
    - Appears to use sentence length as implicit feature

4. **Positive examples**: 4/5 analyzed samples predicted correctly (all controls)

**Actionable recommendations:**
- Remove special token biases in future iterations
- Focus on semantic features over positional artifacts
- Consider attention-based architectures that emphasize content over structure

**Artifacts:** Word-level attribution visualizations in `runs/text_baseline/explanations/`

## Limitations

### Model Limitations
1. **Severe overfitting** (train 88.8% vs valid 60.9%)
2. **Spurious correlations**: Relies on [SEP] token and punctuation
3. **Poor sensitivity**: Misses 75% of dementia cases in validation
4. **Small training set**: Only 249 samples

### Data Limitations
1. **Class imbalance**: Test set has only 3 dementia cases (6.5%)
2. **ASR errors**: Performance upper-bounded by Whisper quality
3. **Domain**: English interview speech only
4. **Sample size**: 341 files (~184 subjects total)

### Fairness Limitations
1. **Not evaluated for demographic bias** (age, gender, ethnicity)
2. **No analysis of different dementia types** (Alzheimer's vs others)
3. **Potential socioeconomic bias** in speech patterns

## Ethical Considerations

1. **Privacy:** Trained on public dataset; clinical deployment requires consent
2. **Bias:** Performance not validated across demographics
3. **Clinical use:** **NOT validated for diagnosis** - research only
4. **Transparency:** Explainability provided but shows model limitations
5. **Harm potential:** False negatives (missed diagnoses) are concerning


## Comparison to Baselines

| Model | Valid Acc | Valid F1 | Notes |
|-------|-----------|----------|-------|
| **Text-only (DistilBERT)** | **60.9%** | **0.357** | This model |
| MFCC + Logistic Regression | 63.0% | 0.485 | Audio-only baseline |
| Wav2Vec2 + Logistic Regression | TBD | TBD | Audio-only baseline |
| DenseNet Spectrogram | TBD | TBD | Audio-only baseline |

**Insight:** Text-only baseline **underperforms** simple audio baselines, suggesting acoustic features (prosody, pauses) are more informative than linguistic content alone for this task.

## Future Work

1. **Address overfitting**: More data, regularization, data augmentation
2. **Remove spurious features**: Mask special tokens, focus on semantic features
3. **Multimodal fusion**: Combine with acoustic features (main project goal)
4. **Fairness evaluation**: Test across demographics
5. **Better ASR**: Use larger Whisper models or fine-tuned ASR
6. **Ensemble methods**: Combine multiple text models

## References

- **DistilBERT:** Sanh et al. (2019) "DistilBERT, a distilled version of BERT"
- **Whisper ASR:** Radford et al. (2022) "Robust Speech Recognition via Large-Scale Weak Supervision"
- **Captum:** Kokhlikyan et al. (2020) "Captum: A unified and generic model interpretability library for PyTorch"
- **DementiaNet:** Fraser et al. (2016) "Linguistic Features Identify Alzheimer's Disease in Narrative Speech"

## Model Access

**Checkpoint:** `runs/text_baseline/model.pth`
**Config:** `runs/text_baseline/config.json`
**Metrics:** `runs/text_baseline/metrics.json`

---

**Last updated:** December 19th, 2025
**Model version:** 1.0

---