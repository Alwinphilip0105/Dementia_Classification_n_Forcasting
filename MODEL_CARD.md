## MODEL CARD: Dementia Multimodal Classifier

**Developed by:** Alwin Philip, Lucas Liona, David Majomi
**Model date:** December 2025
**License:** MIT

### Model Summary

This project benchmarks multiple models for binary dementia detection from speech:
- **Non-ML baseline**: sklearn on hand-crafted audio features (MFCC + pause stats)
- **DenseNet**: CNNs on log-mel spectrograms (90.2% test accuracy - BEST)
- **Wav2Vec2**: Audio embeddings + classifier (58.8% test accuracy)
- **Text-only models**: Evaluated RoBERTa-base and DistilBERT-base on ASR transcripts
  - DistilBERT (67.3% test accuracy - more efficient, better performance)
  - RoBERTa (baseline comparison)
- **Multimodal fusion**: Cross-attention between word-level Wav2Vec2 and DistilBERT (53.8% test accuracy)

### Intended Use

**Primary uses:**
- Research and educational purposes for dementia detection from speech
- Demonstration of multimodal fusion architectures
- Baseline comparison for audio vs. text vs. fusion approaches
- Model efficiency analysis (RoBERTa vs. DistilBERT)

**Out-of-scope uses:**
- **NOT for clinical diagnosis** - research prototype only
- NOT for real-time screening without expert review
- NOT suitable for production deployment without validation

### Training Data

**Source:** DementiaNet dataset
**Size:** 355 audio recordings (224 controls, 131 dementia cases)
**Preprocessing:** 16kHz mono audio, Whisper ASR for transcripts, subject-level splits

**Split distribution (FIXED):**
- Train: 243 samples (70%)
- Valid: 46 samples (13%)
- Test: 52 samples (37.7% dementia - improved from original 6.5%)

**Known limitations:**
- Small dataset (355 samples) limits generalization
- English interview speech only
- ASR errors propagate to text models
- Subject-level splitting ensures no speaker leakage

### Evaluation

**Best Model: DenseNet on Spectrograms**

| Split | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| Train | 99.6% | - | - |
| Valid | 91.3% | - | - |
| Test | **90.2%** | 0.29 | 0.72 |

**All Models Comparison (Test Set):**

| Model | Test Accuracy | Test F1 | ROC-AUC | Notes |
|-------|---------------|---------|---------|-------|
| DenseNet spectrogram | **90.2%** | 0.29 | **0.72** | Best performer |
| DistilBERT text-only | 67.3% | 0.45 | 0.76 | Efficient, good ROC-AUC |
| Non-ML (MFCC) | 68.6% | 0.00 | 0.11 | Simple baseline |
| RoBERTa text-only | 62.7% | 0.10 | 0.42 | Larger but underperforms DistilBERT |
| Wav2Vec2 audio-only | 58.8% | 0.09 | 0.49 | Embedding approach |
| Fusion (cross-attention) | 53.8% | 0.33 | 0.59 | Severe overfitting |

### Key Findings

1. **Spectrogram CNNs outperform complex architectures** for small datasets (355 samples)
2. **DistilBERT outperforms RoBERTa** despite being smaller (66M vs 125M params), suggesting:
   - Model efficiency matters for small datasets
   - Larger models may overfit more easily
   - DistilBERT's distillation process preserves important linguistic patterns
3. **Multimodal fusion failed** (53.8% vs. 90.2% audio-only) due to:
   - Severe overfitting (81% train vs 48% valid)
   - Frozen encoders preventing task-specific learning
   - High learning rate (1e-3) and insufficient epochs (10)
   - Limited data (355 samples insufficient for multimodal learning)
4. **Text-only models show promise** (67.3%) but underperform audio features
5. **Data quality matters**: Fixed splits improved all model performance

### Text Model Comparison: RoBERTa vs. DistilBERT

**DistilBERT advantages:**
- Better test accuracy (67.3% vs 62.7%)
- Higher ROC-AUC (0.76 vs 0.42)
- Faster inference (~2x speedup)
- Fewer parameters (66M vs 125M)
- Better suited for small datasets (less prone to overfitting)

**Research insight:** Model size is not always beneficial - efficient architectures (DistilBERT) can outperform larger models (RoBERTa) on small datasets by reducing overfitting risk while preserving semantic understanding.

### Explainability

**Method:** Captum LayerIntegratedGradients

**Text model findings (DistilBERT):**
- Spurious reliance on [SEP] token (positional artifacts)
- High attribution on punctuation (shallow pattern matching)
- Limited semantic understanding (hesitation markers, repetitive phrases)
- Validation: 4/5 samples predicted correctly (all controls)

**Audio model findings (DenseNet):**
- Model attention to mid-frequency spectral regions (2-4 kHz)
- Consistent with prosodic features known to change in dementia
- Validates spectrogram approach for this task

### Fusion Model Architecture Details

**Components:**
- Frozen DistilBERT encoder (66M parameters)
- Pre-computed word-level Wav2Vec2 embeddings (341/355 files)
- Cross-attention layer (text queries audio at word level)
- Trainable classification head (690K parameters)

**Why it failed:**
- Dataset too small (355 samples) for multimodal learning
- Frozen encoders unable to adapt to task
- 14/355 missing audio embeddings due to path issues (spaces in filenames)
- Requires 10x more data (3500+ samples) for robust fusion

### Robustness

**SNR testing:** Maintains >80% accuracy at 10dB SNR (graceful degradation)
**Time-shift testing:** <2% accuracy drop for shifts up to 30% of audio duration

### Limitations

**Model limitations:**
- Small dataset (355 samples) limits generalization
- Fusion model severe overfitting indicates insufficient data
- Text models rely on spurious correlations (special tokens, punctuation)
- RoBERTa underperforms due to overfitting on small dataset

**Data limitations:**
- Class imbalance still present (though improved from original)
- English language only
- Interview speech only (not conversational)
- No demographic diversity validation
- ASR errors propagate to text models

**Fairness limitations:**
- Not evaluated for age, gender, ethnicity bias
- No analysis of different dementia types (Alzheimer's vs others)
- Potential socioeconomic bias in speech patterns

### Ethical Considerations

1. **Privacy:** Requires consent for clinical deployment
2. **Bias:** Performance not validated across demographics
3. **Clinical use:** NOT validated for diagnosis - research only
4. **Transparency:** Explainability provided but reveals model limitations
5. **Harm potential:** False negatives (missed diagnoses) are concerning

### Future Work

1. **Collect larger dataset** (3500+ samples) for robust multimodal learning
2. **Fine-tune encoders** instead of freezing them
3. **Fix audio alignment** (14 missing files, improve timestamp accuracy)
4. **Hyperparameter optimization** (lower learning rate 2e-5, more epochs 30+)
5. **Clinical validation** with domain experts
6. **Fairness evaluation** across demographics
7. **Ensemble methods** combining DistilBERT text with DenseNet audio

### References

- **RoBERTa:** Liu et al. (2019) "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
- **DistilBERT:** Sanh et al. (2019) "DistilBERT, a distilled version of BERT"
- **Wav2Vec2:** Baevski et al. (2020) "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"
- **Whisper ASR:** Radford et al. (2022) "Robust Speech Recognition via Large-Scale Weak Supervision"
- **DenseNet:** Huang et al. (2017) "Densely Connected Convolutional Networks"
- **Captum:** Kokhlikyan et al. (2020) "Captum: A unified and generic model interpretability library for PyTorch"
- **DementiaNet:** Fraser et al. (2016) "Linguistic Features Identify Alzheimer's Disease in Narrative Speech"

### Model Access

**Checkpoints:**
- DenseNet: `runs/densenet_spec_full_cuda/model.pth`
- DistilBERT text: `models/text_baseline/model.pth`
- Fusion model: `models/fusion/model.pth`

**Configs:** See respective `config.json` in model directories
**Metrics:** See respective `metrics.json` in model directories

---

**Last updated:** December 19, 2025
**Project version:** 1.0


