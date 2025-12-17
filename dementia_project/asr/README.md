## ASR module

Goal: generate transcripts + word timestamps for each audio file to enable word-level
audio segmentation and multimodal fusion.

Planned backends:
- WhisperX (preferred): transcript + forced alignment word timestamps.
- Fallback: Whisper segment-level timestamps (less granular; no forced alignment).

This repo will cache ASR outputs under `data/processed/asr/`.


