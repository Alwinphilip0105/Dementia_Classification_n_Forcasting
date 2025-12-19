"""Wav2Vec2 embedding extraction for audio-only baselines."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import AutoFeatureExtractor, Wav2Vec2Model


@dataclass(frozen=True)
class Wav2Vec2EmbedConfig:
    model_name: str = "facebook/wav2vec2-base-960h"
    target_sample_rate_hz: int = 16_000
    max_audio_sec: float = 10.0


def _deterministic_offset(num_samples: int, crop_samples: int, key: str) -> int:
    if num_samples <= crop_samples:
        return 0
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    x = int(digest[:16], 16)
    return int(x % (num_samples - crop_samples))


def load_mono_resampled(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
    return wav


@torch.no_grad()
def embed_file_mean_pool(
    path: Path,
    cfg: Wav2Vec2EmbedConfig,
    model: Wav2Vec2Model,
    feature_extractor,
    device: torch.device,
) -> np.ndarray:
    wav = load_mono_resampled(path, cfg.target_sample_rate_hz)
    wav = wav.to(device)

    crop_samples = int(cfg.max_audio_sec * cfg.target_sample_rate_hz)
    if crop_samples > 0 and wav.numel() > crop_samples:
        off = _deterministic_offset(wav.numel(), crop_samples, str(path))
        wav = wav[off : off + crop_samples]

    inputs = feature_extractor(
        wav.cpu().numpy(), sampling_rate=cfg.target_sample_rate_hz, return_tensors="pt"
    )
    input_values = inputs["input_values"].to(device)

    out = model(input_values=input_values)
    hidden = out.last_hidden_state  # (B, T, H)
    pooled = hidden.mean(dim=1).squeeze(0)
    return np.asarray(pooled.detach().cpu().numpy(), dtype=np.float32)


def load_wav2vec2(cfg: Wav2Vec2EmbedConfig, device: torch.device):
    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.model_name)
    model = Wav2Vec2Model.from_pretrained(cfg.model_name)
    model.eval()
    model.to(device)
    return model, feature_extractor
