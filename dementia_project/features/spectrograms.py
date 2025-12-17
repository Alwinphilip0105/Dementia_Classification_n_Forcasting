"""Spectrogram feature utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio


@dataclass(frozen=True)
class MelSpecConfig:
    sample_rate_hz: int = 16_000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    f_min: float = 20.0
    f_max: float = 7600.0
    max_audio_sec: float = 10.0


def load_mono_resampled(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
    return wav


def log_mel_spectrogram(wav: torch.Tensor, cfg: MelSpecConfig) -> torch.Tensor:
    # Crop / pad
    max_samples = int(cfg.max_audio_sec * cfg.sample_rate_hz)
    if max_samples > 0:
        if wav.numel() > max_samples:
            wav = wav[:max_samples]
        elif wav.numel() < max_samples:
            wav = torch.nn.functional.pad(wav, (0, max_samples - wav.numel()))

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate_hz,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
        power=2.0,
    )(wav)
    log_mel = torch.log(mel + 1e-6)
    return log_mel  # (n_mels, time)


