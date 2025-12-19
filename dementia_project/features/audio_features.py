"""Hand-crafted audio features for non-ML baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass(frozen=True)
class MfccConfig:
    sample_rate_hz: int = 16_000
    n_mfcc: int = 13
    n_fft: int = 1024
    hop_length: int = 256


def _summarize_matrix(mat: np.ndarray, prefix: str) -> dict[str, float]:
    """Summarize a feature matrix into scalar statistics."""
    # mat: (features, frames)
    means = mat.mean(axis=1)
    stds = mat.std(axis=1)
    out: dict[str, float] = {}
    for i, (m, s) in enumerate(zip(means, stds)):
        out[f"{prefix}_{i:02d}_mean"] = float(m)
        out[f"{prefix}_{i:02d}_std"] = float(s)
    return out


def extract_mfcc_pause_features(path: Path, cfg: MfccConfig) -> dict[str, float]:
    """Extract MFCC summary stats + simple pause/energy stats."""
    try:
        y, sr = librosa.load(path, sr=cfg.sample_rate_hz, mono=True)
    except Exception:  # noqa: BLE001
        # Keep the feature vector numeric-only for sklearn pipelines.
        return {"audio_load_error": 1.0}

    if y.size == 0:
        return {"audio_empty": 1.0}

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=cfg.n_mfcc, n_fft=cfg.n_fft, hop_length=cfg.hop_length
    )
    rms = librosa.feature.rms(y=y, frame_length=cfg.n_fft, hop_length=cfg.hop_length)[0]

    # Pause proxy: fraction of frames below an adaptive threshold.
    thr = float(np.percentile(rms, 20.0))
    low_energy_frac = float(np.mean(rms <= thr))

    feats = {}
    feats.update(_summarize_matrix(mfcc, "mfcc"))
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"] = float(np.std(rms))
    feats["low_energy_frac_p20"] = low_energy_frac
    feats["duration_sec_sr16k"] = float(len(y) / sr)
    return feats
