"""Inference utilities for trained models.

Separate from training code per ML best practices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import densenet121

from dementia_project.features.spectrograms import (
    MelSpecConfig,
    load_mono_resampled,
    log_mel_spectrogram,
)


def load_densenet_model(
    checkpoint_path: Path, device: torch.device | None = None
) -> nn.Module:
    """Load trained DenseNet model from checkpoint.

    Args:
        checkpoint_path: Path to PyTorch checkpoint.
        device: Device to load model on.

    Returns:
        Loaded model in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    model.eval()
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


def predict_audio_file(
    audio_path: Path | str,
    model: nn.Module,
    cfg: MelSpecConfig | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Predict dementia probability for a single audio file.

    Args:
        audio_path: Path to audio file.
        model: Trained DenseNet model.
        cfg: Spectrogram configuration.
        device: Device to run inference on.

    Returns:
        Dictionary with prediction results:
        - predicted_class: 0 (control) or 1 (dementia)
        - probabilities: [prob_control, prob_dementia]
        - confidence: max probability
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg is None:
        cfg = MelSpecConfig(max_audio_sec=10.0)

    model.eval()
    model = model.to(device)

    # Load and process audio
    wav = load_mono_resampled(str(audio_path), cfg.sample_rate_hz)
    spec = log_mel_spectrogram(wav, cfg)
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    x = spec.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # [1, 3, H, W]

    # Predict
    with torch.no_grad():
        x_tensor = x.to(device)
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1).item()
        probabilities = probs[0].cpu().numpy().tolist()

    return {
        "predicted_class": int(pred_class),
        "probabilities": probabilities,
        "confidence": float(probabilities[pred_class]),
        "audio_path": str(audio_path),
    }


def batch_predict(
    audio_paths: list[Path | str],
    checkpoint_path: Path,
    cfg: MelSpecConfig | None = None,
    device: torch.device | None = None,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Predict dementia probability for multiple audio files.

    Args:
        audio_paths: List of paths to audio files.
        checkpoint_path: Path to trained model checkpoint.
        cfg: Spectrogram configuration.
        device: Device to run inference on.
        batch_size: Batch size for processing.

    Returns:
        List of prediction dictionaries (one per audio file).
    """
    model = load_densenet_model(checkpoint_path, device=device)
    results = []

    for audio_path in audio_paths:
        result = predict_audio_file(audio_path, model, cfg, device)
        results.append(result)

    return results


def save_predictions(predictions: list[dict[str, Any]], output_path: Path) -> None:
    """Save predictions to JSON file.

    Args:
        predictions: List of prediction dictionaries.
        output_path: Path to save JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions, indent=2))
    print(f"Saved {len(predictions)} predictions to: {output_path}")
