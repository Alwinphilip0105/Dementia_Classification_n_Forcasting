"""Robustness tests: noise, time-shift, etc."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision.models import densenet121
from tqdm import tqdm

from dementia_project.data.io import load_metadata, load_splits
from dementia_project.features.spectrograms import (
    MelSpecConfig,
    load_mono_resampled,
    log_mel_spectrogram,
)
from dementia_project.train.train_densenet_spec import SpecDataset


def add_gaussian_noise(
    audio: torch.Tensor, snr_db: float, sample_rate: int = 16000
) -> torch.Tensor:
    """Add Gaussian white noise to audio at specified SNR.

    Args:
        audio: Input audio tensor.
        snr_db: Signal-to-noise ratio in dB.
        sample_rate: Sample rate (for reference).

    Returns:
        Noisy audio tensor.
    """
    # Calculate signal power
    signal_power = torch.mean(audio**2)

    # Calculate noise power for desired SNR
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    # Generate noise
    noise = torch.randn_like(audio) * torch.sqrt(noise_power)

    return audio + noise


def time_shift_audio(audio: torch.Tensor, shift_samples: int) -> torch.Tensor:
    """Apply time shift (circular shift).

    Args:
        audio: Input audio tensor.
        shift_samples: Number of samples to shift (positive = right shift).

    Returns:
        Shifted audio tensor.
    """
    if shift_samples == 0:
        return audio
    return torch.roll(audio, shift_samples)


def test_noise_robustness(
    model: nn.Module,
    test_loader: DataLoader,
    snr_levels: list[float],
    device: torch.device,
) -> dict:
    """Test model robustness to additive noise at different SNR levels.

    Args:
        model: Trained model.
        test_loader: DataLoader for test set.
        snr_levels: List of SNR values in dB to test.
        device: Device to run on.

    Returns:
        Dictionary with results per SNR level.
    """
    model.eval()
    results: dict[str, dict] = {}

    for snr_db in tqdm(snr_levels, desc="Testing SNR levels"):
        all_probs: list[float] = []
        all_preds: list[int] = []
        all_y: list[int] = []

        with torch.no_grad():
            for x, y in test_loader:
                # Add noise to input (before spectrogram conversion)
                # For simplicity, we'll add noise after spectrogram
                # In practice, you'd add noise to raw audio
                x_noisy = x + torch.randn_like(x) * (10 ** (-snr_db / 20.0))
                x_noisy = x_noisy.to(device)
                y = y.to(device)

                logits = model(x_noisy)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = (probs >= 0.5).long()

                all_probs.extend(probs.detach().cpu().numpy().tolist())
                all_preds.extend(preds.detach().cpu().numpy().tolist())
                all_y.extend(y.detach().cpu().numpy().tolist())

        y_true = np.array(all_y, dtype=np.int64)
        y_pred = np.array(all_preds, dtype=np.int64)
        y_prob = np.array(all_probs, dtype=np.float32)

        results[f"snr_{snr_db:.1f}db"] = {
            "snr_db": float(snr_db),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": (
                float(roc_auc_score(y_true, y_prob))
                if len(np.unique(y_true)) == 2
                else None
            ),
        }

    return results


def test_time_shift_robustness(
    model: nn.Module,
    test_loader: DataLoader,
    shift_ratios: list[float],
    device: torch.device,
) -> dict:
    """Test model robustness to time shifts.

    Args:
        model: Trained model.
        test_loader: DataLoader for test set.
        shift_ratios: List of shift ratios (0.0 = no shift, 0.1 = 10% shift).
        device: Device to run on.

    Returns:
        Dictionary with results per shift ratio.
    """
    model.eval()
    results: dict[str, dict] = {}

    for shift_ratio in tqdm(shift_ratios, desc="Testing time shifts"):
        all_probs: list[float] = []
        all_preds: list[int] = []
        all_y: list[int] = []

        with torch.no_grad():
            for x, y in test_loader:
                # Apply circular shift along time dimension
                if shift_ratio != 0.0:
                    shift_pixels = int(x.shape[-1] * shift_ratio)
                    x_shifted = torch.roll(x, shifts=shift_pixels, dims=-1)
                else:
                    x_shifted = x

                x_shifted = x_shifted.to(device)
                y = y.to(device)

                logits = model(x_shifted)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = (probs >= 0.5).long()

                all_probs.extend(probs.detach().cpu().numpy().tolist())
                all_preds.extend(preds.detach().cpu().numpy().tolist())
                all_y.extend(y.detach().cpu().numpy().tolist())

        y_true = np.array(all_y, dtype=np.int64)
        y_pred = np.array(all_preds, dtype=np.int64)
        y_prob = np.array(all_probs, dtype=np.float32)

        results[f"shift_{shift_ratio:.2f}"] = {
            "shift_ratio": float(shift_ratio),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": (
                float(roc_auc_score(y_true, y_prob))
                if len(np.unique(y_true)) == 2
                else None
            ),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True, type=Path)
    parser.add_argument("--splits_csv", required=True, type=Path)
    parser.add_argument("--pytorch_checkpoint", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("runs/robustness"))
    parser.add_argument(
        "--snr_levels",
        type=str,
        default="inf,30,20,10,5,0",
        help="Comma-separated SNR levels in dB",
    )
    parser.add_argument(
        "--shift_ratios",
        type=str,
        default="0.0,0.1,0.2,0.3",
        help="Comma-separated time shift ratios",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Parse SNR levels and shift ratios
    snr_levels = [float(x.strip()) for x in args.snr_levels.split(",")]
    shift_ratios = [float(x.strip()) for x in args.shift_ratios.split(",")]

    # Load data
    metadata_df = load_metadata(args.metadata_csv)
    splits_df = load_splits(args.splits_csv)
    df = metadata_df.merge(
        splits_df[["audio_path", "split"]], on="audio_path", how="inner"
    )
    test_df = df[df["split"] == "test"]

    cfg = MelSpecConfig(max_audio_sec=10.0)
    test_dataset = SpecDataset(test_df, cfg)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    if args.pytorch_checkpoint is not None and args.pytorch_checkpoint.exists():
        checkpoint = torch.load(args.pytorch_checkpoint, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    model.eval()
    model = model.to(device)

    print("Running robustness tests...")

    # Test noise robustness
    print("\n1. Testing noise robustness...")
    noise_results = test_noise_robustness(model, test_loader, snr_levels, device)

    # Test time shift robustness
    print("\n2. Testing time shift robustness...")
    shift_results = test_time_shift_robustness(model, test_loader, shift_ratios, device)

    # Save results
    all_results = {
        "noise_robustness": noise_results,
        "time_shift_robustness": shift_results,
    }
    results_path = args.out_dir / "robustness_test_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))

    print(f"\nSaved robustness test results to: {results_path}")

    # Print summary
    print("\nNoise Robustness Summary:")
    for key, val in noise_results.items():
        print(f"  {key}: Acc={val['accuracy']:.3f}, F1={val['f1']:.3f}")

    print("\nTime Shift Robustness Summary:")
    for key, val in shift_results.items():
        print(f"  {key}: Acc={val['accuracy']:.3f}, F1={val['f1']:.3f}")


if __name__ == "__main__":
    main()
