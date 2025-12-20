"""Conformance test: Compare PyTorch vs ONNX model outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision.models import densenet121


def test_densenet_onnx_conformance(
    onnx_path: Path,
    pytorch_model_path: Path | None = None,
    num_test_samples: int = 5,
    tolerance: float = 1e-4,
) -> dict[str, float | bool]:
    """Test ONNX model conformance against PyTorch model.

    Args:
        onnx_path: Path to ONNX model.
        pytorch_model_path: Path to PyTorch checkpoint (optional).
        num_test_samples: Number of test samples to compare.
        tolerance: Maximum allowed absolute difference.

    Returns:
        Dictionary with test results.
    """
    device = "cpu"

    # Load ONNX model
    ort_session = ort.InferenceSession(str(onnx_path))

    # Create or load PyTorch model
    # For conformance test, we need the same model weights as ONNX
    # If no checkpoint, we'll export a dummy model and test it
    pytorch_model = densenet121(weights=None)
    pytorch_model.classifier = nn.Linear(pytorch_model.classifier.in_features, 2)
    if pytorch_model_path is not None and pytorch_model_path.exists():
        checkpoint = torch.load(pytorch_model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            pytorch_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            pytorch_model.load_state_dict(checkpoint)
    pytorch_model.eval()

    pytorch_model = pytorch_model.to(device)

    # Test on random inputs
    max_diff = 0.0
    all_diffs: list[float] = []

    for i in range(num_test_samples):
        # Generate random input
        dummy_input = torch.randn(1, 3, 128, 500).to(device)

        # PyTorch forward
        with torch.no_grad():
            pytorch_output = pytorch_model(dummy_input)
            pytorch_logits = pytorch_output.cpu().numpy()

        # ONNX forward
        onnx_input = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        onnx_output = ort_session.run(None, onnx_input)
        onnx_logits = onnx_output[0]

        # Compare
        diff = np.abs(pytorch_logits - onnx_logits).max()
        max_diff = max(max_diff, diff)
        all_diffs.append(float(diff))

    passed = max_diff < tolerance

    results = {
        "max_absolute_difference": float(max_diff),
        "mean_absolute_difference": float(np.mean(all_diffs)),
        "tolerance": tolerance,
        "passed": passed,
        "num_test_samples": num_test_samples,
    }

    print("ONNX Conformance Test Results:")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {np.mean(all_diffs):.6f}")
    print(f"  Tolerance: {tolerance}")
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")

    return results


def test_fusion_onnx_conformance(
    onnx_path: Path,
    pytorch_model_path: Path | None = None,
    num_test_samples: int = 5,
    tolerance: float = 1e-4,
) -> dict[str, float | bool]:
    """Test fusion model ONNX conformance.

    Args:
        onnx_path: Path to ONNX model.
        pytorch_model_path: Path to PyTorch checkpoint (optional).
        num_test_samples: Number of test samples to compare.
        tolerance: Maximum allowed absolute difference.

    Returns:
        Dictionary with test results.
    """
    from dementia_project.models.fusion_model import MultimodalFusionClassifier

    device = "cpu"

    # Load ONNX model
    ort_session = ort.InferenceSession(str(onnx_path))

    # Create or load PyTorch model
    pytorch_model = MultimodalFusionClassifier(
        text_encoder_dim=768, audio_encoder_dim=768, hidden_dim=256
    )
    if pytorch_model_path is not None and pytorch_model_path.exists():
        checkpoint = torch.load(pytorch_model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            pytorch_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            pytorch_model.load_state_dict(checkpoint)
    pytorch_model.eval()

    pytorch_model = pytorch_model.to(device)

    # Test on random inputs
    max_diff = 0.0
    all_diffs: list[float] = []

    for i in range(num_test_samples):
        # Generate random inputs
        audio_seq_len = np.random.randint(10, 50)
        dummy_audio = torch.randn(1, audio_seq_len, 768).to(device)
        dummy_text = torch.randn(1, 1, 768).to(device)

        # PyTorch forward
        with torch.no_grad():
            pytorch_output = pytorch_model(dummy_audio, dummy_text)
            pytorch_logits = pytorch_output.cpu().numpy()

        # ONNX forward
        input_names = [inp.name for inp in ort_session.get_inputs()]
        onnx_input = {
            input_names[0]: dummy_audio.cpu().numpy(),
            input_names[1]: dummy_text.cpu().numpy(),
        }
        onnx_output = ort_session.run(None, onnx_input)
        onnx_logits = onnx_output[0]

        # Compare
        diff = np.abs(pytorch_logits - onnx_logits).max()
        max_diff = max(max_diff, diff)
        all_diffs.append(float(diff))

    passed = max_diff < tolerance

    results = {
        "max_absolute_difference": float(max_diff),
        "mean_absolute_difference": float(np.mean(all_diffs)),
        "tolerance": tolerance,
        "passed": passed,
        "num_test_samples": num_test_samples,
    }

    print("ONNX Conformance Test Results:")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {np.mean(all_diffs):.6f}")
    print(f"  Tolerance: {tolerance}")
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")

    return results
