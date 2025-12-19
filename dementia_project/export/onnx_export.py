"""ONNX export for PyTorch models."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import densenet121


def export_densenet_to_onnx(
    model_path: Path | None,
    output_path: Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 128, 500),
    device: str = "cpu",
) -> None:
    """Export DenseNet model to ONNX format.

    Args:
        model_path: Path to saved PyTorch model checkpoint. If None, creates a new model.
        output_path: Path to save ONNX model.
        input_shape: Input shape (batch, channels, height, width).
        device: Device to run export on.
    """
    # Load or create model
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    if model_path is not None and model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    model.eval()

    model = model.to(device)

    # Create dummy input
    dummy_input = torch.randn(*input_shape).to(device)

    # Export to ONNX
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["spectrogram"],
        output_names=["logits"],
        dynamic_axes={
            "spectrogram": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=13,
        do_constant_folding=True,
    )

    print(f"Exported ONNX model to: {output_path}")


def export_fusion_model_to_onnx(
    model_path: Path | None,
    output_path: Path,
    audio_seq_len: int = 50,
    device: str = "cpu",
) -> None:
    """Export fusion model to ONNX format.

    Args:
        model_path: Path to saved PyTorch model checkpoint. If None, creates a new model.
        output_path: Path to save ONNX model.
        audio_seq_len: Maximum audio sequence length (words).
        device: Device to run export on.
    """
    from dementia_project.models.fusion_model import MultimodalFusionClassifier

    # Load or create model
    model = MultimodalFusionClassifier(
        text_encoder_dim=768, audio_encoder_dim=768, hidden_dim=256
    )
    if model_path is not None and model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    model.eval()

    model = model.to(device)

    # Create dummy inputs
    dummy_audio = torch.randn(1, audio_seq_len, 768).to(
        device
    )  # [batch, words, audio_dim]
    dummy_text = torch.randn(1, 1, 768).to(device)  # [batch, 1, text_dim]

    # Export to ONNX
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_audio, dummy_text),
        str(output_path),
        input_names=["audio_embeddings", "text_embeddings"],
        output_names=["logits"],
        dynamic_axes={
            "audio_embeddings": {0: "batch_size", 1: "sequence_length"},
            "text_embeddings": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=13,
        do_constant_folding=True,
    )

    print(f"Exported ONNX model to: {output_path}")
