"""Explainability analysis using Captum."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, Saliency
from captum.attr import visualization as viz


def explain_densenet_with_integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int = 1,
    device: torch.device | None = None,
) -> tuple[np.ndarray, dict]:
    """Compute Integrated Gradients attribution for DenseNet model.

    Args:
        model: Trained DenseNet model.
        input_tensor: Input spectrogram tensor [1, 3, H, W].
        target_class: Target class index (0 or 1).
        device: Device to run on.

    Returns:
        Tuple of (attributions, metadata dict).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Integrated Gradients
    ig = IntegratedGradients(model)
    attributions = ig.attribute(
        input_tensor,
        target=target_class,
        n_steps=50,
        internal_batch_size=1,
    )

    # Convert to numpy
    attributions_np = attributions.detach().cpu().numpy()[0]  # [3, H, W]
    # Average over channels for visualization
    attributions_avg = attributions_np.mean(axis=0)  # [H, W]

    metadata = {
        "target_class": int(target_class),
        "attribution_shape": list(attributions_np.shape),
        "attribution_min": float(attributions_avg.min()),
        "attribution_max": float(attributions_avg.max()),
        "attribution_mean": float(attributions_avg.mean()),
    }

    return attributions_avg, metadata


def visualize_attributions(
    input_tensor: torch.Tensor,
    attributions: np.ndarray,
    output_path: Path,
    title: str = "Integrated Gradients Attribution",
) -> None:
    """Visualize attributions as a heatmap.

    Args:
        input_tensor: Original input [1, 3, H, W].
        attributions: Attribution map [H, W].
        output_path: Path to save visualization.
        title: Plot title.
    """
    # Get input for visualization (average over channels)
    input_np = input_tensor.detach().cpu().numpy()[0].mean(axis=0)  # [H, W]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original input
    im1 = axes[0].imshow(input_np, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("Input Spectrogram")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Frequency")
    plt.colorbar(im1, ax=axes[0])

    # Attributions (use symmetric colormap centered at 0)
    from matplotlib.colors import TwoSlopeNorm

    vmax = max(abs(attributions.min()), abs(attributions.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im2 = axes[1].imshow(
        attributions, aspect="auto", origin="lower", cmap="RdBu_r", norm=norm
    )
    axes[1].set_title(title)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Frequency")
    plt.colorbar(im2, ax=axes[1])

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def explain_fusion_model_attention(
    model: nn.Module,
    audio_emb: torch.Tensor,
    text_emb: torch.Tensor,
    target_class: int = 1,
    device: torch.device | None = None,
) -> dict:
    """Extract attention weights from fusion model for explainability.

    Args:
        model: Trained fusion model.
        audio_emb: Audio embeddings [1, seq_len, audio_dim].
        text_emb: Text embeddings [1, 1, text_dim].
        target_class: Target class index.
        device: Device to run on.

    Returns:
        Dictionary with attention weights and metadata.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model = model.to(device)
    audio_emb = audio_emb.to(device)
    text_emb = text_emb.to(device)

    # Hook to capture attention weights
    attention_weights: list[torch.Tensor] = []

    def attention_hook(module, input, output):
        # output is (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) > 1:
            attention_weights.append(output[1].detach().cpu())

    # Register hook on cross-attention layer
    hook_handle = model.fusion.cross_attn.register_forward_hook(attention_hook)

    # Forward pass
    with torch.no_grad():
        logits = model(audio_emb, text_emb)
        probs = torch.softmax(logits, dim=1)

    hook_handle.remove()

    # Extract attention if available
    attn_data = {}
    if attention_weights:
        attn = attention_weights[0]  # [batch, num_heads, seq_len, seq_len]
        # Average over heads
        attn_avg = attn.mean(dim=1)[0]  # [seq_len, seq_len]
        attn_data = {
            "attention_weights": attn_avg.numpy().tolist(),
            "audio_seq_len": int(audio_emb.shape[1]),
            "text_seq_len": int(text_emb.shape[1]),
        }

    return {
        "target_class": int(target_class),
        "predicted_class": int(logits.argmax(dim=1).item()),
        "probabilities": probs[0].cpu().numpy().tolist(),
        "attention": attn_data,
    }


def visualize_fusion_attention(
    attention_weights: np.ndarray,
    output_path: Path,
    title: str = "Cross-Attention Weights (Audio → Text)",
) -> None:
    """Visualize cross-attention weights.

    Args:
        attention_weights: Attention matrix [audio_seq_len, text_seq_len].
        output_path: Path to save visualization.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attention_weights, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Text Position")
    ax.set_ylabel("Audio Word Position")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
