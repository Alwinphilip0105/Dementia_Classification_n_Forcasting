"""Multimodal fusion model with cross-attention between text and word-level audio."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion layer.

    Given text embeddings and audio embeddings, applies cross-attention to align them.
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize cross-attention fusion layer.

        Args:
            text_dim: Dimension of text embeddings.
            audio_dim: Dimension of audio embeddings.
            hidden_dim: Hidden dimension for attention.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim

        # Project text and audio to same dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Layer norm and dropout
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_emb: torch.Tensor, audio_emb: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention fusion.

        Args:
            text_emb: Text embeddings [batch, text_seq_len, text_dim] or [batch, text_dim]
            audio_emb: Audio embeddings [batch, audio_seq_len, audio_dim]

        Returns:
            Fused embeddings [batch, hidden_dim]
        """
        # Handle case where text_emb is [batch, text_dim] -> expand to [batch, 1, text_dim]
        if text_emb.ndim == 2:
            text_emb = text_emb.unsqueeze(1)  # [batch, 1, text_dim]

        # Project to same dimension
        text_proj = self.text_proj(text_emb)  # [batch, text_seq_len, hidden_dim]
        audio_proj = self.audio_proj(audio_emb)  # [batch, audio_seq_len, hidden_dim]

        # Cross-attention: audio attends to text
        # Query: audio, Key/Value: text
        fused, _ = self.cross_attn(query=audio_proj, key=text_proj, value=text_proj)
        # fused: [batch, audio_seq_len, hidden_dim]

        # Mean pool over sequence
        pooled = fused.mean(dim=1)  # [batch, hidden_dim]
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)

        return pooled


class MultimodalFusionClassifier(nn.Module):
    """Multimodal classifier fusing text and word-level audio embeddings."""

    def __init__(
        self,
        text_encoder_dim: int = 768,  # RoBERTa-base
        audio_encoder_dim: int = 768,  # Wav2Vec2-base
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize multimodal fusion classifier.

        Args:
            text_encoder_dim: Output dimension of text encoder.
            audio_encoder_dim: Output dimension of audio encoder.
            hidden_dim: Hidden dimension for fusion.
            num_heads: Number of attention heads.
            num_classes: Number of output classes.
            dropout: Dropout rate.
        """
        super().__init__()

        self.fusion = CrossAttentionFusion(
            text_dim=text_encoder_dim,
            audio_dim=audio_encoder_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        text_emb: torch.Tensor,
        audio_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            text_emb: Text embeddings [batch, text_seq_len, text_encoder_dim]
            audio_emb: Audio embeddings [batch, audio_seq_len, audio_encoder_dim]

        Returns:
            Logits [batch, num_classes]
        """
        # Fuse modalities
        fused = self.fusion(text_emb, audio_emb)  # [batch, hidden_dim]

        # Classify
        logits = self.classifier(fused)  # [batch, num_classes]

        return logits
