"""Multimodal fusion model with cross-attention between text and word-level audio.

Architecture:
1. Frozen DistilBERT encoder for text
2. Pre-computed Wav2Vec2 word embeddings for audio
3. Cross-attention: text (query) attends to audio (key/value)
4. Classification head on fused representation
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion layer: text queries audio features."""

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
            text_dim: Dimension of text embeddings
            audio_dim: Dimension of audio embeddings
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim

        # Project text and audio to same dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # Multi-head cross-attention: text queries audio
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Layer norm and dropout
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-attention fusion.

        Args:
            text_emb: Text embeddings [batch, text_seq_len, text_dim]
            audio_emb: Audio word embeddings [batch, audio_seq_len, audio_dim]
            audio_mask: Audio attention mask [batch, audio_seq_len] (1=valid, 0=padding)

        Returns:
            Tuple of:
                - Fused embeddings [batch, hidden_dim]
                - Attention weights [batch, text_seq_len, audio_seq_len]
        """
        # Project to same dimension
        text_proj = self.text_proj(text_emb)  # [batch, text_seq_len, hidden_dim]
        audio_proj = self.audio_proj(audio_emb)  # [batch, audio_seq_len, hidden_dim]

        # Create attention mask for cross-attention
        # key_padding_mask: True where padding (0 in audio_mask)
        key_padding_mask = None
        if audio_mask is not None:
            key_padding_mask = (audio_mask == 0)  # [batch, audio_seq_len]

        # Cross-attention: text (query) attends to audio (key/value)
        fused, attn_weights = self.cross_attn(
            query=text_proj,
            key=audio_proj,
            value=audio_proj,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # Return per-head weights
        )
        # fused: [batch, text_seq_len, hidden_dim]
        # attn_weights: [batch, num_heads, text_seq_len, audio_seq_len]

        # Mean pool over text sequence
        pooled = fused.mean(dim=1)  # [batch, hidden_dim]
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)

        # Average attention weights across heads
        attn_weights = attn_weights.mean(dim=1)  # [batch, text_seq_len, audio_seq_len]

        return pooled, attn_weights


class FusionClassifier(nn.Module):
    """Multimodal fusion classifier with frozen encoders.

    Uses:
    - Frozen DistilBERT for text encoding
    - Pre-computed Wav2Vec2 word embeddings
    - Cross-attention fusion
    - Learnable classification head
    """

    def __init__(
        self,
        text_model_name: str = "distilbert-base-uncased",
        text_dim: int = 768,
        audio_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.1,
        freeze_text_encoder: bool = True,
    ):
        """Initialize fusion classifier.

        Args:
            text_model_name: HuggingFace model name for text encoder
            text_dim: Output dimension of text encoder
            audio_dim: Dimension of audio word embeddings
            hidden_dim: Hidden dimension for fusion
            num_heads: Number of attention heads
            num_classes: Number of output classes
            dropout: Dropout rate
            freeze_text_encoder: Whether to freeze text encoder weights
        """
        super().__init__()

        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.freeze_text_encoder = freeze_text_encoder

        # Text encoder (DistilBERT)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        if freeze_text_encoder:
            # Freeze all text encoder parameters
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            text_dim=text_dim,
            audio_dim=audio_dim,
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
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_embeddings: torch.Tensor,
        audio_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            text_input_ids: Text token IDs [batch, text_seq_len]
            text_attention_mask: Text attention mask [batch, text_seq_len]
            audio_embeddings: Pre-computed word embeddings [batch, audio_seq_len, 768]
            audio_mask: Audio mask [batch, audio_seq_len] (1=valid, 0=padding)
            return_attention: If True, return attention weights

        Returns:
            Logits [batch, num_classes]
            Or tuple of (logits, attention_weights) if return_attention=True
        """
        # Encode text (frozen or fine-tuned)
        if self.freeze_text_encoder:
            with torch.no_grad():
                text_outputs = self.text_encoder(
                    input_ids=text_input_ids, attention_mask=text_attention_mask
                )
        else:
            text_outputs = self.text_encoder(
                input_ids=text_input_ids, attention_mask=text_attention_mask
            )

        text_embeddings = text_outputs.last_hidden_state  # [batch, text_seq_len, 768]

        # Fuse modalities with cross-attention
        fused, attn_weights = self.fusion(
            text_embeddings, audio_embeddings, audio_mask
        )  # [batch, hidden_dim]

        # Classify
        logits = self.classifier(fused)  # [batch, num_classes]

        if return_attention:
            return logits, attn_weights
        return logits
