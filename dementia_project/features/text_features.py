"""Text feature extraction from ASR transcripts using Transformer encoders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class TextEmbedConfig:
    """Configuration for text embedding extraction."""

    model_name: str = "roberta-base"
    max_length: int = 512
    device: str | None = None


def load_text_model(
    cfg: TextEmbedConfig, device: torch.device | None = None
) -> tuple[nn.Module, AutoTokenizer]:
    """Load a Transformer model and tokenizer for text encoding.

    Args:
        cfg: Configuration for the text model.
        device: Device to load the model on (default: auto-detect).

    Returns:
        Tuple of (model, tokenizer).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModel.from_pretrained(cfg.model_name).to(device)
    model.eval()

    return model, tokenizer


def load_transcript(transcript_json: Path) -> str:
    """Load transcript text from a JSON file.

    Args:
        transcript_json: Path to transcript.json file.

    Returns:
        Transcript text string.
    """
    data = json.loads(transcript_json.read_text(encoding="utf-8"))
    text = str(data.get("text", "")).strip()
    return text


@torch.no_grad()
def embed_text_mean_pool(
    text: str,
    cfg: TextEmbedConfig,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> np.ndarray:
    """Extract mean-pooled text embeddings from a transcript.

    Args:
        text: Input text string.
        cfg: Configuration for embedding extraction.
        model: Pre-loaded Transformer model.
        tokenizer: Pre-loaded tokenizer.
        device: Device to run inference on.

    Returns:
        Mean-pooled embedding vector (numpy array).
    """
    if not text:
        # Return zero vector if text is empty
        model_output = model.embeddings.word_embeddings(
            torch.zeros(1, 1, dtype=torch.long, device=device)
        )
        return model_output.squeeze(0).mean(dim=0).cpu().numpy()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=cfg.max_length,
        truncation=True,
        padding="max_length",
    ).to(device)

    outputs = model(**inputs)
    # Mean pool over sequence length (excluding padding)
    # Shape: [batch, seq_len, hidden] -> [batch, hidden]
    embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]
    # Mask out padding tokens
    masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
    pooled = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
    result: np.ndarray = pooled.squeeze(0).cpu().numpy()
    return result


def embed_file_mean_pool(
    transcript_json: Path,
    cfg: TextEmbedConfig,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> np.ndarray:
    """Extract mean-pooled embeddings from a transcript JSON file.

    Args:
        transcript_json: Path to transcript.json file.
        cfg: Configuration for embedding extraction.
        model: Pre-loaded Transformer model.
        tokenizer: Pre-loaded tokenizer.
        device: Device to run inference on.

    Returns:
        Mean-pooled embedding vector (numpy array).
    """
    text = load_transcript(transcript_json)
    return embed_text_mean_pool(text, cfg, model, tokenizer, device)
