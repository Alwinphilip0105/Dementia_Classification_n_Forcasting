"""Fusion dataset: Combines text transcripts with word-level audio embeddings.

This dataset loads:
1. Text transcripts (for DistilBERT encoding)
2. Pre-computed word-level audio embeddings (from align_audio.py)
3. Labels (dementia vs control)

For the fusion model with cross-attention between text and audio.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from dementia_project.features.text_features import build_text_dataframe


class FusionDataset(Dataset):
    """Dataset for multimodal fusion training.

    Combines text transcripts with word-level audio embeddings.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        word_embed_dir: Path,
        tokenizer: AutoTokenizer,
        max_text_length: int = 512,
        max_audio_words: int = 256,
    ):
        """Initialize fusion dataset.

        Args:
            df: DataFrame with columns: audio_path, text, label
            word_embed_dir: Directory containing .pt word embedding files
            tokenizer: Text tokenizer (e.g., DistilBERT)
            max_text_length: Maximum text sequence length
            max_audio_words: Maximum number of audio word embeddings
        """
        self.df = df.reset_index(drop=True)
        self.word_embed_dir = Path(word_embed_dir)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.max_audio_words = max_audio_words

    def __len__(self) -> int:
        return len(self.df)

    def _get_word_embed_path(self, audio_path: str) -> Path:
        """Convert audio_path to word embedding .pt filename.

        Args:
            audio_path: Original audio path (e.g., "dementia-20251218.../file.wav")

        Returns:
            Path to corresponding .pt file
        """
        # Match the naming convention from align_audio.py
        safe_name = audio_path.replace("/", "__").replace(" ", "_")
        return self.word_embed_dir / f"{safe_name}.pt"

    def _load_word_embeddings(self, audio_path: str) -> torch.Tensor:
        """Load word-level audio embeddings from .pt file.

        Args:
            audio_path: Original audio path

        Returns:
            Tensor of shape [num_words, 768] (padded/truncated to max_audio_words)
        """
        embed_path = self._get_word_embed_path(audio_path)

        if not embed_path.exists():
            # Return zero embeddings if file not found
            return torch.zeros(self.max_audio_words, 768)

        # Load embeddings
        data = torch.load(embed_path, weights_only=False)
        num_words = data.get("num_words", 0)

        if num_words == 0:
            return torch.zeros(self.max_audio_words, 768)

        # Extract word embeddings in order
        word_vectors = []
        for i in range(num_words):
            key = f"word_{i}"
            if key in data:
                word_vectors.append(data[key])

        if len(word_vectors) == 0:
            return torch.zeros(self.max_audio_words, 768)

        # Stack into [num_words, 768]
        embeddings = torch.stack(word_vectors, dim=0)

        # Truncate or pad to max_audio_words
        if embeddings.shape[0] > self.max_audio_words:
            embeddings = embeddings[: self.max_audio_words, :]
        elif embeddings.shape[0] < self.max_audio_words:
            pad_size = self.max_audio_words - embeddings.shape[0]
            padding = torch.zeros(pad_size, 768)
            embeddings = torch.cat([embeddings, padding], dim=0)

        return embeddings

    def _create_audio_mask(self, audio_path: str) -> torch.Tensor:
        """Create attention mask for audio word embeddings.

        Args:
            audio_path: Original audio path

        Returns:
            Binary mask [max_audio_words] where 1 = valid word, 0 = padding
        """
        embed_path = self._get_word_embed_path(audio_path)

        if not embed_path.exists():
            return torch.zeros(self.max_audio_words, dtype=torch.long)

        data = torch.load(embed_path, weights_only=False)
        num_words = min(data.get("num_words", 0), self.max_audio_words)

        # Create mask: 1 for valid words, 0 for padding
        mask = torch.zeros(self.max_audio_words, dtype=torch.long)
        if num_words > 0:
            mask[:num_words] = 1

        return mask

    def __getitem__(self, idx: int):
        """Get a single sample.

        Returns:
            Dictionary with keys:
                - text_input_ids: [max_text_length]
                - text_attention_mask: [max_text_length]
                - audio_embeddings: [max_audio_words, 768]
                - audio_mask: [max_audio_words]
                - label: scalar
        """
        row = self.df.iloc[idx]

        audio_path = row["audio_path"]
        text = str(row["text"])
        label = int(row["label"])

        # Tokenize text
        text_encoding = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Load word-level audio embeddings
        audio_embeddings = self._load_word_embeddings(audio_path)
        audio_mask = self._create_audio_mask(audio_path)

        return {
            "text_input_ids": text_encoding["input_ids"].squeeze(0),
            "text_attention_mask": text_encoding["attention_mask"].squeeze(0),
            "audio_embeddings": audio_embeddings,  # [max_audio_words, 768]
            "audio_mask": audio_mask,  # [max_audio_words]
            "label": torch.tensor(label, dtype=torch.long),
        }


def build_fusion_dataloaders(
    metadata_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    asr_manifest_df: pd.DataFrame,
    word_embed_dir: Path,
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_text_length: int = 512,
    max_audio_words: int = 256,
):
    """Build train/valid/test dataloaders for fusion model.

    Args:
        metadata_df: Metadata with audio_path, label
        splits_df: Splits with audio_path, split
        asr_manifest_df: ASR manifest with audio_path, transcript_json
        word_embed_dir: Directory with word embedding .pt files
        tokenizer: Text tokenizer
        batch_size: Batch size for dataloaders
        max_text_length: Max text sequence length
        max_audio_words: Max audio word embeddings

    Returns:
        Dictionary with keys: train_loader, valid_loader, test_loader
    """
    # Build text dataframe
    text_df = build_text_dataframe(metadata_df, asr_manifest_df)

    # Merge with splits
    df = text_df.merge(splits_df[["audio_path", "split"]], on="audio_path", how="inner")

    # Create datasets for each split
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    valid_df = df[df["split"] == "valid"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    train_dataset = FusionDataset(
        train_df, word_embed_dir, tokenizer, max_text_length, max_audio_words
    )
    valid_dataset = FusionDataset(
        valid_df, word_embed_dir, tokenizer, max_text_length, max_audio_words
    )
    test_dataset = FusionDataset(
        test_df, word_embed_dir, tokenizer, max_text_length, max_audio_words
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
        "train_size": len(train_df),
        "valid_size": len(valid_df),
        "test_size": len(test_df),
    }
