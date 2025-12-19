"""Dataset for multimodal fusion (word-level audio + text)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from dementia_project.features.text_features import (
    TextEmbedConfig,
    embed_text_mean_pool,
    load_text_model,
    load_transcript,
)
from dementia_project.features.wav2vec2_embed import (
    Wav2Vec2EmbedConfig,
    embed_file_mean_pool,
    load_wav2vec2,
)


class MultimodalFusionDataset(Dataset):
    """Dataset for multimodal fusion at word level.

    For each audio file:
    - Load word-level segments from word_segments.csv
    - Extract Wav2Vec2 embeddings for each word's audio
    - Load full transcript and extract text embeddings
    - Return aligned word-level audio embeddings + text embedding
    """

    def __init__(
        self,
        word_segments_df: pd.DataFrame,
        asr_manifest_df: pd.DataFrame,
        text_cfg: TextEmbedConfig,
        audio_cfg: Wav2Vec2EmbedConfig,
        text_model: torch.nn.Module,
        text_tokenizer,
        audio_model: torch.nn.Module,
        audio_feature_extractor,
        device: torch.device,
        max_words_per_sample: int = 50,
    ):
        """Initialize multimodal fusion dataset.

        Args:
            word_segments_df: DataFrame with word-level segments.
            asr_manifest_df: ASR manifest with transcript_json paths.
            text_cfg: Configuration for text embedding.
            audio_cfg: Configuration for audio embedding.
            text_model: Pre-loaded text encoder model.
            text_tokenizer: Pre-loaded text tokenizer.
            audio_model: Pre-loaded audio encoder model.
            audio_feature_extractor: Pre-loaded audio feature extractor.
            device: Device to run inference on.
            max_words_per_sample: Maximum words per sample (truncate if longer).
        """
        self.word_segments_df = word_segments_df.reset_index(drop=True)
        self.asr_manifest_df = asr_manifest_df.set_index("audio_path")
        self.text_cfg = text_cfg
        self.audio_cfg = audio_cfg
        self.text_model = text_model
        self.text_tokenizer = text_tokenizer
        self.audio_model = audio_model
        self.audio_feature_extractor = audio_feature_extractor
        self.device = device
        self.max_words_per_sample = max_words_per_sample

        # Group by audio_path to process words per audio file
        self.audio_groups = self.word_segments_df.groupby("audio_path")

    def __len__(self) -> int:
        return len(self.audio_groups)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample.

        Returns:
            Tuple of (word_audio_embeddings, text_embedding, label)
            - word_audio_embeddings: [num_words, audio_dim]
            - text_embedding: [text_dim] (mean-pooled)
            - label: scalar
        """
        audio_path = list(self.audio_groups.groups.keys())[idx]
        words_df = self.audio_groups.get_group(audio_path)

        # Get label (should be same for all words in same audio)
        label = int(words_df.iloc[0]["label"])

        # Load transcript
        if audio_path not in self.asr_manifest_df.index:
            raise ValueError(f"Audio path not in ASR manifest: {audio_path}")

        transcript_json = Path(
            str(self.asr_manifest_df.loc[audio_path, "transcript_json"])
        )
        if not transcript_json.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_json}")

        transcript_text = load_transcript(transcript_json)
        text_emb = embed_text_mean_pool(
            transcript_text,
            self.text_cfg,
            self.text_model,
            self.text_tokenizer,
            self.device,
        )
        text_emb_tensor = torch.from_numpy(text_emb).float()

        # Extract word-level audio embeddings
        audio_path_obj = Path(str(audio_path))
        word_audio_embs: list[torch.Tensor] = []

        for _, word_row in words_df.iterrows():
            start_sec = float(word_row["start_sec"])
            end_sec = float(word_row["end_sec"])

            # Load audio and extract segment
            try:
                wav, sr = torchaudio.load(str(audio_path_obj))
                if wav.ndim == 2 and wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                wav = wav.squeeze(0)

                # Resample to 16kHz if needed
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=16000
                    )
                    wav = resampler(wav)

                # Extract segment
                start_sample = int(start_sec * 16000)
                end_sample = int(end_sec * 16000)
                segment = wav[start_sample:end_sample]

                # Pad or truncate to fixed length if needed
                target_length = int(self.audio_cfg.max_audio_sec * 16000)
                if len(segment) < target_length:
                    segment = torch.nn.functional.pad(
                        segment, (0, target_length - len(segment))
                    )
                elif len(segment) > target_length:
                    segment = segment[:target_length]

                # Extract embedding using Wav2Vec2
                # Save segment to temp file for wav2vec2 function
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    torchaudio.save(tmp.name, segment.unsqueeze(0), 16000)
                    word_emb = embed_file_mean_pool(
                        Path(tmp.name),
                        self.audio_cfg,
                        self.audio_model,
                        self.audio_feature_extractor,
                        self.device,
                    )
                    Path(tmp.name).unlink()

                word_audio_embs.append(torch.from_numpy(word_emb).float())

            except Exception:
                # Skip words with errors, use zero embedding
                # Default Wav2Vec2-base dimension is 768
                word_audio_embs.append(torch.zeros(768))

        # Stack word embeddings
        if not word_audio_embs:
            # Fallback: single zero embedding
            word_audio_embs = [torch.zeros(768)]

        # Truncate if too many words
        if len(word_audio_embs) > self.max_words_per_sample:
            word_audio_embs = word_audio_embs[: self.max_words_per_sample]

        word_audio_stack = torch.stack(word_audio_embs)  # [num_words, audio_dim]

        # For cross-attention, we need:
        # - text_emb: [1, text_dim] (single embedding representing full transcript)
        # - audio_emb: [num_words, audio_dim] (word-level embeddings)
        # The model will expand text_emb to match audio sequence length
        text_emb_expanded = text_emb_tensor.unsqueeze(0)  # [1, text_dim]

        return (
            word_audio_stack,
            text_emb_expanded,
            torch.tensor(label, dtype=torch.long),
        )
