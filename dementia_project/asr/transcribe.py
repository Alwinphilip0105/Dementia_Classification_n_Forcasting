"""ASR transcription interface.

Primary backend: HuggingFace Transformers Whisper pipeline with word timestamps.
This keeps the project self-contained (no WhisperX dependency needed to start).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from transformers import pipeline


@dataclass(frozen=True)
class AsrResult:
    text: str
    # List of dicts like {"start": float, "end": float, "text": str}
    segments: list[dict[str, Any]]
    # Optional list of dicts like {"start": float, "end": float, "word": str}
    words: list[dict[str, Any]] | None = None


def save_asr_result(out_dir: Path, result: AsrResult) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transcript.json").write_text(
        json.dumps({"text": result.text, "segments": result.segments}, indent=2),
        encoding="utf-8",
    )
    if result.words is not None:
        (out_dir / "words.json").write_text(
            json.dumps({"words": result.words}, indent=2), encoding="utf-8"
        )


def _load_audio_16k_mono(path: Path) -> np.ndarray:
    wav, sr = torchaudio.load(str(path))
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    target_sr = 16_000
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
    result: np.ndarray = wav.detach().cpu().numpy().astype(np.float32)
    return result


@torch.no_grad()
def transcribe_with_whisper_pipeline(
    audio_path: Path,
    model_name: str = "openai/whisper-tiny",
    device: str | None = None,
    chunk_length_s: int = 30,
    language: str = "en",
    task: str = "transcribe",
) -> AsrResult:
    """Transcribe audio using a Whisper ASR pipeline.

    Returns word-level timestamps when supported by the model/pipeline via
    `return_timestamps="word"`.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_dtype = torch.float16 if (device == "cuda") else torch.float32

    asr = pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        device=0 if device == "cuda" else -1,
        torch_dtype=torch_dtype,
        ignore_warning=True,
    )

    audio = _load_audio_16k_mono(audio_path)
    out_raw = asr(
        audio,
        return_timestamps="word",
        chunk_length_s=chunk_length_s,
        generate_kwargs={"language": language, "task": task},
    )

    # Normalize pipeline output to a single dict for downstream processing.
    if isinstance(out_raw, list):
        out: dict[str, Any] = out_raw[0] if out_raw else {}
    elif isinstance(out_raw, dict):
        out = out_raw
    else:
        out = {}

    text = str(out.get("text", "")).strip()
    chunks = out.get("chunks", []) or []

    words: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []

    # In "word" mode, chunks are per-word.
    for ch in chunks:
        w = str(ch.get("text", "")).strip()
        ts = ch.get("timestamp")
        if not isinstance(ts, (list, tuple)) or len(ts) != 2:
            continue
        start, end = ts
        if start is None or end is None:
            continue
        words.append({"word": w, "start": float(start), "end": float(end)})

    # Coarse segments (optional): use one segment spanning the full recognized text.
    # Downstream tasks can rely on words.json for alignment.
    if words:
        segments = [
            {
                "start": float(words[0]["start"]),
                "end": float(words[-1]["end"]),
                "text": text,
            }
        ]

    return AsrResult(text=text, segments=segments, words=words if words else None)
