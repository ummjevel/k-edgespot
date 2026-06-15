from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Qwen3TTS:
    """Thin adapter for Qwen3-TTS.

    Qwen3-TTS runtime APIs are still moving. This adapter first tries a small
    conventional Python package surface. If your installed runtime exposes a
    different API, change this file only; the manifest generation pipeline will
    stay stable.
    """

    model_id: str = "Qwen/Qwen3-TTS"
    device: str = "cuda"
    qwen_source: str = "/data/users/voice/zoey/Qwen3-TTS"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    cache_dir: str | None = "models/huggingface"
    max_new_tokens: int = 128
    temperature: float = 0.9
    top_p: float = 0.9

    def __post_init__(self) -> None:
        self._engine = self._load_engine()

    def synthesize(
        self,
        text: str,
        voice_prompt: str,
        language: str = "ko",
        sample_rate: int = 16000,
    ) -> tuple[np.ndarray, int]:
        language_name = _language_name(language)
        if hasattr(self._engine, "generate_voice_design"):
            result = self._engine.generate_voice_design(
                text=[text],
                language=[language_name],
                instruct=[voice_prompt],
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        elif hasattr(self._engine, "synthesize"):
            result = self._engine.synthesize(
                text=text,
                voice_prompt=voice_prompt,
                language=language_name,
                sample_rate=sample_rate,
            )
        elif callable(self._engine):
            result = self._engine(
                text=text,
                voice_prompt=voice_prompt,
                language=language_name,
                sample_rate=sample_rate,
            )
        else:
            raise RuntimeError("Unsupported Qwen3-TTS engine object")

        if isinstance(result, tuple):
            wav, sr = result
        elif isinstance(result, dict):
            wav = result.get("audio") or result.get("wav") or result.get("waveform")
            sr = result.get("sample_rate", sample_rate)
        else:
            wav, sr = result, sample_rate

        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.squeeze()
        return wav, int(sr)

    def synthesize_batch(self, rows: list[dict]) -> tuple[list[np.ndarray], int]:
        texts = [row["text"] for row in rows]
        languages = [_language_name(row.get("language", "ko")) for row in rows]
        instructions = [row["voice_prompt"] for row in rows]
        if hasattr(self._engine, "generate_voice_design"):
            wavs, sr = self._engine.generate_voice_design(
                text=texts,
                language=languages,
                instruct=instructions,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            return [_to_float32(wav) for wav in wavs], int(sr)

        wavs = []
        sample_rate = rows[0].get("sample_rate", 16000)
        for row in rows:
            wav, sample_rate = self.synthesize(
                text=row["text"],
                voice_prompt=row["voice_prompt"],
                language=row.get("language", "ko"),
                sample_rate=row.get("sample_rate", sample_rate),
            )
            wavs.append(wav)
        return wavs, int(sample_rate)

    def _load_engine(self):
        try:
            source = Path(self.qwen_source)
            if source.exists():
                sys.path.insert(0, str(source))
            from qwen_tts import Qwen3TTSModel

            dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[self.dtype]
            return Qwen3TTSModel.from_pretrained(
                self.model_id,
                device_map=self.device,
                dtype=dtype,
                attn_implementation=self.attn_implementation,
                cache_dir=self.cache_dir,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-TTS runtime is not importable. Expected local source at "
                f"{self.qwen_source} exposing `qwen_tts.Qwen3TTSModel`, or adapt "
                "src/edgespot/tts/qwen3.py to your runtime API."
            ) from exc


def _language_name(language: str) -> str:
    normalized = language.lower()
    if normalized in {"ko", "kor", "korean", "한국어"}:
        return "Korean"
    return language


def _to_float32(wav) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.squeeze()
    return wav
