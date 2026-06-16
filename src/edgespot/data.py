from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from edgespot.features import LogMel


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        sample_rate: int = 16000,
        teacher_embeddings: str | Path | None = None,
    ) -> None:
        self.items = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line]
        self.extract = LogMel(sample_rate=sample_rate)
        self.labels = sorted({item["label"] for item in self.items})
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.teacher_embeddings = None
        self.teacher_dim = None
        if teacher_embeddings:
            data = np.load(teacher_embeddings)
            self.teacher_embeddings = {key: torch.from_numpy(data[key]) for key in data.files}
            if self.teacher_embeddings:
                self.teacher_dim = int(next(iter(self.teacher_embeddings.values())).numel())

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        item = self.items[idx]
        wav, sr = _read_audio(item)
        wav = _crop_audio(wav, sr, item)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav_t = torch.from_numpy(wav)
        mel = self.extract(wav_t, sr)
        label = self.label_to_idx[item["label"]]
        audio_path = item.get("audio_path") or f"{item.get('zip_path')}::{item.get('zip_member')}"
        sample = {
            "features": mel,
            "label": torch.tensor(label, dtype=torch.long),
            "label_name": item["label"],
            "audio_path": audio_path,
        }
        if self.teacher_embeddings is not None:
            if audio_path not in self.teacher_embeddings:
                raise KeyError(f"Missing teacher embedding for {audio_path}")
            sample["teacher_embedding"] = self.teacher_embeddings[audio_path]
        return sample


def _read_audio(item: dict) -> tuple[np.ndarray, int]:
    if item.get("audio_path"):
        return sf.read(item["audio_path"], dtype="float32", always_2d=False)
    if item.get("zip_path") and item.get("zip_member"):
        with ZipFile(item["zip_path"]) as archive:
            payload = archive.read(item["zip_member"])
        return sf.read(BytesIO(payload), dtype="float32", always_2d=False)
    raise KeyError("Manifest row must contain audio_path or zip_path/zip_member")


def _crop_audio(wav: np.ndarray, sample_rate: int, item: dict) -> np.ndarray:
    if "start_sec" not in item and "end_sec" not in item:
        return wav
    start_sec = float(item.get("start_sec", 0.0) or 0.0)
    end_sec = item.get("end_sec")
    start = max(0, int(round(start_sec * sample_rate)))
    end = wav.shape[0] if end_sec is None else int(round(float(end_sec) * sample_rate))
    end = max(start, min(wav.shape[0], end))
    return wav[start:end]
