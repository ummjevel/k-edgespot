from __future__ import annotations

import json
from pathlib import Path

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
        wav, sr = sf.read(item["audio_path"], dtype="float32", always_2d=False)
        wav_t = torch.from_numpy(wav)
        mel = self.extract(wav_t, sr)
        label = self.label_to_idx[item["label"]]
        sample = {
            "features": mel,
            "label": torch.tensor(label, dtype=torch.long),
            "label_name": item["label"],
            "audio_path": item["audio_path"],
        }
        if self.teacher_embeddings is not None:
            if item["audio_path"] not in self.teacher_embeddings:
                raise KeyError(f"Missing teacher embedding for {item['audio_path']}")
            sample["teacher_embedding"] = self.teacher_embeddings[item["audio_path"]]
        return sample
