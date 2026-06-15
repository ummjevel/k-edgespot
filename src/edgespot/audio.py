from __future__ import annotations

import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_wav(path: str, target_sr: int) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    tensor = torch.from_numpy(wav)
    if sr != target_sr:
        tensor = torchaudio.functional.resample(tensor, sr, target_sr)
    return tensor.numpy()
