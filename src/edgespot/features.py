from __future__ import annotations

import torch
import torchaudio


class LogMel:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 40,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        target_frames: int = 101,
    ) -> None:
        self.sample_rate = sample_rate
        self.target_frames = target_frames
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            center=True,
            power=2.0,
            mel_scale="slaney",
            norm="slaney",
        )

    def __call__(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if wav.ndim == 2:
            wav = wav.mean(dim=0)
        if sample_rate != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, self.sample_rate)
        wav = _fit_length(wav, self.sample_rate)
        mel = self.mel(wav).clamp_min(1e-6)
        if mel.shape[-1] < self.target_frames:
            mel = torch.nn.functional.pad(mel, (0, self.target_frames - mel.shape[-1]))
        mel = mel[..., : self.target_frames]
        return mel.unsqueeze(0)


def _fit_length(wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
    target = sample_rate
    if wav.numel() >= target:
        return wav[:target]
    return torch.nn.functional.pad(wav, (0, target - wav.numel()))
