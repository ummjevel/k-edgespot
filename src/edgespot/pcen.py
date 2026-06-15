from __future__ import annotations

import torch
from torch import nn


class TrainablePCEN(nn.Module):
    """Trainable per-channel energy normalization for mel spectrograms.

    Input shape is expected to be ``[batch, 1, freq, time]`` with non-negative
    mel energies. The smoothing scalar and compression parameters are shared
    across channels, matching the EdgeSpot paper description.
    """

    def __init__(
        self,
        alpha: float = 0.98,
        delta: float = 2.0,
        root: float = 0.5,
        smooth: float = 0.025,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.logit_alpha = nn.Parameter(_inv_sigmoid(torch.tensor(alpha)))
        self.log_delta = nn.Parameter(torch.log(torch.tensor(delta)))
        self.logit_root = nn.Parameter(_inv_sigmoid(torch.tensor(root)))
        self.logit_smooth = nn.Parameter(_inv_sigmoid(torch.tensor(smooth)))
        self.eps = eps

    def forward(self, energy: torch.Tensor) -> torch.Tensor:
        if energy.ndim != 4:
            raise ValueError(f"Expected [B, 1, F, T], got {tuple(energy.shape)}")

        energy = energy.clamp_min(self.eps)
        alpha = torch.sigmoid(self.logit_alpha)
        delta = torch.exp(self.log_delta)
        root = torch.sigmoid(self.logit_root).clamp_min(1e-3)
        smooth = torch.sigmoid(self.logit_smooth)

        smoother = []
        prev = energy[..., 0]
        smoother.append(prev)
        for t in range(1, energy.shape[-1]):
            prev = (1.0 - smooth) * prev + smooth * energy[..., t]
            smoother.append(prev)
        smooth_energy = torch.stack(smoother, dim=-1)

        normalized = energy / (self.eps + smooth_energy).pow(alpha)
        return (normalized + delta).pow(root) - delta.pow(root)


def _inv_sigmoid(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(1e-5, 1.0 - 1e-5)
    return torch.log(value / (1.0 - value))
