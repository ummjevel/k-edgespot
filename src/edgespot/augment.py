from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F


def apply_specaugment(
    features: torch.Tensor,
    freq_mask_width: int = 6,
    time_mask_width: int = 8,
    time_stretch_min: float = 0.9,
    time_stretch_max: float = 1.1,
) -> torch.Tensor:
    """Apply the EdgeSpot paper's lightweight feature-domain augmentation.

    The paper samples waveform time-stretch factors from [0.9, 1.1]. This
    implementation applies the stretch on the 40 x 101 feature map so the
    manifest dataset can stay waveform-agnostic during batching.
    """

    out = features
    out = _time_stretch(out, random.uniform(time_stretch_min, time_stretch_max))
    out = _mask_along_dim(out, dim=2, max_width=freq_mask_width)
    out = _mask_along_dim(out, dim=3, max_width=time_mask_width)
    return out


def load_device_like_profile(path: str | Path, device: str | torch.device) -> dict[str, torch.Tensor]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "freq_bias_mean": torch.tensor(data["freq_bias_mean"], dtype=torch.float32, device=device),
        "freq_bias_std": torch.tensor(data["freq_bias_std"], dtype=torch.float32, device=device),
        "global_mean_std": torch.tensor(float(data["global_mean_std"]), device=device),
        "frame_noise_std": torch.tensor(float(data["frame_noise_std"]), device=device),
    }


def apply_device_like_augmentation(
    features: torch.Tensor,
    profile: dict[str, torch.Tensor],
    probability: float = 0.5,
    coloration_scale: float = 0.5,
    noise_scale: float = 0.05,
    gain_scale: float = 0.2,
) -> torch.Tensor:
    """Apply feature-domain coloration/noise derived from device recordings."""

    if probability <= 0:
        return features
    batch = features.shape[0]
    mask = torch.rand(batch, device=features.device) < probability
    if not bool(mask.any()):
        return features

    out = features.clone()
    selected = int(mask.sum().item())
    freq_mean = profile["freq_bias_mean"].view(1, 1, -1, 1)
    freq_std = profile["freq_bias_std"].clamp_min(1e-6).view(1, 1, -1, 1)
    coloration = freq_mean + torch.randn(
        selected,
        1,
        freq_mean.shape[2],
        1,
        device=features.device,
    ) * freq_std
    gain = torch.randn(selected, 1, 1, 1, device=features.device) * profile["global_mean_std"]
    noise = torch.randn_like(out[mask]) * profile["frame_noise_std"]
    out[mask] = (
        out[mask]
        + coloration_scale * coloration
        + gain_scale * gain
        + noise_scale * noise
    )
    return out


def _time_stretch(features: torch.Tensor, factor: float) -> torch.Tensor:
    if factor == 1.0:
        return features
    batch, channels, freq, time = features.shape
    stretched_time = max(1, round(time * factor))
    stretched = F.interpolate(
        features,
        size=(freq, stretched_time),
        mode="bilinear",
        align_corners=False,
    )
    if stretched_time >= time:
        return stretched[..., :time]
    return F.pad(stretched, (0, time - stretched_time))


def _mask_along_dim(features: torch.Tensor, dim: int, max_width: int) -> torch.Tensor:
    if max_width <= 0:
        return features
    out = features.clone()
    size = out.shape[dim]
    width = random.randint(0, min(max_width, size))
    if width == 0:
        return out
    start = random.randint(0, size - width)
    index = [slice(None)] * out.ndim
    index[dim] = slice(start, start + width)
    out[tuple(index)] = 0
    return out
