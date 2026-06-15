from __future__ import annotations

import random

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
