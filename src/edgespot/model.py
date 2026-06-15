from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from edgespot.pcen import TrainablePCEN


class SubSpectralNorm(nn.Module):
    """Sub-spectral normalization used by the Qualcomm BC-ResNet implementation."""

    def __init__(self, channels: int, sub_bands: int = 5) -> None:
        super().__init__()
        self.sub_bands = sub_bands
        self.bn = nn.BatchNorm2d(channels * sub_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, freq, time = x.shape
        if freq % self.sub_bands != 0:
            return F.batch_norm(
                x,
                None,
                None,
                training=True,
            )
        x = x.view(batch, channels * self.sub_bands, freq // self.sub_bands, time)
        x = self.bn(x)
        return x.view(batch, channels, freq, time)


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | None = None,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        activation: str | None = "relu",
        ssn: bool = False,
    ) -> None:
        super().__init__()
        if padding is None:
            if isinstance(kernel_size, tuple):
                if isinstance(dilation, tuple):
                    padding = tuple(
                        (k - 1) // 2 * d for k, d in zip(kernel_size, dilation, strict=True)
                    )
                else:
                    padding = tuple((k - 1) // 2 * dilation for k in kernel_size)
            else:
                padding = (kernel_size - 1) // 2

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias=False,
            )
        ]
        layers.append(SubSpectralNorm(out_channels) if ssn else nn.BatchNorm2d(out_channels))
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "silu":
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BCResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stage_idx: int,
        stride: tuple[int, int] = (1, 1),
        fused: bool = False,
    ) -> None:
        super().__init__()
        self.transition = in_channels != out_channels
        self.fused = fused

        layers: list[nn.Module] = []
        if self.transition:
            layers.append(ConvBNAct(in_channels, out_channels, 1, activation="relu"))
            in_channels = out_channels
        layers.append(
            ConvBNAct(
                in_channels,
                out_channels,
                (3, 1),
                (stride[0], 1),
                groups=in_channels,
                activation=None,
                ssn=True,
            )
        )
        self.freq_path = nn.Sequential(*layers)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        dilation = (1, 2**stage_idx)
        temporal_padding = (0, dilation[1])
        if fused:
            self.time_path = ConvBNAct(
                out_channels,
                out_channels,
                (1, 3),
                (1, stride[1]),
                padding=temporal_padding,
                dilation=dilation,
                groups=1,
                activation="silu",
            )
        else:
            self.time_path = nn.Sequential(
                ConvBNAct(
                    out_channels,
                    out_channels,
                    (1, 3),
                    (1, stride[1]),
                    padding=temporal_padding,
                    dilation=dilation,
                    groups=out_channels,
                    activation="silu",
                ),
                nn.Conv2d(out_channels, out_channels, 1, bias=False),
                nn.Dropout2d(0.1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        aux = self.freq_path(x)
        y = self.freq_pool(aux)
        y = self.time_path(y)
        y = y + aux
        if not self.transition and y.shape == shortcut.shape:
            y = y + shortcut
        return F.relu(y, inplace=True)


class TemporalAttentionHead(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int = 64, pos_kernel: int = 16) -> None:
        super().__init__()
        self.pos = nn.Conv1d(
            in_channels,
            in_channels,
            pos_kernel,
            padding=pos_kernel // 2,
            groups=in_channels,
        )
        self.q = nn.Linear(in_channels, embedding_dim)
        self.k = nn.Linear(in_channels, embedding_dim)
        self.v = nn.Linear(in_channels, embedding_dim)
        self.prelu = nn.PReLU()
        self.proj = nn.Conv1d(embedding_dim, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] != 1:
            x = F.adaptive_avg_pool2d(x, (1, None))
        x = x.squeeze(2)
        x = x + self.pos(x)[..., : x.shape[-1]]
        seq = x.transpose(1, 2)
        q = self.q(seq)
        k = self.k(seq)
        v = self.v(seq)
        attn = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(q.shape[-1]), dim=-1)
        z = self.prelu(attn @ v)
        weights = torch.softmax(self.proj(z.transpose(1, 2)).squeeze(1), dim=-1)
        emb = torch.sum(z * weights.unsqueeze(-1), dim=1)
        return F.normalize(emb, p=2.0, dim=-1)


class EdgeSpot(nn.Module):
    """EdgeSpot-style embedding model for 40 x 101 mel spectrograms."""

    def __init__(self, tau: int = 1, embedding_dim: int = 64, use_pcen: bool = True) -> None:
        super().__init__()
        base = 8 * tau
        channels = [base * 2, base, int(base * 1.5), base * 2, int(base * 2.5), base * 4]
        self.pcen = TrainablePCEN() if use_pcen else nn.Identity()
        self.head = ConvBNAct(1, channels[0], 5, stride=(2, 1), padding=2, activation="relu")
        self.stages = nn.ModuleList(
            [
                _make_stage(channels[0], channels[1], 2, 0, False, True),
                _make_stage(channels[1], channels[2], 2, 1, True, True),
                _make_stage(channels[2], channels[3], 4, 2, True, False),
                _make_stage(channels[3], channels[4], 4, 3, False, False),
            ]
        )
        self.tail = nn.Sequential(
            nn.Conv2d(channels[4], channels[4], 5, padding=(0, 2), groups=channels[4], bias=False),
            nn.Conv2d(channels[4], channels[5], 1, bias=False),
            nn.BatchNorm2d(channels[5]),
            nn.ReLU(inplace=True),
        )
        self.embedding = TemporalAttentionHead(channels[5], embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pcen(x)
        x = self.head(x)
        for stage in self.stages:
            for block in stage:
                x = block(x)
        x = self.tail(x)
        return self.embedding(x)


def _make_stage(
    in_channels: int,
    out_channels: int,
    num_layers: int,
    stage_idx: int,
    use_stride: bool,
    fused: bool,
) -> nn.ModuleList:
    blocks = nn.ModuleList()
    for idx in range(num_layers):
        stride = (2, 1) if use_stride and idx == 0 else (1, 1)
        blocks.append(
            BCResBlock(
                in_channels if idx == 0 else out_channels,
                out_channels,
                stage_idx,
                stride,
                fused,
            )
        )
    return blocks
