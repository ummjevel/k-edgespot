from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from edgespot.pcen import TrainablePCEN


def parallel_scan_log(log_a: torch.Tensor, log_b: torch.Tensor) -> torch.Tensor:
    """Parallel log-space scan for h[t] = a[t] * h[t - 1] + b[t], h[-1] = 0."""
    a_star = torch.cumsum(log_a, dim=1)
    log_h = a_star + torch.logcumsumexp(log_b - a_star, dim=1)
    return torch.exp(log_h)


def positive_candidate(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


def log_positive_candidate(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.log(F.relu(x) + 0.5), -F.softplus(-x))


class MinGRU(nn.Module):
    """minGRU with parallel training path and stateful streaming step path."""

    def __init__(self, dim: int, expansion: float = 1.0) -> None:
        super().__init__()
        self.inner = int(dim * expansion)
        self.to_z = nn.Linear(dim, self.inner)
        self.to_h = nn.Linear(dim, self.inner)
        self.out = nn.Linear(self.inner, dim) if self.inner != dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.to_z(x)
        h_pre = self.to_h(x)
        log_a = -F.softplus(k)
        log_b = -F.softplus(-k) + log_positive_candidate(h_pre)
        return self.out(parallel_scan_log(log_a, log_b))

    def step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if h_prev is None:
            h_prev = x_t.new_zeros(x_t.shape[0], self.inner)
        k = self.to_z(x_t)
        z = torch.sigmoid(k)
        h = (1.0 - z) * h_prev + z * positive_candidate(self.to_h(x_t))
        return self.out(h), h


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class MinGRUMixerBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = RMSNorm(dim)
        self.gru = MinGRU(dim)
        self.norm2 = RMSNorm(dim)
        self.mix = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gru(self.norm1(x))
        x = x + self.mix(self.norm2(x))
        return x

    def step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gru_out, h = self.gru.step(self.norm1(x_t), h_prev)
        x_t = x_t + gru_out
        x_t = x_t + self.mix(self.norm2(x_t))
        return x_t, h


class CausalConvBackbone(nn.Module):
    """Small frequency-only conv backbone. Time kernel is 1, so it is causal."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, (5, 1), stride=(2, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, (3, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, dim, (3, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x.mean(dim=2)


class CausalPosEnc(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 16) -> None:
        super().__init__()
        self.left_pad = kernel_size - 1
        self.dw = nn.Conv1d(dim, dim, kernel_size, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dw(F.pad(x, (self.left_pad, 0)))


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.left_pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_pad, 0)))


class EdgeSpotMinGRU(nn.Module):
    """Experimental streaming EdgeSpot variant with minGRU replacing temporal SDPA."""

    def __init__(
        self,
        dim: int = 64,
        num_blocks: int = 3,
        embedding_dim: int = 64,
        use_pcen: bool = True,
    ) -> None:
        super().__init__()
        self.pcen = TrainablePCEN() if use_pcen else nn.Identity()
        self.backbone = CausalConvBackbone(dim)
        self.posenc = CausalPosEnc(dim)
        self.blocks = nn.ModuleList(MinGRUMixerBlock(dim) for _ in range(num_blocks))
        self.act = nn.PReLU()
        self.head = nn.Sequential(
            CausalConv1d(dim, embedding_dim, kernel_size=3),
            nn.PReLU(),
        )
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, 1, F, T], got {tuple(x.shape)}")
        seq = self.frontend_sequence(x)
        for block in self.blocks:
            seq = block(seq)
        seq = self.act(seq)
        seq = self.head(seq.transpose(1, 2))
        emb = self.proj(seq.mean(dim=-1))
        return F.normalize(emb, p=2.0, dim=-1)

    def frontend_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pcen(x)
        x = self.backbone(x)
        x = self.posenc(x)
        return x.transpose(1, 2)

    @torch.no_grad()
    def stream_mixer(self, seq: torch.Tensor) -> torch.Tensor:
        states: list[torch.Tensor | None] = [None] * len(self.blocks)
        outputs = []
        for t in range(seq.shape[1]):
            x_t = seq[:, t]
            for idx, block in enumerate(self.blocks):
                x_t, states[idx] = block.step(x_t, states[idx])
            outputs.append(x_t)
        return torch.stack(outputs, dim=1)
