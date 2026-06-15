from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class SubCenterArcFaceLoss(nn.Module):
    """Sub-center ArcFace classification loss.

    Each class owns multiple normalized centers. The closest center logit is used
    for the ArcFace margin transform.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        num_centers: int = 3,
        scale: float = 30.0,
        margin: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_centers = num_centers
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes * num_centers, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        centers = F.normalize(self.weight, p=2, dim=-1)
        logits = embeddings @ centers.t()
        logits = logits.view(-1, self.num_classes, self.num_centers).max(dim=-1).values

        index = torch.arange(labels.numel(), device=labels.device)
        target_logits = logits[index, labels].clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(target_logits)
        margin_logits = torch.cos(theta + self.margin)
        logits = logits.clone()
        logits[index, labels] = margin_logits
        return F.cross_entropy(logits * self.scale, labels)

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, num_centers={self.num_centers}, "
            f"scale={self.scale}, margin={self.margin}"
        )


def cosine_distillation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    student = F.normalize(student, p=2, dim=-1)
    teacher = F.normalize(teacher, p=2, dim=-1)
    return 1.0 - F.cosine_similarity(student, teacher, dim=-1).mean()


def mse_distillation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student, teacher)


def recommended_arcface_scale(num_classes: int) -> float:
    return max(10.0, math.sqrt(2.0) * math.log(max(2, num_classes - 1)))
