from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel


class AttentionReductionHead(nn.Module):
    """Attention pooling head that maps SSL frame features to 64-D embeddings."""

    def __init__(self, input_dim: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.PReLU(),
        )
        self.attn = nn.Linear(embedding_dim, 1)

    def forward(self, frames: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        z = self.proj(frames)
        scores = self.attn(z).squeeze(-1)
        if frame_mask is not None:
            scores = scores.masked_fill(~frame_mask, -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        emb = torch.sum(z * weights.unsqueeze(-1), dim=1)
        return F.normalize(emb, p=2.0, dim=-1)


class Wav2VecTeacher(nn.Module):
    """Pretrained Wav2Vec2-family encoder plus attention reduction head."""

    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-base",
        embedding_dim: int = 64,
        freeze_encoder: bool = True,
        encoder_layer: int | None = 16,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
        self.encoder = AutoModel.from_pretrained(model_id, cache_dir=cache_dir)
        self.head = AttentionReductionHead(config.hidden_size, embedding_dim)
        self.model_id = model_id
        self.embedding_dim = embedding_dim
        self.encoder_layer = encoder_layer
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=self.encoder_layer is not None,
        )
        frames = output.last_hidden_state
        if self.encoder_layer is not None and output.hidden_states is not None:
            layer = min(self.encoder_layer, len(output.hidden_states) - 1)
            frames = output.hidden_states[layer]
        frame_mask = None
        if attention_mask is not None:
            frame_mask = _feature_mask(frames, attention_mask)
        return self.head(frames, frame_mask)


def _feature_mask(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # Wav2Vec2 conv frontend stride is 320 samples for standard HF models.
    output_lengths = torch.div(attention_mask.sum(-1) - 1, 320, rounding_mode="floor") + 1
    max_len = hidden.shape[1]
    arange = torch.arange(max_len, device=hidden.device).unsqueeze(0)
    return arange < output_lengths.unsqueeze(1)
