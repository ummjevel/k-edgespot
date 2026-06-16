from __future__ import annotations

import torch

from edgespot.model_mingru import CausalConv1d, EdgeSpotMinGRU, MinGRU


def main() -> None:
    torch.manual_seed(0)

    gru = MinGRU(32).eval()
    x = torch.randn(4, 50, 32)
    with torch.no_grad():
        parallel = gru(x)
        h = None
        stepped = []
        for t in range(x.shape[1]):
            y_t, h = gru.step(x[:, t], h)
            stepped.append(y_t)
        sequential = torch.stack(stepped, dim=1)
    diff = (parallel - sequential).abs().max().item()
    print(f"minGRU parallel-vs-step max_abs_diff={diff:.3e}")
    if diff >= 1e-4:
        raise SystemExit("minGRU parity check failed")

    model = EdgeSpotMinGRU(num_blocks=3).eval()
    seq = torch.randn(2, 101, 64)
    with torch.no_grad():
        parallel = seq
        for block in model.blocks:
            parallel = block(parallel)
        streamed = model.stream_mixer(seq)
    diff = (parallel - streamed).abs().max().item()
    print(f"mixer parallel-vs-stream max_abs_diff={diff:.3e}")
    if diff >= 1e-4:
        raise SystemExit("mixer parity check failed")

    causal = CausalConv1d(8, 8, kernel_size=5).eval()
    prefix = torch.randn(2, 8, 37)
    suffix = torch.randn(2, 8, 13)
    with torch.no_grad():
        y_prefix = causal(prefix)
        y_joined = causal(torch.cat([prefix, suffix], dim=-1))[..., : prefix.shape[-1]]
    diff = (y_prefix - y_joined).abs().max().item()
    print(f"causal conv prefix max_abs_diff={diff:.3e}")
    if diff >= 1e-6:
        raise SystemExit("causal convolution prefix check failed")

    mel = torch.rand(2, 1, 40, 101)
    with torch.no_grad():
        embedding = model(mel)
    param_count = sum(param.numel() for param in model.parameters())
    norms = embedding.norm(dim=-1)
    print(
        "forward ok "
        f"embedding_shape={tuple(embedding.shape)} "
        f"norm_min={norms.min().item():.6f} "
        f"norm_max={norms.max().item():.6f} "
        f"params={param_count / 1000:.1f}K"
    )


if __name__ == "__main__":
    main()
