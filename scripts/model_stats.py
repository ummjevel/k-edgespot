from __future__ import annotations

import argparse
import json
from pathlib import Path

from edgespot.model import EdgeSpot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/model_stats.json"))
    parser.add_argument("--taus", nargs="+", type=int, default=[1, 2, 3, 4])
    args = parser.parse_args()

    rows = []
    for tau in args.taus:
        model = EdgeSpot(tau=tau)
        params = sum(param.numel() for param in model.parameters())
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        rows.append(
            {
                "tau": tau,
                "parameters": params,
                "trainable_parameters": trainable,
                "size_fp32_mb": params * 4 / 1_000_000,
                "size_int8_mb": params / 1_000_000,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
