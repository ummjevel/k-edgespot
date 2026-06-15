from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", required=True)
    parser.add_argument("--negative", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rows = []
    rows.extend(_read(Path(args.positive)))
    rows.extend(_read(Path(args.negative)))
    random.Random(args.seed).shuffle(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(rows), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {out}")


def _read(path: Path) -> list[str]:
    return [line if line.endswith("\n") else line + "\n" for line in path.read_text().splitlines()]


if __name__ == "__main__":
    main()
