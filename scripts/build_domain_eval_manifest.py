from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-query", required=True)
    parser.add_argument("--negative-query", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--negative-limit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    positives = [
        row
        for row in _read_jsonl(Path(args.positive_query))
        if row.get("label") != "__negative__"
    ]
    negatives = [
        row
        for row in _read_jsonl(Path(args.negative_query))
        if row.get("label") == "__negative__"
    ]
    rng = random.Random(args.seed)
    rng.shuffle(negatives)
    rows = positives + negatives[: args.negative_limit]
    rng.shuffle(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"wrote {len(rows)} rows to {out} "
        f"({len(positives)} positives, {min(len(negatives), args.negative_limit)} negatives)"
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    main()
