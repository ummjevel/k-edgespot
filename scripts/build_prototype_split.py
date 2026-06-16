from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--support-out", required=True)
    parser.add_argument("--query-out", required=True)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--negative-query-limit", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    positives_by_label = defaultdict(list)
    negatives = []
    for row in _read_jsonl(Path(args.manifest)):
        if row.get("label") == "__negative__":
            negatives.append(row)
        else:
            positives_by_label[row["label"]].append(row)

    support = []
    query = []
    skipped_labels = {}
    for label, rows in sorted(positives_by_label.items()):
        rng.shuffle(rows)
        if len(rows) <= args.k_shot:
            skipped_labels[label] = len(rows)
            continue
        support.extend(rows[: args.k_shot])
        query.extend(rows[args.k_shot :])

    rng.shuffle(negatives)
    if args.negative_query_limit is not None:
        negatives = negatives[: args.negative_query_limit]
    query.extend(negatives)
    rng.shuffle(support)
    rng.shuffle(query)

    _write_jsonl(Path(args.support_out), support)
    _write_jsonl(Path(args.query_out), query)
    summary = {
        "support_rows": len(support),
        "query_rows": len(query),
        "k_shot": args.k_shot,
        "support_labels": dict(Counter(row["label"] for row in support).most_common()),
        "query_labels": dict(Counter(row["label"] for row in query).most_common()),
        "skipped_labels": skipped_labels,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
