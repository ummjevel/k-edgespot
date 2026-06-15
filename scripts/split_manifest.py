from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--group-key",
        default="voice_prompt",
        help="Keep rows sharing this key in the same split when present.",
    )
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.manifest))
    rng = random.Random(args.seed)
    grouped = _group_by_label_and_key(rows, args.group_key)

    splits = {"train": [], "val": [], "test": []}
    for label, groups in grouped.items():
        rng.shuffle(groups)
        n = len(groups)
        train_end = int(n * args.train_ratio)
        val_end = train_end + int(n * args.val_ratio)
        if label != "__negative__" and n >= 3:
            train_end = max(1, min(train_end, n - 2))
            val_end = max(train_end + 1, min(val_end, n - 1))

        for split_name, selected in [
            ("train", groups[:train_end]),
            ("val", groups[train_end:val_end]),
            ("test", groups[val_end:]),
        ]:
            for group_rows in selected:
                splits[split_name].extend(group_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split_name, split_rows in splits.items():
        rng.shuffle(split_rows)
        path = out_dir / f"{split_name}.jsonl"
        _write_jsonl(path, split_rows)
        summary[split_name] = {
            "rows": len(split_rows),
            "labels": _label_counts(split_rows),
        }

    (out_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _group_by_label_and_key(rows: list[dict], key: str) -> dict[str, list[list[dict]]]:
    by_label_key: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for idx, row in enumerate(rows):
        group_value = str(row.get(key) or row.get("speaker_id") or row.get("id") or idx)
        by_label_key[row["label"]][group_value].append(row)
    return {label: list(groups.values()) for label, groups in by_label_key.items()}


def _label_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["label"]] += 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    main()
