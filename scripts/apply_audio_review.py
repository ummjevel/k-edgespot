from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--review-tsv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out")
    parser.add_argument(
        "--mode",
        choices=["all", "conservative"],
        default="all",
        help=(
            "all applies every review label decision. conservative excludes bad audio "
            "and applies map_positive plus false_accept keep_negative only."
        ),
    )
    args = parser.parse_args()

    decisions = _read_decisions(Path(args.review_tsv))
    rows = []
    stats = {
        "input_rows": 0,
        "output_rows": 0,
        "reviewed_rows": 0,
        "excluded_rows": 0,
        "excluded_bad_audio": 0,
        "excluded_unclear": 0,
        "decision_counts": Counter(),
        "label_changes": Counter(),
        "ignored_label_changes": Counter(),
        "output_labels": Counter(),
    }

    for row in _read_jsonl(Path(args.manifest)):
        stats["input_rows"] += 1
        key = row.get("audio_path")
        decision = decisions.get(key)
        if decision is None:
            rows.append(row)
            stats["output_labels"][row["label"]] += 1
            continue

        stats["reviewed_rows"] += 1
        stats["decision_counts"][decision["decision"]] += 1
        if decision["decision"] == "bad_audio":
            stats["excluded_rows"] += 1
            stats["excluded_bad_audio"] += 1
            continue

        updated = dict(row)
        old_label = updated.get("label")
        updated["review_decision"] = decision["decision"]
        updated["review_type"] = decision["type"]
        updated["review_note"] = decision["note"]
        if decision["decision"] == "keep_negative":
            if args.mode == "all" or decision["type"] == "false_accept":
                updated["label"] = "__negative__"
            else:
                stats["ignored_label_changes"][f"{old_label}->__negative__"] += 1
        elif decision["decision"] == "map_positive":
            updated["label"] = decision["target_label"]
        elif decision["decision"] == "exclude_unclear":
            stats["excluded_rows"] += 1
            stats["excluded_unclear"] += 1
            continue
        elif decision["decision"] == "keep_positive":
            pass
        else:
            raise ValueError(f"Unsupported decision: {decision['decision']}")
        if updated["label"] != old_label:
            stats["label_changes"][f"{old_label}->{updated['label']}"] += 1
        rows.append(updated)
        stats["output_labels"][updated["label"]] += 1

    stats["output_rows"] = len(rows)
    _write_jsonl(Path(args.out), rows)
    summary = {
        key: dict(value) if isinstance(value, Counter) else value for key, value in stats.items()
    }
    if args.summary_out:
        Path(args.summary_out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _read_decisions(path: Path) -> dict[str, dict]:
    decisions = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            decision = (row.get("decision") or "").strip()
            if not decision:
                continue
            source_audio = row["source_audio"]
            target_label = (row.get("note") or "").strip() or row.get("matched_label") or ""
            decisions[source_audio] = {
                "decision": decision,
                "type": row.get("type") or "",
                "note": row.get("note") or "",
                "target_label": target_label,
            }
    return decisions


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
