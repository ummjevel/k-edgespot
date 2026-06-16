from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RUN_RE = re.compile(r"edgespot-ko-(?P<family>.+)-tau(?P<tau>[0-9]+)$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("docs/results_summary.md"))
    args = parser.parse_args()

    rows = []
    for path in sorted(args.runs_root.glob("edgespot-ko-*-tau*/prototype_eval_k*.json")):
        match = RUN_RE.match(path.parent.name)
        if not match:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        thresholds = data["thresholds"]
        rows.append(
            {
                "family": match.group("family"),
                "tau": int(match.group("tau")),
                "k_shot": int(data["k_shot"]),
                "auc": data["auc"],
                "recall_far_0_1": thresholds["0.001"]["recall"],
                "recall_far_1": thresholds["0.01"]["recall"],
                "recall_far_5": thresholds["0.05"]["recall"],
                "path": path,
            }
        )

    rows.sort(key=lambda row: (row["family"], row["tau"], row["k_shot"]))
    best_rows = sorted(
        rows,
        key=lambda row: (
            row["auc"] if row["auc"] is not None else -1.0,
            row["recall_far_1"],
            row["recall_far_5"],
        ),
        reverse=True,
    )

    lines = [
        "# EdgeSpot Korean Results",
        "",
        "Generated from prototype evaluation JSON files under `runs/`.",
        "",
        "## Best Runs",
        "",
        "| Rank | Family | Tau | K-shot | AUC | Recall@FAR0.1% | Recall@FAR1% | Recall@FAR5% |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(best_rows[:8], start=1):
        lines.append(_row_line(rank, row))

    lines.extend(
        [
            "",
            "## All Runs",
            "",
            "| Family | Tau | K-shot | AUC | Recall@FAR0.1% | Recall@FAR1% | Recall@FAR5% |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(_row_line(None, row))

    lines.extend(
        [
            "",
            "## Current Takeaways",
            "",
            "- The best SCAF-only baseline is `scaf`, tau 2, 10-shot.",
            "- The best distilled model is `distill`, tau 4, 10-shot.",
            "- Distillation improves this TTS split substantially, "
            "but the evaluation is still TTS-domain only.",
            "- Next validation should use hard negatives and real-recording "
            "negative/domain-test data.",
            "",
        ]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


def _row_line(rank: int | None, row: dict) -> str:
    fields = [
        row["family"],
        str(row["tau"]),
        str(row["k_shot"]),
        _fmt(row["auc"]),
        _fmt(row["recall_far_0_1"]),
        _fmt(row["recall_far_1"]),
        _fmt(row["recall_far_5"]),
    ]
    if rank is not None:
        fields.insert(0, str(rank))
    return "| " + " | ".join(fields) + " |"


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
