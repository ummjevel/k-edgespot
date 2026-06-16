from __future__ import annotations

import argparse
import json
from pathlib import Path

BENCHMARKS = [
    (
        "TTS held-out test",
        "Pre-AIHub hard-negative tau4",
        Path("runs/edgespot-ko-distill-hard-tau4/prototype_eval_k10.json"),
        "TTS test split, 384 positives + 1,251 TTS negatives.",
    ),
    (
        "Real-recording negative domain test",
        "Pre-AIHub hard-negative tau4",
        Path("runs/edgespot-ko-distill-hard-tau4/prototype_eval_realneg5k_k10.json"),
        "Original TTS positives plus 5,000 real-recording Korean negatives.",
    ),
    (
        "AIHub target-keyword full",
        "Pre-AIHub hard-negative tau4",
        Path("runs/edgespot-ko-distill-hard-tau4/prototype_eval_aihub71405_target_keyword_k10.json"),
        "984 mapped AIHub target-keyword rows, all S/N variants.",
    ),
    (
        "Augmented held-out test",
        "Post-AIHub tau4",
        Path("runs/edgespot-ko-distill-hard-aihub-tau4/prototype_eval_augmented_test_k10.json"),
        "Held-out TTS + AIHub mapped test split.",
    ),
    (
        "AIHub target-keyword full",
        "Post-AIHub tau4",
        Path("runs/edgespot-ko-distill-hard-aihub-tau4/prototype_eval_aihub71405_target_keyword_k10.json"),
        "984 mapped AIHub target-keyword rows, all S/N variants.",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/benchmark_results.md"))
    args = parser.parse_args()

    rows = []
    for benchmark, model, path, notes in BENCHMARKS:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        thresholds = data["thresholds"]
        rows.append(
            {
                "benchmark": benchmark,
                "model": model,
                "k_shot": data["k_shot"],
                "queries": data["num_queries"],
                "auc": data["auc"],
                "recall_0_1": thresholds["0.001"]["recall"],
                "recall_1": thresholds["0.01"]["recall"],
                "recall_5": thresholds["0.05"]["recall"],
                "path": path,
                "notes": notes,
            }
        )

    lines = [
        "# EdgeSpot Korean Benchmark Results",
        "",
        "All rows use prototype evaluation with `k=10` unless otherwise noted.",
        "",
        "| Benchmark | Model | Queries | AUC | Recall@FAR0.1% | Recall@FAR1% | Recall@FAR5% |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {benchmark} | {model} | {queries} | {auc} | {r01} | {r1} | {r5} |".format(
                benchmark=row["benchmark"],
                model=row["model"],
                queries=row["queries"],
                auc=_fmt(row["auc"]),
                r01=_fmt(row["recall_0_1"]),
                r1=_fmt(row["recall_1"]),
                r5=_fmt(row["recall_5"]),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- The pre-AIHub hard-negative model remains strong on the TTS held-out "
                "and real-recording negative tests."
            ),
            (
                "- Adding AIHub mapped data improves the held-out augmented split "
                "substantially, but full AIHub target-keyword generalization remains weak."
            ),
            (
                "- The gap between augmented held-out test and full AIHub target-keyword "
                "test suggests the next iteration should focus on AIHub label quality, "
                "support/query protocol, and hard negative mining rather than simply "
                "increasing epochs."
            ),
            "",
            "## Fixed Benchmark Protocol",
            "",
            "| Benchmark | Query manifest | Purpose |",
            "|---|---|---|",
            (
                "| TTS held-out test | `data/manifests/splits/test.jsonl` | "
                "Check generated-command baseline behavior. |"
            ),
            (
                "| Real-recording negative domain test | "
                "`data/manifests/splits/test_real_negative_5k.jsonl` | "
                "Check false accepts on real Korean speech negatives. |"
            ),
            (
                "| AIHub mapped held-out test | "
                "`data/manifests/splits/test_aihub_mapped.jsonl` | "
                "Check held-out AIHub split after training with mapped AIHub rows. |"
            ),
            (
                "| AIHub target-keyword full scan | "
                "`data/manifests/aihub_71405_validation_seed.extracted."
                "target_keyword_short.mapped.jsonl` | "
                "Stress-test all mapped AIHub short target-keyword rows. |"
            ),
            "",
            "## Source Files",
            "",
        ]
    )
    for row in rows:
        lines.append(f"- `{row['path']}`: {row['notes']}")
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
