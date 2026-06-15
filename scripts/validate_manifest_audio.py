from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-sample-rate", type=int, default=16000)
    parser.add_argument("--min-duration", type=float, default=0.15)
    parser.add_argument("--max-duration", type=float, default=8.0)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.manifest))
    summary = {
        "manifest": args.manifest,
        "total_rows": len(rows),
        "ok_rows": 0,
        "missing_audio": 0,
        "bad_audio": 0,
        "bad_sample_rate": 0,
        "bad_duration": 0,
        "labels": Counter(),
        "durations": [],
        "errors": [],
    }

    for row in tqdm(rows, desc="validate"):
        label = row.get("label", "")
        summary["labels"][label] += 1
        audio_path = row.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            summary["missing_audio"] += 1
            _record_error(summary, row, "missing_audio")
            continue
        try:
            info = sf.info(audio_path)
        except Exception as exc:
            summary["bad_audio"] += 1
            _record_error(summary, row, f"bad_audio:{type(exc).__name__}:{exc}")
            continue

        duration = info.frames / max(1, info.samplerate)
        summary["durations"].append(duration)
        ok = True
        if info.samplerate != args.expected_sample_rate:
            summary["bad_sample_rate"] += 1
            _record_error(summary, row, f"sample_rate:{info.samplerate}")
            ok = False
        if not args.min_duration <= duration <= args.max_duration:
            summary["bad_duration"] += 1
            _record_error(summary, row, f"duration:{duration:.3f}")
            ok = False
        if ok:
            summary["ok_rows"] += 1

    durations = summary.pop("durations")
    if durations:
        summary["duration_sec"] = {
            "min": min(durations),
            "max": max(durations),
            "mean": sum(durations) / len(durations),
        }
    else:
        summary["duration_sec"] = {}
    summary["labels"] = dict(summary["labels"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.strict and summary["ok_rows"] != summary["total_rows"]:
        raise SystemExit(1)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _record_error(summary: dict, row: dict, error: str) -> None:
    if len(summary["errors"]) >= 100:
        return
    summary["errors"].append(
        {
            "id": row.get("id"),
            "label": row.get("label"),
            "audio_path": row.get("audio_path"),
            "error": error,
        }
    )


if __name__ == "__main__":
    main()
