from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

RULES = [
    (
        "music_stop",
        [r"(음악|노래).*(그만|멈춰|중지|정지|꺼)", r"(그만|멈춰|중지|정지).*(음악|노래)"],
        [],
    ),
    (
        "music_play",
        [
            r"(음악|노래|찬양|찬불가).*(틀|시작|들려|듣고 싶|들을래|재생)",
            r"(틀|시작|들려|재생).*(음악|노래|찬양|찬불가)",
        ],
        [r"찾|리스트|뭐야|있나|있어\?|어때|어땠|보여|넣어|보관|즐겨찾기|싶지|할게"],
    ),
    (
        "lights_off",
        [r"(불|조명).*(꺼|끄|소등)", r"(꺼|끄|소등).*(불|조명)"],
        [r"불러|불고기|가스불|꺼졌|꺼져 있|꺼진|봐줘|맞아|나왔어|합니다|드릴게요"],
    ),
    (
        "lights_on",
        [r"(불|조명).*(켜|점등)", r"(켜|점등).*(불|조명)"],
        [r"불러|불고기|가스불|켜졌|켜져 있|켜진|봐줘|맞아|나왔어|합니다|드릴게요"],
    ),
    (
        "alarm_set",
        [r"알람.*(맞춰|설정|울려|깨워)", r"(맞춰|설정).*(알람)"],
        [r"꺼|종료|대답|알림|확인|있|알람음"],
    ),
    (
        "timer_stop",
        [r"타이머.*(중지|종료|꺼|멈춰)", r"(중지|종료|꺼|멈춰).*(타이머)"],
        [],
    ),
    (
        "timer_start",
        [r"타이머.*(시작|맞춰|설정)", r"(시작|맞춰|설정).*(타이머)"],
        [],
    ),
    (
        "volume_up",
        [r"(볼륨|소리).*(올려|높여|키워|크게)", r"(올려|높여|키워|크게).*(볼륨|소리)"],
        [r"드릴게요|합니다|입니다"],
    ),
    (
        "volume_down",
        [r"(볼륨|소리).*(내려|낮춰|줄여|작게)", r"(내려|낮춰|줄여|작게).*(볼륨|소리)"],
        [r"드릴게요|합니다|입니다"],
    ),
    (
        "next_track",
        [r"다음\s*(곡|노래)", r"이 노래는 다음 노래"],
        [r"일정|수업|숙제|경기|내용|할게"],
    ),
    (
        "previous_track",
        [r"이전\s*(곡|노래)"],
        [r"수업|일정"],
    ),
    (
        "weather",
        [
            r"날씨.*(알려|말|어때|예보|알아봐|조회|확인|검색)",
            r"(알려|말|어때|예보|알아봐|조회|확인|검색).*(날씨)",
        ],
        [r"합니다|입니다|에요|예요|볼게요|할게요|드릴게요|검색해서|괜찮|봤어|봤니"],
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--positives-out")
    parser.add_argument("--negatives-out")
    parser.add_argument("--summary-out")
    parser.add_argument(
        "--keep-variants",
        default="S,N",
        help="Comma-separated variants to keep after mapping. Default keeps both S and N.",
    )
    args = parser.parse_args()

    keep_variants = {item.strip() for item in args.keep_variants.split(",") if item.strip()}
    rows = []
    stats = {
        "input_rows": 0,
        "kept_rows": 0,
        "labels": Counter(),
        "variants": Counter(),
        "texts_by_label": defaultdict(Counter),
    }

    for row in _read_jsonl(Path(args.manifest)):
        stats["input_rows"] += 1
        if keep_variants and row.get("variant") not in keep_variants:
            continue
        mapped_label, rule = _map_label(row.get("text") or "")
        mapped = dict(row)
        mapped["original_label"] = row.get("label")
        mapped["label"] = mapped_label
        mapped["aihub_mapping_rule"] = rule
        mapped["aihub_pair_id"] = _pair_id(mapped)
        rows.append(mapped)
        stats["kept_rows"] += 1
        stats["labels"][mapped_label] += 1
        stats["variants"][mapped.get("variant") or ""] += 1
        stats["texts_by_label"][mapped_label][mapped.get("text") or ""] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out, rows)
    if args.positives_out:
        positive_rows = [row for row in rows if row["label"] != "__negative__"]
        _write_jsonl(Path(args.positives_out), positive_rows)
    if args.negatives_out:
        negative_rows = [row for row in rows if row["label"] == "__negative__"]
        _write_jsonl(Path(args.negatives_out), negative_rows)
    summary = _summary_json(stats)
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _map_label(text: str) -> tuple[str, str]:
    for label, include_patterns, exclude_patterns in RULES:
        if not any(re.search(pattern, text) for pattern in include_patterns):
            continue
        if any(re.search(pattern, text) for pattern in exclude_patterns):
            return "__negative__", f"{label}:excluded"
        return label, label
    return "__negative__", "unmapped"


def _pair_id(row: dict) -> str:
    row_id = str(row.get("id") or "")
    if row_id.endswith("-S") or row_id.endswith("-N"):
        return row_id.rsplit("-", 1)[0]
    return row_id or str(row.get("audio_path") or "")


def _summary_json(stats: dict) -> dict:
    return {
        "input_rows": stats["input_rows"],
        "kept_rows": stats["kept_rows"],
        "labels": dict(stats["labels"].most_common()),
        "variants": dict(stats["variants"].most_common()),
        "top_texts_by_label": {
            label: dict(counter.most_common(20))
            for label, counter in sorted(stats["texts_by_label"].items())
        },
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
