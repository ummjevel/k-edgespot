from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--negative-metadata")
    parser.add_argument("--negative-limit", type=int, default=0)
    parser.add_argument("--negative-seed", type=int, default=2026)
    parser.add_argument("--negative-texts")
    parser.add_argument("--negative-text-takes", type=int, default=1)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for command in config["commands"]:
        for voice_idx, voice_prompt in enumerate(config["voice_design_prompts"]):
            for take in range(config["clips_per_command_per_voice"]):
                rows.append(
                    {
                        "id": f"{command['id']}_v{voice_idx:03d}_{take:03d}",
                        "label": command["id"],
                        "text": command["text"],
                        "language": "ko",
                        "voice_prompt": voice_prompt,
                        "take": take,
                        "sample_rate": config.get("sample_rate", 16000),
                    }
                )

    if args.negative_metadata:
        rows.extend(
            _negative_rows_from_metadata(
                metadata=Path(args.negative_metadata),
                voice_prompts=config["voice_design_prompts"],
                sample_rate=config.get("sample_rate", 16000),
                limit=args.negative_limit,
                seed=args.negative_seed,
            )
        )

    if args.negative_texts:
        rows.extend(
            _negative_rows_from_texts(
                text_path=Path(args.negative_texts),
                voice_prompts=config["voice_design_prompts"],
                sample_rate=config.get("sample_rate", 16000),
                takes=args.negative_text_takes,
            )
        )

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


def _negative_rows_from_metadata(
    metadata: Path,
    voice_prompts: list[str],
    sample_rate: int,
    limit: int,
    seed: int,
) -> list[dict]:
    candidates = []
    for line in metadata.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        rel_path, text = parts[:2]
        text = text.strip()
        if not text:
            continue
        candidates.append((Path(rel_path).stem, text))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    if limit > 0:
        candidates = candidates[:limit]

    rows = []
    for idx, (utt_id, text) in enumerate(candidates):
        voice_idx = idx % len(voice_prompts)
        rows.append(
            {
                "id": f"negative_{utt_id}_v{voice_idx:03d}",
                "label": "__negative__",
                "text": text,
                "language": "ko",
                "voice_prompt": voice_prompts[voice_idx],
                "take": 0,
                "sample_rate": sample_rate,
                "source_text_id": utt_id,
            }
        )
    return rows


def _negative_rows_from_texts(
    text_path: Path,
    voice_prompts: list[str],
    sample_rate: int,
    takes: int,
) -> list[dict]:
    texts = [
        line.strip()
        for line in text_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    rows = []
    for text_idx, text in enumerate(texts):
        for voice_idx, voice_prompt in enumerate(voice_prompts):
            for take in range(takes):
                rows.append(
                    {
                        "id": f"hard_negative_{text_idx:04d}_v{voice_idx:03d}_{take:03d}",
                        "label": "__negative__",
                        "text": text,
                        "language": "ko",
                        "voice_prompt": voice_prompt,
                        "take": take,
                        "sample_rate": sample_rate,
                        "source": "hard_negative_text",
                    }
                )
    return rows


if __name__ == "__main__":
    main()
