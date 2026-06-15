from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    metadata = Path(args.metadata)
    audio_root = Path(args.audio_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for line in metadata.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        rel_path, transcript, speaker_id = parts[:3]
        rows.append(
            {
                "id": Path(rel_path).stem,
                "label": "__negative__",
                "text": transcript,
                "speaker_id": speaker_id,
                "audio_path": str(audio_root / rel_path),
                "source": "korean_style_tts_metadata",
            }
        )

    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit]

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
