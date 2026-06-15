from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True, help="Glob pattern for shard done manifests.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected", type=int)
    args = parser.parse_args()

    paths = sorted(Path().glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files matched {args.pattern}")

    rows = []
    seen_audio = set()
    duplicates = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            audio_path = row.get("audio_path")
            if audio_path in seen_audio:
                duplicates += 1
                continue
            seen_audio.add(audio_path)
            rows.append(row)

    if args.expected is not None and len(rows) != args.expected:
        raise SystemExit(f"Expected {args.expected} rows, collected {len(rows)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"collected {len(rows)} rows from {len(paths)} shards into {out}; "
        f"duplicates={duplicates}"
    )


if __name__ == "__main__":
    main()
