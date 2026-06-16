from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--errors", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--types", default="false_accept,false_reject")
    parser.add_argument("--limit-per-type", type=int, default=50)
    args = parser.parse_args()

    wanted_types = {item.strip() for item in args.types.split(",") if item.strip()}
    out_dir = Path(args.out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in Path(args.errors).read_text(encoding="utf-8").splitlines()]
    selected = []
    counts = {item: 0 for item in wanted_types}
    for row in rows:
        row_type = row.get("type")
        if row_type not in wanted_types:
            continue
        if counts[row_type] >= args.limit_per_type:
            continue
        counts[row_type] += 1
        selected.append(row)

    review_path = out_dir / "review.tsv"
    with review_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "\t".join(
                [
                    "review_id",
                    "type",
                    "label",
                    "matched_label",
                    "score",
                    "text",
                    "variant",
                    "domain",
                    "subdomain",
                    "audio_link",
                    "source_audio",
                    "decision",
                    "note",
                ]
            )
            + "\n"
        )
        for idx, row in enumerate(selected, start=1):
            source = Path(row["audio_path"])
            suffix = source.suffix or ".wav"
            review_id = f"{idx:03d}_{row['type']}_{row.get('matched_label') or 'none'}{suffix}"
            link = audio_dir / review_id
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(source, link)
            handle.write(
                "\t".join(
                    [
                        review_id,
                        str(row.get("type") or ""),
                        str(row.get("label") or ""),
                        str(row.get("matched_label") or ""),
                        f"{float(row.get('score', 0.0)):.6f}",
                        _clean_tsv(row.get("text") or ""),
                        str(row.get("variant") or ""),
                        _clean_tsv(row.get("command_domain") or ""),
                        _clean_tsv(row.get("command_subdomain") or ""),
                        str(link),
                        str(source),
                        "",
                        "",
                    ]
                )
                + "\n"
            )

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Audio Review",
                "",
                f"Source errors: `{args.errors}`",
                f"Review table: `{review_path}`",
                f"Audio symlinks: `{audio_dir}`",
                "",
                "Listen to the files in `audio/` and mark decisions in `review.tsv`.",
                (
                    "Suggested decisions: `keep_negative`, `map_positive`, "
                    "`exclude_unclear`, `bad_audio`."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {review_path}")
    print(f"linked {len(selected)} audio files under {audio_dir}")


def _clean_tsv(value: str) -> str:
    return value.replace("\t", " ").replace("\n", " ").strip()


if __name__ == "__main__":
    main()
