from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--selection")
    parser.add_argument("--domain-regex")
    parser.add_argument("--subdomain-regex")
    parser.add_argument("--text-regex")
    parser.add_argument("--max-speech-length", type=float)
    parser.add_argument("--min-speech-length", type=float)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    filters = {
        "domain": re.compile(args.domain_regex) if args.domain_regex else None,
        "subdomain": re.compile(args.subdomain_regex) if args.subdomain_regex else None,
        "text": re.compile(args.text_regex) if args.text_regex else None,
    }

    kept = 0
    total = 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.manifest).open(encoding="utf-8") as inp, out.open("w", encoding="utf-8") as dst:
        for line in inp:
            total += 1
            row = json.loads(line)
            if not _keep(row, args, filters):
                continue
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            if args.limit and kept >= args.limit:
                break
    print(f"wrote {kept} / {total} rows to {out}")


def _keep(row: dict, args: argparse.Namespace, filters: dict) -> bool:
    if args.selection and row.get("selection") != args.selection:
        return False
    speech_length = row.get("speech_length")
    if args.min_speech_length is not None and (
        speech_length is None or float(speech_length) < args.min_speech_length
    ):
        return False
    if args.max_speech_length is not None and (
        speech_length is None or float(speech_length) > args.max_speech_length
    ):
        return False
    if filters["domain"] and not filters["domain"].search(row.get("command_domain") or ""):
        return False
    if filters["subdomain"] and not filters["subdomain"].search(row.get("command_subdomain") or ""):
        return False
    if filters["text"] and not filters["text"].search(row.get("text") or ""):
        return False
    return True


if __name__ == "__main__":
    main()
