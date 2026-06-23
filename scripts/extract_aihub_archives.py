from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zipfile import ZipFile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="AIHub downloaded dataset root.")
    parser.add_argument("--out-root", required=True, help="Directory for extracted files.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)
    archives = sorted(root.rglob("*.zip"))
    stats = {
        "root": str(root),
        "out_root": str(out_root),
        "archives": len(archives),
        "source_archives": sum(1 for path in archives if _archive_kind(path) == "source"),
        "label_archives": sum(1 for path in archives if _archive_kind(path) == "label"),
        "other_archives": sum(1 for path in archives if _archive_kind(path) == "other"),
        "workers": args.workers,
        "members": 0,
        "extracted": 0,
        "skipped_existing": 0,
        "bad_archives": [],
    }

    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_extract_archive, archive_path, out_root, args.overwrite)
            for archive_path in archives
        ]
        for future in as_completed(futures):
            result = future.result()
            stats["members"] += result["members"]
            stats["extracted"] += result["extracted"]
            stats["skipped_existing"] += result["skipped_existing"]
            if result.get("error"):
                stats["bad_archives"].append(
                    {"path": result["path"], "error": result["error"]}
                )

    print(json.dumps(stats, ensure_ascii=False, indent=2))


def _extract_archive(archive_path: Path, out_root: Path, overwrite: bool) -> dict:
    kind = _archive_kind(archive_path)
    target_dir = out_root / kind / archive_path.stem
    target_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "path": str(archive_path),
        "members": 0,
        "extracted": 0,
        "skipped_existing": 0,
        "error": None,
    }
    try:
        with ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                name = Path(member.filename).name
                if not name:
                    continue
                out_path = target_dir / name
                result["members"] += 1
                if out_path.exists() and not overwrite:
                    result["skipped_existing"] += 1
                    continue
                with archive.open(member) as src, out_path.open("wb") as dst:
                    dst.write(src.read())
                result["extracted"] += 1
    except Exception as exc:  # pragma: no cover - defensive for third-party archives.
        result["error"] = repr(exc)
    return result


def _archive_kind(path: Path) -> str:
    text = str(path)
    name = path.name
    if "라벨" in text or name.startswith(("TL_", "VL_")):
        return "label"
    if "원천" in text or name.startswith(("TS_", "VS_")):
        return "source"
    return "other"


if __name__ == "__main__":
    main()
