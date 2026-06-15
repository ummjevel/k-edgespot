from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


DEFAULT_DATA_ROOT = Path("/data/datasets/voice")
DEFAULT_CACHE_ROOT = DEFAULT_DATA_ROOT / ".cache"

FILES = {
    "gsc_v2": {
        "url": "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
        "out": "google_speech_commands/speech_commands_v0.02.tar.gz",
    },
    "mswc_metadata": {
        "url": "https://mswc.mlcommons-storage.org/metadata.json.gz",
        "out": "mswc/metadata.json.gz",
    },
    "mswc_en_audio": {
        "url": "https://mswc.mlcommons-storage.org/audio/en.tar.gz",
        "out": "mswc/en/audio/en.tar.gz",
    },
    "mswc_en_splits": {
        "url": "https://mswc.mlcommons-storage.org/splits/en.tar.gz",
        "out": "mswc/en/splits/en.tar.gz",
    },
    "mswc_en_alignments": {
        "url": "https://mswc.mlcommons-storage.org/alignments/en.tar.gz",
        "out": "mswc/en/alignments/en.tar.gz",
    },
}

PRESETS = {
    "small": ["gsc_v2", "mswc_metadata"],
    "mswc_english": ["mswc_metadata", "mswc_en_audio", "mswc_en_splits", "mswc_en_alignments"],
    "default": [
        "gsc_v2",
        "mswc_metadata",
        "mswc_en_audio",
        "mswc_en_splits",
        "mswc_en_alignments",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="default")
    parser.add_argument("--only", action="append", choices=sorted(FILES))
    parser.add_argument("--no-clobber", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    downloader = _downloader()
    keys = args.only or PRESETS[args.preset]
    args.data_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    for key in keys:
        item = FILES[key]
        out = args.data_root / item["out"]
        out.parent.mkdir(parents=True, exist_ok=True)
        if args.no_clobber and out.exists():
            print(f"skip existing {key}: {out}")
            continue
        print(f"download {key}: {item['url']} -> {out}")
        if args.dry_run:
            continue
        _download(downloader, item["url"], out)


def _downloader() -> str:
    for name in ["wget", "curl"]:
        path = shutil.which(name)
        if path:
            return name
    raise RuntimeError("Neither wget nor curl is available")


def _download(downloader: str, url: str, out: Path) -> None:
    if downloader == "wget":
        cmd = [
            "wget",
            "--continue",
            "--tries=20",
            "--timeout=30",
            "--waitretry=10",
            "--output-document",
            str(out),
            url,
        ]
    else:
        cmd = [
            "curl",
            "--location",
            "--continue-at",
            "-",
            "--retry",
            "20",
            "--retry-delay",
            "10",
            "--output",
            str(out),
            url,
        ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
