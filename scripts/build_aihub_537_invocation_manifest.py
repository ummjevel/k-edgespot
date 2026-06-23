from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
LABEL_SUFFIXES = {".json"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Downloaded AIHub 537 root.")
    parser.add_argument("--extracted-root", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--source-name", default="aihub_537_invocation")
    parser.add_argument("--min-label-count", type=int, default=20)
    parser.add_argument("--max-labels", type=int, default=2048)
    parser.add_argument("--max-rows-per-label", type=int, default=500)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--include-non-invocation",
        action="store_true",
        help="Keep rows without invocation markers. Default keeps only invocation-like rows.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    extracted_root = Path(args.extracted_root)
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    audio_index = _audio_index(extracted_root)
    label_paths = sorted(
        path for path in extracted_root.rglob("*") if path.suffix.lower() in LABEL_SUFFIXES
    )
    rows = []
    stats = {
        "root": str(root),
        "extracted_root": str(extracted_root),
        "label_jsons": len(label_paths),
        "audio_files_indexed": sum(len(paths) for paths in audio_index.values()),
        "missing_audio": 0,
        "missing_text": 0,
        "non_invocation_filtered": 0,
        "raw_rows": 0,
        "kept_rows": 0,
        "labels_before_filter": 0,
        "labels_after_filter": 0,
        "workers": args.workers,
        "splits": Counter(),
        "category_values": Counter(),
    }

    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _row_for_label_path,
                label_path,
                audio_index,
                args.include_non_invocation,
                args.source_name,
            )
            for label_path in label_paths
        ]
        for future in as_completed(futures):
            status, row = future.result()
            stats[status] += 1
            if row is None:
                continue
            category = row.get("source_category")
            if category:
                stats["category_values"][category] += 1
            rows.append(row)

    split_rows, split_summary = _stratified_split(
        rows=rows,
        min_label_count=args.min_label_count,
        max_labels=args.max_labels,
        max_rows_per_label=args.max_rows_per_label,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    stats["kept_rows"] = len(split_rows["all"])
    stats["labels_before_filter"] = len({row["label"] for row in rows})
    stats["labels_after_filter"] = len({row["label"] for row in split_rows["all"]})
    stats["splits"].update({name: len(items) for name, items in split_rows.items()})
    stats["split_summary"] = split_summary

    _write_jsonl(manifest_dir / "all.jsonl", split_rows["all"])
    _write_jsonl(manifest_dir / "train.jsonl", split_rows["train"])
    _write_jsonl(manifest_dir / "val.jsonl", split_rows["val"])
    _write_json(manifest_dir / "summary.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def _row_for_label_path(
    label_path: Path,
    audio_index: dict[str, list[Path]],
    include_non_invocation: bool,
    source_name: str,
) -> tuple[str, dict | None]:
    try:
        doc = json.loads(label_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        doc = json.loads(label_path.read_text(encoding="cp949"))
    flat = list(_flatten(doc))
    text = _extract_text(doc, flat)
    if not text:
        return "missing_text", None
    audio_path = _match_audio(label_path, doc, flat, audio_index)
    if audio_path is None:
        return "missing_audio", None
    is_invocation = _has_invocation_marker(label_path, flat)
    if not is_invocation and not include_non_invocation:
        return "non_invocation_filtered", None
    category = _first_value(flat, ("category", "카테고리", "domain", "도메인", "type", "유형"))
    label = _label_for_text(text)
    row = {
        "id": _row_id(audio_path),
        "label": label,
        "text": text,
        "sample_rate": 16000,
        "audio_path": str(audio_path),
        "label_path": str(label_path),
        "source": source_name,
        "source_dataset": "aihub_537",
        "source_label": text,
        "source_category": category,
        "speaker_id": _first_value(flat, ("speaker_id", "speakerId", "speaker", "화자", "id")),
        "split_hint": _split_hint(label_path),
        "is_invocation_marker": is_invocation,
    }
    return "raw_rows", row


def _audio_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        keys = {
            path.stem.lower(),
            path.name.lower(),
            path.stem.replace("-S", "").replace("-N", "").lower(),
        }
        for key in keys:
            index[key].append(path)
    return index


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    out = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.extend(_flatten(child, child_prefix))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            out.extend(_flatten(child, f"{prefix}[{idx}]"))
    elif value is not None:
        out.append((prefix, str(value).strip()))
    return out


def _extract_text(doc: dict, flat: list[tuple[str, str]]) -> str | None:
    direct_paths = (
        ("command", "text"),
        ("script", "text"),
        ("file", "text"),
        ("utterance", "text"),
        ("sentence", "text"),
    )
    for path in direct_paths:
        value = doc
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, str) and _looks_like_text(value):
            return _normalize_text(value)
    preferred = ("text", "transcript", "transcription", "sentence", "script", "utterance", "발화", "문장", "호출어")
    for key, value in flat:
        key_l = key.lower()
        if any(token in key_l for token in preferred) and _looks_like_text(value):
            return _normalize_text(value)
    return None


def _match_audio(
    label_path: Path,
    doc: dict,
    flat: list[tuple[str, str]],
    audio_index: dict[str, list[Path]],
) -> Path | None:
    candidates = []
    for key, value in flat:
        if any(token in key.lower() for token in ("file", "audio", "wav", "filename", "name")):
            candidates.append(Path(value).name)
    file_info = doc.get("file") if isinstance(doc, dict) else None
    if isinstance(file_info, dict):
        for value in file_info.values():
            if isinstance(value, str):
                candidates.append(Path(value).name)
    candidates.extend([label_path.stem, label_path.stem.removesuffix("-J")])
    for candidate in candidates:
        keys = {candidate.lower(), Path(candidate).stem.lower()}
        keys.add(Path(candidate).stem.replace("-J", "").lower())
        for key in keys:
            matches = audio_index.get(key)
            if matches:
                return matches[0]
    return None


def _has_invocation_marker(label_path: Path, flat: list[tuple[str, str]]) -> bool:
    haystack = " ".join([str(label_path), *[key for key, _ in flat], *[value for _, value in flat]])
    return any(marker in haystack for marker in ("호출어", "호출", "wake", "Wake", "WAKE"))


def _first_value(flat: list[tuple[str, str]], keys: tuple[str, ...]) -> str | None:
    key_lowers = tuple(key.lower() for key in keys)
    for key, value in flat:
        key_lower = key.lower()
        if any(token in key_lower for token in key_lowers) and value:
            return value
    return None


def _split_hint(path: Path) -> str | None:
    text = str(path)
    if "Training" in text or "train" in text.lower():
        return "Training"
    if "Validation" in text or "valid" in text.lower() or "val" in text.lower():
        return "Validation"
    return None


def _looks_like_text(value: str) -> bool:
    value = _normalize_text(value)
    if not value or len(value) > 80:
        return False
    if value.lower().endswith((".wav", ".json", ".zip")):
        return False
    return bool(re.search(r"[가-힣A-Za-z]", value))


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().strip(".?!")


def _label_for_text(text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^0-9A-Za-z가-힣]+", "_", text).strip("_")[:48]
    return f"inv_{slug}_{digest}"


def _row_id(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]


def _stratified_split(
    rows: list[dict],
    min_label_count: int,
    max_labels: int,
    max_rows_per_label: int,
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, list[dict]], dict]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    labels = [
        label
        for label, items in sorted(by_label.items(), key=lambda item: (-len(item[1]), item[0]))
        if len(items) >= min_label_count
    ][:max_labels]
    train = []
    val = []
    label_counts = {}
    for label in labels:
        items = list(by_label[label])
        rng.shuffle(items)
        items = items[:max_rows_per_label]
        val_count = max(1, int(round(len(items) * val_ratio)))
        val_count = min(val_count, len(items) - 1)
        val_items = items[:val_count]
        train_items = items[val_count:]
        for row in train_items:
            row["split"] = "train"
        for row in val_items:
            row["split"] = "val"
        train.extend(train_items)
        val.extend(val_items)
        label_counts[label] = {"train": len(train_items), "val": len(val_items)}
    rng.shuffle(train)
    rng.shuffle(val)
    all_rows = train + val
    return {"all": all_rows, "train": train, "val": val}, {"labels": label_counts}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, value: dict) -> None:
    def default(obj: object) -> object:
        if isinstance(obj, Counter):
            return dict(obj)
        raise TypeError(type(obj).__name__)

    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default), encoding="utf-8")


if __name__ == "__main__":
    main()
