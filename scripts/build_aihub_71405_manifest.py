from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/data/datasets/voice/aihub_71405/validation_seed",
        help="AIHub 71405 download root containing Validation/01 and 02 zip files.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--stats-out")
    parser.add_argument(
        "--extracted-root",
        help=(
            "Optional root produced by extracting zips. When set, rows include direct "
            "audio_path/label_path fields while retaining zip metadata."
        ),
    )
    parser.add_argument("--label", default="__negative__")
    parser.add_argument("--variants", default="S,N", help="Comma-separated wav variants to emit.")
    parser.add_argument("--short-sec", type=float, default=2.0)
    parser.add_argument("--max-rows", type=int, help="Optional cap for quick inspection.")
    args = parser.parse_args()

    root = Path(args.root)
    extracted_root = Path(args.extracted_root) if args.extracted_root else None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    variants = [variant.strip() for variant in args.variants.split(",") if variant.strip()]

    label_zips = sorted(root.rglob("Validation/02.라벨링데이터/VL_*.zip"))
    rows = []
    stats = {
        "root": str(root),
        "extracted_root": str(extracted_root) if extracted_root else None,
        "label_zips": len(label_zips),
        "rows": 0,
        "missing_source_zip": 0,
        "missing_extracted_wav": 0,
        "missing_wav_member": 0,
        "variants": Counter(),
        "domains": Counter(),
        "subdomains": Counter(),
        "noise_types": Counter(),
        "speech_length_bucket": Counter(),
        "selection": Counter(),
    }

    for label_zip in label_zips:
        source_zip = _source_zip_for_label_zip(label_zip)
        if not source_zip.exists():
            stats["missing_source_zip"] += 1
            continue
        with ZipFile(label_zip) as label_archive, ZipFile(source_zip) as source_archive:
            source_members = {member.lstrip("/"): member for member in source_archive.namelist()}
            for label_member in label_archive.namelist():
                if not label_member.lower().endswith(".json"):
                    continue
                label_doc = json.loads(label_archive.read(label_member).decode("utf-8"))
                for row in _rows_for_label(
                    label_doc=label_doc,
                    source_zip=source_zip,
                    source_members=source_members,
                    label_zip=label_zip,
                    label_member=label_member,
                    extracted_root=extracted_root,
                    label=args.label,
                    variants=variants,
                    short_sec=args.short_sec,
                ):
                    if row is None:
                        stats["missing_wav_member"] += 1
                        continue
                    if row.pop("_missing_extracted_wav", False):
                        stats["missing_extracted_wav"] += 1
                    rows.append(row)
                    _update_stats(stats, row)
                    if args.max_rows and len(rows) >= args.max_rows:
                        _write_outputs(rows, stats, out, args.stats_out)
                        return

    _write_outputs(rows, stats, out, args.stats_out)


def _source_zip_for_label_zip(label_zip: Path) -> Path:
    text = str(label_zip)
    text = text.replace("/02.라벨링데이터/", "/01.원천데이터/")
    text = text.replace("/VL_", "/VS_")
    return Path(text)


def _rows_for_label(
    label_doc: dict,
    source_zip: Path,
    source_members: dict[str, str],
    label_zip: Path,
    label_member: str,
    extracted_root: Path | None,
    label: str,
    variants: list[str],
    short_sec: float,
) -> list[dict | None]:
    file_info = label_doc.get("file", {})
    speaker = label_doc.get("speaker", {})
    noise = label_doc.get("noise", {})
    command = label_doc.get("command", {})
    base_name = file_info.get("name") or Path(label_member).stem.removesuffix("-J")
    start_sec = _float_or_none(file_info.get("beginOfSpeech")) or 0.0
    end_sec = _float_or_none(file_info.get("endOfSpeech"))
    speech_length = _float_or_none(file_info.get("speechLength"))
    if end_sec is None and speech_length is not None:
        end_sec = start_sec + speech_length
    selection = (
        "short_command_like" if (speech_length or 0.0) <= short_sec else "long_hard_negative"
    )

    rows = []
    for variant in variants:
        wav_name = f"{base_name}-{variant}.wav"
        zip_member = source_members.get(wav_name)
        if zip_member is None:
            rows.append(None)
            continue
        extracted_wav = _extracted_file(
            extracted_root=extracted_root,
            kind="source",
            zip_stem=source_zip.stem,
            member_name=wav_name,
        )
        extracted_label = _extracted_file(
            extracted_root=extracted_root,
            kind="labels",
            zip_stem=label_zip.stem,
            member_name=Path(label_member.lstrip("/")).name,
        )
        missing_extracted_wav = bool(extracted_root and extracted_wav is None)
        row = {
            "id": f"{base_name}-{variant}",
            "label": label,
            "text": command.get("text", ""),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "speech_length": speech_length,
            "sample_rate": 16000,
            "zip_path": str(source_zip),
            "zip_member": zip_member,
            "label_zip_path": str(label_zip),
            "label_zip_member": label_member,
            "source": "aihub_71405_validation",
            "variant": variant,
            "selection": selection,
            "speaker_id": speaker.get("id"),
            "speaker_age": speaker.get("age"),
            "speaker_gender": speaker.get("gender"),
            "recording_device": speaker.get("recordingDevice"),
            "noise_type": _nested(noise, "multipleSpeakersNoise", "type"),
            "noise_distance": _nested(noise, "multipleSpeakersNoise", "distance"),
            "car_speed": _nested(noise, "carNoise", "speed"),
            "car_type": _nested(noise, "carNoise", "carType"),
            "noise_level": noise.get("level"),
            "command_category": command.get("category"),
            "command_domain": command.get("domain"),
            "command_subdomain": command.get("subDomain"),
            "command_id": command.get("commandId"),
            "script_id": command.get("scriptId"),
            "snr": command.get("snr"),
            "command_level": command.get("level"),
            "qna": command.get("qna"),
            "_missing_extracted_wav": missing_extracted_wav,
        }
        if extracted_wav:
            row["audio_path"] = str(extracted_wav)
        if extracted_label:
            row["label_path"] = str(extracted_label)
        rows.append(
            row
        )
    return rows


def _extracted_file(
    extracted_root: Path | None,
    kind: str,
    zip_stem: str,
    member_name: str,
) -> Path | None:
    if extracted_root is None:
        return None
    path = extracted_root / kind / zip_stem / Path(member_name.lstrip("/")).name
    return path if path.exists() else None


def _nested(doc: dict, *keys: str) -> str | None:
    value = doc
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _float_or_none(value: object) -> float | None:
    if value in (None, "", "Null"):
        return None
    try:
        return float(str(value).replace("sec", "").strip())
    except ValueError:
        return None


def _update_stats(stats: dict, row: dict) -> None:
    stats["rows"] += 1
    stats["variants"][row["variant"]] += 1
    stats["domains"][row.get("command_domain") or ""] += 1
    stats["subdomains"][row.get("command_subdomain") or ""] += 1
    stats["noise_types"][row.get("noise_type") or row.get("car_type") or ""] += 1
    stats["selection"][row["selection"]] += 1
    length = row.get("speech_length") or 0.0
    if length <= 1.0:
        bucket = "<=1s"
    elif length <= 2.0:
        bucket = "1-2s"
    elif length <= 3.0:
        bucket = "2-3s"
    elif length <= 5.0:
        bucket = "3-5s"
    else:
        bucket = ">5s"
    stats["speech_length_bucket"][bucket] += 1


def _write_outputs(rows: list[dict], stats: dict, out: Path, stats_out: str | None) -> None:
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats_json = {
        key: dict(value) if isinstance(value, Counter) else value for key, value in stats.items()
    }
    if stats_out:
        stats_path = Path(stats_out)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(stats_json, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(stats_json, ensure_ascii=False, indent=2))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
