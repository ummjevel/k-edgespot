from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from edgespot.tts.qwen3 import Qwen3TTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--done-manifest")
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument("--cache-dir", default="models/huggingface")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qwen-source", default="/data/users/voice/zoey/Qwen3-TTS")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")

    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done_manifest = (
        Path(args.done_manifest) if args.done_manifest else manifest.with_suffix(".done.jsonl")
    )

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    rows = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
    if args.limit:
        rows = rows[: args.limit]

    tts = Qwen3TTS(
        model_id=args.model_id,
        device=args.device,
        qwen_source=args.qwen_source,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        cache_dir=args.cache_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    done_rows = []
    for start in tqdm(range(0, len(rows), args.batch_size), desc="synth"):
        batch = rows[start : start + args.batch_size]
        wavs, sample_rate = tts.synthesize_batch(batch)
        for row, wav in zip(batch, wavs, strict=True):
            audio_path = out_dir / row["label"] / f"{row['id']}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(audio_path, wav, sample_rate)
            row = dict(row)
            row["audio_path"] = str(audio_path)
            row["source"] = "qwen3_tts"
            done_rows.append(row)

    with done_manifest.open("w", encoding="utf-8") as f:
        for row in done_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(done_rows)} rows to {done_manifest}")


if __name__ == "__main__":
    main()
