from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from edgespot.features import LogMel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-record-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--hop-sec", type=float, default=1.0)
    parser.add_argument("--min-rms", type=float, default=1e-4)
    args = parser.parse_args()

    root = Path(args.device_record_dir)
    wav_paths = sorted(path for path in root.glob("*.wav") if path.is_file())
    if not wav_paths:
        raise FileNotFoundError(f"No top-level wav files found under {root}")

    extractor = LogMel(sample_rate=args.sample_rate)
    features = []
    stats = {
        "source_files": [str(path) for path in wav_paths],
        "num_windows": 0,
        "skipped_low_rms": 0,
        "sample_rate": args.sample_rate,
        "window_sec": args.window_sec,
        "hop_sec": args.hop_sec,
        "min_rms": args.min_rms,
    }
    window = int(round(args.window_sec * args.sample_rate))
    hop = int(round(args.hop_sec * args.sample_rate))
    for path in wav_paths:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav_t = torch.from_numpy(wav)
        if sr != args.sample_rate:
            wav_t = torchaudio.functional.resample(wav_t, sr, args.sample_rate)
        if wav_t.numel() < window:
            starts = [0]
        else:
            starts = list(range(0, wav_t.numel() - window + 1, max(1, hop)))
        for start in starts:
            segment = wav_t[start : start + window]
            if segment.numel() < window:
                segment = torch.nn.functional.pad(segment, (0, window - segment.numel()))
            rms = float(torch.sqrt(torch.mean(segment.square())).item())
            if rms < args.min_rms:
                stats["skipped_low_rms"] += 1
                continue
            mel = extractor(segment, args.sample_rate).squeeze(0).numpy()
            features.append(mel)

    if not features:
        raise RuntimeError("No usable windows found for profile")

    arr = np.stack(features, axis=0)
    freq_means = arr.mean(axis=-1)
    global_means = arr.mean(axis=(1, 2))
    centered_freq = freq_means - global_means[:, None]
    profile = {
        **stats,
        "num_windows": int(arr.shape[0]),
        "freq_bias_mean": centered_freq.mean(axis=0).astype(float).tolist(),
        "freq_bias_std": centered_freq.std(axis=0).astype(float).tolist(),
        "global_mean_std": float(global_means.std()),
        "frame_noise_std": float(arr.std(axis=0).mean()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: profile[k] for k in stats | {"num_windows": None}}, indent=2))


if __name__ == "__main__":
    main()
