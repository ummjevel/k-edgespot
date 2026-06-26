from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy import signal


SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400
TARGET_FRAMES = 101


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to EdgeSpot encoder.onnx")
    parser.add_argument("--device-record-dir", required=True, help="Path to device_record")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument(
        "--support-count",
        type=int,
        default=5,
        help="Number of support wavs per positive label prefix.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["device-split", "all"],
        default="device-split",
        help=(
            "device-split matches the experiment manifests: even positive indices as support, "
            "odd positive and odd hard-negative indices as query. all uses first-k support and "
            "evaluates all remaining positive plus all hard negatives."
        ),
    )
    parser.add_argument(
        "--include-support-in-query",
        action="store_true",
        help="Also evaluate support files as query files.",
    )
    parser.add_argument(
        "--out-prototypes",
        help="Optional path to write prototype vectors as JSON.",
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        help="Run latency benchmark on the first query wav. Measures feature, encoder, and total.",
    )
    args = parser.parse_args()

    root = Path(args.device_record_dir)
    positive_dir = root / "device_positive_eval"
    hard_negative_dir = root / "device_hard_negative_eval"
    if not positive_dir.exists():
        raise FileNotFoundError(f"Missing positive dir: {positive_dir}")
    if not hard_negative_dir.exists():
        raise FileNotFoundError(f"Missing hard negative dir: {hard_negative_dir}")

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    support = build_support(positive_dir, args.support_count, args.eval_mode)
    support_paths = {path for rows in support.values() for path in rows}
    prototypes = {
        label: normalize(np.mean([embed_wav(session, path) for path in paths], axis=0))
        for label, paths in support.items()
    }

    if args.out_prototypes:
        out = Path(args.out_prototypes)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {label: proto.astype(float).tolist() for label, proto in prototypes.items()},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    query_paths = build_queries(positive_dir, hard_negative_dir, args.eval_mode)
    if not args.include_support_in_query:
        query_paths = [path for path in query_paths if path not in support_paths]

    print("model:", args.model)
    print("threshold:", args.threshold)
    print("support:")
    for label, paths in support.items():
        print(f"  {label}: {len(paths)} files")
        for path in paths:
            print(f"    {path.name}")
    print("queries:", len(query_paths))
    print()

    rows = []
    for path in query_paths:
        embedding = embed_wav(session, path)
        label, score = best_score(embedding, prototypes)
        prediction = score >= args.threshold
        expected_positive = path.parent.name == "device_positive_eval"
        rows.append((path, expected_positive, label, score, prediction))
        status = "ACCEPT" if prediction else "reject"
        truth = "positive" if expected_positive else "negative"
        print(f"{status}\t{score:.6f}\t{label}\t{truth}\t{path}")

    summarize(rows)

    if args.benchmark > 0 and query_paths:
        benchmark_pipeline(session, query_paths[0], args.benchmark)


def build_support(positive_dir: Path, support_count: int, eval_mode: str) -> dict[str, list[Path]]:
    if eval_mode == "device-split":
        return {
            "토닥아": indexed_wavs(positive_dir, "todak", parity=0)[:support_count],
            "토닥이": indexed_wavs(positive_dir, "todaki", parity=0)[:support_count],
        }
    return {
        "토닥아": sorted(positive_dir.glob("todak_*.wav"))[:support_count],
        "토닥이": sorted(positive_dir.glob("todaki_*.wav"))[:support_count],
    }


def build_queries(positive_dir: Path, hard_negative_dir: Path, eval_mode: str) -> list[Path]:
    if eval_mode == "device-split":
        positives = (
            indexed_wavs(positive_dir, "todak", parity=1)
            + indexed_wavs(positive_dir, "todaki", parity=1)
        )
        hard_negatives = []
        for prefix in ["toda", "todatoda", "tomatodak", "toyo"]:
            hard_negatives.extend(indexed_wavs(hard_negative_dir, prefix, parity=1))
        return sorted(positives + hard_negatives)
    return sorted(positive_dir.glob("*.wav")) + sorted(hard_negative_dir.glob("*.wav"))


def indexed_wavs(root: Path, prefix: str, parity: int | None = None) -> list[Path]:
    rows = []
    for path in root.glob(f"{prefix}_*.wav"):
        try:
            index = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if parity is None or index % 2 == parity:
            rows.append((index, path))
    return [path for _, path in sorted(rows)]


def embed_wav(session: ort.InferenceSession, path: Path) -> np.ndarray:
    features = log_mel(path)
    embedding = session.run(None, {"features": features})[0][0]
    return normalize(embedding)


def benchmark_pipeline(session: ort.InferenceSession, path: Path, loops: int) -> None:
    print()
    print(f"benchmark_file: {path}")
    for _ in range(3):
        features = log_mel(path)
        session.run(None, {"features": features})

    feature_latencies = []
    encoder_latencies = []
    total_latencies = []
    for _ in range(loops):
        total_start = time.perf_counter()

        feature_start = time.perf_counter()
        features = log_mel(path)
        feature_latencies.append((time.perf_counter() - feature_start) * 1000.0)

        encoder_start = time.perf_counter()
        session.run(None, {"features": features})
        encoder_latencies.append((time.perf_counter() - encoder_start) * 1000.0)

        total_latencies.append((time.perf_counter() - total_start) * 1000.0)

    print_latency("feature_latency_ms", feature_latencies)
    print_latency("encoder_latency_ms", encoder_latencies)
    print_latency("total_latency_ms", total_latencies)


def best_score(
    embedding: np.ndarray,
    prototypes: dict[str, np.ndarray],
) -> tuple[str, float]:
    scores = [
        (label, float(np.dot(normalize(embedding), normalize(prototype))))
        for label, prototype in prototypes.items()
    ]
    return max(scores, key=lambda item: item[1])


def summarize(rows: list[tuple[Path, bool, str, float, bool]]) -> None:
    positives = [row for row in rows if row[1]]
    negatives = [row for row in rows if not row[1]]
    true_accept = sum(1 for _, expected, _, _, pred in rows if expected and pred)
    false_reject = sum(1 for _, expected, _, _, pred in rows if expected and not pred)
    false_accept = sum(1 for _, expected, _, _, pred in rows if not expected and pred)
    true_reject = sum(1 for _, expected, _, _, pred in rows if not expected and not pred)

    print()
    print("summary:")
    print(f"  positives: {len(positives)}")
    print(f"  negatives: {len(negatives)}")
    print(f"  true_accept: {true_accept}")
    print(f"  false_reject: {false_reject}")
    print(f"  false_accept: {false_accept}")
    print(f"  true_reject: {true_reject}")
    if positives:
        print(f"  positive_recall: {true_accept / len(positives):.4f}")
    if negatives:
        print(f"  negative_fp_rate: {false_accept / len(negatives):.4f}")


def log_mel(path: Path) -> np.ndarray:
    wav, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        gcd = math.gcd(sample_rate, SAMPLE_RATE)
        wav = signal.resample_poly(wav, SAMPLE_RATE // gcd, sample_rate // gcd).astype(np.float32)
    wav = fit_length(wav, SAMPLE_RATE)

    padded = np.pad(wav, (N_FFT // 2, N_FFT // 2), mode="reflect")
    window = signal.get_window("hann", WIN_LENGTH, fftbins=True).astype(np.float32)
    frames = []
    for start in range(0, len(padded) - N_FFT + 1, HOP_LENGTH):
        frame = padded[start : start + N_FFT].copy()
        frame[:WIN_LENGTH] *= window
        spectrum = np.fft.rfft(frame, n=N_FFT)
        frames.append((np.abs(spectrum) ** 2).astype(np.float32))
    spec = np.stack(frames, axis=1)

    mel = mel_filterbank() @ spec
    mel = np.maximum(mel, 1e-6)
    if mel.shape[-1] < TARGET_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, TARGET_FRAMES - mel.shape[-1])))
    mel = mel[:, :TARGET_FRAMES]
    return mel[None, None, :, :].astype(np.float32)


def fit_length(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    target = sample_rate
    if wav.shape[0] >= target:
        return wav[:target]
    return np.pad(wav, (0, target - wav.shape[0]))


def mel_filterbank() -> np.ndarray:
    fft_freqs = np.linspace(0.0, SAMPLE_RATE / 2.0, 1 + N_FFT // 2)
    mel_min = hz_to_mel(0.0)
    mel_max = hz_to_mel(SAMPLE_RATE / 2.0)
    mel_points = np.linspace(mel_min, mel_max, N_MELS + 2)
    hz_points = mel_to_hz(mel_points)

    fdiff = np.diff(hz_points)
    ramps = hz_points[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (hz_points[2 : N_MELS + 2] - hz_points[:N_MELS])
    weights *= enorm[:, None]
    return weights.astype(np.float32)


def hz_to_mel(freq: float | np.ndarray) -> float | np.ndarray:
    scalar_input = np.isscalar(freq)
    freq = np.atleast_1d(np.asarray(freq, dtype=np.float64))
    f_sp = 200.0 / 3
    mels = freq / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    mask = freq >= min_log_hz
    if np.any(mask):
        mels = mels.copy()
        mels[mask] = min_log_mel + np.log(freq[mask] / min_log_hz) / logstep
    return float(mels[0]) if scalar_input else mels


def mel_to_hz(mels: float | np.ndarray) -> float | np.ndarray:
    scalar_input = np.isscalar(mels)
    mels = np.atleast_1d(np.asarray(mels, dtype=np.float64))
    f_sp = 200.0 / 3
    freqs = mels * f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    mask = mels >= min_log_mel
    if np.any(mask):
        freqs = freqs.copy()
        freqs[mask] = min_log_hz * np.exp(logstep * (mels[mask] - min_log_mel))
    return float(freqs[0]) if scalar_input else freqs


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def print_latency(name: str, latencies: list[float]) -> None:
    values = np.asarray(latencies, dtype=np.float64)
    print(f"  {name}_avg: {values.mean():.4f}")
    print(f"  {name}_min: {values.min():.4f}")
    print(f"  {name}_max: {values.max():.4f}")
    print(f"  {name}_p50: {np.percentile(values, 50):.4f}")
    print(f"  {name}_p95: {np.percentile(values, 95):.4f}")


if __name__ == "__main__":
    main()
