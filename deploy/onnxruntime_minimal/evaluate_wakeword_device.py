from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
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
THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.99,
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate EdgeSpot ONNX prototype wake-word models on all device positive "
            "and hard-negative wavs, using the same report shape as the openWakeWord "
            "Todak device evaluation."
        )
    )
    parser.add_argument("--device-record-dir", required=True, type=Path)
    parser.add_argument("--model", action="append", nargs=2, metavar=("NAME", "MODEL_PATH"))
    parser.add_argument("--support-count", type=int, default=5)
    parser.add_argument(
        "--support-mode",
        choices=["device-split", "first-k"],
        default="device-split",
        help="device-split uses even positive indices as support; first-k uses sorted first k.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    if not args.model:
        raise SystemExit("At least one --model NAME MODEL_PATH is required.")

    positive_dir = args.device_record_dir / "device_positive_eval"
    hard_negative_dir = args.device_record_dir / "device_hard_negative_eval"
    if not positive_dir.exists():
        raise FileNotFoundError(f"Missing positive dir: {positive_dir}")
    if not hard_negative_dir.exists():
        raise FileNotFoundError(f"Missing hard negative dir: {hard_negative_dir}")

    results = {}
    for name, model_path_value in args.model:
        model_path = Path(model_path_value)
        print(f"{name}: evaluating {model_path}")
        results[name] = evaluate_model(
            model_path=model_path,
            positive_dir=positive_dir,
            hard_negative_dir=hard_negative_dir,
            support_count=args.support_count,
            support_mode=args.support_mode,
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"saved: {args.output_json}")

    report = render_markdown(
        results,
        positive_dir=positive_dir,
        hard_negative_dir=hard_negative_dir,
        support_count=args.support_count,
        support_mode=args.support_mode,
    )
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report, encoding="utf-8")
        print(f"saved: {args.output_md}")
    else:
        print(report)


def evaluate_model(
    *,
    model_path: Path,
    positive_dir: Path,
    hard_negative_dir: Path,
    support_count: int,
    support_mode: str,
) -> dict:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    support = build_support(positive_dir, support_count, support_mode)
    prototypes = {
        label: normalize(np.mean([embed_wav(session, path) for path in paths], axis=0))
        for label, paths in support.items()
    }

    positive_rows = score_wavs(session, sorted(positive_dir.glob("*.wav")), prototypes)
    hard_negative_rows = score_wavs(session, sorted(hard_negative_dir.glob("*.wav")), prototypes)

    pos_scores = np.asarray([row["score"] for row in positive_rows], dtype=np.float32)
    neg_scores = np.asarray([row["score"] for row in hard_negative_rows], dtype=np.float32)
    thresholds = []
    for threshold in THRESHOLDS:
        pos_recall = float(np.mean(pos_scores >= threshold)) if pos_scores.size else 0.0
        neg_fp = int(np.sum(neg_scores >= threshold)) if neg_scores.size else 0
        thresholds.append(
            {
                "threshold": threshold,
                "device_positive_recall": pos_recall,
                "device_hard_negative_fp": neg_fp,
                "device_hard_negative_fp_rate": float(neg_fp / neg_scores.size)
                if neg_scores.size
                else 0.0,
            }
        )

    return {
        "model": str(model_path),
        "support_mode": support_mode,
        "support_count": support_count,
        "support": {label: [path.name for path in paths] for label, paths in support.items()},
        "positive_summary": summary(pos_scores),
        "hard_negative_summary": summary(neg_scores),
        "best_thresholds": best_thresholds(pos_scores, neg_scores),
        "thresholds": thresholds,
        "positive_scores": sorted(positive_rows, key=lambda row: row["score"]),
        "hard_negative_scores": sorted(hard_negative_rows, key=lambda row: row["score"], reverse=True),
    }


def build_support(positive_dir: Path, support_count: int, support_mode: str) -> dict[str, list[Path]]:
    if support_mode == "device-split":
        return {
            "토닥아": indexed_wavs(positive_dir, "todak", parity=0)[:support_count],
            "토닥이": indexed_wavs(positive_dir, "todaki", parity=0)[:support_count],
        }
    return {
        "토닥아": sorted(positive_dir.glob("todak_*.wav"))[:support_count],
        "토닥이": sorted(positive_dir.glob("todaki_*.wav"))[:support_count],
    }


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


def score_wavs(
    session: ort.InferenceSession,
    wavs: Iterable[Path],
    prototypes: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for wav in wavs:
        label, score = best_score(embed_wav(session, wav), prototypes)
        rows.append({"wav": str(wav), "matched_label": label, "score": score})
    return rows


def embed_wav(session: ort.InferenceSession, path: Path) -> np.ndarray:
    features = log_mel(path)
    embedding = session.run(None, {"features": features})[0][0]
    return normalize(embedding)


def best_score(
    embedding: np.ndarray,
    prototypes: dict[str, np.ndarray],
) -> tuple[str, float]:
    scores = [
        (label, float(np.dot(normalize(embedding), normalize(prototype))))
        for label, prototype in prototypes.items()
    ]
    return max(scores, key=lambda item: item[1])


def summary(scores: np.ndarray) -> dict[str, float | int]:
    if scores.size == 0:
        return {"n": 0, "min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "n": int(scores.shape[0]),
        "min": float(np.min(scores)),
        "median": float(np.median(scores)),
        "max": float(np.max(scores)),
    }


def best_thresholds(pos_scores: np.ndarray, neg_scores: np.ndarray) -> dict[str, dict]:
    candidate_thresholds = threshold_candidates(pos_scores, neg_scores)
    best_zero_fp = None
    best_f1 = None
    best_youden = None
    for threshold in candidate_thresholds:
        row = threshold_metrics(pos_scores, neg_scores, threshold)
        if row["false_accept"] == 0 and (
            best_zero_fp is None
            or (row["recall"], -row["threshold"]) > (best_zero_fp["recall"], -best_zero_fp["threshold"])
        ):
            best_zero_fp = row
        if best_f1 is None or (
            row["f1"],
            row["recall"],
            -row["false_accept"],
            -row["threshold"],
        ) > (
            best_f1["f1"],
            best_f1["recall"],
            -best_f1["false_accept"],
            -best_f1["threshold"],
        ):
            best_f1 = row
        if best_youden is None or (
            row["youden"],
            row["recall"],
            -row["false_accept"],
            -row["threshold"],
        ) > (
            best_youden["youden"],
            best_youden["recall"],
            -best_youden["false_accept"],
            -best_youden["threshold"],
        ):
            best_youden = row
    return {
        "max_recall_at_zero_fp": best_zero_fp or threshold_metrics(pos_scores, neg_scores, 1.0),
        "max_f1": best_f1 or threshold_metrics(pos_scores, neg_scores, 1.0),
        "max_youden": best_youden or threshold_metrics(pos_scores, neg_scores, 1.0),
    }


def threshold_candidates(pos_scores: np.ndarray, neg_scores: np.ndarray) -> list[float]:
    scores = sorted(set(float(score) for score in np.concatenate([pos_scores, neg_scores])))
    candidates = {0.0, 1.0}
    candidates.update(scores)
    candidates.update(score + 1e-6 for score in scores if score < 1.0)
    candidates.update((left + right) / 2.0 for left, right in zip(scores, scores[1:]))
    return sorted(candidates)


def threshold_metrics(pos_scores: np.ndarray, neg_scores: np.ndarray, threshold: float) -> dict:
    true_accept = int(np.sum(pos_scores >= threshold)) if pos_scores.size else 0
    false_reject = int(pos_scores.size - true_accept)
    false_accept = int(np.sum(neg_scores >= threshold)) if neg_scores.size else 0
    true_reject = int(neg_scores.size - false_accept)
    recall = float(true_accept / pos_scores.size) if pos_scores.size else 0.0
    fpr = float(false_accept / neg_scores.size) if neg_scores.size else 0.0
    precision = float(true_accept / (true_accept + false_accept)) if true_accept + false_accept else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "true_accept": true_accept,
        "false_reject": false_reject,
        "false_accept": false_accept,
        "true_reject": true_reject,
        "recall": recall,
        "false_positive_rate": fpr,
        "precision": precision,
        "f1": f1,
        "youden": recall - fpr,
    }


def render_markdown(
    results: dict,
    *,
    positive_dir: Path,
    hard_negative_dir: Path,
    support_count: int,
    support_mode: str,
) -> str:
    lines = ["# EdgeSpot Device Recording Evaluation Results", ""]
    lines.append(f"Positive dir: `{positive_dir}`")
    lines.append(f"Hard negative dir: `{hard_negative_dir}`")
    lines.append(f"Support mode: `{support_mode}`")
    lines.append(f"Support count per label: `{support_count}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| model | @0.50 pos recall | @0.50 hard FP | @0.90 pos recall | "
        "@0.90 hard FP | @0.95 pos recall | @0.95 hard FP |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, result in results.items():
        rows = result["thresholds"]
        row50 = closest_threshold(rows, 0.50)
        row90 = closest_threshold(rows, 0.90)
        row95 = closest_threshold(rows, 0.95)
        lines.append(
            f"| {name} | {row50['device_positive_recall']:.4f} | "
            f"{row50['device_hard_negative_fp']} | {row90['device_positive_recall']:.4f} | "
            f"{row90['device_hard_negative_fp']} | {row95['device_positive_recall']:.4f} | "
            f"{row95['device_hard_negative_fp']} |"
        )
    lines.append("")

    for name, result in results.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"Model: `{result['model']}`")
        lines.append("")
        lines.append("### Support")
        lines.append("")
        for label, filenames in result["support"].items():
            lines.append(f"- `{label}`: {', '.join(f'`{filename}`' for filename in filenames)}")
        lines.append("")
        lines.append("### Score Summary")
        lines.append("")
        lines.append("| set | n | min | median | max |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for label, score_summary in [
            ("device_positive", result["positive_summary"]),
            ("device_hard_negative", result["hard_negative_summary"]),
        ]:
            lines.append(
                f"| {label} | {score_summary['n']} | {score_summary['min']:.4f} | "
                f"{score_summary['median']:.4f} | {score_summary['max']:.4f} |"
            )
        lines.append("")
        lines.append("### Thresholds")
        lines.append("")
        lines.append("| threshold | device positive recall | hard negative FP | hard negative FP rate |")
        lines.append("| ---: | ---: | ---: | ---: |")
        for row in result["thresholds"]:
            lines.append(
                f"| {row['threshold']:.2f} | {row['device_positive_recall']:.4f} | "
                f"{row['device_hard_negative_fp']} | {row['device_hard_negative_fp_rate']:.4f} |"
            )
        lines.append("")
        lines.append("### Best Free Thresholds")
        lines.append("")
        lines.append(
            "| criterion | threshold | recall | hard FP | hard FP rate | precision | F1 | Youden |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for criterion, row in result["best_thresholds"].items():
            lines.append(
                f"| {criterion} | {row['threshold']:.6f} | {row['recall']:.4f} | "
                f"{row['false_accept']} | {row['false_positive_rate']:.4f} | "
                f"{row['precision']:.4f} | {row['f1']:.4f} | {row['youden']:.4f} |"
            )
        lines.append("")
        lines.append("### Lowest Positive Scores")
        lines.append("")
        lines.append("| score | matched label | wav |")
        lines.append("| ---: | --- | --- |")
        for row in result["positive_scores"][:10]:
            lines.append(f"| {row['score']:.6f} | `{row['matched_label']}` | `{row['wav']}` |")
        lines.append("")
        lines.append("### Highest Hard Negative Scores")
        lines.append("")
        lines.append("| score | matched label | wav |")
        lines.append("| ---: | --- | --- |")
        for row in result["hard_negative_scores"][:10]:
            lines.append(f"| {row['score']:.6f} | `{row['matched_label']}` | `{row['wav']}` |")
        lines.append("")
    return "\n".join(lines) + "\n"


def closest_threshold(rows: list[dict], threshold: float) -> dict:
    return min(rows, key=lambda row: abs(row["threshold"] - threshold))


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


if __name__ == "__main__":
    main()
