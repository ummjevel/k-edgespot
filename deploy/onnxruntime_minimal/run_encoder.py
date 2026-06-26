from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to EdgeSpot encoder.onnx")
    parser.add_argument(
        "--features-npy",
        help="Optional .npy containing log-mel features shaped [1,1,40,101] or [40,101].",
    )
    parser.add_argument(
        "--prototype-json",
        help="Optional JSON mapping label -> 64-D prototype vector for cosine scoring.",
    )
    parser.add_argument("--benchmark", type=int, default=100, help="Number of benchmark loops.")
    args = parser.parse_args()

    features = load_features(args.features_npy)
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    embedding = session.run(None, {"features": features})[0]
    print("model:", args.model)
    print("input:", features.shape, features.dtype)
    print("embedding:", embedding.shape, embedding.dtype)
    print("embedding_norm:", float(np.linalg.norm(embedding[0])))

    if args.prototype_json:
        scores = score_prototypes(embedding[0], Path(args.prototype_json))
        print("scores:")
        for label, score in scores:
            print(f"  {label}: {score:.6f}")

    if args.benchmark > 0:
        for _ in range(10):
            session.run(None, {"features": features})
        latencies = []
        for _ in range(args.benchmark):
            start = time.perf_counter()
            session.run(None, {"features": features})
            latencies.append((time.perf_counter() - start) * 1000.0)
        print_latency("latency_ms", latencies)


def load_features(path: str | None) -> np.ndarray:
    if path:
        features = np.load(path).astype(np.float32)
    else:
        features = np.ones((1, 1, 40, 101), dtype=np.float32)

    if features.shape == (40, 101):
        features = features[None, None, :, :]
    if features.shape != (1, 1, 40, 101):
        raise ValueError(f"Expected [1,1,40,101] or [40,101], got {features.shape}")
    return features


def score_prototypes(embedding: np.ndarray, prototype_path: Path) -> list[tuple[str, float]]:
    data = json.loads(prototype_path.read_text(encoding="utf-8"))
    embedding = normalize(embedding)
    rows = []
    for label, vector in data.items():
        prototype = normalize(np.asarray(vector, dtype=np.float32))
        rows.append((label, float(np.dot(embedding, prototype))))
    return sorted(rows, key=lambda item: item[1], reverse=True)


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def print_latency(name: str, latencies: list[float]) -> None:
    values = np.asarray(latencies, dtype=np.float64)
    print(f"{name}_avg: {values.mean():.4f}")
    print(f"{name}_min: {values.min():.4f}")
    print(f"{name}_max: {values.max():.4f}")
    print(f"{name}_p50: {np.percentile(values, 50):.4f}")
    print(f"{name}_p95: {np.percentile(values, 95):.4f}")


if __name__ == "__main__":
    main()
