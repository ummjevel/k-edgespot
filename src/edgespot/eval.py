from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from edgespot.data import ManifestDataset
from edgespot.train import Classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--support-manifest", required=True)
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = Classifier(
        tau=ckpt["tau"],
        num_classes=len(ckpt["labels"]),
        teacher_dim=ckpt.get("teacher_dim"),
    )
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    support = ManifestDataset(args.support_manifest)
    query = ManifestDataset(args.query_manifest)
    support_embeddings = _embed(model, support, args.batch_size, args.num_workers, args.device)
    query_embeddings = _embed(model, query, args.batch_size, args.num_workers, args.device)

    prototypes = _build_prototypes(support_embeddings, args.k_shot)
    scores, targets = _score_queries(query_embeddings, prototypes)
    metrics = _metrics(scores, targets)
    metrics["k_shot"] = args.k_shot
    metrics["num_prototypes"] = len(prototypes)
    metrics["num_queries"] = len(targets)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


@torch.no_grad()
def _embed(
    model: Classifier,
    dataset: ManifestDataset,
    batch_size: int,
    num_workers: int,
    device: str,
) -> list[dict]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    rows = []
    offset = 0
    for batch in tqdm(loader, desc="embed"):
        x = batch["features"].to(device)
        emb = model.encoder(x).cpu().numpy()
        labels = list(batch["label_name"])
        paths = list(batch["audio_path"])
        for idx in range(emb.shape[0]):
            rows.append(
                {
                    "embedding": emb[idx],
                    "label": labels[idx],
                    "audio_path": paths[idx],
                    "index": offset + idx,
                }
            )
        offset += emb.shape[0]
    return rows


def _build_prototypes(rows: list[dict], k_shot: int) -> dict[str, np.ndarray]:
    by_label: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        if row["label"] == "__negative__":
            continue
        if len(by_label[row["label"]]) < k_shot:
            by_label[row["label"]].append(row["embedding"])
    return {
        label: _normalize(np.mean(np.stack(embs), axis=0))
        for label, embs in by_label.items()
        if len(embs) > 0
    }


def _score_queries(
    rows: list[dict],
    prototypes: dict[str, np.ndarray],
) -> tuple[list[float], list[int]]:
    scores = []
    targets = []
    proto_items = list(prototypes.items())
    for row in rows:
        emb = _normalize(row["embedding"])
        best_score = max(float(np.dot(emb, proto)) for _, proto in proto_items)
        scores.append(best_score)
        targets.append(0 if row["label"] == "__negative__" else 1)
    return scores, targets


def _metrics(scores: list[float], targets: list[int]) -> dict:
    scores_np = np.asarray(scores)
    targets_np = np.asarray(targets)
    result = {
        "auc": float(roc_auc_score(targets_np, scores_np)) if len(set(targets)) == 2 else None,
        "thresholds": {},
    }
    neg_scores = scores_np[targets_np == 0]
    pos_scores = scores_np[targets_np == 1]
    for far in [0.001, 0.005, 0.01, 0.05]:
        threshold = float(np.quantile(neg_scores, 1.0 - far)) if len(neg_scores) else 1.0
        frr = float(np.mean(pos_scores < threshold)) if len(pos_scores) else None
        result["thresholds"][str(far)] = {
            "threshold": threshold,
            "far": far,
            "frr": frr,
            "recall": None if frr is None else 1.0 - frr,
        }
    return result


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / max(1e-8, float(np.linalg.norm(x)))


if __name__ == "__main__":
    main()
