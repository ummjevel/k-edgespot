from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
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
    parser.add_argument("--k-shot", type=int, default=10)
    parser.add_argument("--far", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = Classifier(
        tau=checkpoint["tau"],
        num_classes=len(checkpoint["labels"]),
        teacher_dim=checkpoint.get("teacher_dim"),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(args.device).eval()

    support = ManifestDataset(args.support_manifest)
    query = ManifestDataset(args.query_manifest)
    support_rows = _embed(model, support, args.batch_size, args.num_workers, args.device)
    query_rows = _embed(model, query, args.batch_size, args.num_workers, args.device)
    prototypes = _build_prototypes(support_rows, args.k_shot)
    scored = _score(query_rows, prototypes)

    neg_scores = np.asarray([row["score"] for row in scored if row["label"] == "__negative__"])
    threshold = float(np.quantile(neg_scores, 1.0 - args.far))
    false_accepts = [
        row for row in scored if row["label"] == "__negative__" and row["score"] >= threshold
    ]
    false_accepts.sort(key=lambda row: row["score"], reverse=True)
    false_accepts = false_accepts[: args.top_n]
    false_rejects = [
        row for row in scored if row["label"] != "__negative__" and row["score"] < threshold
    ]
    false_rejects.sort(key=lambda row: row["score"])
    false_rejects = false_rejects[: args.top_n]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        summary = {
            "checkpoint": args.checkpoint,
            "support_manifest": args.support_manifest,
            "query_manifest": args.query_manifest,
            "k_shot": args.k_shot,
            "far": args.far,
            "threshold": threshold,
            "num_false_accepts_at_threshold": int(
                sum(
                    1
                    for row in scored
                    if row["label"] == "__negative__" and row["score"] >= threshold
                )
            ),
            "num_false_rejects_at_threshold": int(
                sum(
                    1
                    for row in scored
                    if row["label"] != "__negative__" and row["score"] < threshold
                )
            ),
        }
        handle.write(json.dumps({"type": "summary", **summary}, ensure_ascii=False) + "\n")
        for row in false_accepts:
            row = {key: value for key, value in row.items() if key != "embedding"}
            handle.write(json.dumps({"type": "false_accept", **row}, ensure_ascii=False) + "\n")
        for row in false_rejects:
            row = {key: value for key, value in row.items() if key != "embedding"}
            handle.write(json.dumps({"type": "false_reject", **row}, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


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
        features = batch["features"].to(device)
        embeddings = model.encoder(features).cpu().numpy()
        labels = list(batch["label_name"])
        paths = list(batch["audio_path"])
        for idx in range(embeddings.shape[0]):
            item = dataset.items[offset + idx]
            rows.append(
                {
                    "embedding": embeddings[idx],
                    "label": labels[idx],
                    "audio_path": paths[idx],
                    "index": offset + idx,
                    "text": item.get("text"),
                    "command_domain": item.get("command_domain"),
                    "command_subdomain": item.get("command_subdomain"),
                    "variant": item.get("variant"),
                    "speaker_id": item.get("speaker_id"),
                    "noise_type": item.get("noise_type"),
                    "aihub_mapping_rule": item.get("aihub_mapping_rule"),
                }
            )
        offset += embeddings.shape[0]
    return rows


def _build_prototypes(rows: list[dict], k_shot: int) -> dict[str, np.ndarray]:
    by_label: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        if row["label"] == "__negative__":
            continue
        if len(by_label[row["label"]]) < k_shot:
            by_label[row["label"]].append(row["embedding"])
    return {
        label: _normalize(np.mean(np.stack(embeddings), axis=0))
        for label, embeddings in by_label.items()
        if embeddings
    }


def _score(rows: list[dict], prototypes: dict[str, np.ndarray]) -> list[dict]:
    scored = []
    proto_items = list(prototypes.items())
    for row in rows:
        emb = _normalize(row["embedding"])
        matches = [(label, float(np.dot(emb, proto))) for label, proto in proto_items]
        match_label, score = max(matches, key=lambda item: item[1])
        scored.append(
            {
                **row,
                "score": score,
                "matched_label": match_label,
            }
        )
    return scored


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / max(1e-8, float(np.linalg.norm(x)))


if __name__ == "__main__":
    main()
