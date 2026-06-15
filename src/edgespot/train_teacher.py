from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoFeatureExtractor

from edgespot.audio import load_wav
from edgespot.losses import SubCenterArcFaceLoss
from edgespot.teacher_model import Wav2VecTeacher


class TeacherDataset(Dataset):
    def __init__(self, manifest: str | Path, model_id: str, cache_dir: str | Path) -> None:
        self.rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line]
        self.labels = sorted({row["label"] for row in self.rows})
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id, cache_dir=cache_dir)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        wav = load_wav(row["audio_path"], self.extractor.sampling_rate)
        return {
            "wav": wav,
            "label": self.label_to_idx[row["label"]],
            "audio_path": row["audio_path"],
        }

    def collate(self, batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
        inputs = self.extractor(
            [item["wav"] for item in batch],
            sampling_rate=self.extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        return {
            "input_values": inputs["input_values"],
            "attention_mask": inputs.get("attention_mask"),
            "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
            "audio_path": [item["audio_path"] for item in batch],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-id", default="facebook/wav2vec2-base")
    parser.add_argument("--cache-dir", default="models/huggingface")
    parser.add_argument("--encoder-layer", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--arcface-centers", type=int, default=3)
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--arcface-margin", type=float, default=0.2)
    parser.add_argument("--unfreeze-encoder", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = TeacherDataset(args.manifest, args.model_id, args.cache_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=dataset.collate,
    )
    model = Wav2VecTeacher(
        model_id=args.model_id,
        freeze_encoder=not args.unfreeze_encoder,
        encoder_layer=args.encoder_layer,
        cache_dir=args.cache_dir,
    ).to(args.device)
    criterion = SubCenterArcFaceLoss(
        embedding_dim=64,
        num_classes=len(dataset.labels),
        num_centers=args.arcface_centers,
        scale=args.arcface_scale,
        margin=args.arcface_margin,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        list(model.head.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    if args.unfreeze_encoder:
        optimizer.add_param_group({"params": model.encoder.parameters(), "lr": args.lr * 0.1})

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        count = 0
        for batch in tqdm(loader, desc=f"teacher {epoch + 1}/{args.epochs}"):
            input_values = batch["input_values"].to(args.device)
            attention_mask = batch["attention_mask"]
            if attention_mask is not None:
                attention_mask = attention_mask.to(args.device)
            labels = batch["label"].to(args.device)
            emb = model(input_values=input_values, attention_mask=attention_mask)
            loss = criterion(emb, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * labels.numel()
            count += labels.numel()

        mean_loss = total / max(1, count)
        print(f"epoch={epoch + 1} loss={mean_loss:.4f}")
        if mean_loss <= best_loss:
            best_loss = mean_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "arcface": criterion.state_dict(),
                    "labels": dataset.labels,
                    "model_id": args.model_id,
                    "encoder_layer": args.encoder_layer,
                    "cache_dir": args.cache_dir,
                },
                out_dir / "best_teacher.pt",
            )


if __name__ == "__main__":
    main()
