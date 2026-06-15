from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from edgespot.augment import apply_specaugment
from edgespot.data import ManifestDataset
from edgespot.losses import SubCenterArcFaceLoss, cosine_distillation_loss, mse_distillation_loss
from edgespot.model import EdgeSpot


class Classifier(nn.Module):
    def __init__(self, tau: int, num_classes: int, teacher_dim: int | None = None) -> None:
        super().__init__()
        self.encoder = EdgeSpot(tau=tau)
        self.classifier = nn.Linear(64, num_classes)
        self.teacher_projection = (
            nn.Linear(64, teacher_dim) if teacher_dim is not None and teacher_dim != 64 else None
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(x)
        return self.classifier(emb), emb

    def project_teacher(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.teacher_projection is None:
            return embeddings
        return self.teacher_projection(embeddings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tau", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=4e-5)
    parser.add_argument(
        "--loss",
        choices=["cross_entropy", "subcenter_arcface"],
        default="cross_entropy",
    )
    parser.add_argument("--arcface-centers", type=int, default=3)
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--arcface-margin", type=float, default=0.2)
    parser.add_argument("--teacher-embeddings")
    parser.add_argument(
        "--objective",
        choices=["proxy", "paper_distill"],
        default="proxy",
        help=(
            "proxy: classifier/SCAF with optional auxiliary KD. "
            "paper_distill: MSE KD + lambda*SCAF."
        ),
    )
    parser.add_argument("--distill-loss", choices=["mse", "cosine"], default="mse")
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--scaf-weight", type=float, default=5e-5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--specaugment", choices=["auto", "off", "on"], default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.objective == "paper_distill":
        if not args.teacher_embeddings:
            raise ValueError("--objective paper_distill requires --teacher-embeddings")
        args.loss = "subcenter_arcface"
    if args.distill_weight > 0 and not args.teacher_embeddings:
        raise ValueError("--distill-weight > 0 requires --teacher-embeddings")

    dataset = ManifestDataset(args.manifest, teacher_embeddings=args.teacher_embeddings)
    valid_len = max(1, int(len(dataset) * 0.1))
    train_len = len(dataset) - valid_len
    train_set, valid_set = random_split(
        dataset,
        [train_len, valid_len],
        generator=torch.Generator().manual_seed(2026),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    teacher_dim = dataset.teacher_dim if args.teacher_embeddings else None
    model = Classifier(args.tau, len(dataset.labels), teacher_dim=teacher_dim).to(args.device)
    arcface = None
    if args.loss == "subcenter_arcface":
        arcface = SubCenterArcFaceLoss(
            embedding_dim=64,
            num_classes=len(dataset.labels),
            num_centers=args.arcface_centers,
            scale=args.arcface_scale,
            margin=args.arcface_margin,
        ).to(args.device)
        parameters = list(model.parameters()) + list(arcface.parameters())
    else:
        parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = min(total_steps, max(0, len(train_loader) * args.warmup_epochs))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _warmup_cosine_factor(step, warmup_steps, total_steps),
    )
    use_specaugment = args.specaugment == "on" or (
        args.specaugment == "auto" and args.tau in {2, 3, 4}
    )

    best_score = -float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            x = batch["features"].to(args.device)
            if use_specaugment:
                x = apply_specaugment(x)
            y = batch["label"].to(args.device)
            logits, emb = model(x)
            if args.objective == "paper_distill":
                teacher = batch["teacher_embedding"].to(args.device)
                student = model.project_teacher(emb)
                kd_loss = _distillation_loss(student, teacher, args.distill_loss)
                scaf_loss = arcface(emb, y)
                loss = kd_loss + args.scaf_weight * scaf_loss
            elif arcface is None:
                loss = F.cross_entropy(logits, y)
            else:
                loss = arcface(emb, y)
            if args.distill_weight > 0:
                teacher = batch["teacher_embedding"].to(args.device)
                student = model.project_teacher(emb)
                loss = loss + args.distill_weight * _distillation_loss(
                    student,
                    teacher,
                    args.distill_loss,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * x.shape[0]

        valid_score, valid_name = _validation_score(
            model,
            valid_loader,
            args.device,
            args.objective,
            arcface,
            args.scaf_weight,
            args.distill_loss,
        )
        print(
            f"epoch={epoch + 1} loss={total_loss / len(train_set):.4f} "
            f"{valid_name}={valid_score:.4f}"
        )
        if valid_score >= best_score:
            best_score = valid_score
            torch.save(
                {
                    "model": model.state_dict(),
                    "arcface": None if arcface is None else arcface.state_dict(),
                    "labels": dataset.labels,
                    "tau": args.tau,
                    "loss": args.loss,
                    "objective": args.objective,
                    "teacher_dim": teacher_dim,
                },
                out_dir / "best.pt",
            )


@torch.no_grad()
def _accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        x = batch["features"].to(device)
        y = batch["label"].to(device)
        logits, _ = model(x)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


@torch.no_grad()
def _validation_score(
    model: Classifier,
    loader: DataLoader,
    device: str,
    objective: str,
    arcface: SubCenterArcFaceLoss | None,
    scaf_weight: float,
    distill_loss: str,
) -> tuple[float, str]:
    if objective != "paper_distill":
        return _accuracy(model, loader, device), "valid_acc"

    if arcface is None:
        raise ValueError("paper_distill validation requires Sub-center ArcFace")
    model.eval()
    total_loss = 0.0
    total = 0
    for batch in loader:
        x = batch["features"].to(device)
        y = batch["label"].to(device)
        teacher = batch["teacher_embedding"].to(device)
        _, emb = model(x)
        student = model.project_teacher(emb)
        kd_loss = _distillation_loss(student, teacher, distill_loss)
        scaf_loss = arcface(emb, y)
        loss = kd_loss + scaf_weight * scaf_loss
        total_loss += loss.item() * y.numel()
        total += y.numel()
    return -(total_loss / max(1, total)), "neg_valid_loss"


def _distillation_loss(student: torch.Tensor, teacher: torch.Tensor, name: str) -> torch.Tensor:
    if name == "mse":
        return mse_distillation_loss(student, teacher)
    if name == "cosine":
        return cosine_distillation_loss(student, teacher)
    raise ValueError(f"Unknown distillation loss: {name}")


def _warmup_cosine_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(1e-8, float(step + 1) / float(warmup_steps))
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


if __name__ == "__main__":
    main()
