from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

from edgespot.data import _crop_audio, _read_audio
from edgespot.teacher_model import Wav2VecTeacher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--model-id", default="facebook/wav2vec2-base")
    parser.add_argument("--cache-dir", default="models/huggingface")
    parser.add_argument("--encoder-layer", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line]
    if args.teacher_checkpoint:
        embeddings = _extract_trained_teacher_embeddings(rows, args)
    else:
        embeddings = _extract_mean_pooled_ssl_embeddings(rows, args)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **embeddings)
    print(f"wrote {len(embeddings)} teacher embeddings to {out}")


def _extract_trained_teacher_embeddings(
    rows: list[dict],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu")
    model_id = checkpoint.get("model_id", args.model_id)
    encoder_layer = checkpoint.get("encoder_layer", args.encoder_layer)
    extractor = AutoFeatureExtractor.from_pretrained(model_id, cache_dir=args.cache_dir)
    model = Wav2VecTeacher(
        model_id=model_id,
        encoder_layer=encoder_layer,
        cache_dir=args.cache_dir,
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    embeddings = {}
    for batch in _iter_batches(rows, args.batch_size):
        inputs = _batch_inputs(batch, extractor, args.device)
        with torch.no_grad():
            emb = model(**inputs).cpu().numpy().astype("float32")
        for row, vec in zip(batch, emb, strict=True):
            embeddings[row["audio_path"]] = vec
    return embeddings


def _extract_mean_pooled_ssl_embeddings(
    rows: list[dict],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    extractor = AutoFeatureExtractor.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    model = (
        AutoModel.from_pretrained(args.model_id, cache_dir=args.cache_dir)
        .to(args.device)
        .eval()
    )

    embeddings = {}
    for batch in _iter_batches(rows, args.batch_size):
        inputs = _batch_inputs(batch, extractor, args.device)
        with torch.no_grad():
            output = model(
                **inputs,
                output_hidden_states=args.encoder_layer is not None,
            )
            hidden = output.last_hidden_state
            if args.encoder_layer is not None and output.hidden_states is not None:
                layer = min(args.encoder_layer, len(output.hidden_states) - 1)
                hidden = output.hidden_states[layer]
            if "attention_mask" in inputs:
                mask = _get_feature_mask(hidden, inputs["attention_mask"])
                emb = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            else:
                emb = hidden.mean(dim=1)
            emb = F.normalize(emb, p=2, dim=-1).cpu().numpy().astype("float32")
        for row, vec in zip(batch, emb, strict=True):
            embeddings[row["audio_path"]] = vec
    return embeddings


def _iter_batches(rows: list[dict], batch_size: int) -> tqdm:
    return tqdm(
        (rows[start : start + batch_size] for start in range(0, len(rows), batch_size)),
        total=(len(rows) + batch_size - 1) // batch_size,
        desc="teacher",
    )


def _batch_inputs(batch: list[dict], extractor: AutoFeatureExtractor, device: str) -> dict:
    wavs = [_load_manifest_wav(row, extractor.sampling_rate) for row in batch]
    inputs = extractor(
        wavs,
        sampling_rate=extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    return {key: value.to(device) for key, value in inputs.items()}


def _load_manifest_wav(row: dict, target_sr: int) -> np.ndarray:
    wav, sr = _read_audio(row)
    wav = _crop_audio(wav, sr, row)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    tensor = torch.from_numpy(wav)
    if sr != target_sr:
        tensor = torchaudio.functional.resample(tensor, sr, target_sr)
    return tensor.numpy()


def _get_feature_mask(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    output_lengths = torch.div(attention_mask.sum(-1) - 1, 320, rounding_mode="floor") + 1
    max_len = hidden.shape[1]
    arange = torch.arange(max_len, device=hidden.device).unsqueeze(0)
    return arange < output_lengths.unsqueeze(1)


if __name__ == "__main__":
    main()
