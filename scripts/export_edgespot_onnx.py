from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from edgespot.train import Classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="Use the newer torch.export-based ONNX exporter. Requires onnxscript.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--freq-bins", type=int, default=40)
    parser.add_argument("--frames", type=int, default=101)
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export with a dynamic batch dimension. Frequency/time remain fixed.",
    )
    parser.add_argument(
        "--metadata-out",
        help="Optional JSON sidecar with checkpoint labels and export shape.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = Classifier(
        tau=int(checkpoint["tau"]),
        num_classes=len(checkpoint["labels"]),
        teacher_dim=checkpoint.get("teacher_dim"),
    )
    model.load_state_dict(checkpoint["model"])
    encoder = model.encoder.eval()

    dummy = torch.ones(args.batch_size, 1, args.freq_bins, args.frames, dtype=torch.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"features": {0: "batch"}, "embedding": {0: "batch"}}

    torch.onnx.export(
        encoder,
        dummy,
        out,
        input_names=["features"],
        output_names=["embedding"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        dynamo=args.dynamo,
        do_constant_folding=True,
    )

    metadata = {
        "checkpoint": str(checkpoint_path),
        "onnx": str(out),
        "tau": int(checkpoint["tau"]),
        "labels": checkpoint["labels"],
        "input_shape": [args.batch_size, 1, args.freq_bins, args.frames],
        "output_shape": [args.batch_size, 64],
        "opset": args.opset,
        "dynamo_exporter": args.dynamo,
        "dynamic_batch": args.dynamic_batch,
        "export": "encoder_only",
        "score_path": "Compute cosine similarity between the 64-D embedding and wake-word prototypes.",
    }
    metadata_out = Path(args.metadata_out) if args.metadata_out else out.with_suffix(".json")
    metadata_out.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
