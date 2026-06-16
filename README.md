# EdgeSpot Korean

This repository is a Korean command keyword-spotting scaffold based on the
BC-ResNet family from Qualcomm AI Research and the EdgeSpot paper architecture:

- trainable PCEN frontend
- early fused BC-ResNet blocks
- lightweight temporal self-attention
- 64-dimensional embedding head for prototype-based few-shot KWS

The Qualcomm reference implementation is tracked as a git submodule under
`third_party/bcresnet`.

## Setup

```bash
git submodule update --init --recursive
UV_CACHE_DIR=.uv-cache uv sync
```

Downloaded model weights are kept under the project directory by default:

```text
models/huggingface/
```

The directory is ignored by git except for `models/.gitkeep`. Override it with
`--cache-dir` or `CACHE_DIR` when needed.

For development checks:

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra dev
UV_CACHE_DIR=.uv-cache uv run python -m compileall src scripts
UV_CACHE_DIR=.uv-cache uv run ruff check .
```

## Data Plan

The first Korean experiment is intentionally small:

1. Generate positive Korean command utterances with Qwen3-TTS.
2. Generate or collect Korean non-command utterances for open-set negatives.
3. Train EdgeSpot embeddings with a supervised proxy objective or the
   paper-aligned distillation objective.
4. Evaluate with K-shot prototype matching and FAR-controlled thresholds.

MSWC can also be used. It is directly relevant because the EdgeSpot paper trains
on the English portion of MLCommons Multilingual Spoken Words and evaluates
cross-domain on Google Speech Commands. MLCommons describes MSWC as a 50-language
spoken-word dataset for keyword spotting and spoken term search, with 23.4M
1-second spoken examples and more than 6,000 hours. The full dataset is 124 GB;
the English audio alone is about 32.45 GB.

Source: https://mlcommons.org/datasets/multilingual-spoken-words/

For this repo:

- Use MSWC English for paper-aligned pretraining and later distillation work.
- Use Korean Qwen3-TTS command audio for positive Korean commands.
- Use Korean non-command utterances as negatives and hard negatives.
- Add any Korean MSWC-style word corpus later if a suitable Korean split is
  available, but do not block the first Korean experiment on it.

Negative text does not have to come from the local emotion/style speech
synthesis metadata. That file is only a convenient source of Korean
non-command transcripts. It can be skipped if we generate negative prompts
ourselves or use another Korean text corpus.

Optional local non-command transcript source:

```text
/data/datasets/voice/015.감성_및_발화_스타일별_음성합성_데이터/train_tsv/valid_metadata_processed.txt
```

Expected metadata format:

```text
relative/path.wav|transcript|speaker_id
```

## Korean Command Manifest

Create a TTS synthesis plan:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/build_tts_manifest.py \
  --config configs/korean_commands.yaml \
  --out data/manifests/tts_commands.jsonl
```

Optionally add TTS negatives sampled from the local transcript metadata. This is
not required; use it only when we want quick non-command text without writing a
separate negative prompt list:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/build_tts_manifest.py \
  --config configs/korean_commands.yaml \
  --negative-metadata /data/datasets/voice/015.감성_및_발화_스타일별_음성합성_데이터/train_tsv/valid_metadata_processed.txt \
  --negative-limit 10000 \
  --out data/manifests/tts_commands_and_negatives.jsonl
```

If that metadata is skipped, provide another negative manifest before
paper-style open-set evaluation. The simplest alternatives are:

- Generate Korean general-speech negative prompts and synthesize them with the
  same Qwen3-TTS pipeline.
- Use real Korean recordings as validation-only or hard-negative data.
- Use MSWC/GSC-style non-target spoken words for paper-aligned experiments.

Synthesize audio with Qwen3-TTS:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/synthesize_qwen3_tts.py \
  --manifest data/manifests/tts_commands.jsonl \
  --out-dir data/audio/commands \
  --model-id Qwen/Qwen3-TTS \
  --cache-dir models/huggingface \
  --device cuda
```

Shard synthesis across GPUs 4,5,6,7:

```bash
GPU_IDS=4,5,6,7 sbatch slurm/synthesize_qwen3_tts.sbatch
```

On clusters that remap allocated GPUs to `CUDA_VISIBLE_DEVICES=0,1,2,3`,
submit with:

```bash
GPU_IDS=0,1,2,3 sbatch slurm/synthesize_qwen3_tts.sbatch
```

Alternatively, let Slurm assign one GPU per shard:

```bash
sbatch slurm/synthesize_qwen3_tts_array.sbatch
```

The script is written as an adapter because Qwen3-TTS package APIs may differ
between releases. If the installed runtime exposes a different entry point, only
`src/edgespot/tts/qwen3.py` should need changes.

Optionally build a real-recording negative speech manifest for validation or
domain testing. Do not mix this into the first TTS-only training run unless we
intentionally want source-mismatched negatives:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/build_negative_manifest.py \
  --metadata /data/datasets/voice/015.감성_및_발화_스타일별_음성합성_데이터/train_tsv/valid_metadata_processed.txt \
  --audio-root /data/datasets/voice/015.감성_및_발화_스타일별_음성합성_데이터 \
  --out data/manifests/negative_speech.jsonl \
  --limit 20000
```

Merge positive and negative manifests:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/merge_manifests.py \
  --positive data/manifests/tts_commands.done.jsonl \
  --negative data/manifests/negative_speech.jsonl \
  --out data/manifests/train.jsonl
```

For a source-balanced TTS-only model, use the `.done.jsonl` produced by the TTS
synthesis job directly as the training manifest instead of mixing TTS positives
with real-recording negatives.

Collect shard outputs, validate audio, and split manifests. The `13072`
expected count is for the current optional-metadata run:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/collect_shards.py \
  --pattern 'data/manifests/tts_commands_and_negatives.shard*.done.jsonl' \
  --out data/manifests/tts_commands_and_negatives.done.jsonl \
  --expected 13072

UV_CACHE_DIR=.uv-cache uv run python scripts/validate_manifest_audio.py \
  --manifest data/manifests/tts_commands_and_negatives.done.jsonl \
  --out data/manifests/tts_commands_and_negatives.validation.json \
  --strict

UV_CACHE_DIR=.uv-cache uv run python scripts/split_manifest.py \
  --manifest data/manifests/tts_commands_and_negatives.done.jsonl \
  --out-dir data/manifests/splits
```

For a commands-only run without generated negatives, use
`tts_commands*.done.jsonl` and the expected count from the command config
instead.

## Train

```bash
UV_CACHE_DIR=.uv-cache uv run python -m edgespot.train \
  --manifest data/manifests/splits/train.jsonl \
  --out-dir runs/edgespot-ko-small \
  --tau 1 \
  --epochs 40
```

Sub-center ArcFace training:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m edgespot.train \
  --manifest data/manifests/splits/train.jsonl \
  --valid-manifest data/manifests/splits/val.jsonl \
  --out-dir runs/edgespot-ko-arcface \
  --tau 1 \
  --loss subcenter_arcface \
  --arcface-centers 3 \
  --batch-size 64 \
  --num-workers 2 \
  --epochs 40
```

Train EdgeSpot-1/2/3/4 in parallel on GPUs 4,5,6,7 with conservative CPU
memory usage:

```bash
GPU_IDS=4,5,6,7 BATCH_SIZE=512 sbatch slurm/train_edgespot_scaf.sbatch
```

The Slurm script keeps audio loading streaming from disk, uses `batch_size=512`,
and starts only two DataLoader workers per model by default.

TensorBoard logs are written under each run directory:

```bash
UV_CACHE_DIR=.uv-cache uv run tensorboard --logdir runs --port 6006
```

Summarize prototype-evaluation results across run directories:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/summarize_results.py
```

The generated report is written to:

```text
docs/results_summary.md
```

Count EdgeSpot-1/2/3/4 model parameters:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/model_stats.py
```

Inspect high-scoring false accepts for the current best distilled model:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/inspect_false_accepts.py \
  --checkpoint runs/edgespot-ko-distill-tau4/best.pt \
  --support-manifest data/manifests/splits/val.jsonl \
  --query-manifest data/manifests/splits/test.jsonl \
  --out docs/false_accepts_distill_tau4_k10_far1.jsonl \
  --k-shot 10 \
  --far 0.01 \
  --top-n 100 \
  --batch-size 512 \
  --num-workers 0
```

Use `--num-workers 0` in local sandboxed sessions. Slurm jobs can use worker
processes when the cluster environment permits multiprocessing sockets.

Run few-shot prototype evaluation after training:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m edgespot.eval \
  --checkpoint runs/edgespot-ko-scaf-tau1/best.pt \
  --support-manifest data/manifests/splits/val.jsonl \
  --query-manifest data/manifests/splits/test.jsonl \
  --out runs/edgespot-ko-scaf-tau1/prototype_eval_k5.json \
  --k-shot 5 \
  --batch-size 512 \
  --num-workers 2
```

Or submit all tau 1/2/3/4 evaluations for 1-shot, 5-shot, and 10-shot:

```bash
sbatch slurm/eval_edgespot_prototypes.sbatch
```

Paper-aligned teacher training and distillation. The EdgeSpot paper uses a
pretrained Wav2Vec2.0 encoder up to the 16th transformer layer, an
attention-based reduction head that maps frame-level SSL features to 64-D
embeddings, and Sub-center ArcFace training for the teacher. The student is
then trained with `MSE(student, teacher) + 5e-5 * SCAF(student)`.

Train the teacher head:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m edgespot.train_teacher \
  --manifest data/manifests/splits/train.jsonl \
  --out-dir runs/teacher-wav2vec2-scaf \
  --model-id facebook/wav2vec2-base \
  --cache-dir models/huggingface \
  --encoder-layer 16 \
  --epochs 10
```

For Korean experiments, `facebook/wav2vec2-xls-r-300m` is a configurable
multilingual Wav2Vec2-family option. The EdgeSpot paper itself says Wav2Vec2.0
and does not state XLS-R as the reported teacher.

Project model cache path:

```text
/data/users/voice/zoey/todak/impl/EdgeSpot/models
```

Shared dataset root and cache paths:

```text
/data/datasets/voice
/data/datasets/voice/.cache
```

Export trained 64-D teacher embeddings:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m edgespot.teacher \
  --manifest data/manifests/splits/train.jsonl \
  --out data/teacher/train_teacher64.npz \
  --teacher-checkpoint runs/teacher-wav2vec2-scaf/best_teacher.pt \
  --cache-dir models/huggingface \
  --batch-size 16
```

Train the student with the paper objective:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m edgespot.train \
  --manifest data/manifests/splits/train.jsonl \
  --valid-manifest data/manifests/splits/val.jsonl \
  --out-dir runs/edgespot-ko-paper-kd \
  --tau 1 \
  --objective paper_distill \
  --teacher-embeddings data/teacher/train_teacher64.npz \
  --epochs 40
```

Few-shot prototype evaluation:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m edgespot.eval \
  --checkpoint runs/edgespot-ko-arcface/best.pt \
  --support-manifest data/manifests/splits/val.jsonl \
  --query-manifest data/manifests/splits/test.jsonl \
  --out runs/edgespot-ko-arcface/prototype_eval.json \
  --k-shot 5
```

## Notes

- The paper states that knowledge distillation uses a self-supervised teacher
  model. The full paper uses Wav2Vec2.0, a 64-D attention reduction head trained
  with Sub-center ArcFace, and MSE distillation into the EdgeSpot student.
- The feature-domain time-stretch implementation approximates the paper's
  waveform time-stretch setting. Replace it with waveform-level augmentation if
  exact reproduction becomes the priority.
- The first target is a working Korean KWS pipeline. Distillation can be added
  after manifest generation, training, and prototype evaluation are stable.
