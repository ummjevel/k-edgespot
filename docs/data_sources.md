# Data Sources

## Korean Commands

Positive labels are generated from `configs/korean_commands.yaml` and synthesized
with Qwen3-TTS using Korean voice-design prompts derived from the LiveKit
VoxCPM config style.

## Korean Non-Commands

Use:

```text
/data/datasets/voice/015.감성_및_발화_스타일별_음성합성_데이터/train_tsv/valid_metadata_processed.txt
```

Rows are expected as:

```text
relative/path.wav|transcript|speaker_id
```

For the first model, prefer using this file as a text source for Qwen3-TTS
negative synthesis. If positives are Qwen3-TTS but negatives are direct
recordings, the model can learn source artifacts instead of command semantics.

The original wav paths are still useful as a domain-generalization validation
set and for later hardening.

## MSWC

MLCommons Multilingual Spoken Words is useful for EdgeSpot because the paper
uses MSWC English for training. The dataset page describes it as a large spoken
word corpus in 50 languages for KWS and spoken term search, licensed CC-BY 4.0.

Recommended use here:

- Download metadata first.
- Download English audio/splits only for paper-aligned experiments.
- Avoid the full 124 GB download until training scripts are stable.

Dataset page:

```text
https://mlcommons.org/datasets/multilingual-spoken-words/
```

## Later Model Candidate: NanoWakeWord

NanoWakeWord is worth keeping as a later baseline/candidate. Its README
positions it as a lightweight custom wake word framework with multiple
architectures, synthetic/adversarial negative generation, augmentation,
streaming inference, and ONNX/PyTorch export.

Use it later as a practical wakeword baseline against EdgeSpot-style training,
not as a replacement for the current EdgeSpot reproduction path.

```text
https://github.com/arcosoph/nanowakeword
```
