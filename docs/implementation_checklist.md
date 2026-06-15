# EdgeSpot Korean Implementation Checklist

## Done Today

- [x] Created the `EdgeSpot` project scaffold with `uv` support.
- [x] Added Qualcomm BC-ResNet as `third_party/bcresnet` git submodule.
- [x] Added Korean command and voice-design config for Qwen3-TTS.
- [x] Built the TTS synthesis manifest with Korean commands and TTS non-command text.
- [x] Added Qwen3-TTS synthesis scripts and Slurm jobs for GPUs 4,5,6,7.
- [x] Submitted the Qwen3-TTS Slurm synthesis job.
- [x] Completed Qwen3-TTS synthesis:
  - [x] 13,072 total wav files generated.
  - [x] 3,072 command utterances generated.
  - [x] 10,000 negative utterances generated.
  - [x] Four shards completed with 3,268 rows each.
  - [x] Combined shard manifests into `tts_commands_and_negatives.done.jsonl`.
- [x] Validated generated audio:
  - [x] 13,072/13,072 files present and readable.
  - [x] Source wav files are 24 kHz.
  - [x] 24 kHz validation passed with no missing or bad audio.
  - [x] Training loader will resample to 16 kHz and crop/pad to 1 second.
- [x] Split generated data into train/val/test manifests:
  - [x] Train: 10,212 rows.
  - [x] Val: 1,225 rows.
  - [x] Test: 1,635 rows.
- [x] Added manifest collection, validation, splitting, and negative-manifest utilities.
- [x] Implemented the EdgeSpot-style student model:
  - [x] 40 x 101 mel input path.
  - [x] Trainable PCEN frontend.
  - [x] BC-ResNet-style acoustic backbone.
  - [x] Fused early BC-ResBlocks.
  - [x] Depthwise temporal relative positional convolution.
  - [x] Single-head temporal scaled dot-product attention.
  - [x] 64-D normalized embedding output.
- [x] Implemented Sub-center ArcFace loss.
- [x] Implemented few-shot prototype evaluation with FAR-threshold reporting.
- [x] Read the local EdgeSpot paper PDF at `docs/2601.16316v1.pdf`.
- [x] Updated the teacher pipeline toward the paper:
  - [x] Wav2Vec2-family encoder.
  - [x] Optional 16th-transformer-layer feature selection.
  - [x] Attention-based 64-D dimensionality reduction head.
  - [x] Teacher training with Sub-center ArcFace.
  - [x] Export of trained 64-D teacher embeddings.
- [x] Updated student training toward the paper:
  - [x] `MSE(student, teacher) + 5e-5 * SCAF(student)` objective.
  - [x] 5-epoch linear warmup.
  - [x] Step-wise cosine learning-rate decay.
  - [x] Tau-aware SpecAugment enablement.
  - [x] Explicit `--valid-manifest` support.
  - [x] Configurable `--num-workers` to limit CPU memory/process pressure.
- [x] Added conservative SCAF training Slurm script:
  - [x] `slurm/train_edgespot_scaf.sbatch`.
  - [x] Default GPU mapping: 5,6,7,8.
  - [x] Default tau mapping: 1,2,3,4.
  - [x] Default batch size: 64.
  - [x] Default DataLoader workers per model: 2.
- [x] Documented MSWC usage and NanoWakeWord as a later candidate.

## Paper Items Implemented

- [x] BC-ResNet acoustic backbone.
- [x] Trainable PCEN before the backbone.
- [x] Fused BC-ResBlocks in early stages.
- [x] Relative positional encoding with depthwise Conv1D, kernel size 16.
- [x] Temporal SDPA head.
- [x] 64-D embedding head.
- [x] Wav2Vec2 teacher with attention reduction head.
- [x] Teacher SCAF training path.
- [x] Student distillation from teacher embeddings.
- [x] Student SCAF term with paper lambda `5e-5`.
- [x] 40-band mel spectrogram with centered STFT and 101 frames.
- [x] AdamW/Adam-style optimizer path with weight decay `4e-5`.
- [x] 40-epoch default training.
- [x] Prototype-based K-shot evaluation.
- [x] External train/validation split support.

## Remaining Work

- [ ] Start SCAF baseline training on GPUs 5,6,7,8.
- [ ] Monitor `runs/edgespot-ko-scaf-tau{1,2,3,4}` and per-tau logs.
- [ ] Run a small smoke training job first if cluster load or memory pressure is uncertain.
- [ ] Run few-shot prototype evaluation for SCAF baseline checkpoints.
- [ ] Train the Wav2Vec2 teacher head on the Korean manifest.
- [ ] Export teacher embeddings with `edgespot.teacher --teacher-checkpoint`.
- [ ] Train the EdgeSpot student with `--objective paper_distill`.
- [ ] Run few-shot prototype evaluation for 1-shot, 5-shot, and 10-shot.
- [ ] Add Slurm scripts for teacher training, teacher embedding export, distillation training, and evaluation.
- [ ] Add MAC/parameter counting to compare EdgeSpot-1/2/3/4 against the paper table.
- [ ] Add a BC-ResNet baseline trained with the same KD+SCAF protocol.
- [ ] Replace feature-domain time-stretch with waveform-level time-stretch for stricter paper reproduction.
- [ ] Confirm exact teacher model from prior work [6] if exact reproduction is required.
- [ ] Decide Korean experiment protocol:
  - [ ] TTS-only train and TTS validation.
  - [ ] Real-recording negative validation.
  - [ ] Korean few-shot command enrollment trials.
- [ ] Add Korean hard negatives and command-like confuser words.
- [ ] Add final model export path for on-device inference.
- [ ] Push the local training-prep commit if remote synchronization is needed.

## Open Technical Notes

- The EdgeSpot paper says Wav2Vec2.0, not XLS-R. XLS-R remains a practical
  multilingual option for Korean, but it should be treated as a Korean extension
  experiment rather than a confirmed paper setting.
- The paper's teacher uses the Wav2Vec2 encoder up to the 16th transformer layer.
  The code now supports `--encoder-layer 16`; if the selected checkpoint has
  fewer layers, the highest available layer is used.
- The paper applies waveform time-stretch for tau 2/3/4. The current
  implementation approximates this with feature-domain interpolation.
- Generated Qwen3-TTS wav files are 24 kHz. The current dataset loader resamples
  to 16 kHz at load time and crops/pads each waveform to 1 second. This avoids
  creating another full normalized audio copy, keeping CPU memory usage lower at
  the cost of per-epoch resampling work.
- The current SCAF training script is intentionally conservative for CPU memory:
  it streams audio from disk, uses `batch_size=64`, and uses two DataLoader
  workers per model by default.
- Local branch `main` currently includes a training-prep commit that has not
  been pushed after the interrupted push attempt.
