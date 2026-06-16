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
  - [x] Default GPU mapping: 4,5,6,7.
  - [x] Default tau mapping: 1,2,3,4.
  - [x] Default batch size: 512.
  - [x] Default DataLoader workers per model: 2.
- [x] Added TensorBoard logging:
  - [x] Per-epoch train loss.
  - [x] Per-epoch validation metric.
  - [x] Per-epoch learning rate.
- [x] Added prototype evaluation Slurm script:
  - [x] `slurm/eval_edgespot_prototypes.sbatch`.
  - [x] Evaluates tau 1,2,3,4 checkpoints.
  - [x] Evaluates 1-shot, 5-shot, and 10-shot prototype matching.
- [x] Submitted dependent prototype evaluation job:
  - [x] Training job: `3262`.
  - [x] Evaluation job: `3263`, waiting on `afterok:3262`.
- [x] Documented MSWC usage and NanoWakeWord as a later candidate.
- [x] Completed SCAF baseline training and prototype evaluation.
- [x] Added teacher training, teacher embedding export, distillation training, and distillation evaluation Slurm scripts.
- [x] Trained XLS-R 300M teacher head with SCAF:
  - [x] `runs/teacher-xls-r-300m-scaf/best_teacher.pt`.
- [x] Exported teacher embeddings:
  - [x] `data/teacher/xls-r-300m_teacher64_all.npz`.
- [x] Trained distilled EdgeSpot-1/2/3/4 student runs:
  - [x] `runs/edgespot-ko-distill-tau1/best.pt`.
  - [x] `runs/edgespot-ko-distill-tau2/best.pt`.
  - [x] `runs/edgespot-ko-distill-tau3/best.pt`.
  - [x] `runs/edgespot-ko-distill-tau4/best.pt`.
- [x] Evaluated distilled student checkpoints with 1-shot, 5-shot, and 10-shot prototype matching.
- [x] Downloaded external data for later experiments:
  - [x] Google Speech Commands v2 tarball under `/data/datasets/voice/google_speech_commands`.
  - [x] MSWC metadata under `/data/datasets/voice/mswc`.
  - [x] MSWC English audio, splits, and alignments under `/data/datasets/voice/mswc/en`.
- [x] Added result analysis utilities:
  - [x] `scripts/summarize_results.py`.
  - [x] `scripts/model_stats.py`.
  - [x] `scripts/inspect_false_accepts.py`.
- [x] Generated current result reports:
  - [x] `docs/results_summary.md`.
  - [x] `docs/model_stats.json`.
  - [x] `docs/false_accepts_distill_tau4_k10_far1.jsonl`.
- [x] Added short Korean hard-negative seed prompts from false accepts:
  - [x] `configs/korean_hard_negatives.txt`.
  - [x] `scripts/build_tts_manifest.py --negative-texts`.
  - [x] `scripts/build_tts_manifest.py --skip-commands`.
  - [x] `docs/false_accept_review.md`.
- [x] Synthesized hard-negative audio from `configs/korean_hard_negatives.txt`:
  - [x] 1,920 hard-negative wav files generated.
  - [x] `data/manifests/tts_hard_negatives.done.jsonl`.
  - [x] 24 kHz audio validation passed with no missing or bad audio.
- [x] Built train-only augmented manifest:
  - [x] `data/manifests/splits/train_with_hard_negatives.jsonl`.
  - [x] 12,132 train rows: 10,212 original train + 1,920 hard negatives.
- [x] Submitted teacher embedding export for augmented train:
  - [x] Job `3276`.
  - [x] Output target `data/teacher/xls-r-300m_teacher64_train_hard.npz`.
- [x] Built all-split teacher embedding manifest for distill validation lookup:
  - [x] `data/manifests/splits/all_with_hard_negatives.jsonl`.
  - [x] 14,992 rows: augmented train + original val/test.
- [x] Exported all-hard teacher embeddings:
  - [x] Job `3279`.
  - [x] `data/teacher/xls-r-300m_teacher64_all_hard.npz`.
- [x] Submitted hard-negative tau 4 distill training:
  - [x] Job `3280`.
  - [x] Run target `runs/edgespot-ko-distill-hard-tau4`.
  - [x] Epoch 1 validation passed.
- [x] Submitted dependent hard-negative tau 4 prototype evaluation:
  - [x] Job `3281`, dependency `afterok:3280`.

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

- [ ] Wait for hard-negative tau 4 distill job `3280`.
- [ ] Review dependent prototype eval job `3281`.
- [ ] Review top false accepts after hard-negative training and expand command-like confusers.
- [ ] Evaluate the best distilled checkpoint against real-recording Korean negative/domain-test data.
- [ ] Build manifests/extraction scripts for downloaded GSC v2 and MSWC English.
- [ ] Add MAC counting to compare EdgeSpot-1/2/3/4 against the paper table.
- [ ] Add a BC-ResNet baseline trained with the same KD+SCAF protocol.
- [ ] Replace feature-domain time-stretch with waveform-level time-stretch for stricter paper reproduction.
- [ ] Confirm exact teacher model from prior work [6] if exact reproduction is required.
- [ ] Decide Korean experiment protocol:
  - [ ] TTS-only train and TTS validation.
  - [ ] Real-recording negative validation.
  - [ ] Korean few-shot command enrollment trials.
- [ ] Add Korean hard negatives and command-like confuser words.
- [ ] Add final model export path for on-device inference.
- [ ] Add quantization/export benchmark for actual on-device target constraints.

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
  it streams audio from disk, uses `batch_size=512`, and uses two DataLoader
  workers per model by default.
