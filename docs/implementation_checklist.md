# EdgeSpot Korean Implementation Checklist

## Done Today

- [x] Created the `EdgeSpot` project scaffold with `uv` support.
- [x] Added Qualcomm BC-ResNet as `third_party/bcresnet` git submodule.
- [x] Added Korean command and voice-design config for Qwen3-TTS.
- [x] Built the TTS synthesis manifest with Korean commands and TTS non-command text.
- [x] Added Qwen3-TTS synthesis scripts and Slurm jobs for GPUs 4,5,6,7.
- [x] Submitted the Qwen3-TTS Slurm synthesis job.
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

## Remaining Work

- [ ] Finish and validate the current Qwen3-TTS synthesis job.
- [ ] Collect shard manifests into `data/manifests/tts_commands_and_negatives.done.jsonl`.
- [ ] Run audio validation and fix any bad/missing generated clips.
- [ ] Split generated data into train/val/test manifests.
- [ ] Run a small smoke training job on a subset before full training.
- [ ] Train the Wav2Vec2 teacher head on the Korean manifest.
- [ ] Export teacher embeddings with `edgespot.teacher --teacher-checkpoint`.
- [ ] Train the EdgeSpot student with `--objective paper_distill`.
- [ ] Run few-shot prototype evaluation for 1-shot, 5-shot, and 10-shot.
- [ ] Add Slurm scripts for teacher training, teacher embedding export, student training, and evaluation.
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

## Open Technical Notes

- The EdgeSpot paper says Wav2Vec2.0, not XLS-R. XLS-R remains a practical
  multilingual option for Korean, but it should be treated as a Korean extension
  experiment rather than a confirmed paper setting.
- The paper's teacher uses the Wav2Vec2 encoder up to the 16th transformer layer.
  The code now supports `--encoder-layer 16`; if the selected checkpoint has
  fewer layers, the highest available layer is used.
- The paper applies waveform time-stretch for tau 2/3/4. The current
  implementation approximates this with feature-domain interpolation.
