# Command-like Keyword vs Ultra-short Wake-word Notes

This note captures the working interpretation behind the Todak device experiments.

## Core Difference

Command-like keyword spotting and ultra-short wake-word spotting look similar at the embedding/prototype
level, but the error modes are different.

Command-like keyword spotting usually has a wider semantic surface. A command such as turning
on a light can appear as:

```text
불 켜줘
불 좀 켜줘
불 켜줄래
불 켰으면 좋겠어
불 좀 밝게 해줘
```

These utterances vary acoustically, but they share semantic and lexical evidence
across a longer span. The model can use words like `불`, `켜`, and `밝게`, plus
sentence-level context, to place them near the same command region in embedding
space.

Wake words are much shorter. For Todak, the important cases are closer to:

```text
토닥아
토닥이
토닥
토닥토닥
토마토닥
도닥아
토다
```

Here, semantic context does not help much. The decision boundary is mostly
phonetic and acoustic: onset, coda, syllable count, rhythm, vowel/consonant
transients, and endpoint timing. A near-miss hard negative can share most of the
same acoustic material as the wake word.

## Why Near-Miss Negatives Are Hard

The current EdgeSpot wake-word evaluation uses prototype cosine similarity:

```text
support wavs -> encoder -> embeddings
label-wise mean embeddings -> prototypes

query wav -> encoder -> embedding
score = max cosine(query_embedding, positive_prototypes)
accept if score >= threshold
```

The output label is therefore the nearest support prototype name, not a decoded
text label. A row such as:

```text
ACCEPT 0.954425 토닥이 negative todatoda_3.wav
```

means `todatoda_3.wav` landed close to the `토닥이` prototype. It does not mean
the model transcribed the audio as `토닥이`.

This is acceptable for command-like keyword spotting when broad semantic regions are well
separated. It is harder for wake words because near-miss phrases can be close in
the same embedding space:

- `토닥` is a prefix-like fragment of `토닥아`.
- `토닥토닥` repeats a wake-word-like unit.
- `토마토닥` contains a wake-word-like ending.
- `토닥아` and `토닥이` are both positive wake-word variants, so separating them
  too aggressively is not the real goal.

## Device And Channel Effects

Large noise-environment datasets can improve general command/negative ranking
without solving device wake-word robustness. The missing factor may be channel
diversity, not only noise diversity.

Separate dimensions:

```text
Noise/environment diversity:
  background noise, room, distance, SNR, speaker environment

Device/channel diversity:
  microphone frequency response, AGC, noise suppression, AEC, compression,
  clipping, resampling, input gain, enclosure resonance, band limits
```

If most training data is phone-recorded, the model may learn a broad set of
phone-like acoustic conditions but still fail on the deployment device's
coloration. This matches the observed pattern:

- large general AUC can improve after adding more noise data;
- device split recall can still degrade;
- positive-only device-like augmentation can improve large ranking while
  worsening practical device wake-word tradeoffs.

## Current Experiment Implication

For a leakage-free comparison with openWakeWord v3/v5-style results, labeled
device evaluation clips should stay out of training:

```text
Do not train on:
  device_positive_eval/*.wav
  device_hard_negative_eval/*.wav

Allowed for device-like augmentation:
  top-level device_record/*.wav profile/noise/channel statistics
```

The current both-side device-like augmentation experiment follows that rule. It
uses only the top-level device profile and applies the same feature-domain
coloration/noise/gain augmentation to both positive and negative training rows.
The goal is to avoid the shortcut where `device-like sound` becomes correlated
only with the wake word.

## AIHub Invocation Data Hypothesis

The AIHub invocation dataset discussed for follow-up work is better aligned with
Todak than the command/noise dataset because its largest reported slice is
`호출어` data, not arbitrary command intent data:

```text
호출어: 5,269 hours
공통문장: 370 hours
랜덤텍스트: 2,667 hours
호출어+공통문장: 446 hours
```

That makes it a strong candidate for wake-word-form training. The important
constraint is label handling. If the invocation phrases are not literally Todak
variants, they should not all be merged into the Todak positive class. Doing so
would teach the model that many unrelated invocation words are acceptable
matches, which can increase false accepts on wake-word-like Korean phrases.

Preferred uses:

1. Use invocation labels as separate few-shot keyword classes so same-phrase
   examples are pulled together and different invocation phrases remain apart.
2. Use the corpus for wake-word-form pretraining or auxiliary batches, then keep
   Todak recognition driven by Todak support prototypes.
3. Mix it at a larger rate than generic command/noise positives, because the
   acoustic and lexical shape is closer to ultra-short wake-word spotting.
4. Keep generated/confusable near-miss negatives in the mix, because invocation
   positives alone do not teach the boundary against `토닥`, `토닥토닥`, or
   `토마토닥`.
5. Continue holding out `device_positive_eval` and `device_hard_negative_eval`
   entirely for final device comparison.

This should be treated as a higher-priority data experiment than adding more
generic command-like positives. It tests whether the model is missing
short-invocation structure, while the current both-side device augmentation
tests whether it is missing deployment-channel robustness.

## Practical Direction

The likely sequence is:

1. Apply device-like augmentation to both positive and negative rows.
2. Add AIHub invocation data as label-aware auxiliary/pretraining data.
3. Add or strengthen generated non-device near-miss hard negatives.
4. Evaluate unified wake-word prototype scoring, because `토닥아` and `토닥이`
   are both accepted variants.
5. Use anti-saturation loss carefully or on a schedule to avoid collapsing
   positive recall.
6. Consider a verifier/calibration layer only after the embedding separation
   improves.

The key insight is that command-like keyword spotting benefits from broad semantic grouping,
while wake-word spotting depends on tight acoustic boundaries. EdgeSpot's
few-shot prototype setup may work well for command spotting but needs extra
near-miss and channel-robustness work to compete with a direct wake-word
classifier on ultra-short Korean wake words.
