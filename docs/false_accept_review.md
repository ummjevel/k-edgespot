# False Accept Review

Manual notes for high-scoring negative utterances from:

```text
docs/false_accepts_distill_tau4_k10_far1.jsonl
```

These cases are useful hard-negative seeds because short Korean state
utterances can sound command-like under prototype matching.

| Audio ID | Text | Matched label | Note |
|---|---|---|---|
| `negative_0049_G2A4E2S0C3_HSJ_000297_v026` | 괴롭다 | `lights_off` | Very short one-word state utterance. Similar duration/rhythm to short commands. |
| `negative_0039_G1A3E2S0C6_LSJ_000600_v026` | 두렵다 | `lights_off` | Very short one-word state utterance. Similar command-like cadence. |
| `negative_0049_G2A4E2S0C3_HSJ_001026_v004` | 뼈아프다 | `next_track` | Syllable-separated delivery can sound like a short command phrase. |
| `negative_0039_G1A3E2S0C6_LSJ_000349_v026` | 그립다 | `lights_off` | Very short one-word state utterance. Good hard-negative candidate. |

Current seed file:

```text
configs/korean_hard_negatives.txt
```

## After Hard-Negative Training

Compared file:

```text
docs/false_accepts_distill_hard_tau4_k10_far1.jsonl
```

At k=10 and FAR=1%, the number of false accepts at threshold remains 13 because
the threshold is selected from the negative score quantile. The important change
is which negatives are now above threshold.

| Seed text | Original top false accept | After hard-negative training |
|---|---|---|
| 괴롭다 | Present | Removed from top FAR=1% false accepts |
| 두렵다 | Present | Removed from top FAR=1% false accepts |
| 뼈아프다 | Present | Removed from top FAR=1% false accepts |
| 그립다 | Present | Still appears once with another sample |

The new top false accepts are no longer dominated by the original short state
utterances. Remaining examples include other short or command-like phrases such
as `즐겁다` and `아이 추운데.` plus longer sentences whose acoustic rhythm still
matches command prototypes.

These remaining patterns were added to `configs/korean_hard_negatives.txt`:

```text
즐겁다
아이 추운데
추운데
뭐 하는 짓입니까
불길하다
젊다
죽었다
위험하다
공부했습니다
```

## Real-Recording Negative Domain-Test

The best hard-negative distilled checkpoint was evaluated against the original
test positives plus 5,000 real-recording Korean negatives:

```text
runs/edgespot-ko-distill-hard-tau4/prototype_eval_realneg5k_k10.json
```

At k=10, AUC was 0.9222, Recall@FAR1% was 0.6276, and Recall@FAR5% was
0.8229. This is comparable to the TTS-negative test result, so the current
prototype model does not show an immediate collapse on real-recording negatives.
The next useful review item is extracting false accepts from this domain-test
set and adding the strongest confusers into another hard-negative synthesis
round.
