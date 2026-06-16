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
