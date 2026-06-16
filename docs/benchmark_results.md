# EdgeSpot Korean Benchmark Results

All rows use prototype evaluation with `k=10` unless otherwise noted.

| Benchmark | Model | Queries | AUC | Recall@FAR0.1% | Recall@FAR1% | Recall@FAR5% |
|---|---|---:|---:|---:|---:|---:|
| TTS held-out test | Pre-AIHub hard-negative tau4 | 1635 | 0.9191 | 0.1510 | 0.5911 | 0.8177 |
| Real-recording negative domain test | Pre-AIHub hard-negative tau4 | 5384 | 0.9222 | 0.1354 | 0.6276 | 0.8229 |
| AIHub target-keyword full | Pre-AIHub hard-negative tau4 | 984 | 0.4969 | 0.0096 | 0.0192 | 0.0705 |
| Augmented held-out test | Post-AIHub tau4 | 1745 | 0.9118 | 0.1368 | 0.7406 | 0.8255 |
| AIHub target-keyword full | Post-AIHub tau4 | 984 | 0.5445 | 0.0096 | 0.0481 | 0.1122 |
| Reviewed augmented held-out test | Reviewed tau4 | 1743 | 0.7072 | 0.2601 | 0.4821 | 0.5680 |
| Reviewed AIHub target-keyword full | Reviewed tau4 | 953 | 0.5330 | 0.0043 | 0.0255 | 0.1021 |
| Conservative-review augmented held-out test | Conservative-review tau4 | 1742 | 0.7376 | 0.0993 | 0.4586 | 0.6028 |
| Conservative-review AIHub target-keyword full | Conservative-review tau4 | 953 | 0.4879 | 0.0106 | 0.0142 | 0.0496 |

## Interpretation

- The pre-AIHub hard-negative model remains strong on the TTS held-out and real-recording negative tests.
- Adding AIHub mapped data improves the held-out augmented split substantially, but full AIHub target-keyword generalization remains weak.
- Applying all audio-review decisions directly made the held-out score worse. The likely issue is that many false rejects were relabeled as negatives, narrowing the positive boundary too aggressively.
- The conservative-review ablation recovered a small amount of held-out AUC versus direct review application, but it still underperforms the original post-AIHub model and worsens the full target scan.
- The gap between augmented held-out test and full AIHub target-keyword test suggests the next iteration should focus on AIHub label quality, support/query protocol, and hard negative mining rather than simply increasing epochs.

## Fixed Benchmark Protocol

| Benchmark | Query manifest | Purpose |
|---|---|---|
| TTS held-out test | `data/manifests/splits/test.jsonl` | Check generated-command baseline behavior. |
| Real-recording negative domain test | `data/manifests/splits/test_real_negative_5k.jsonl` | Check false accepts on real Korean speech negatives. |
| AIHub mapped held-out test | `data/manifests/splits/test_aihub_mapped.jsonl` | Check held-out AIHub split after training with mapped AIHub rows. |
| AIHub target-keyword full scan | `data/manifests/aihub_71405_validation_seed.extracted.target_keyword_short.mapped.jsonl` | Stress-test all mapped AIHub short target-keyword rows. |
| Reviewed AIHub mapped held-out test | `data/manifests/splits/test_aihub_reviewed.jsonl` | Check the direct application of audio-review decisions. |
| Reviewed AIHub target-keyword full scan | `data/manifests/aihub_71405_validation_seed.extracted.target_keyword_short.reviewed.jsonl` | Stress-test the direct review-decision manifest after bad-audio removal. |
| Conservative-review AIHub mapped held-out test | `data/manifests/splits/test_aihub_reviewed_conservative.jsonl` | Check review ablation without false-reject negative relabeling. |
| Conservative-review AIHub target-keyword full scan | `data/manifests/aihub_71405_validation_seed.extracted.target_keyword_short.reviewed_conservative.jsonl` | Stress-test conservative review decisions after bad-audio removal. |

## Source Files

- `runs/edgespot-ko-distill-hard-tau4/prototype_eval_k10.json`: TTS test split, 384 positives + 1,251 TTS negatives.
- `runs/edgespot-ko-distill-hard-tau4/prototype_eval_realneg5k_k10.json`: Original TTS positives plus 5,000 real-recording Korean negatives.
- `runs/edgespot-ko-distill-hard-tau4/prototype_eval_aihub71405_target_keyword_k10.json`: 984 mapped AIHub target-keyword rows, all S/N variants.
- `runs/edgespot-ko-distill-hard-aihub-tau4/prototype_eval_augmented_test_k10.json`: Held-out TTS + AIHub mapped test split.
- `runs/edgespot-ko-distill-hard-aihub-tau4/prototype_eval_aihub71405_target_keyword_k10.json`: 984 mapped AIHub target-keyword rows, all S/N variants.
- `runs/edgespot-ko-distill-hard-aihub-reviewed-tau4/prototype_eval_augmented_reviewed_test_k10.json`: 1,743 held-out TTS + reviewed AIHub test rows.
- `runs/edgespot-ko-distill-hard-aihub-reviewed-tau4/prototype_eval_aihub71405_target_keyword_reviewed_k10.json`: 953 reviewed AIHub target-keyword rows after bad-audio removal.
- `runs/edgespot-ko-distill-hard-aihub-conservative-tau4/prototype_eval_augmented_conservative_test_k10.json`: 1,742 held-out TTS + conservative-reviewed AIHub test rows.
- `runs/edgespot-ko-distill-hard-aihub-conservative-tau4/prototype_eval_aihub71405_target_keyword_conservative_k10.json`: 953 conservative-reviewed AIHub target-keyword rows after bad-audio removal.
