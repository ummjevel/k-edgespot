# EdgeSpot Korean Results

Generated from prototype evaluation JSON files under `runs/`.

## Best Runs

| Rank | Family | Tau | K-shot | AUC | Recall@FAR0.1% | Recall@FAR1% | Recall@FAR5% |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | distill | 4 | 10 | 0.9325 | 0.0938 | 0.4948 | 0.7917 |
| 2 | scaf | 2 | 10 | 0.8663 | 0.0156 | 0.1458 | 0.4036 |
| 3 | scaf | 2 | 5 | 0.8295 | 0.0130 | 0.0990 | 0.3177 |
| 4 | distill | 4 | 5 | 0.7857 | 0.0365 | 0.4297 | 0.6536 |
| 5 | distill | 3 | 10 | 0.7412 | 0.0078 | 0.3229 | 0.6016 |
| 6 | scaf | 2 | 1 | 0.7006 | 0.0469 | 0.1068 | 0.2057 |
| 7 | scaf | 1 | 1 | 0.6476 | 0.0104 | 0.0703 | 0.1641 |
| 8 | distill | 3 | 5 | 0.6390 | 0.0078 | 0.1432 | 0.5391 |

## All Runs

| Family | Tau | K-shot | AUC | Recall@FAR0.1% | Recall@FAR1% | Recall@FAR5% |
|---|---:|---:|---:|---:|---:|---:|
| distill | 1 | 1 | 0.2935 | 0.0000 | 0.0026 | 0.0130 |
| distill | 1 | 5 | 0.3394 | 0.0000 | 0.0052 | 0.0443 |
| distill | 1 | 10 | 0.3729 | 0.0000 | 0.0052 | 0.0417 |
| distill | 2 | 1 | 0.1481 | 0.0000 | 0.0026 | 0.0052 |
| distill | 2 | 5 | 0.3146 | 0.0078 | 0.0104 | 0.0677 |
| distill | 2 | 10 | 0.4377 | 0.0078 | 0.0234 | 0.1380 |
| distill | 3 | 1 | 0.1538 | 0.0000 | 0.0026 | 0.0104 |
| distill | 3 | 5 | 0.6390 | 0.0078 | 0.1432 | 0.5391 |
| distill | 3 | 10 | 0.7412 | 0.0078 | 0.3229 | 0.6016 |
| distill | 4 | 1 | 0.1671 | 0.0078 | 0.0104 | 0.0286 |
| distill | 4 | 5 | 0.7857 | 0.0365 | 0.4297 | 0.6536 |
| distill | 4 | 10 | 0.9325 | 0.0938 | 0.4948 | 0.7917 |
| scaf | 1 | 1 | 0.6476 | 0.0104 | 0.0703 | 0.1641 |
| scaf | 1 | 5 | 0.5368 | 0.0000 | 0.0156 | 0.0599 |
| scaf | 1 | 10 | 0.5815 | 0.0052 | 0.0365 | 0.0833 |
| scaf | 2 | 1 | 0.7006 | 0.0469 | 0.1068 | 0.2057 |
| scaf | 2 | 5 | 0.8295 | 0.0130 | 0.0990 | 0.3177 |
| scaf | 2 | 10 | 0.8663 | 0.0156 | 0.1458 | 0.4036 |
| scaf | 3 | 1 | 0.5157 | 0.0000 | 0.0104 | 0.0625 |
| scaf | 3 | 5 | 0.5179 | 0.0026 | 0.0078 | 0.0365 |
| scaf | 3 | 10 | 0.5961 | 0.0052 | 0.0182 | 0.0625 |
| scaf | 4 | 1 | 0.6100 | 0.0026 | 0.0182 | 0.1146 |
| scaf | 4 | 5 | 0.5746 | 0.0026 | 0.0312 | 0.1042 |
| scaf | 4 | 10 | 0.5724 | 0.0000 | 0.0260 | 0.0938 |

## Current Takeaways

- The best SCAF-only baseline is `scaf`, tau 2, 10-shot.
- The best distilled model is `distill`, tau 4, 10-shot.
- Distillation improves this TTS split substantially, but the evaluation is still TTS-domain only.
- Next validation should use hard negatives and real-recording negative/domain-test data.
