# EdgeSpot Korean Results

Generated from prototype evaluation JSON files under `runs/`.

## Latest Practical Ranking

2026-06-23 기준 최신 후보를 device wake-word 관점으로 다시 정리했다.

랭킹 기준:

1. `Device Split Holdout`의 AUC와 Recall@FAR1%.
2. `OpenWakeWord-Matched all45`의 Recall@FAR1%와 hard-negative 분리.
3. `Large General Conservative`는 일반 command/negative ranking 보조지표로만 사용.
4. `AIHub validation FAR noise-only`는 Qwen3-TTS 생성 음성과 녹음 음성 positive를
   제외하고, 소음환경 negative만 222,044개 평가한 별도 false-accept 압력 지표로
   사용한다. 이 평가는 positive가 없으므로 AUC/recall은 없다.

| Rank | Checkpoint | 역할 | Device split AUC | Device split Recall@FAR1% | all45 AUC | all45 Recall@FAR1% | Large AUC | Large Recall@FAR1% | Noise-only Threshold@FAR1% | 판단 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `edgespot-ko-proxy-noise50far-margin-confusable-tau4` | 현재 baseline / 운영 후보 | 0.7818 | 0.6000 | 0.5540 | 0.1000 | 0.9349 | 0.8652 | 0.3519 | device split 기준 가장 안정적이다. all45는 약하지만 현재 후보 중 device holdout 균형이 제일 낫다. |
| 2 | `edgespot-ko-proxy-noise50far-margin-confusable-posdeviceaug-w35-tau4` | positive-only device-like augmentation | 0.5455 | 0.4000 | 0.5920 | 0.2000 | 0.9601 | 0.8605 | 0.4516 | large/all45 ranking은 좋아졌지만 device split AUC가 낮아 운영 후보로는 baseline보다 약하다. |
| 3 | `edgespot-ko-proxy-noise50far-margin-confusable-posdeviceaug-w50-antisat-w01-tau4` | exported second candidate | 0.6818 | 0.3000 | 0.5580 | 0.2000 | 0.9686 | 0.8558 | 0.4608 | large AUC는 가장 높지만 device hard negative에서 threshold tradeoff가 나쁘다. |
| 4 | `edgespot-ko-proxy-noise50far-margin-confusable-posdeviceaug-w25-tau4` | weak positive-only device-like augmentation | 0.6727 | 0.3000 | 0.6080 | 0.1000 | 0.9554 | 0.8487 | 0.4812 | all45 AUC는 높지만 Recall@FAR1%가 낮다. |
| 5 | `edgespot-ko-proxy-noise50far-clean-margin-confusable-tau4` | clean/noise50far ablation | 0.6636 | 0.2000 | 0.5580 | 0.0500 | 0.9216 | 0.5366 | n/a | device split은 일부 남지만 large recall이 낮다. |
| 6 | `edgespot-ko-proxy-noise-plus50-margin-confusable-tau4` | AIHub 소음환경 +50GB | 0.5000 | 0.1000 | 0.5800 | 0.0000 | 0.9469 | 0.8463 | 0.5100 | large AUC는 올라갔지만 device split이 무너졌다. |
| 7 | `edgespot-ko-cmdaux-pretrain-noise50far-finetune-tau4` | command auxiliary pretrain 후 fine-tune | 0.5182 | 0.1000 | 0.5220 | 0.0500 | 0.9394 | 0.7943 | 0.5798 | command auxiliary만으로 device robustness가 개선되지는 않았다. |
| 8 | `edgespot-ko-proxy-noise-plus100-margin-confusable-tau4` | AIHub 소음환경 +100GB | 0.5545 | 0.0000 | 0.5680 | 0.0000 | 0.9474 | 0.6359 | 0.4407 | negative 양 증가만으로 device 성능이 좋아지지 않았다. |
| 9 | `edgespot-ko-cmdaux-deviceaug-pretrain-noise50far-finetune-weakdeviceaug-tau4` | command auxiliary + weak device-like fine-tune | 0.5273 | 0.0000 | 0.5760 | 0.0500 | 0.9457 | 0.7801 | 0.5392 | large는 괜찮지만 device split Recall@FAR1%가 0이다. |

최신 결론:

- 현재 운영 후보는 여전히 `edgespot-ko-proxy-noise50far-margin-confusable-tau4`다.
- large general AUC만 보면 `posdeviceaug-w50-antisat-w01`가 가장 좋지만, device
  hard negative와 positive score가 겹쳐 실제 wake-word threshold 선택이 어렵다.
- positive에만 device-like augmentation을 주는 실험은 large AUC를 올릴 수 있지만,
  device split 일반화는 개선하지 못했다.
- 다음 실험은 labeled `device_positive_eval`/`device_hard_negative_eval` 45개를
  학습에 넣지 않고, top-level `device_record/*.wav` profile만 사용해 positive와
  hard negative 양쪽에 같은 device-like augmentation을 적용하는 것이 우선이다.

## Large General And Noise-Only Negative Results

`Large General Conservative`는 Qwen3-TTS/generated command positive와 large
negative가 섞인 26,742개 평가다. 반면 `AIHub validation FAR noise-only`는
Qwen3-TTS 생성 음성과 녹음 음성 positive를 제외하고 AIHub 소음환경 negative만
222,044개 평가한 것이다. Noise-only에는 positive가 없으므로 threshold quantile만
기록한다. 낮은 `Threshold@FARx`일수록 해당 noise-only set의 negative score가 낮게
눌렸다는 뜻이다.

| Checkpoint | Large AUC | Large Recall@FAR0.1% | Large Recall@FAR1% | Large Recall@FAR5% | Noise-only Th@FAR0.1% | Noise-only Th@FAR1% | Noise-only Th@FAR5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| `noise50far-margin-confusable` | 0.9349 | 0.5201 | 0.8652 | 0.9007 | 0.5619 | 0.3519 | 0.3314 |
| `posdeviceaug-w25` | 0.9554 | 0.3522 | 0.8487 | 0.9007 | 0.8111 | 0.4812 | 0.3574 |
| `posdeviceaug-w35` | 0.9601 | 0.3783 | 0.8605 | 0.8983 | 0.7216 | 0.4516 | 0.3159 |
| `posdeviceaug-w50-antisat-w01` | 0.9686 | 0.3333 | 0.8558 | 0.9078 | 0.7282 | 0.4608 | 0.3765 |
| `plus50-noise` | 0.9469 | 0.3168 | 0.8463 | 0.9007 | 0.6250 | 0.5100 | 0.4770 |
| `plus100-noise` | 0.9474 | 0.1631 | 0.6359 | 0.8652 | 0.7593 | 0.4407 | 0.3633 |
| `cmdaux-finetune` | 0.9394 | 0.4799 | 0.7943 | 0.8629 | 0.7306 | 0.5798 | 0.4532 |
| `cmdaux-deviceaug-finetune` | 0.9457 | 0.5225 | 0.7801 | 0.8652 | 0.6877 | 0.5392 | 0.4174 |
| `aihub-command-aux` | 0.8858 | 0.0686 | 0.1631 | 0.5083 | 0.8603 | 0.8295 | 0.6980 |
| `aihub-command-aux-deviceaug` | 0.8921 | 0.0307 | 0.0969 | 0.5366 | 0.8858 | 0.8636 | 0.6966 |

해석:

- `posdeviceaug-w50-antisat-w01`는 large AUC 0.9686으로 가장 높지만, noise-only
  Threshold@FAR1%가 0.4608로 baseline 0.3519보다 높다. 즉 large ranking은
  좋아졌지만 pure noise negative 점수는 더 높아졌다.
- `noise50far-margin-confusable` baseline은 large AUC가 최고는 아니지만,
  noise-only Threshold@FAR1%가 낮고 device split Recall@FAR1%가 가장 높아 현재
  균형이 제일 좋다.
- command auxiliary pretrain 계열은 그대로 쓰면 noise-only threshold가 높다.
  fine-tune 후에도 device split 개선은 제한적이었다.
- AIHub 소음환경 negative를 단순히 더 많이 추가한 `plus50`/`plus100`은 large AUC를
  올렸지만 device split recall을 악화시켰다.

## Best Runs

| Rank | Family | Tau | K-shot | AUC | Recall@FAR0.1% | Recall@FAR1% | Recall@FAR5% |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | distill | 4 | 10 | 0.9325 | 0.0938 | 0.4948 | 0.7917 |
| 2 | distill-hard | 4 | 10 | 0.9191 | 0.1510 | 0.5911 | 0.8177 |
| 3 | scaf | 2 | 10 | 0.8663 | 0.0156 | 0.1458 | 0.4036 |
| 4 | scaf | 2 | 5 | 0.8295 | 0.0130 | 0.0990 | 0.3177 |
| 5 | distill | 4 | 5 | 0.7857 | 0.0365 | 0.4297 | 0.6536 |
| 6 | distill-hard | 4 | 5 | 0.7610 | 0.0964 | 0.5495 | 0.6615 |
| 7 | distill | 3 | 10 | 0.7412 | 0.0078 | 0.3229 | 0.6016 |
| 8 | scaf | 2 | 1 | 0.7006 | 0.0469 | 0.1068 | 0.2057 |

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
| distill-hard | 4 | 1 | 0.2149 | 0.0208 | 0.0625 | 0.0859 |
| distill-hard | 4 | 5 | 0.7610 | 0.0964 | 0.5495 | 0.6615 |
| distill-hard | 4 | 10 | 0.9191 | 0.1510 | 0.5911 | 0.8177 |
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
- The hard-negative tau 4 run lowers AUC slightly but improves recall at strict FARs.
- Real-recording negative domain-test has now been run for `distill-hard`, tau 4,
  10-shot:
  - Query set: original test positives plus 5,000 real-recording Korean negatives.
  - AUC: 0.9222.
  - Recall@FAR0.1%: 0.1354.
  - Recall@FAR1%: 0.6276.
  - Recall@FAR5%: 0.8229.
- Compared with the TTS-negative test for the same checkpoint, real-recording
  negatives did not cause an immediate domain collapse. FAR1 recall improved
  from 0.5911 to 0.6276, while FAR0.1 recall dropped from 0.1510 to 0.1354.
- Next validation should add real-recording false-accept inspection and
  streaming-style metrics.

## 이번 Device 실험 요약

이번 실험의 목적은 실제 기기 녹음에서 `토닥아`/`토닥이` positive와
`토닥`, `토닥토닥`, `토마토닥`, `도요` 같은 hard negative를 EdgeSpot이
얼마나 구분하는지 확인하는 것이다.

평가 기준은 두 가지로 나눴다.

- `Device Split Holdout`: support와 query가 분리된 더 깨끗한 평가.
  - support: `device_split/train_positive` 10개.
  - query: `device_split/holdout_positive` 10개 +
    `device_split/holdout_hard_negative` 11개.
- `OpenWakeWord-Matched Device Eval`: openWakeWord 문서와 개수를 맞춘 비교용
  평가.
  - query: `device_positive_eval` 20개 + `device_hard_negative_eval` 25개.
  - 단, EdgeSpot은 prototype 생성을 위해 positive support가 필요하므로 일부
    positive query가 support와 겹친다. 학습 누수는 아니지만, prototype 평가
    기준으로는 positive recall이 낙관적으로 보일 수 있다.

현재까지의 핵심 결론:

- `margin+confusable`는 device split holdout에서 AUC가 가장 높고, high
  threshold sweep에서 `recall 0.6000 / hard FP 1/11`이 나와 현재 가장 가능성
  있는 후보이다.
- 하지만 `@0.95` 고정 threshold에서는 hard FP가 `11/11`이다. 즉 positive도
  잘 듣지만 hard negative도 전부 wake word처럼 본다.
- `deviceaug` 단독 실험은 이번 설정에서는 도움이 되지 않았다. 점수 포화가 더
  심해졌고 device split holdout 성능이 나빠졌다.
- `antisat` 단독 실험은 hard negative 점수를 낮추는 효과는 있었지만 너무
  강했다. hard FP는 줄었지만 positive recall도 같이 무너졌다.
- 다음 anti-saturation 실험은 weight를 낮추거나 후반 epoch에만 켜는 식으로
  약하게 적용해야 한다.

## OpenWakeWord 개수 맞춤 Device Eval

이 평가는 `openWakeWord/docs/todak_device_eval_corrected_all_models.md`에서 쓴
평가 corpus 개수와 맞춘 것이다.

- Query positive: `device_positive_eval`, 20개.
- Query hard negative: `device_hard_negative_eval`, 25개.
- Query 전체: 45개.
- Support: `device_todak_support_k5.jsonl`, positive 10개.
  - `토닥아` 5개 + `토닥이` 5개.

주의할 점:

- EdgeSpot은 prototype 생성을 위해 positive support 샘플이 필요하다.
- 45개 query에는 support로 쓴 positive 파일 일부가 다시 들어간다.
- 모델 weight를 업데이트하는 학습 누수는 아니다.
- 하지만 prototype 평가 관점에서는 같은 positive 파일을 기준점 생성에도 쓰고
  query 평가에도 쓰는 overlap이므로 positive recall이 낙관적으로 보일 수 있다.
- openWakeWord와 개수 맞춤 비교를 할 때만 이 섹션을 보고, 독립 holdout 판단은
  `Device Split Holdout`을 우선한다.

| Run | AUC | Recall@FAR0.1% | Threshold@FAR0.1% | Recall@FAR1% | Threshold@FAR1% | Recall@FAR5% | Threshold@FAR5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| `distill-hard` tau4 | 0.5920 | 0.0500 | 0.993344 | 0.0500 | 0.993287 | 0.1000 | 0.992740 |
| `hard+aihub` tau4 | 0.6380 | 0.0500 | 0.993238 | 0.0500 | 0.992518 | 0.2000 | 0.989967 |
| `conservative` tau4 | 0.5400 | 0.0000 | 0.997132 | 0.0000 | 0.997117 | 0.0000 | 0.996064 |
| `margin+confusable` tau4 | 0.5500 | 0.0000 | 0.999930 | 0.0000 | 0.999919 | 0.0000 | 0.999878 |

같은 45개 파일에 대한 고정 threshold 관점:

| Run | @0.50 pos recall | @0.50 hard FP | @0.90 pos recall | @0.90 hard FP | @0.95 pos recall | @0.95 hard FP |
|---|---:|---:|---:|---:|---:|---:|
| `distill-hard` tau4 | 0.9500 | 25/25 | 0.9500 | 25/25 | 0.8500 | 22/25 |
| `hard+aihub` tau4 | 1.0000 | 25/25 | 0.9000 | 24/25 | 0.8000 | 18/25 |
| `conservative` tau4 | 1.0000 | 25/25 | 0.9500 | 25/25 | 0.8000 | 18/25 |
| `margin+confusable` tau4 | 1.0000 | 25/25 | 1.0000 | 25/25 | 1.0000 | 25/25 |

해석:

- openWakeWord 문서와 직접 숫자를 비교하려면 이 20 positive + 25 hard
  negative 기준을 봐야 한다.
- 이 기준에서는 `hard+aihub` tau4가 AUC `0.6380`으로 가장 높다.
- 하지만 FAR1% recall은 `distill-hard`와 `hard+aihub` 모두 `0.0500`으로 낮다.
- 고정 threshold `0.95` 기준 hard FP는 `hard+aihub`와 `conservative`가
  `18/25`로 가장 낮지만, 여전히 hard negative가 많이 통과한다.
- `margin+confusable`는 모든 positive를 높게 올리지만 hard negative도 전부
  높게 올려서, 45개 기준에서는 좋은 모델로 보기 어렵다.

## Device Split Holdout

평가 구성:

- Support: `data/manifests/device_record/device_split_support_train_positive.jsonl`
  - `device_split/train_positive`에서 온 기기 녹음 positive 10개.
  - `토닥아` 5개 + `토닥이` 5개.
- Query: `data/manifests/device_record/device_split_query_holdout.jsonl`
  - holdout positive 10개.
  - holdout hard negative 11개.
- K-shot: 5.

| Run | AUC | Recall@FAR0.1% | Threshold@FAR0.1% | Recall@FAR1% | Threshold@FAR1% | Recall@FAR5% | Threshold@FAR5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| `distill-hard` tau4 | 0.5636 | 0.2000 | 0.990697 | 0.2000 | 0.990536 | 0.2000 | 0.989819 |
| `hard+aihub` tau4 | 0.5727 | 0.0000 | 0.992264 | 0.0000 | 0.992052 | 0.1000 | 0.991113 |
| `conservative` tau4 | 0.4727 | 0.0000 | 0.996534 | 0.0000 | 0.996464 | 0.0000 | 0.996156 |
| `margin+confusable` tau4 | 0.6818 | 0.0000 | 0.999869 | 0.0000 | 0.999839 | 0.5000 | 0.999706 |

고정 threshold 관점:

| Run | @0.50 pos recall | @0.50 hard FP | @0.90 pos recall | @0.90 hard FP | @0.95 pos recall | @0.95 hard FP |
|---|---:|---:|---:|---:|---:|---:|
| `distill-hard` tau4 | 0.9000 | 11/11 | 0.9000 | 10/11 | 0.8000 | 10/11 |
| `hard+aihub` tau4 | 1.0000 | 11/11 | 0.9000 | 11/11 | 0.8000 | 8/11 |
| `conservative` tau4 | 1.0000 | 11/11 | 0.9000 | 11/11 | 0.7000 | 9/11 |
| `margin+confusable` tau4 | 1.0000 | 11/11 | 1.0000 | 11/11 | 1.0000 | 11/11 |

같은 holdout에서 `0.99` 이상 threshold를 훑어본 결과:

| Run | Best threshold >=0.99 under hard FP<=1 | Pos recall | Hard FP |
|---|---:|---:|---:|
| `distill-hard` tau4 | 0.994114 | 0.2000 | 0/11 |
| `hard+aihub` tau4 | 0.990000 | 0.1000 | 1/11 |
| `conservative` tau4 | 0.997000 | 0.0000 | 0/11 |
| `margin+confusable` tau4 | 0.999648 | 0.6000 | 1/11 |

이 high-threshold sweep은 holdout 점수를 직접 보고 threshold를 고른 것이다.
따라서 진단용으로는 유용하지만, 실제 운영점 추정으로는 낙관적일 수 있다.

openWakeWord의 sigmoid 출력과 달리, EdgeSpot의 고정 threshold는 prototype
cosine similarity 점수에 적용된다. 이 점수는 device holdout에서 많이 포화되어
있어서 `0.90`, `0.95` 같은 threshold가 openWakeWord threshold와 직접 비교되면
안 된다. 다만 score calibration을 보는 데는 도움이 된다. 예를 들어
`margin+confusable`는 모든 positive를 `0.95` 위로 올렸지만, hard negative도
전부 `0.95` 위로 올렸다.

점수 해석:

- openWakeWord의 sigmoid 점수는 비교적 "확률처럼" 해석할 수 있는 출력이다.
  예를 들어 threshold `0.90`은 모델이 해당 구간을 wake word라고 강하게 보는
  기준으로 이해하기 쉽다.
- EdgeSpot의 현재 평가 점수는 sigmoid 확률이 아니라 support positive
  prototype과 query embedding 사이의 cosine similarity이다. 따라서 `0.95`가
  "95% 확률"이라는 뜻은 아니며, 모델이 바뀌면 같은 `0.95`라도 의미가 달라질
  수 있다.
- 고정 threshold 표는 직관적으로 보기 좋지만, 점수가 calibration되어 있지
  않으면 실제 운영 성능 순위를 그대로 말해주지 않는다.
- FAR 기반 표는 hard negative 점수 분포를 보고 "false accept를 이 정도만
  허용하려면 threshold를 어디에 둬야 하는가"를 계산한 것이다. 실제 wake-word
  운영점 판단에는 이쪽이 더 중요하다.
- 이번 결과에서 `margin+confusable`는 AUC가 가장 높다. 즉 positive가 negative
  보다 위에 오는 ranking은 가장 좋아졌다.
- 하지만 hard negative 점수도 같이 매우 높아져서, FAR1%를 맞추려면 threshold를
  `0.999839`까지 올려야 한다. 그 threshold에서는 positive recall이 `0.0`이다.
- 그래서 `margin+confusable`를 "제일 잘 나온 모델"이라고 단정하기 어렵다.
  ranking 관점에서는 가장 좋지만, strict FAR 운영점에서는 아직 좋지 않다.
- 현재 device holdout 기준 운영점만 보면 `distill-hard` tau4가 FAR1%에서
  recall `0.2`라도 남기 때문에 더 실용적인 baseline으로 볼 수 있다.

`margin+confusable` 실험 설정:

- negative-to-positive batch prototype hinge penalty:
  - `--negative-prototype-margin-weight 0.1`
  - `--negative-prototype-margin 0.35`
- confusable command-pair prototype separation:
  - `configs/confusable_command_pairs.tsv`
  - `--confusable-margin-weight 0.05`
  - `--confusable-margin 0.25`

Device holdout 해석:

- `margin+confusable` objective는 이 작은 holdout에서 ranking을 개선했다.
  AUC `0.6818`로 비교한 run 중 가장 높다.
- 하지만 strict operating point는 좋아지지 않았다. FAR1% recall은 `0.0`이다.
- 핵심 문제는 score saturation이다. 가장 높은 held-out hard negative인
  `todatoda_3.wav`가 `0.999873`까지 올라가면서 Threshold@FAR1%가
  `0.999839`까지 올라간다.
- FAR5%에서는 recall이 `0.5000`까지 올라가므로 일부 positive 분리는 좋아졌지만,
  낮은 FAR의 wake-word 운영점으로는 아직 부족하다.

현재 결론:

- loss만 조정한 2 -> 1 실험은 ranking 개선 방향으로는 의미가 있었지만, strict
  FAR에서 hard-negative rejection을 해결하지 못했다.
- 다음 단계는 hard negative score를 낮추되 positive를 같이 죽이지 않는 방식이다.
  anti-saturation을 더 약하게 적용하거나, controlled near-miss hard negative
  데이터를 단계적으로 추가해야 한다.

## 단일 요인 후속 실험

아래 실험들은 `margin+confusable` 설정을 baseline으로 유지하고 한 번에 하나만
바꾼 것이다. 원인 분리를 위해 device augmentation과 anti-saturation을 동시에
섞지 않았다.

- `deviceaug`: top-level `device_record/*.wav`에서 뽑은 기기 녹음 profile로
  feature-domain coloration/noise/gain을 추가했다.
- `antisat`: hard-negative anti-saturation loss만 추가했다.

두 job 모두 정상 완료됐다.

- `deviceaug`: Slurm job `3304`, 40 epochs.
- `antisat`: Slurm job `3305`, 40 epochs.

Device split holdout 결과:

| Run | AUC | Recall@FAR0.1% | Threshold@FAR0.1% | Recall@FAR1% | Threshold@FAR1% | Recall@FAR5% | Threshold@FAR5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| `margin+confusable` tau4 | 0.6818 | 0.0000 | 0.999869 | 0.0000 | 0.999839 | 0.5000 | 0.999706 |
| `deviceaug` tau4 | 0.5000 | 0.0000 | 0.999935 | 0.0000 | 0.999935 | 0.0000 | 0.999934 |
| `antisat` tau4 | 0.6636 | 0.1000 | 0.563311 | 0.1000 | 0.560203 | 0.2000 | 0.546387 |

Device split 고정 threshold 결과:

| Run | @0.50 pos recall | @0.50 hard FP | @0.90 pos recall | @0.90 hard FP | @0.95 pos recall | @0.95 hard FP | Best threshold >=0.99 under hard FP<=1 | Pos recall | Hard FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `margin+confusable` tau4 | 1.0000 | 11/11 | 1.0000 | 11/11 | 1.0000 | 11/11 | 0.999648 | 0.6000 | 1/11 |
| `deviceaug` tau4 | 1.0000 | 11/11 | 1.0000 | 11/11 | 1.0000 | 11/11 | 0.999935 | 0.0000 | 1/11 |
| `antisat` tau4 | 0.4000 | 2/11 | 0.0000 | 0/11 | 0.0000 | 0/11 | 0.990000 | 0.0000 | 0/11 |

OpenWakeWord 개수 맞춤 45개 평가:

| Run | AUC | Recall@FAR0.1% | Threshold@FAR0.1% | Recall@FAR1% | Threshold@FAR1% | Recall@FAR5% | Threshold@FAR5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| `margin+confusable` tau4 | 0.5500 | 0.0000 | 0.999930 | 0.0000 | 0.999919 | 0.0000 | 0.999878 |
| `deviceaug` tau4 | 0.5180 | 0.1500 | 0.999895 | 0.2000 | 0.999890 | 0.2000 | 0.999866 |
| `antisat` tau4 | 0.5620 | 0.1500 | 0.782539 | 0.1500 | 0.775805 | 0.1500 | 0.748886 |

OpenWakeWord 개수 맞춤 고정 threshold 결과:

| Run | @0.50 pos recall | @0.50 hard FP | @0.90 pos recall | @0.90 hard FP | @0.95 pos recall | @0.95 hard FP | Best threshold >=0.99 under hard FP<=1 | Pos recall | Hard FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `margin+confusable` tau4 | 1.0000 | 25/25 | 1.0000 | 25/25 | 1.0000 | 25/25 | n/a | n/a | n/a |
| `deviceaug` tau4 | 1.0000 | 25/25 | 1.0000 | 25/25 | 1.0000 | 25/25 | 0.999895 | 0.2000 | 1/25 |
| `antisat` tau4 | 0.5000 | 10/25 | 0.0000 | 0/25 | 0.0000 | 0/25 | 0.990000 | 0.0000 | 0/25 |

해석:

- `deviceaug`는 이번 설정에서는 도움이 되지 않았다. score saturation이 유지되었고
  독립 device split holdout은 오히려 나빠졌다.
- `antisat`는 hard negative 점수를 크게 낮췄지만 너무 강했다. `0.90`, `0.95`
  같은 고정 threshold에서 positive도 대부분 아래로 떨어졌다.
- 현재 운영점 후보는 여전히 기존 `margin+confusable` run이다. 아주 높은 tuned
  threshold를 써야 하지만, hard FP<=1 조건에서 의미 있는 device split recall을
  유지한 유일한 run이다.
- 다음 anti-saturation 실험은 더 약하게 가야 한다. 예를 들면
  `--negative-antisaturation-weight 0.1`,
  `--negative-antisaturation-margin 0.9`,
  `--negative-antisaturation-top-k 8`처럼 hard negative만 살짝 누르는 설정을
  먼저 확인하는 것이 좋다.

## Anti-Saturation Grid Results

추가 grid는 데이터 추가 없이 loss 설정만 바꾼 실험이다. 목표는 고정 threshold
`0.90`, `0.95` 자체가 아니라, hard negative와 positive 점수 분포가
`0.999...` 포화 구간에서 내려오면서 validation 기반 threshold를 잡을 수 있는지
확인하는 것이었다.

가장 의미 있었던 run은 1차 grid의 `w01-m097-k4`였다.

- 설정: `--negative-antisaturation-weight 0.1`,
  `--negative-antisaturation-margin 0.97`,
  `--negative-antisaturation-top-k 4`.
- Device split holdout: AUC `0.6818`, Recall@FAR1% `0.6000`,
  Threshold@FAR1% `0.612756`.
- OpenWakeWord 개수 맞춤 all45: AUC `0.6160`, Recall@FAR1% `0.3500`,
  Threshold@FAR1% `0.710312`.
- 기존 `margin+confusable`처럼 모든 점수가 `0.999...` 근처에 몰리는 문제는
  줄었지만, all45 recall은 아직 낮다.

Device split holdout 주요 결과:

| Run | AUC | Recall@FAR1% | Threshold@FAR1% | Recall@FAR5% | @0.50 pos recall | @0.50 hard FP |
|---|---:|---:|---:|---:|---:|---:|
| `margin+confusable` k5 | n/a | n/a | n/a | n/a | 1.0000 | 11/11 |
| `w01-m097-k4` 1차 | 0.6818 | 0.6000 | 0.612756 | 0.6000 | 0.7000 | 10/11 |
| `w005-m097-k4` 2차 | 0.4455 | 0.1000 | 0.999973 | 0.2000 | 1.0000 | 11/11 |
| `w0075-m097-k4` 2차 | 0.6091 | 0.0000 | 0.999976 | 0.1000 | 1.0000 | 11/11 |
| `w01-m096-k4` 2차 | 0.4909 | 0.1000 | 0.935232 | 0.1000 | 0.9000 | 9/11 |
| `w01-m098-k4` 2차 | 0.3545 | 0.0000 | 0.999889 | 0.0000 | 1.0000 | 11/11 |
| `antisat` aggressive | 0.6636 | 0.1000 | 0.560203 | 0.2000 | 0.4000 | 2/11 |

OpenWakeWord 개수 맞춤 all45 주요 결과:

| Run | AUC | Recall@FAR1% | Threshold@FAR1% | Recall@FAR5% | @0.50 pos recall | @0.50 hard FP |
|---|---:|---:|---:|---:|---:|---:|
| `margin+confusable` k5 | n/a | n/a | n/a | n/a | 1.0000 | 25/25 |
| `w01-m097-k4` 1차 | 0.6160 | 0.3500 | 0.710312 | 0.4000 | 0.7000 | 17/25 |
| `w005-m097-k4` 2차 | 0.4480 | 0.1500 | 0.999971 | 0.2000 | 1.0000 | 25/25 |
| `w0075-m097-k4` 2차 | 0.5960 | 0.2000 | 0.999977 | 0.2000 | 1.0000 | 25/25 |
| `w01-m096-k4` 2차 | 0.5400 | 0.1000 | 0.937047 | 0.1500 | 0.9000 | 23/25 |
| `w01-m098-k4` 2차 | 0.4820 | 0.1000 | 0.999944 | 0.1500 | 1.0000 | 25/25 |
| `antisat` aggressive | 0.5620 | 0.1500 | 0.775805 | 0.1500 | 0.5000 | 10/25 |

해석:

- 2차 grid는 1차 `w01-m097-k4`를 넘지 못했다.
- `w=0.05`, `w=0.075`로 낮추면 다시 `0.999...` saturation으로 돌아갔다.
- `margin=0.96`은 saturation은 일부 줄였지만 recall과 hard FP 균형이 좋지 않았다.
- `margin=0.98`은 다시 saturation이 심해졌다.
- 현재 best candidate는 `w01-m097-k4`다. 다만 all45 Recall@FAR1% `0.3500`은
  아직 낮으므로, 운영 후보라기보다는 다음 실험의 기준점으로 보는 것이 맞다.

## Remaining Work

우선순위가 높은 남은 작업:

1. `w01-m097-k4`를 기준점으로 두고 다음 실험 방향을 정한다.
   - 2차 grid는 1차 best를 넘지 못했다.
   - 데이터 추가 전에는 schedule 방식이 가장 자연스러운 다음 후보이다.

2. anti-saturation schedule 실험
   - 목적: 초반 embedding 형성을 방해하지 않고 후반에만 hard negative를 누른다.
   - 방식:
     - 처음 20~30 epoch는 기존 `margin+confusable`처럼 학습.
     - 후반 epoch에만 `w=0.1`, `margin=0.97`, `top-k=4` 계열의
       anti-saturation loss를 켠다.
   - 기대 효과:
     - 초반 positive embedding 형성은 유지한다.
     - 후반에 hard negative score saturation만 누른다.

3. controlled near-miss hard negative 추가
   - 목적: loss만으로 부족한 near-miss 분리를 데이터로 직접 보강한다.
   - 우선순위:
     - 먼저 generated/non-device near-miss hard negatives.
     - 그 다음 필요하면 `device_split/train_hard_negative`를 직접 학습에 추가.
   - 주의:
     - `device_split/holdout_*`는 평가용으로 유지한다.
     - 실제 wav/checkpoint/log는 git에 커밋하지 않는다.

4. calibration layer 검토
   - 목적: cosine score를 openWakeWord sigmoid처럼 더 해석 가능한 점수로 바꾼다.
   - 단, 현재는 positive/hard negative 분리 자체가 부족하므로 calibration은
     성능 개선 이후에 적용한다.

보류할 작업:

- 현재 방식의 `deviceaug` 재시도는 우선순위를 낮춘다.
  - 이번 feature-domain profile augmentation은 score saturation을 줄이지 못했다.
  - 다시 한다면 waveform-level window/noise mixing이나 더 정교한 device impulse
    approximation으로 재설계해야 한다.
- positive device 녹음을 학습 데이터로 직접 넣는 실험은 뒤로 미룬다.
  - 현재 EdgeSpot 구조에서는 positive device sample은 support/enrollment로 쓰는
    것이 자연스럽다.
  - 직접 학습에 넣으면 평가 누수와 실제 사용자 등록 시나리오가 섞일 수 있다.

## 2026-06-22 AIHub Training Noise Scale 결과

목적:

- 기존 `noise50far-margin-confusable-tau4`에서 AIHub training 소음환경 데이터를
  더 많이 붙였을 때 성능이 좋아지는지 확인했다.
- 비교 모델:
  - `50far`: 기존 best 후보
  - `plus50`: 추가 training 소음환경 약 +50GB
  - `plus100`: 추가 training 소음환경 약 +100GB
- 모두 `margin+confusable`, `tau4`, 30 epoch 계열이다.

평가 결과:

| model | eval | AUC | Recall@FAR0.1% | Recall@FAR1% | Recall@FAR5% |
|---|---:|---:|---:|---:|---:|
| 50far | device split | 0.7818 | 0.6000 | 0.6000 | 0.6000 |
| plus50 | device split | 0.5000 | 0.1000 | 0.1000 | 0.1000 |
| plus100 | device split | 0.5545 | 0.0000 | 0.0000 | 0.0000 |
| 50far | all45 | 0.5540 | 0.1000 | 0.1000 | 0.1500 |
| plus50 | all45 | 0.5800 | 0.0000 | 0.0000 | 0.1000 |
| plus100 | all45 | 0.5680 | 0.0000 | 0.0000 | 0.0500 |
| 50far | large | 0.9349 | 0.5201 | 0.8652 | 0.9007 |
| plus50 | large | 0.9469 | 0.3168 | 0.8463 | 0.9007 |
| plus100 | large | 0.9474 | 0.1631 | 0.6359 | 0.8652 |

해석:

- 대량 negative 추가는 large general AUC를 올렸다.
  - `50far` large AUC 0.9349에서 `plus50` 0.9469, `plus100` 0.9474로 상승했다.
- 하지만 실제로 중요한 device split recall은 크게 나빠졌다.
  - `50far`는 Recall@FAR1% 0.6000이다.
  - `plus50`은 0.1000, `plus100`은 0.0000이다.
- 따라서 단순히 AIHub 소음환경 명령어를 대량 negative로 덧붙이는 방향은
  운영 후보로 부적합하다.
- 현재 운영 후보 기준은 여전히 `noise50far-margin-confusable-tau4`가 가장 낫다.

후속 방향:

- `토닥아/토닥이` positive를 직접 증강하지 않는다.
  - 호출어가 바뀔 수 있으므로 특정 호출어에 종속된 학습을 피한다.
- AIHub JSON의 `beginOfSpeech`, `speechLength`, `endOfSpeech` 기준으로 발화 구간을
  잘라 사용한다.
- AIHub 발화 구간은 wake word positive가 아니라 command auxiliary objective로 사용한다.
  - 같은 command label의 여러 소음/거리/기기 조건 녹음은 가깝게 만든다.
  - 다른 command label은 분리한다.
- device 녹음 negative는 학습 샘플로 직접 넣지 않고, device-like augmentation profile
  추정용으로만 사용한다.
  - 목적은 특정 발화를 외우는 것이 아니라 기기 coloration/level/noise 변화에 덜 흔들리는
    embedding을 만드는 것이다.

체크포인트별 짧은 해석:

| checkpoint | 역할 | 해석 |
|---|---|---|
| `edgespot-ko-proxy-noise50far-margin-confusable-tau4` | 현재 baseline / 운영 후보 | device split과 large general의 균형이 가장 낫다. 대량 추가 실험 전 기준점으로 유지한다. |
| `edgespot-ko-proxy-noise50far-margin-confusable-antisat-w01-m097-k4-tau4` | anti-saturation 후보 | hard negative score saturation을 누르려 했지만 device recall이 크게 개선되지는 않았다. |
| `edgespot-ko-proxy-noise50far-clean-margin-confusable-tau4` | AIHub training-only 비교군 | 기존 hard-negative 구성 없이 AIHub training noise만 붙인 비교군이다. 운영 후보보다는 ablation 성격이다. |
| `edgespot-ko-proxy-noise50far-clean-margin-confusable-antisat-w01-m097-k4-tau4` | clean + anti-saturation 비교군 | large AUC는 좋았지만 device split이 약하다. general ranking과 device robustness가 분리된다는 근거로 본다. |
| `edgespot-ko-proxy-noise-plus50-margin-confusable-tau4` | 대량 negative +50GB 실험 | large AUC는 올랐지만 device split recall이 무너졌다. 단순 negative 확장은 부적합하다. |
| `edgespot-ko-proxy-noise-plus100-margin-confusable-tau4` | 대량 negative +100GB 실험 | +50GB보다 더 강하게 device recall이 악화됐다. 데이터 양 증가만으로 해결되지 않는다는 실패 사례다. |
| `edgespot-ko-aihub-command-aux-tau4` | command auxiliary pretrain 후보 | 같은 AIHub command끼리 뭉치게 학습한 encoder 초기화 후보이다. 그대로 운영하기에는 AIHub noise-only negative score가 높다. |
| `edgespot-ko-aihub-command-aux-deviceaug-tau4` | command auxiliary + device-like pretrain 후보 | command auxiliary에 device-like augmentation을 더한 pretrain 후보이다. 기기 불변 표현을 기대했지만 그대로 쓰면 false accept 위험이 커 fine-tune 전제가 필요하다. |
