# EdgeSpot ONNXRuntime Minimal

CPU ONNXRuntime만으로 `encoder.onnx`를 실행하는 최소 Docker 예시입니다.

## Build

```bash
cd /data/users/voice/zoey/todak/impl/EdgeSpot
docker build -t edgespot-ort-minimal deploy/onnxruntime_minimal
```

## Run With Dummy Input

```bash
docker run --rm \
  -v "$PWD/runs:/models:ro" \
  edgespot-ort-minimal \
  --model /models/edgespot-ko-proxy-noise50far-margin-confusable-tau4/encoder.onnx
```

## Run The Other Exported Model

```bash
docker run --rm \
  -v "$PWD/runs:/models:ro" \
  edgespot-ort-minimal \
  --model /models/edgespot-ko-proxy-noise50far-margin-confusable-posdeviceaug-w50-antisat-w01-tau4/encoder.onnx
```

## Run The Paper-Style Candidates

```bash
docker run --rm \
  -v "$PWD/runs:/models:ro" \
  edgespot-ort-minimal \
  --model /models/edgespot-ko-aihub537-paper-distill-noise50far-finetune-tau4/encoder.onnx
```

```bash
docker run --rm \
  -v "$PWD/runs:/models:ro" \
  edgespot-ort-minimal \
  --model /models/edgespot-ko-aihub537-paper-distill-noise50far-bothdevaug-p20-finetune-tau4/encoder.onnx
```

## Real Input

`encoder.onnx` 입력은 log-mel feature입니다.

- shape: `[1, 1, 40, 101]`
- dtype: `float32`

이미 만든 feature를 `.npy`로 저장했다면:

```bash
docker run --rm \
  -v "$PWD/runs:/models:ro" \
  -v "$PWD/tmp:/data:ro" \
  edgespot-ort-minimal \
  --model /models/edgespot-ko-proxy-noise50far-margin-confusable-tau4/encoder.onnx \
  --features-npy /data/features.npy
```

출력은 64차원 L2-normalized embedding입니다. 실제 wake-word 판정은 이 embedding과
support 음성들로 만든 prototype 사이 cosine similarity를 계산해서 threshold와 비교하면 됩니다.

## Optional Prototype Scoring

prototype JSON 형식:

```json
{
  "토닥아": [0.01, 0.02, "... 64 floats ..."],
  "토닥이": [0.03, 0.04, "... 64 floats ..."]
}
```

실행:

```bash
docker run --rm \
  -v "$PWD/runs:/models:ro" \
  -v "$PWD/tmp:/data:ro" \
  edgespot-ort-minimal \
  --model /models/edgespot-ko-proxy-noise50far-margin-confusable-tau4/encoder.onnx \
  --features-npy /data/features.npy \
  --prototype-json /data/prototypes.json
```

## Device Record Wake-Word Demo

`device_record/device_positive_eval`에서 support 음성을 골라 prototype을 만들고,
`device_positive_eval`의 나머지 wav와 `device_hard_negative_eval` wav를 query로 평가합니다.

기본 support는 다음 파일입니다.

- `todak_0.wav`, `todak_2.wav`, `todak_4.wav`, `todak_6.wav`, `todak_8.wav` -> `토닥아`
- `todaki_0.wav`, `todaki_2.wav`, `todaki_4.wav`, `todaki_6.wav`, `todaki_8.wav` -> `토닥이`

기본 query는 기존 device split 평가와 맞춰 홀수 index positive와 홀수 index hard negative를 사용합니다.

실행:

```bash
docker run --rm \
  -v "$PWD/runs:/models:ro" \
  -v "/data/users/voice/zoey/todak/openWakeWord/data/device_record:/device_record:ro" \
  --entrypoint python \
  edgespot-ort-minimal \
  /app/run_wakeword.py \
  --model /models/edgespot-ko-proxy-noise50far-margin-confusable-tau4/encoder.onnx \
  --device-record-dir /device_record \
  --threshold 0.95
```

두 번째 후보 모델:

```bash
docker run --rm \
  -v "$PWD/runs:/models:ro" \
  -v "/data/users/voice/zoey/todak/openWakeWord/data/device_record:/device_record:ro" \
  --entrypoint python \
  edgespot-ort-minimal \
  /app/run_wakeword.py \
  --model /models/edgespot-ko-proxy-noise50far-margin-confusable-posdeviceaug-w50-antisat-w01-tau4/encoder.onnx \
  --device-record-dir /device_record \
  --threshold 0.9663
```

출력:

- 각 wav의 best prototype label
- cosine similarity
- threshold 기준 `ACCEPT` / `reject`
- positive recall
- hard negative false positive rate

레이턴시까지 보려면:

```bash
docker run --rm \
  -v "$PWD/runs:/models:ro" \
  -v "/data/users/voice/zoey/todak/openWakeWord/data/device_record:/device_record:ro" \
  --entrypoint python \
  edgespot-ort-minimal \
  /app/run_wakeword.py \
  --model /models/edgespot-ko-proxy-noise50far-margin-confusable-tau4/encoder.onnx \
  --device-record-dir /device_record \
  --threshold 0.95 \
  --benchmark 100
```

출력되는 latency는 다음을 분리합니다.

- `feature_latency_ms`: wav -> log-mel
- `encoder_latency_ms`: ONNXRuntime encoder
- `total_latency_ms`: feature + encoder
