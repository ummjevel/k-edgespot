# AIHub Post-Augmentation Error Review

## Summary

- Source: `docs/false_accepts_aihub_augmented_target_keyword_k10_far1.jsonl`
- FAR: `0.01`
- Threshold: `0.5995`
- False accepts: `7`
- False rejects: `297`
- Audio review folder: `docs/audio_review_aihub_augmented_target_keyword_far1/audio`
- Review TSV: `docs/audio_review_aihub_augmented_target_keyword_far1/review.tsv`

## False Accepts

- Count in review file: `7`

### Labels / Matches

- True labels: `{'__negative__': 7}`
- Matched labels: `{'weather': 3, 'volume_down': 3, 'music_stop': 1}`

### Top Texts

- `2` x 날씨는 괜찮아? -> [('weather', 2)]
- `1` x 내일 날씨를 확인합니다. -> [('weather', 1)]
- `1` x 다음 달 수업에 뭐 챙겨가야해? -> [('music_stop', 1)]
- `1` x 다음에 배울 거 알려 주세요. -> [('volume_down', 1)]
- `1` x 이제 어린이집 알림이 뭐야? -> [('volume_down', 1)]
- `1` x 월요일 알림에 알았다고 보내 줄래? -> [('volume_down', 1)]

## False Rejects

- Count in review file: `200`

### Labels / Matches

- True labels: `{'music_play': 82, 'weather': 49, 'lights_off': 29, 'lights_on': 22, 'alarm_set': 9, 'music_stop': 5, 'next_track': 2, 'timer_start': 2}`
- Matched labels: `{'volume_down': 99, 'music_stop': 77, 'lights_off': 22, 'volume_up': 1, 'weather': 1}`

### Top Texts

- `8` x 오늘 날씨 어때? -> [('music_stop', 5), ('volume_down', 2), ('lights_off', 1)]
- `7` x 지금 노래 시작해 주라. -> [('volume_down', 5), ('music_stop', 2)]
- `6` x 아까 들은 노래 다시 틀어 줘. -> [('music_stop', 3), ('volume_down', 2), ('lights_off', 1)]
- `5` x 이젠 그만 불 끄자. -> [('music_stop', 4), ('lights_off', 1)]
- `4` x 지금 노래 땡벌 노래 틀어 줘. -> [('volume_down', 2), ('music_stop', 1), ('lights_off', 1)]
- `4` x 고독한 노래 틀어줘. -> [('music_stop', 2), ('volume_down', 2)]
- `4` x 두 시간 후에 조명 다 켜줘. -> [('music_stop', 2), ('volume_down', 2)]
- `4` x 삼십 분 뒤로 알람 맞춰. -> [('volume_down', 3), ('music_stop', 1)]
- `4` x 음악 그만 꺼 놔. -> [('music_stop', 2), ('lights_off', 1), ('volume_down', 1)]
- `4` x 영어 노래 다시 들을래요. -> [('volume_down', 3), ('music_stop', 1)]
- `3` x 안방 조명 다 꺼줘. -> [('music_stop', 1), ('lights_off', 1), ('volume_down', 1)]
- `3` x 십분 간격으로 알람 울려. -> [('music_stop', 2), ('lights_off', 1)]
- `3` x 당근 송 노래 시작. -> [('volume_down', 2), ('music_stop', 1)]
- `2` x 유튜브 뿡뿡이 노래 시작해. -> [('music_stop', 1), ('volume_down', 1)]
- `2` x 내일 용당동 날씨 어때? -> [('music_stop', 1), ('volume_down', 1)]
- `2` x 요번 주 날씨 예보 좀 알려줘 봐. -> [('music_stop', 1), ('volume_down', 1)]
- `2` x 안방 조명 켜줘. -> [('music_stop', 1), ('volume_up', 1)]
- `2` x 이번 주 날씨 어떤지 알려줘. -> [('music_stop', 1), ('volume_down', 1)]
- `2` x 홍길동 노래 재생해 줄래? -> [('music_stop', 1), ('volume_down', 1)]
- `2` x 나 지금 홍길동 음악이 듣고 싶어. -> [('music_stop', 1), ('volume_down', 1)]

## Listening Decisions

Use `review.tsv` to mark one of:

- `keep_negative`: correctly negative; mine as hard negative.
- `map_positive`: should be remapped as a positive command.
- `exclude_unclear`: ambiguous or outside benchmark scope.
- `bad_audio`: audio/content problem.
