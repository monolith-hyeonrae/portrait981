# State Timeline

## 문서의 목적

이 문서는 p981-state가 생성하는 상태 타임라인의 구조를 정의한다.

---

## 입력

- stage가 전달하는 media_handle 또는 frame_source
- sampling fps/구간은 stage(또는 state 요구)에서 결정한다.
- state는 제공된 frame_source를 소비하며 원본 video_ref를 직접 읽지 않는다.
- 모든 points는 media_handle의 video time(ms) 기준이며, 다른 fps 결과와 결합할 때는 timestamp_ms로 정합한다.

---

## 출력

- state_timeline_ref (asset)
- 상태 타임라인은 meta에 저장된다.

---

## 관측 항목 (MVP)

- emotion: happy/angry/neutral 확률
- hand: 손 위치/상태
- quality: 얼굴/프레임 품질 점수

---

## 공통 포맷 (권장)

- 각 track은 track_id + points[]로 구성한다.
- points는 t_ms(정수), value, confidence를 포함한다.
- t_ms는 video time 기준의 ms 정수다.

### emotion/quality 예시

```
{
  "track_id": "emotion.happy",
  "points": [
    {"t_ms": 0, "value": 0.82, "confidence": 0.9}
  ]
}
```

---

## hand 트랙 (MVP)

- 좌표는 normalized [0..1] 기준이다.
- state는 down | up | unknown 으로 제한한다.

```
{
  "track_id": "hand",
  "points": [
    {
      "t_ms": 0,
      "hands": [
        {
          "hand": "left",
          "present": true,
          "conf": 0.9,
          "center_x": 0.4,
          "center_y": 0.6,
          "bbox_w": 0.1,
          "bbox_h": 0.2,
          "state": "up"
        }
      ]
    }
  ]
}
```

- hands는 최대 2개(좌/우)까지 포함할 수 있다.
- bbox_w/bbox_h는 없으면 null로 둘 수 있다.
