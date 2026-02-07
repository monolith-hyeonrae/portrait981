# facemoment-gesture

제스처 감지 extractor. MediaPipe Hands 백엔드.

## ML 의존성

- `mediapipe>=0.10.32`

## 핵심 파일

| 파일 | 역할 |
|------|------|
| `src/.../extractors/gesture.py` | GestureExtractor |
| `src/.../extractors/backends/hand_backends.py` | MediaPipe Hands 백엔드 |

## Processing Steps

```
hand_detection → gesture_classification → aggregation
```

- **hand_detection**: 손 영역 검출 (MediaPipe)
- **gesture_classification**: 손 랜드마크 기반 제스처 분류
- **aggregation**: 프레임 전체 제스처 요약

## 트리거

- `gesture_vsign`: V사인 (GR차량 시나리오)
- `gesture_thumbsup`: 엄지척 (GR차량 시나리오)

## Entry Point

```toml
[project.entry-points."visualpath.extractors"]
gesture = "facemoment.moment_detector.extractors.gesture:GestureExtractor"
```
