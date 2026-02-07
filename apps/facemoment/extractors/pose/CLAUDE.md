# facemoment-pose

포즈 추정 extractor. YOLO-Pose 백엔드.

## ML 의존성

- `ultralytics>=8.0.0` (PyTorch 기반)

## 핵심 파일

| 파일 | 역할 |
|------|------|
| `src/.../extractors/pose.py` | PoseExtractor |
| `src/.../extractors/backends/pose_backends.py` | YOLO-Pose 백엔드 |

## Processing Steps

```
pose_estimation → hands_raised_check / wave_detection → aggregation
```

- **pose_estimation**: COCO 17 keypoints 추출
- **hands_raised_check**: 손 위치가 어깨 위인지 판정
- **wave_detection**: 손 흔들기 동작 감지
- **aggregation**: 프레임 전체 포즈 요약

## 트리거

- `hand_wave`: 손 흔들기 감지 시 발생

## 시각화

상반신 스켈레톤 (COCO keypoints 0-10):
- 머리: 코(흰), 눈(노랑), 귀(노랑)
- 상반신: 어깨(초록), 팔꿈치(노랑), 손목(노랑, 큰 원)
- 연결선: 하늘색

## Entry Point

```toml
[project.entry-points."visualpath.extractors"]
pose = "facemoment.moment_detector.extractors.pose:PoseExtractor"
```
