# facemoment-face-detect

얼굴 검출 extractor. InsightFace SCRFD 백엔드.

## ML 의존성

- `insightface>=0.7.3`
- `onnxruntime-gpu>=1.16.0` (GPU 가속)

Production에서 별도 venv 권장 (onnxruntime CPU와 충돌 방지).

## 핵심 파일

| 파일 | 역할 |
|------|------|
| `src/.../extractors/face_detect.py` | FaceDetectionExtractor |
| `src/.../extractors/backends/insightface.py` | InsightFaceSCRFD 백엔드 |

## Processing Steps

```
detect → tracking → roi_filter
```

- **detect**: SCRFD로 얼굴 bbox + head pose 검출
- **tracking**: face_id 기반 추적 (프레임 간 동일 인물 매칭)
- **roi_filter**: ROI 영역 내 얼굴만 필터링

## Output

`FaceDetectOutput` (정의: facemoment core의 `outputs.py`):
- `faces`: List[FaceObservation] — bbox, confidence, head_pose
- `detected_faces`: List[DetectedFace] — face_id, bbox, area
- `image_size`: tuple[int, int]

## Entry Point

```toml
[project.entry-points."visualpath.extractors"]
face_detect = "facemoment.moment_detector.extractors.face_detect:FaceDetectionExtractor"
```
