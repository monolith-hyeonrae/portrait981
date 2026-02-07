# facemoment-face

Legacy composite extractor (검출 + 표정을 하나의 extractor로 결합).

**새 코드에서는 `facemoment-face-detect` + `facemoment-expression` 분리 사용 권장.**

## 의존성

- `facemoment-face-detect` + `facemoment-expression` (두 패키지를 합침)

## 핵심 파일

| 파일 | 역할 |
|------|------|
| `src/.../extractors/face.py` | FaceExtractor (composite) |
| `src/.../extractors/backends/face_backends.py` | 통합 백엔드 래퍼 |

## Entry Point

```toml
[project.entry-points."visualpath.extractors"]
face = "facemoment.moment_detector.extractors.face:FaceExtractor"
```
