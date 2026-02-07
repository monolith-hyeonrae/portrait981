# facemoment-expression

표정 분석 extractor. HSEmotion 백엔드 (PyFeat 대체 백엔드 포함).

## 의존성

- `depends = ["face_detect"]` — FaceDetectionExtractor 결과 필요
- `hsemotion-onnx>=0.3` (onnxruntime CPU 사용)

Production에서 별도 venv 권장 (onnxruntime-gpu와 충돌 방지).

## 핵심 파일

| 파일 | 역할 |
|------|------|
| `src/.../extractors/expression.py` | ExpressionExtractor |
| `src/.../extractors/backends/hsemotion.py` | HSEmotionBackend |
| `src/.../extractors/backends/pyfeat.py` | PyFeatBackend (대체) |

## Processing Steps

```
expression → aggregation
```

- **expression**: 검출된 얼굴별 감정 분류 (happy, sad, angry, surprise 등)
- **aggregation**: 프레임 전체 표정 요약

## Output

`ExpressionOutput` (정의: facemoment core의 `outputs.py`):
- 얼굴별 감정 확률, 지배 감정, 변화량

## Entry Point

```toml
[project.entry-points."visualpath.extractors"]
expression = "facemoment.moment_detector.extractors.expression:ExpressionExtractor"
```
