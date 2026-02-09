# FaceMoment

981파크 Portrait981 파이프라인의 **얼굴/장면 분석 앱**.
visualpath 프레임워크 기반으로 얼굴 감지, 표정 분석, 포즈 추정, 하이라이트 순간 감지를 수행.
GR차량 시나리오 특화 기능 포함.

## 디렉토리 구조 (core 패키지)

```
src/facemoment/
├── __init__.py                # fm.run() high-level API
├── main.py                    # run(), Result, ANALYZERS, FUSIONS 상수
├── cli/
│   ├── __init__.py            # main(), argparse
│   ├── utils.py               # visualbase 호환성 레이어
│   └── commands/
│       ├── info.py            # facemoment info
│       ├── debug.py           # facemoment debug
│       ├── process.py         # facemoment process
│       └── benchmark.py       # facemoment benchmark
├── pipeline/
│   ├── config.py              # AnalyzerConfig, PipelineConfig
│   ├── orchestrator.py        # PipelineOrchestrator (Distributed 모드)
│   └── pathway_pipeline.py    # FacemomentPipeline (Pathway 통합)
├── moment_detector/
│   ├── detector.py            # MomentDetector (Library 모드)
│   ├── visualize/             # DebugVisualizer, 타이밍 오버레이, stats_panel
│   ├── analyzers/
│   │   ├── __init__.py        # Lazy import, Output 타입 re-export
│   │   ├── base.py            # Module, @processing_step, ProcessingStep
│   │   ├── outputs.py         # FaceDetectOutput, ExpressionOutput 등
│   │   ├── types.py           # FaceObservation, DetectedFace 등
│   │   ├── quality.py         # QualityAnalyzer
│   │   ├── face_classifier.py # FaceClassifierAnalyzer
│   │   ├── dummy.py           # DummyAnalyzer
│   │   └── source.py          # SourceProcessor, BackendPreprocessor
│   ├── fusion/
│   │   ├── highlight.py       # HighlightFusion
│   │   └── dummy.py           # DummyFusion
│   └── scoring/               # FrameScorer
├── observability/
│   ├── __init__.py            # ObservabilityHub
│   ├── records.py             # TriggerFireRecord 등
│   └── sinks.py               # MemorySink, ConsoleSink
└── process/
    ├── __init__.py            # re-export from visualpath
    └── mappers.py             # FacemomentMapper
```

ML 의존성이 필요한 analyzer는 별도 패키지로 분리됨 (face_detect, expression, face, pose, gesture).

## High-Level API

```python
import facemoment as fm

result = fm.run("video.mp4")
result = fm.run("video.mp4", fps=10, cooldown=3.0, backend="pathway", output_dir="./clips")
```

설정 상수: `DEFAULT_FPS=10`, `DEFAULT_COOLDOWN=2.0`, `DEFAULT_BACKEND="pathway"`

## A-B*-C-A 파이프라인

```
A: Video Input (visualbase)
     │ Frame
     ▼
B* Analyzers (VenvWorker/InlineWorker)
     face.detect ─deps─▶ face.expression, face.classify
     body.pose, hand.gesture, frame.quality
     │ Observations
     ▼
C: HighlightFusion (main_only=True)
     │ Trigger
     ▼
A: Clip Output → clips/highlight_001.mp4
```

## deps 패턴

Analyzer 간 데이터 전달. `depends` 선언 → 실행 시 `deps` dict로 이전 결과 수신:

```python
class ExpressionAnalyzer(Module):
    depends = ["face.detect"]

    def process(self, frame, deps=None):
        face_obs = deps["face.detect"] if deps else None
```

모든 실행 경로 (Pathway, Simple, Worker, VenvWorker ZMQ)에서 동일한 deps 누적 패턴 적용.
의존성 순서는 Path 초기화 시 자동 검증.

## Analyzer 전체 목록

| Analyzer | name | depends | 패키지 | Backend | Steps |
|----------|------|---------|--------|---------|-------|
| FaceDetectionAnalyzer | `face.detect` | - | vpx-face-detect | InsightFace SCRFD | detect → tracking → roi_filter |
| ExpressionAnalyzer | `face.expression` | face.detect | vpx-expression | HSEmotion | expression → aggregation |
| FaceClassifierAnalyzer | `face.classify` | face.detect | facemoment (core) | 내장 로직 | track_update → classify → role_assignment |
| FaceAnalyzer | `face` | - | vpx-face | InsightFace + HSEmotion | detect → expression/tracking → roi_filter |
| PoseAnalyzer | `body.pose` | - | vpx-pose | YOLO-Pose | pose_estimation → hands_raised/wave → aggregation |
| GestureAnalyzer | `hand.gesture` | - | vpx-gesture | MediaPipe Hands | hand_detection → gesture_classification → aggregation |
| QualityAnalyzer | `frame.quality` | - | facemoment (core) | OpenCV | grayscale → blur/brightness/contrast → quality_gate |
| DummyAnalyzer | `mock.dummy` | - | facemoment (core) | - | 테스트용 |

## FaceClassifierAnalyzer

탑승자 역할 분류 (core 패키지, ML 의존성 없음). depends=["face.detect"].

| 역할 | 조건 | 인원 |
|------|------|------|
| `main` | 안정적 위치 + 큰 얼굴 | 정확히 1명 |
| `passenger` | 안정적 위치 + 두 번째 후보 | 0~1명 |
| `transient` | 위치 변화 큼 / 짧은 등장 | 0~N명 |
| `noise` | 작은 얼굴 / 낮은 confidence | 0~N명 |

점수 가중치: 위치 안정성 40%, 얼굴 크기 30%, 프레임 중앙 20%, 프레임 내부 10%.

## 트리거 유형

| 트리거 | 소스 | 설명 |
|--------|------|------|
| expression_spike | FaceAnalyzer | 표정 급변 |
| head_turn | FaceAnalyzer | 빠른 머리 회전 |
| hand_wave | PoseAnalyzer | 손 흔들기 |
| camera_gaze | HighlightFusion | 카메라 응시 (gokart) |
| passenger_interaction | HighlightFusion | 동승자 상호작용 (gokart) |
| gesture_vsign | GestureAnalyzer | V사인 (gokart) |
| gesture_thumbsup | GestureAnalyzer | 엄지척 (gokart) |

HighlightFusion `main_only=True` (기본): 주탑승자만 트리거. 동승자 표정/헤드턴 무시.

## 시각화 컬러 코딩

| 역할 | 색상 (BGR) |
|------|------------|
| main | 초록 (0,255,0) 두꺼운 선 |
| passenger | 주황 (0,165,255) |
| transient | 노랑 (0,255,255) |
| noise | 회색 (128,128,128) |

포즈: 상반신 스켈레톤 (COCO 0-10) — 코, 눈, 귀, 어깨, 팔꿈치, 손목. 하늘색 연결선.

## Observability

| Trace Level | 용도 | 오버헤드 |
|-------------|------|----------|
| OFF | 프로덕션 기본 | 0% |
| MINIMAL | Trigger만 | <1% |
| NORMAL | 프레임 요약 + Gate | ~5% |
| VERBOSE | 모든 Signal + 타이밍 | ~15% |

## 백엔드

| 백엔드 | 설명 |
|--------|------|
| `pathway` | Pathway 스트리밍 엔진 (process 기본) |
| `simple` | 순차 실행 fallback |
| inline | 프레임별 순차 (debug 기본, 부드러운 시각화) |

## CLI 명령어

```bash
facemoment info                              # analyzer/backend 상태
facemoment info --deps                       # 의존성 그래프
facemoment info --steps                      # 처리 단계 DAG

facemoment debug video.mp4                   # 모든 analyzer (inline)
facemoment debug video.mp4 -e face           # face만 (+ classifier 자동)
facemoment debug video.mp4 -e pose           # pose만
facemoment debug video.mp4 -e face,pose      # 복수 선택
facemoment debug video.mp4 --no-ml           # dummy 모드
facemoment debug video.mp4 -e face --profile # 성능 프로파일링
facemoment debug video.mp4 --backend pathway # Pathway 엔진 직접
facemoment debug video.mp4 --backend simple  # SimpleDebugSession
facemoment debug video.mp4 --distributed     # 분산 모드
facemoment debug video.mp4 -o out.mp4        # 파일 저장

facemoment process video.mp4 -o ./clips      # 클립 추출
facemoment process video.mp4 --gokart        # GR차량 모드
facemoment process video.mp4 --backend simple
facemoment process video.mp4 --distributed
facemoment process video.mp4 --config pipeline.yaml
facemoment process video.mp4 --trace verbose --trace-output trace.jsonl

facemoment benchmark video.mp4 --frames 100
```

## 테스트

```bash
cd /home/hyeonrae/repo/monolith/portrait981
uv sync --all-packages --all-extras
uv run pytest apps/facemoment/core/tests/ -v
```
