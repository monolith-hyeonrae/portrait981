# MomentScan

981파크 Portrait981 파이프라인의 **얼굴/장면 분석 앱**.
visualpath 프레임워크 기반으로 얼굴 감지, 표정 분석, 포즈 추정, 하이라이트 순간 감지를 수행.
GR차량 시나리오 특화 기능 포함.

## 디렉토리 구조

```
src/momentscan/
├── __init__.py                # ms.run() re-export
├── main.py                    # MomentscanApp, run(), Result
├── cli/
│   ├── __init__.py            # main(), argparse
│   ├── debug_handler.py       # debug 프레임 핸들러
│   ├── utils.py               # visualbase 호환성, 노이즈 억제
│   └── commands/
│       ├── info.py            # momentscan info
│       ├── debug.py           # momentscan debug
│       └── process.py         # momentscan process
├── algorithm/
│   ├── analyzers/
│   │   ├── __init__.py        # Lazy import, Output 타입 re-export
│   │   ├── face_classifier/   # vpx 플러그인 구조 (momentscan 전용)
│   │   │   ├── __init__.py    # re-export: FaceClassifierAnalyzer, ClassifiedFace, FaceClassifierOutput
│   │   │   ├── analyzer.py    # FaceClassifierAnalyzer
│   │   │   ├── output.py      # FaceClassifierOutput
│   │   │   └── types.py       # ClassifiedFace
│   │   ├── quality/           # vpx 플러그인 구조 (momentscan 전용)
│   │   │   ├── __init__.py    # re-export: QualityAnalyzer, QualityOutput
│   │   │   ├── analyzer.py    # QualityAnalyzer
│   │   │   └── output.py      # QualityOutput
│   │   ├── highlight/         # HighlightFusion + trigger/gate records
│   │   │   ├── __init__.py    # re-export: HighlightFusion, records
│   │   │   ├── analyzer.py    # HighlightFusion
│   │   │   ├── records.py     # TriggerFireRecord, GateChangeRecord 등
│   │   │   └── types.py       # ExpressionState, AdaptiveEmotionState
│   │   ├── frame_scoring/     # FrameScoringAnalyzer, FrameScorer
│   │   └── source.py          # SourceProcessor, BackendPreprocessor
│   ├── monitoring/            # 파이프라인 성능 모니터링
│   │   ├── __init__.py        # re-export: PipelineMonitor, records
│   │   ├── monitor.py         # PipelineMonitor
│   │   └── records.py         # PathwayFrameRecord, BackpressureRecord 등
│   ├── frame_selector.py      # FrameSelector
│   └── source.py              # SourceProcessor
├── cli/
│   ├── __init__.py            # main(), argparse
│   ├── sinks.py               # ConsoleSink, MemorySink (확장)
│   ├── debug_handler.py       # debug 프레임 핸들러
│   ├── utils.py               # visualbase 호환성, 노이즈 억제
│   └── commands/
│       ├── info.py            # momentscan info
│       ├── debug.py           # momentscan debug
│       └── process.py         # momentscan process
├── visualize/                 # DebugVisualizer, 타이밍 오버레이, stats_panel
```

ML 의존성이 필요한 analyzer는 별도 패키지로 분리됨 (vpx-face-detect, vpx-face-expression, vpx-body-pose, vpx-hand-gesture).

## High-Level API

```python
import momentscan as ms

result = ms.run("video.mp4")
result = ms.run("video.mp4", fps=10, cooldown=3.0, backend="pathway", output_dir="./clips")
```

설정 상수: `DEFAULT_FPS=10`, `DEFAULT_COOLDOWN=2.0`, `DEFAULT_BACKEND="pathway"`

## FlowGraph 파이프라인

```
Source: Video Input (visualbase)
     │ Frame
     ▼
PathNode: Modules (inline / Worker 격리)
     face.detect ─deps─▶ face.expression, face.classify
     body.pose, hand.gesture, frame.quality
     │ Observations
     ▼
     HighlightFusion (main_only=True)
     │ Trigger
     ▼
on_trigger → Clip Output (clips/highlight_001.mp4)
```

Backend가 FlowGraph를 해석하여 실행:
- **SimpleBackend**: 순차/병렬 실행 (개발, 디버그)
- **WorkerBackend**: 격리 필요 모듈을 WorkerModule로 래핑 → SimpleBackend 위임
- **PathwayBackend**: 스트리밍 실행 (실시간 처리)

## deps 패턴

Analyzer 간 데이터 전달. `depends` 선언 → 실행 시 `deps` dict로 이전 결과 수신:

```python
class ExpressionAnalyzer(Module):
    depends = ["face.detect"]

    def process(self, frame, deps=None):
        face_obs = deps["face.detect"] if deps else None
```

모든 Backend (Simple, Worker, Pathway)에서 동일한 deps 누적 패턴 적용.
의존성 순서는 Path 초기화 시 자동 검증.

## Analyzer 전체 목록

| Analyzer | name | depends | 패키지 | Backend | Steps |
|----------|------|---------|--------|---------|-------|
| FaceDetectionAnalyzer | `face.detect` | - | vpx-face-detect | InsightFace SCRFD | detect → tracking → roi_filter |
| ExpressionAnalyzer | `face.expression` | face.detect | vpx-face-expression | HSEmotion | expression → aggregation |
| FaceClassifierAnalyzer | `face.classify` | face.detect | momentscan (core) | 내장 로직 | track_update → classify → role_assignment |
| PoseAnalyzer | `body.pose` | - | vpx-body-pose | YOLO-Pose | pose_estimation → hands_raised/wave → aggregation |
| GestureAnalyzer | `hand.gesture` | - | vpx-hand-gesture | MediaPipe Hands | hand_detection → gesture_classification → aggregation |
| QualityAnalyzer | `frame.quality` | - | momentscan (core) | OpenCV | grayscale → blur/brightness/contrast → quality_gate |

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

`visualpath.observability`가 코어 (Hub, TraceLevel, base sinks/records).
momentscan는 records와 sinks만 확장:

```python
# Core (visualpath)
from visualpath.observability import ObservabilityHub, TraceLevel
from visualpath.observability.sinks import FileSink

# Trigger/Gate records (emitter: highlight/analyzer.py)
from momentscan.algorithm.analyzers.highlight.records import TriggerFireRecord, GateChangeRecord

# Pipeline monitoring records + PipelineMonitor
from momentscan.algorithm.monitoring import PipelineMonitor
from momentscan.algorithm.monitoring.records import BackpressureRecord, PipelineStatsRecord

# Extended sinks (CLI 프레젠테이션)
from momentscan.cli.sinks import ConsoleSink, MemorySink
```

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
momentscan info                              # analyzer/backend 상태
momentscan info --deps                       # 의존성 그래프
momentscan info --steps                      # 처리 단계 DAG

momentscan debug video.mp4                   # 모든 analyzer (inline)
momentscan debug video.mp4 -e face           # face만 (+ classifier 자동)
momentscan debug video.mp4 -e pose           # pose만
momentscan debug video.mp4 -e face,pose      # 복수 선택
momentscan debug video.mp4 -e face --profile # 성능 프로파일링
momentscan debug video.mp4 --backend pathway # Pathway 엔진 직접
momentscan debug video.mp4 --backend simple  # SimpleDebugSession
momentscan debug video.mp4 --distributed     # 분산 모드
momentscan debug video.mp4 -o out.mp4        # 파일 저장

momentscan process video.mp4 -o ./clips      # 클립 추출
momentscan process video.mp4 --backend simple
momentscan process video.mp4 --distributed
momentscan process video.mp4 --trace verbose --trace-output trace.jsonl
```

Analyzer 개별 성능 측정은 `vpx run`을 사용:

```bash
vpx run face.detect --input video.mp4 --max-frames 100
vpx run face.detect,face.expression --input video.mp4 --fps 5
```

새 모듈 스캐폴딩은 `vpx new`를 사용:

```bash
# vpx 플러그인 (libs/vpx/plugins/ 아래 독립 패키지)
vpx new face.landmark                          # 기본 구조 + backends/
vpx new face.landmark --depends face.detect    # 의존 모듈 지정
vpx new face.landmark --no-backend             # backends/ 생략

# momentscan 내부 모듈 (algorithm/analyzers/ 아래)
vpx new scene.transition --internal
vpx new scene.transition --internal --depends face.detect

# 미리보기
vpx new face.landmark --dry-run
```

## 테스트

```bash
cd /home/hyeonrae/repo/monolith/portrait981
uv sync --all-packages --all-extras
uv run pytest apps/momentscan/tests/ -v
```
