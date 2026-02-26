# MomentScan

981파크 Portrait981 파이프라인의 **얼굴/장면 분석 앱**.
visualpath 프레임워크 기반으로 얼굴 감지, 표정 분석, 포즈 추정, 하이라이트 순간 감지를 수행.

## 디렉토리 구조

```
src/momentscan/
├── __init__.py                # ms.run() re-export
├── main.py                    # MomentscanApp, run(), Result
├── algorithm/
│   ├── analyzers/
│   │   ├── __init__.py        # Lazy import, Output 타입 re-export
│   │   ├── face_classifier/   # FaceClassifierAnalyzer (역할 분류)
│   │   ├── face_quality/      # FaceQualityAnalyzer (head crop blur/exposure + seg)
│   │   ├── face_baseline/     # FaceBaselineAnalyzer (Welford online stats)
│   │   ├── frame_gate/        # FrameGateAnalyzer (per-face quality gate)
│   │   ├── quality/           # QualityAnalyzer (프레임 전체 blur/brightness)
│   │   └── source.py          # SourceProcessor, BackendPreprocessor
│   ├── batch/                 # Phase 1: 배치 하이라이트 분석
│   │   ├── types.py           # FrameRecord, HighlightConfig, HighlightWindow, HighlightResult
│   │   ├── extract.py         # FlowData → FrameRecord 변환
│   │   └── highlight.py       # BatchHighlightEngine (per-video 정규화 + peak detection)
│   ├── monitoring/            # 파이프라인 성능 모니터링
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
result = ms.run("video.mp4", fps=10, output_dir="./output")
print(f"Found {len(result.highlights)} highlights")
```

설정 상수: `DEFAULT_FPS=10`, `DEFAULT_BACKEND="simple"`

## 배치 하이라이트 파이프라인 (Phase 1)

```
Video Source
     │ Frame (per-frame)
     ▼
Analyzers (DAG):
  face.detect → face.classify → face.baseline (stateful)
       │ → face.expression, face.quality, face.parse, portrait.score, face.au, head.pose
       └→ face.gate (depends: detect+classify, optional: quality+frame.quality+head.pose)
  body.pose, hand.gesture, frame.quality (independent)
     │ Observations → on_frame() → FrameRecord 축적
     ▼
BatchHighlightEngine (after_run, per-video batch)
     │ 1. Numeric feature delta (EMA baseline)
     │ 2. Per-video 정규화 (MAD z-score)
     │ 3. face.gate 판정 읽기 (gate_passed)
     │ 4. Scoring: 0.35×Quality + 0.65×Impact (가산)
     │ 5. Temporal smoothing (EMA, gate-pass only)
     │ 6. Peak detection (scipy.signal.find_peaks)
     │ 7. Window 생성 + best frame 선택
     ▼
HighlightResult (windows.json)
```

MomentscanApp 생명주기:
- `setup()`: `_frame_records` 초기화
- `on_frame(frame, results)`: FlowData → FrameRecord 변환 후 축적
- `after_run(result)`: BatchHighlightEngine.analyze() 실행, Result 반환
- `teardown()`: `_frame_records` 정리

Backend가 FlowGraph를 해석하여 실행:
- **SimpleBackend**: 순차/병렬 실행 (기본)
- **WorkerBackend**: 격리 필요 모듈을 WorkerModule로 래핑 → SimpleBackend 위임
- **PathwayBackend**: 스트리밍 실행

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
| FaceParseAnalyzer | `face.parse` | face.detect | vpx-face-parse | BiSeNet | 19-class face segmentation |
| FaceQualityAnalyzer | `face.quality` | face.detect | momentscan (core) | OpenCV + BiSeNet mask | head crop blur/exposure + seg ratios (face/eye/mouth/hair) |
| FrameGateAnalyzer | `face.gate` | face.detect, face.classify | momentscan (core) | 내장 로직 | per-face quality gate (confidence, blur, exposure, parsing) |
| FaceBaselineAnalyzer | `face.baseline` | face.detect, face.classify | momentscan (core) | 내장 로직 | Welford online stats (STATEFUL) |
| FaceAUAnalyzer | `face.au` | face.detect | vpx-face-au | ONNX | Action Unit 분석 (AU6/12/25/26) |
| HeadPoseAnalyzer | `head.pose` | face.detect | vpx-head-pose | 6DRepNet | 6DoF head pose (yaw/pitch/roll) |
| PortraitScoreAnalyzer | `portrait.score` | face.detect | vpx-portrait-score | CLIP | 4축 aesthetic scoring + composites |
| PoseAnalyzer | `body.pose` | - | vpx-body-pose | YOLO-Pose | pose_estimation → hands_raised/wave → aggregation |
| GestureAnalyzer | `hand.gesture` | - | vpx-hand-gesture | MediaPipe Hands | hand_detection → gesture_classification → aggregation |
| QualityAnalyzer | `frame.quality` | - | momentscan (core) | OpenCV | grayscale → blur/brightness/contrast |

## FaceClassifierAnalyzer

탑승자 역할 분류 (core 패키지, ML 의존성 없음). depends=["face.detect"].

| 역할 | 조건 | 인원 |
|------|------|------|
| `main` | 안정적 위치 + 큰 얼굴 | 정확히 1명 |
| `passenger` | 안정적 위치 + 두 번째 후보 | 0~1명 |
| `transient` | 위치 변화 큼 / 짧은 등장 | 0~N명 |
| `noise` | 작은 얼굴 / 낮은 confidence | 0~N명 |

점수 가중치: camera_proximity 40%, 얼굴 크기 20%, track_length 0.5×, 위치 안정성 20%.

**Single non-noise → main 승격**: non-noise 얼굴이 정확히 1명이고 transient일 때 → main으로 강제 승격.
1인 탑승 시나리오에서 유일한 얼굴이 transient로 남는 것을 방지.
Lock-in: main이 10프레임 이상 연속 등장 시 고정 (LOCK_THRESHOLD=10).

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

# Pipeline monitoring records + PipelineMonitor
from momentscan.algorithm.monitoring import PipelineMonitor
from momentscan.algorithm.monitoring.records import BackpressureRecord, PipelineStatsRecord

# Extended sinks (CLI 프레젠테이션)
from momentscan.cli.sinks import ConsoleSink, MemorySink
```

| Trace Level | 용도 | 오버헤드 |
|-------------|------|----------|
| OFF | 프로덕션 기본 | 0% |
| MINIMAL | 세션 시작/종료 | <1% |
| NORMAL | 프레임 요약 + 타이밍 | ~5% |
| VERBOSE | 모든 Signal + 상세 | ~15% |

## 백엔드

| 백엔드 | 설명 |
|--------|------|
| `simple` | 순차 실행 (기본) |
| `pathway` | Pathway 스트리밍 엔진 |
| `worker` | 격리 필요 모듈을 WorkerModule로 래핑 |

## CLI 명령어

```bash
momentscan info                              # analyzer/backend 상태
momentscan info --deps                       # 의존성 그래프
momentscan info --steps                      # 처리 단계 DAG

momentscan debug video.mp4                   # 모든 analyzer
momentscan debug video.mp4 -e face           # face만 (+ classifier 자동)
momentscan debug video.mp4 -e pose           # pose만
momentscan debug video.mp4 -e face,pose      # 복수 선택
momentscan debug video.mp4 --backend simple  # SimpleBackend
momentscan debug video.mp4 --distributed     # 분산 모드
momentscan debug video.mp4 -o out.mp4        # 파일 저장

momentscan process video.mp4 -o ./output     # 하이라이트 분석 + 결과 출력
momentscan process video.mp4 --backend simple
momentscan process video.mp4 --distributed
momentscan process video.mp4 --trace verbose --trace-output trace.jsonl
```

## 테스트

```bash
cd /home/hyeonrae/repo/monolith/portrait981
uv sync --all-packages --all-extras
uv run pytest apps/momentscan/tests/ -v
```
