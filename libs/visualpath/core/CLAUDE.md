# VisualPath - Claude Session Context

> 최종 업데이트: 2026-02-07
> 상태: **4패키지 분리 완료 (visualpath, visualpath-isolation, visualpath-pathway, visualpath-cli)**
> 참고: visualpath-isolation = Worker 격리(venv/process) + IPC + Observability

## 프로젝트 역할

**범용 영상 분석 프레임워크** (재사용 가능):
- **통합 Module 인터페이스** (Analyzer/Fusion 통합 완료)
- **통합 Observation 출력** (모든 모듈은 Observation 반환)
- **NodeSpec 기반 선언적 노드 정의** (ModuleSpec 등)
- Worker 격리 실행 (venv, process, thread)
- Plugin discovery (entry_points 기반)
- Observability (트레이싱, 로깅)
- 파이프라인 실행 엔진
- **실행 백엔드 추상화** (Simple, Pathway)

## 아키텍처 위치

```
┌─────────────────────────────────────────────────────────┐
│  범용 레이어                                             │
│  ┌─────────────┐      ┌─────────────┐                   │
│  │ visualbase  │ ───→ │ visualpath  │ ← core 패키지     │
│  │ (미디어 I/O)│      │ (분석 프레임워크)               │
│  └─────────────┘      └──────┬──────┘                   │
│                              │                           │
│        ┌─────────────────────┼───────────────────┐       │
│        │                     │                   │       │
│  ┌──────┴──────┐ ┌──────────┴──────┐  ┌─────────┴───┐  │
│  │ vp-isolation│ │  vp-pathway     │  │  vp-cli     │  │
│  │ (Worker격리)│ │  (Streaming)    │  │  (CLI/설정) │  │
│  └────────────┘  └─────────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────┘
                              │
                    앱에서 사용 (momentscan 등)
```

### 패키지 분리 구조

| 패키지 | 설명 | 의존성 |
|--------|------|--------|
| `visualpath` | core + flow + simple backend + api | visualbase |
| `visualpath-isolation` | Worker/IPC + observability | visualpath, **visualbase.ipc** |
| `visualpath-pathway` | Pathway 스트리밍 백엔드 | visualpath, visualpath-isolation |
| `visualpath-cli` | CLI + config | visualpath, visualpath-isolation |

모든 import 경로는 기존과 동일 (`pkgutil.extend_path` 사용):
```python
from visualpath.core.module import Module          # ← visualpath
from visualpath.process.launcher import VenvWorker # ← visualpath-isolation
from visualpath.backends.pathway import PathwayBackend  # ← visualpath-pathway
from visualpath.cli import main                    # ← visualpath-cli
```

## 핵심 제공 기능

| 모듈 | 기능 | 설명 |
|------|------|------|
| `core.Module` | **통합 모듈 ABC** | Observation 반환 (통합) |
| `core.Observation` | **통합 결과 컨테이너** | 분석 + 트리거 결과 (signals에 trigger info) |
| `core.DummyAnalyzer` | 테스트용 모듈 | 간단한 테스트용 |
| `core.IsolationLevel` | 격리 수준 | INLINE, THREAD, PROCESS, VENV |
| `flow.specs.*` | **NodeSpec 선언적 스펙** | ModuleSpec 등 17종 |
| `process.WorkerLauncher` | Worker 팩토리 | 격리 수준별 Worker 생성 |
| `process.VenvWorker` | venv 격리 Worker | ML 의존성 충돌 해결 |
| `plugin.discover_*` | 플러그인 검색 | entry_points 기반 |
| `observability` | 트레이싱 | TraceLevel, Sink |
| `backends.ExecutionBackend` | 실행 백엔드 ABC | 파이프라인 실행 추상화 |
| `backends.SimpleBackend` | 순차 실행 | 기본 백엔드 |
| `backends.PathwayBackend` | 스트리밍 실행 | Pathway 기반 (옵션) |

## 디렉토리 구조

### visualpath (core) — ~5,800줄
```
visualpath/
├── __init__.py, _version.py, api.py, runner.py
├── core/
│   ├── module.py, observation.py, isolation.py, path.py
├── flow/
│   ├── node.py, specs.py, graph.py, builder.py, viz.py
│   └── nodes/ (source, path, filter, sampler, branch, join)
├── backends/
│   ├── base.py, protocols.py
│   └── simple/backend.py, interpreter.py, executor.py
└── plugin/discovery.py
```

### visualpath-isolation (Worker 격리 + IPC + Observability) — ~5,600줄
```
visualpath-isolation/src/visualpath/
├── process/
│   ├── worker.py, launcher.py, ipc.py
│   ├── mapper.py, orchestrator.py, serialization.py
└── observability/
    ├── __init__.py, records.py, sinks.py
```

### visualpath-pathway — ~2,800줄
```
visualpath-pathway/src/visualpath/backends/pathway/
├── backend.py, connector.py, converter.py, operators.py, stats.py
```

### visualpath-cli — ~2,000줄
```
visualpath-cli/src/visualpath/
├── cli/ (commands: debug, run, validate, plugins, version)
└── config/ (schema, loader)
```

## 사용 예시

### Module API (권장)

```python
from visualpath.core import Module, Observation
from visualpath.flow import FlowGraphBuilder
from visualpath.backends.simple import GraphExecutor
from visualbase import Trigger

# Analyzer 모듈: Observation 반환
class FaceDetector(Module):
    depends = []  # 의존성 없음

    @property
    def name(self) -> str:
        return "face_detect"

    def process(self, frame, deps=None) -> Observation:
        faces = self._detect(frame.data)
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"face_count": len(faces)},
        )

# Trigger 모듈: Observation with trigger info in signals
class SmileTrigger(Module):
    depends = ["face_detect"]  # face_detect 의존

    @property
    def name(self) -> str:
        return "smile_trigger"

    def process(self, frame, deps=None) -> Observation:
        face_obs = deps.get("face_detect") if deps else None
        if not face_obs or face_obs.signals.get("face_count", 0) == 0:
            return Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={"should_trigger": False},
            )
        trigger = Trigger.point(
            event_time_ns=frame.t_src_ns,
            pre_sec=2.0, post_sec=2.0,
            label="smile",
        )
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={
                "should_trigger": True,
                "trigger_score": 0.9,
                "trigger_reason": "smile_detected",
            },
            metadata={"trigger": trigger},
        )

# FlowGraph 빌드
graph = (FlowGraphBuilder()
    .source("frames")
    .path("analysis", modules=[FaceDetector(), SmileTrigger()])
    .on_trigger(lambda data: print(f"Trigger: {data}"))
    .build())

# 실행
with GraphExecutor(graph) as executor:
    for frame in video:
        executor.process(frame)
```

### 간단한 API

```python
import visualpath as vp

# 비디오 처리 (modules 필수)
result = vp.process_video("video.mp4", modules=[face_detector, smile_trigger])
print(f"Found {len(result.triggers)} triggers")
```

## 실행 백엔드

```python
import visualpath as vp

# 기본 (SimpleBackend)
result = vp.process_video("video.mp4", modules=[face_detector, smile_trigger])

# Pathway 백엔드 (스트리밍)
# pip install visualpath[pathway] 필요
result = vp.process_video("video.mp4", modules=[face_detector], backend="pathway")
```

### 백엔드 비교

| 백엔드 | 특징 | 용도 |
|--------|------|------|
| **SimpleBackend** | 모듈식 컴포넌트, 순차/병렬 처리 | 로컬 비디오, 개발/디버깅, 배치 처리 |
| **PathwayBackend** | 스트리밍, 백프레셔, 워터마크 | 실시간 처리, 복잡한 동기화 |

## on_trigger 콜백 (비즈니스 로직은 앱에서)

```python
# visualpath는 Action을 모름 - 콜백만 제공
def run_pipeline(source, on_trigger: Callable[[Trigger], None]):
    for frame in source:
        result = fusion.process(frame, deps)
        if result.should_trigger:
            on_trigger(result.metadata["trigger"])  # 앱에서 처리

# 앱(momentscan)에서 Action 정의
def handle_trigger(trigger):
    clip = clipper.extract(trigger.frame_id)
    save_clip(clip)
```

## 테스트

```bash
# 각 패키지별 독립 테스트 (모두 visualpath venv에서 실행)
cd ~/repo/monolith/visualpath
uv run pytest tests/ -v                                      # core (267 tests)
uv run pytest ../visualpath-isolation/tests/ -v                 # isolation (135 tests)
uv run pytest ../visualpath-pathway/tests/ -v                 # pathway (80 tests)
uv run pytest ../visualpath-cli/tests/ -v                     # cli (45 tests)

# momentscan 통합 테스트
cd ~/repo/monolith/momentscan && uv run pytest tests/ -v     # 329 tests
```

## 의존성

- `visualpath`: visualbase, numpy (코어만)
- `visualpath-isolation`: visualpath + pyzmq(옵션)
- `visualpath-pathway`: visualpath, visualpath-isolation, pathway>=0.8.0
- `visualpath-cli`: visualpath, visualpath-isolation, pyyaml, pydantic

```bash
# 개발 환경 설정 (모든 패키지 editable 설치)
cd ~/repo/monolith/visualpath
uv pip install -e . -e ../visualpath-isolation -e ../visualpath-pathway -e ../visualpath-cli
```

## 문서

문서는 repo root의 `docs/`에 통합 관리:
- `docs/architecture.md`: 아키텍처 및 플러그인 시스템
- `docs/stream-synchronization.md`: 스트림 동기화
- `docs/index.md`: 전체 문서 인덱스

## 관련 패키지

- **visualbase**: 미디어 I/O 기반
- **momentscan**: visualpath를 사용하는 981파크 특화 앱

---

## 개발 철학 및 의사 결정 기록

### Deprecated 클래스 완전 제거 (2026-02-05)

#### 제거된 항목

다음 클래스/파일들이 완전히 제거됨:
- `core/fusion.py` 파일 전체 (FusionResult, BaseFusion)
- `BaseAnalyzer` 클래스
- `vp.run()` 함수
- `vp.process_video()`의 `analyzers`/`fusion` 파라미터

참고: `Module.is_trigger`는 class attribute로 재도입됨 (`bool = False`).
Pathway UDF처럼 프레임 순서를 보장하지 않는 백엔드에서
stateful trigger 모듈을 ordered path로 분리하는 데 사용.

#### 마이그레이션

| 이전 | 이후 |
|------|------|
| `class MyAnalyzer(BaseAnalyzer)` | `class MyAnalyzer(Module)` |
| `def analyze(self, frame)` | `def process(self, frame, deps=None)` |
| `class MyFusion(BaseFusion)` | `class MyFusion(Module)` with `depends=[...]` |
| `def update(self, observation)` | `def process(self, frame, deps=None)` |
| `return FusionResult(should_trigger=True, ...)` | `return Observation(signals={"should_trigger": True, ...})` |
| `result.score` | `result.trigger_score` |
| `result.reason` | `result.trigger_reason` |
| `vp.run(...)` | `vp.process_video(...)` |
| `process_video(analyzers=[...], fusion=...)` | `process_video(modules=[...])` |

### Trigger 컨벤션

| 필드 | Observation 위치 |
|------|------------------|
| `should_trigger` | `signals["should_trigger"]` |
| `score` | `signals["trigger_score"]` |
| `reason` | `signals["trigger_reason"]` |
| `trigger` | `metadata["trigger"]` |

### Observation 헬퍼 프로퍼티

```python
@dataclass
class Observation:
    # ... 기존 필드 ...

    @property
    def should_trigger(self) -> bool:
        return bool(self.signals.get("should_trigger", False))

    @property
    def trigger_score(self) -> float:
        return float(self.signals.get("trigger_score", 0.0))

    @property
    def trigger_reason(self) -> str:
        return str(self.signals.get("trigger_reason", ""))

    @property
    def trigger(self) -> Optional[Trigger]:
        return self.metadata.get("trigger")
```

---

### NodeSpec 선언적 전환 (2026-02-04)

#### 설계 원칙

1. **spec + process() 공존** — spec은 선언적 의미, process()는 실행 fallback.
2. **Fusion 별도 노드 없음** — fusion은 ModuleSpec의 일부로 포함.
3. **Source 외부 주입** — FlowGraph는 소스에 대해 모름.
4. **시간 의미론은 그래프에 속함** — JoinSpec에 window_ns, lateness_ns 포함.
5. **불변 spec** — 모든 spec은 `frozen=True` dataclass.

---

### Pathway 활용 현황

| Pathway 기능 | 현재 상태 | 비고 |
|-------------|----------|------|
| per-module 병렬 UDF | 구현 완료 | 의존성 분석 기반 |
| interval_join (temporal) | 구현 완료 | JoinSpec.window_ns 우선 사용 |
| watermark/late arrival | 부분 구현 | JoinSpec.lateness_ns 전달 가능 |
| stateful fusion | subscribe 콜백 | Pathway UDF는 stateless, 현실적 선택 |
| 윈도우 집계 (windowby) | 미구현 | 향후 과제 |
| 증분 연산 | 미구현 | 실시간 모델 별도 설계 필요 |
