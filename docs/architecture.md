# VisualPath 아키텍처

> 최종 수정: 2026-02-12
> 상태: 구현 완료

## 개요

이 문서는 visualpath 패키지의 아키텍처, 플러그인 시스템, 실행 백엔드를 설명합니다.

---

## 3계층 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    visualbase                            │
│              (미디어 소스 기반 레이어)                    │
│  - Frame, Source, Stream 추상화                          │
│  - 비디오/카메라/스트림 통합 인터페이스                   │
│  - 클립 추출                                             │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    visualpath                            │
│              (분석 플랫폼 코어 레이어)                    │
│  - Module, Observation 통합 인터페이스                   │
│  - FlowGraph (DAG 기반 파이프라인)                       │
│  - 실행 백엔드 (Simple, Worker, Pathway)                 │
│  - Plugin discovery & loading                            │
│  - Worker 격리 실행                                      │
│  - Observability system                                  │
└─────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ vpx-*    │    │momentscan│    │ 앱/플러그 │
    │ plugins  │    │  (앱)    │    │  인 확장  │
    │  - face  │    │  - CLI   │    │          │
    │  - pose  │    │  - 스코링│    │          │
    │  - gesture│   │  - 시각화│    │          │
    └──────────┘    └──────────┘    └──────────┘
```

### 계층별 역할

| 레이어 | 패키지 | 역할 | 의존성 |
|--------|--------|------|--------|
| **Media** | visualbase | 미디어 소스, Frame, 클립 추출, IPC | opencv, numpy |
| **Platform** | visualpath (4패키지) | FlowGraph, Module, 실행 백엔드, Worker 격리 | visualbase |
| **SDK** | vpx-sdk | Module SDK, 공유 타입/프로토콜, 테스트 하네스 | visualpath |
| **Plugin** | vpx-face-detect 등 | ML 분석 모듈 구현 | vpx-sdk, ML libs |
| **App** | momentscan 등 | 비즈니스 로직, CLI, 클립 추출 | visualpath, vpx-* |

---

## visualpath 패키지 구조

4개 패키지로 분리 (`pkgutil.extend_path`로 동일 네임스페이스 공유):

### visualpath (core)

```
visualpath/
├── core/
│   ├── module.py          # Module ABC (통합 인터페이스)
│   ├── observation.py     # Observation, DummyAnalyzer
│   ├── isolation.py       # IsolationLevel, IsolationConfig
│   ├── capabilities.py    # Capability, ModuleCapabilities, PortSchema
│   ├── error_policy.py    # ErrorPolicy (retry, timeout, fallback)
│   ├── profile.py         # ExecutionProfile, ProfileName
│   └── compat.py          # check_compatibility()
├── flow/
│   ├── node.py            # FlowNode, FlowData
│   ├── specs.py           # NodeSpec 17종 (ModuleSpec, FilterSpec 등)
│   ├── graph.py           # FlowGraph (DAG)
│   ├── builder.py         # FlowGraphBuilder
│   ├── viz.py             # to_dot(), print_ascii()
│   └── nodes/             # Source, Path, Filter, Sampler, Branch, Join
├── backends/
│   ├── base.py            # ExecutionBackend ABC, PipelineResult
│   └── simple/
│       ├── backend.py     # SimpleBackend
│       ├── interpreter.py # SimpleInterpreter
│       └── executor.py    # GraphExecutor
├── plugin/
│   └── discovery.py       # PluginRegistry, discover_modules()
└── api.py, app.py         # High-level API
```

### visualpath-isolation (Worker 격리 + Observability)

```
visualpath-isolation/src/visualpath/
├── backends/worker/
│   └── backend.py         # WorkerBackend
├── process/
│   ├── worker.py          # BaseWorker, ThreadWorker, ProcessWorker
│   ├── launcher.py        # WorkerLauncher, VenvWorker
│   ├── worker_module.py   # WorkerModule
│   ├── ipc.py             # IPC 프로토콜 (ZMQ)
│   ├── mapper.py          # Module name → entry point 매핑
│   └── serialization.py   # 직렬화 헬퍼
└── observability/
    ├── __init__.py        # ObservabilityHub, TraceLevel
    ├── records.py         # TraceRecord 타입
    └── sinks.py           # ConsoleSink, FileSink 등
```

### visualpath-pathway (스트리밍 백엔드)

```
visualpath-pathway/src/visualpath/backends/pathway/
├── backend.py             # PathwayBackend
├── connector.py           # VideoConnectorSubject
├── converter.py           # FlowGraph → Pathway 변환
├── operators.py           # Module UDF 래퍼
└── stats.py               # 통계 추적
```

### visualpath-cli (CLI + 설정)

```
visualpath-cli/src/visualpath/
├── cli/                   # commands: debug, run, validate, plugins, version
└── config/                # schema, loader
```

---

## 핵심 인터페이스

### Module

모든 처리 컴포넌트(Analyzer, Trigger/Fusion)의 통합 인터페이스:

```python
from visualpath.core import Module, Observation

class MyModule(Module):
    depends = ["face.detect"]        # 필수 의존
    optional_depends = ["body.pose"] # 선택적 의존
    stateful = False                 # 순서 보장 필요 여부

    @property
    def name(self) -> str:
        return "my.module"

    def initialize(self) -> None:
        """리소스 초기화 (1회)."""
        self.model = load_model()

    def warmup(self, sample_frame=None) -> None:
        """GPU warmup (initialize 후 호출)."""
        pass

    def process(self, frame, deps=None) -> Optional[Observation]:
        """프레임 처리. deps는 의존 모듈의 Observation dict."""
        face_obs = deps.get("face.detect") if deps else None
        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={"score": 0.9},
        )

    def process_batch(self, frames, deps_list) -> list[Optional[Observation]]:
        """배치 처리 (기본: 순차 호출). BATCHING capability 선언 시 호출."""
        return [self.process(f, d) for f, d in zip(frames, deps_list)]

    def cleanup(self) -> None:
        """리소스 해제."""
        self.model.close()

    @property
    def capabilities(self) -> ModuleCapabilities:
        """GPU, BATCHING 등 능력 선언."""
        return ModuleCapabilities(flags=Capability.GPU | Capability.BATCHING)
```

### Trigger 모듈 (Fusion)

별도 클래스 없이 `Module`로 구현. `signals["should_trigger"]`로 트리거 발생:

```python
class SmileTrigger(Module):
    depends = ["face.detect"]

    @property
    def name(self) -> str:
        return "smile_trigger"

    def process(self, frame, deps=None) -> Observation:
        face_obs = deps.get("face.detect") if deps else None
        happy = face_obs.signals.get("happy", 0) if face_obs else 0

        if happy > 0.7:
            trigger = Trigger.point(
                event_time_ns=frame.t_src_ns,
                pre_sec=2.0, post_sec=2.0,
                label="smile",
            )
            return Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={"should_trigger": True, "trigger_score": happy},
                metadata={"trigger": trigger},
            )

        return Observation(
            source=self.name, frame_id=frame.frame_id,
            t_ns=frame.t_src_ns, signals={"should_trigger": False},
        )
```

---

## 플러그인 시스템

### entry_points 등록

통합 그룹 `visualpath.modules`에 등록:

```toml
[project.entry-points."visualpath.modules"]
"face.detect" = "vpx.face_detect:FaceDetectionAnalyzer"
"face.expression" = "vpx.face_expression:ExpressionAnalyzer"
"body.pose" = "vpx.body_pose:PoseAnalyzer"
"hand.gesture" = "vpx.hand_gesture:GestureAnalyzer"
```

### Plugin Discovery

```python
from visualpath.plugin import PluginRegistry

registry = PluginRegistry()

# 설치된 모든 Module 발견
modules = registry.list_modules()
# {'face.detect': <class FaceDetectionAnalyzer>, ...}

# 이름으로 모듈 생성
face = registry.create("face.detect")

# 런타임 등록
registry.register("custom", MyModule)
```

---

## FlowGraph (DAG 파이프라인)

FlowGraph는 모듈 간 의존성과 데이터 흐름을 선언적으로 정의:

```python
from visualpath.flow import FlowGraphBuilder

graph = (FlowGraphBuilder()
    .source("frames")
    .path("analysis", modules=[face_detector, expression, highlight])
    .on_trigger(lambda data: print(f"Trigger: {data}"))
    .build())
```

### FlowGraph.from_modules (간편 팩토리)

```python
graph = FlowGraph.from_modules(
    modules=[face_detector, expression, highlight],
    on_trigger=handle_trigger,
)
```

### 검증

`graph.validate()`가 자동 검증:
- DAG 속성 (사이클 없음)
- 모든 노드 도달 가능
- 모듈 의존성 충족
- PortSchema 호환성 (경고)
- Capability 호환성 (경고)

---

## 실행 백엔드 (ExecutionBackend)

### 백엔드 종류

| 백엔드 | 설명 | 용도 |
|--------|------|------|
| **SimpleBackend** | 순차/병렬 처리 + 배치 | 로컬 비디오, 개발/디버깅 |
| **WorkerBackend** | 격리 모듈을 WorkerModule로 래핑 → SimpleBackend 위임 | 프로세스/venv 격리 |
| **PathwayBackend** | Pathway 스트리밍 엔진 | 실시간 처리 |

### ExecutionBackend ABC

```python
class ExecutionBackend(ABC):
    @abstractmethod
    def execute(
        self,
        frames: Iterator[Frame],
        graph: FlowGraph,
        *,
        on_frame: Optional[Callable[[Frame, List[FlowData]], bool]] = None,
    ) -> PipelineResult:
        """FlowGraph 실행. on_frame이 False 반환 시 조기 종료."""
        ...
```

### PipelineResult

```python
@dataclass
class PipelineResult:
    triggers: List[Trigger]
    frame_count: int
    stats: dict            # 백엔드별 통계
```

### 사용법

```python
import visualpath as vp

# 기본 (SimpleBackend)
result = vp.process_video("video.mp4", modules=[face_detector, smile_trigger])

# 배치 모드 (GPU 배치 추론)
result = vp.process_video("video.mp4", modules=[face_detector], batch_size=4)

# Pathway 백엔드
result = vp.process_video("video.mp4", modules=[face_detector], backend="pathway")
```

### SimpleBackend

```python
from visualpath.backends.simple import SimpleBackend

backend = SimpleBackend(batch_size=1)  # batch_size > 1이면 배치 모드
result = backend.execute(frames, graph, on_frame=callback)
```

실행 흐름:
1. FlowGraph의 모든 노드 `initialize()` + `warmup()`
2. 프레임마다 `GraphExecutor.process(frame)` → `SimpleInterpreter.interpret()`
3. SimpleInterpreter가 ModuleSpec의 모듈을 위상 정렬하여 실행
4. `on_frame` 콜백 호출 (False 반환 시 중단)
5. `cleanup()` (finally)

배치 모드 (`batch_size > 1`):
- `Capability.BATCHING` 선언된 모듈에 `process_batch()` 호출
- 미선언 모듈은 순차 fallback

### WorkerBackend

격리 설정된 모듈을 자동으로 WorkerModule로 래핑 후 SimpleBackend에 위임:

```python
from visualpath.backends.worker import WorkerBackend

backend = WorkerBackend(batch_size=1)
result = backend.execute(frames, graph)
```

실행 흐름:
1. FlowGraph에서 `ModuleSpec.isolation` 설정 스캔
2. `IsolationLevel.PROCESS/VENV` 모듈을 WorkerModule로 래핑
3. 원본 모듈의 `depends`, `optional_depends`, `stateful`, `capabilities` 복사
4. 새 FlowGraph 구성 → SimpleBackend에 위임

### PathwayBackend

```python
from visualpath.backends.pathway import PathwayBackend

backend = PathwayBackend(
    window_ns=100_000_000,           # 100ms 윈도우
    allowed_lateness_ns=50_000_000,  # 50ms 지연 허용
)
```

### 백엔드 비교

| 측면 | SimpleBackend | WorkerBackend | PathwayBackend |
|------|---------------|---------------|----------------|
| 실행 방식 | 순차/병렬 (위상 정렬) | Simple 위임 | 스트리밍 dataflow |
| 격리 | 없음 (INLINE) | PROCESS/VENV 지원 | 없음 |
| 배치 | `batch_size` 파라미터 | Simple 위임 | N/A |
| 윈도우 정렬 | N/A | N/A | TumblingWindow + watermark |
| 백프레셔 | N/A | N/A | Pathway 엔진 내장 |
| 용도 | 개발, 디버깅, 배치 처리 | ML 의존성 충돌 해결 | 실시간 처리 |

---

## Worker 격리 실행

ML 라이브러리 간 의존성 충돌을 해결하기 위해 Worker별 독립 실행을 지원합니다.

### IsolationLevel

```python
from visualpath.core.isolation import IsolationLevel

IsolationLevel.INLINE    # 같은 프로세스, 같은 스레드
IsolationLevel.THREAD    # 같은 프로세스, 별도 스레드
IsolationLevel.PROCESS   # 별도 프로세스
IsolationLevel.VENV      # 별도 프로세스 + 별도 venv
```

### WorkerModule

WorkerBackend가 격리 모듈을 투명하게 래핑:

```
FlowGraph                         WorkerBackend
  ModuleSpec                         _wrap_modules()
    face.detect (VENV)     →     WorkerModule("face.detect", VenvWorker)
    face.expression (INLINE) →   face.expression (그대로)
    highlight (INLINE)       →   highlight (그대로)
```

WorkerModule은 원본 모듈의 속성을 그대로 전달:
- `depends`, `optional_depends`
- `stateful`
- `capabilities` (BATCHING 플래그 포함)

### VenvWorker (ZMQ IPC)

```
Frame → VenvWorker.process()
        → Serialize frame (JPEG + JSON)
        → ZMQ send {"type": "process", "frame": {...}}
        → Subprocess: load module, call process()
        → Serialize Observation (JSON)
        → ZMQ recv → Observation
```

---

## on_trigger 콜백 패턴

visualpath는 **비즈니스 로직을 포함하지 않습니다**. 트리거 발생 시 앱에서 콜백으로 Action을 처리합니다.

```python
# FlowGraph에 on_trigger 등록
graph = FlowGraph.from_modules(
    modules=[face_detector, smile_trigger],
    on_trigger=handle_trigger,
)

# 앱에서 Action 정의
def handle_trigger(flow_data: FlowData):
    trigger = flow_data.observations[-1].trigger
    clip = clipper.extract(trigger)
    save_clip(clip)

# 실행
backend = SimpleBackend()
result = backend.execute(frames, graph)
```

---

## Observability

### TraceLevel

| 레벨 | 용도 | 오버헤드 |
|------|------|----------|
| OFF | 프로덕션 기본 | 0% |
| MINIMAL | Trigger만 로깅 | <1% |
| NORMAL | 프레임 요약 + 상태 전환 | ~5% |
| VERBOSE | 모든 Signal + 타이밍 | ~15% |

### 사용법

```python
from visualpath.observability import ObservabilityHub, TraceLevel

hub = ObservabilityHub(level=TraceLevel.NORMAL)

# Sink 연결
hub.add_sink(FileSink("trace.jsonl"))
hub.add_sink(ConsoleSink())
```

---

## 핵심 데이터 타입

### Observation

모든 모듈의 통합 결과 컨테이너:

```python
@dataclass
class Observation(Generic[T]):
    source: str                      # 모듈 이름
    frame_id: int                    # 프레임 식별자
    t_ns: int                        # 타임스탬프 (나노초)
    signals: Dict[str, Any]          # 스칼라 피처 값
    data: Optional[T] = None         # 도메인별 데이터
    metadata: Dict[str, Any]         # 추가 정보
    timing: Optional[Dict[str, float]] = None  # 처리 시간 (ms)
```

### Trigger 컨벤션

Trigger 정보는 Observation의 signals/metadata에 저장:

| 필드 | 위치 | 헬퍼 프로퍼티 |
|------|------|--------------|
| 트리거 여부 | `signals["should_trigger"]` | `obs.should_trigger` |
| 점수 | `signals["trigger_score"]` | `obs.trigger_score` |
| 사유 | `signals["trigger_reason"]` | `obs.trigger_reason` |
| Trigger 객체 | `metadata["trigger"]` | `obs.trigger` |

### FlowData

FlowGraph 노드 간 전달되는 데이터:

```python
@dataclass
class FlowData:
    frame: Frame
    observations: List[Observation]
    path: str                        # 경유 경로
    metadata: Dict[str, Any]
```

### PipelineResult

```python
@dataclass
class PipelineResult:
    triggers: List[Trigger]
    frame_count: int
    stats: dict
```

---

## 데이터 흐름

### FlowGraph 실행

```
Source (frames)
    │
    ▼
GraphExecutor.process(frame)
    │
    ▼
SimpleInterpreter.interpret(node, flow_data)
    │
    ├── ModuleSpec → 모듈 위상 정렬 → 병렬/순차 실행
    │     ├── Module A (독립) ──┐
    │     ├── Module B (독립) ──┤── ThreadPool (parallel=True)
    │     └── Module C (A 의존) ── 순차 실행
    │
    ├── FilterSpec → 조건 필터링
    ├── SampleSpec → 프레임 샘플링
    ├── BranchSpec → 분기
    └── JoinSpec → 합류 (temporal window)
    │
    ▼
FlowGraph.fire_triggers()
    │ should_trigger == True ?
    ▼
on_trigger(flow_data) → 앱에서 Action 처리
```

### deps 전달 패턴

모듈 간 데이터 전달. `depends` 선언 → 실행 시 `deps` dict로 이전 결과 수신:

```
face.detect.process(frame, deps=None) → Observation(source="face.detect")
    │
    ▼ deps 누적
face.expression.process(frame, deps={"face.detect": obs}) → Observation
    │
    ▼ deps 누적
highlight.process(frame, deps={"face.detect": obs1, "face.expression": obs2})
```

### 배치 실행

```
Frame₁, Frame₂, Frame₃, Frame₄  (batch_size=4)
    │
    ▼
SimpleInterpreter.interpret_modules_batch()
    │
    ├── BATCHING 모듈 → process_batch([f1,f2,f3,f4], [d1,d2,d3,d4])
    └── 일반 모듈 → 순차 process(f1,d1), process(f2,d2), ...
```

---

## 에러 핸들링

### ErrorPolicy

모듈별 에러 정책 선언:

```python
class MyModule(Module):
    @property
    def error_policy(self) -> ErrorPolicy:
        return ErrorPolicy(
            max_retries=2,
            timeout_ms=5000,
            fallback="skip",  # "skip" | "raise" | "default"
        )
```

SimpleInterpreter가 `_call_module()` 시 ErrorPolicy를 적용:
- retry: 지정 횟수만큼 재시도
- timeout: 제한 시간 초과 시 fallback
- fallback: skip(무시), raise(예외), default(기본 Observation)

### 에러 처리 요약

| 상황 | 처리 방식 |
|------|----------|
| **모듈 실패** | ErrorPolicy에 따라 retry/skip/raise |
| **Worker 타임아웃** | WorkerModule이 에러 Observation 반환 |
| **Subprocess 실패** | VenvWorker가 InlineWorker로 폴백 |
| **Sink 실패** | hub.emit()에서 무시 (메인 처리 영향 없음) |
| **on_frame False** | 조기 종료 (cleanup 보장) |

---

## 확장 포인트

### A. 격리 설정

```python
from visualpath.core.isolation import IsolationConfig, IsolationLevel

config = IsolationConfig(
    default_level=IsolationLevel.INLINE,
    overrides={"face.detect": IsolationLevel.VENV},
    venv_paths={"face.detect": "/opt/venvs/face"},
)
```

### B. 플러그인 등록

```toml
# pyproject.toml
[project.entry-points."visualpath.modules"]
"face.detect" = "vpx.face_detect:FaceDetectionAnalyzer"
```

```python
# 런타임 등록
registry = PluginRegistry()
registry.register("custom", MyModule)
```

### C. Observability 설정

```python
hub = ObservabilityHub.get_instance()
hub.configure(
    level=TraceLevel.NORMAL,
    sinks=[FileSink("/tmp/trace.jsonl"), ConsoleSink()]
)
```

### D. Capability 선언

```python
from visualpath.core.capabilities import Capability, ModuleCapabilities, PortSchema

class MyModule(Module):
    @property
    def capabilities(self) -> ModuleCapabilities:
        return ModuleCapabilities(
            flags=Capability.GPU | Capability.BATCHING,
            resource_group="gpu-0",
        )

    @property
    def port_schema(self) -> PortSchema:
        return PortSchema(
            inputs={"frame": "numpy.ndarray"},
            outputs={"face_count": "float", "faces": "list"},
        )
```

---

## 관련 문서

- [왜 visualpath인가](planning/why-visualpath.md): 단순 루프에서 플랫폼이 필요해지는 과정
- [ML 의존성 충돌과 격리](isolation.md): CUDA/onnxruntime 충돌 사례
- [Stream Synchronization](stream-synchronization.md): 스트림 동기화 아키텍처
- [문서 인덱스](index.md): 전체 문서 목록
