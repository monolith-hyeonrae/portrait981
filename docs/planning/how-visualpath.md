# visualpath — 어떻게 풀었는가, 어떻게 풀건가

> `why-visualpath.md`에서 다룬 문제들을 visualpath가 어떤 설계로 해결하는지,
> 그리고 신규 앱 확장에서 어떤 과제가 남아 있는지 정리.
> 문제의식은 `why-visualpath.md`, 해법은 이 문서.

---

# Part 1: 어떻게 풀었는가

## 1. Spec/Interpreter 분리 (핵심 설계)

**해결하는 문제**: 파이프라인 구조와 실행 방식의 결합

FlowGraph의 각 노드는 **선언적 명세(NodeSpec)**만 노출하고, 실행은 백엔드(Interpreter)가 결정합니다.

```python
# 노드: "나는 무엇을 한다" (구조)
@property
def spec(self) -> ModuleSpec:
    return ModuleSpec(module=self.module, depends=["face.detect"])

# 백엔드: spec 타입에 따라 실행 방식 결정 (실행)
match spec:
    case ModuleSpec(): self._interpret_module(...)
    case SampleSpec(): self._interpret_sample(...)
    case JoinSpec():   self._interpret_join(...)
```

FlowGraph는 AST, NodeSpec은 토큰, 백엔드는 인터프리터입니다. 같은 그래프가:
- **SimpleBackend**: 순차 실행 (개발/디버깅)
- **WorkerBackend**: 격리 실행 (프로덕션)
- **Pathway 백엔드**: 스트리밍 실행 (실시간)

파이프라인 코드 변경 없이 실행 전략만 교체됩니다.

## 2. 투명한 격리 (Wrapper 패턴)

**해결하는 문제**: 격리 인프라가 분석 코드에 누출

WorkerModule은 Module 인터페이스를 구현하면서 내부적으로 별도 프로세스/venv에서 실행합니다.

```
SimpleInterpreter
    │
    ├── module.process(frame, deps)    ← Module (인라인)
    ├── module.process(frame, deps)    ← WorkerModule (프로세스 격리)
    └── module.process(frame, deps)    ← WorkerModule (venv 격리)
```

SimpleInterpreter는 `isinstance` 체크 없이 `.process()`를 호출합니다. 격리가 **100% 투명**합니다. 격리 수준 변경은 `IsolationConfig` 설정 한 줄입니다.

여기에 **자동 격리 감지**가 결합됩니다:
- `ModuleCapabilities.resource_groups`로 충돌 그룹 선언
- `check_compatibility()`가 같은 프로세스에 있는 충돌 감지
- `build_conflict_isolation()`이 소수 그룹을 PROCESS로 격리하는 config 자동 생성

## 3. 선언적 메타데이터 조합

**해결하는 문제**: 비기능 관심사(에러 처리, 리소스 관리)가 분석 코드에 섞임

모듈의 모든 비기능 속성이 선언적 데이터클래스입니다:

```python
class FaceDetectionAnalyzer(Module):
    # 분석 로직
    def analyze(self, frame, deps=None): ...

    # 비기능 선언 (프레임워크가 해석)
    error_policy = ErrorPolicy(max_retries=2, on_error="fallback", fallback_signals={"face_count": 0})
    capabilities = ModuleCapabilities(flags=Capability.GPU | Capability.BATCHING, gpu_memory_mb=800)
    port_schema = PortSchema(input_signals=frozenset(), output_signals=frozenset(["face_count"]))
```

| 선언 | 프레임워크 동작 |
|------|----------------|
| `ErrorPolicy(max_retries=2)` | SimpleInterpreter가 실패 시 2회 재시도 |
| `Capability.BATCHING` | SimpleBackend가 프레임 모아서 `process_batch()` 호출 |
| `Capability.GPU` | `check_compatibility()`가 GPU 메모리 합산, 충돌 감지 |
| `PortSchema(input_signals=...)` | `FlowGraph.validate()`가 빌드 타임에 연결 검증 |
| `fallback_signals` | 모든 재시도 실패 시 안전한 기본 Observation 반환 |

모듈 코드에 if/try/retry 로직이 없습니다. 선언만 하면 프레임워크가 처리합니다.

## 4. 경고 우선 정책

**해결하는 문제**: 새 기능 추가가 기존 파이프라인을 깨뜨림

FlowGraph.validate()의 모든 검사 — 호환성, PortSchema, Capability 충돌 — 는 **경고만 출력**합니다:

```python
# PortSchema 불일치: 경고만
logger.warning("module 'face.expression' expects signal 'face_count' not found upstream")

# Capability 충돌: 경고만
logger.warning("resource conflict between face.detect (onnxruntime) and body.pose (torch)")
```

기존 모듈이 capabilities를 선언하지 않아도 파이프라인은 동작합니다. 새 모듈이 port_schema를 추가해도 기존 연결이 깨지지 않습니다. 점진적 도입이 가능합니다.

## 5. 단일 Module 인터페이스

**해결하는 문제**: Analyzer/Trigger/Fusion 별도 추상화의 복잡도

이전에는 BaseExtractor, BaseFusion, TriggerHandler가 각각 다른 인터페이스를 가졌습니다. 현재는 모든 것이 Module.process() → Observation입니다:

```python
# 분석 모듈
return Observation(source="face.detect", signals={"face_count": 2}, data=faces)

# 트리거 모듈 — 같은 인터페이스, signals에 trigger 정보만 추가
return Observation(
    source="highlight",
    signals={"should_trigger": True, "trigger_score": 0.9},
    metadata={"trigger": Trigger.point(timestamp_ns=...)},
)
```

인터프리터에 특수 분기가 없습니다. 트리거 모듈도 "그냥 모듈"입니다.

## 6. 의존성 레벨 병렬 실행

**해결하는 문제**: 독립 모듈이 순차 실행되는 비효율

FlowGraph가 의존성을 레벨별로 정렬합니다:

```
Level 0: [face.detect, body.pose]           ← 독립, 병렬 실행
Level 1: [face.expression, face.classify]   ← 둘 다 face.detect 의존, 병렬 실행
Level 2: [highlight]                        ← expression + classify 의존, 순차
```

같은 레벨 모듈은 ThreadPoolExecutor로 동시 실행. 단일 모듈 레벨은 스레드 오버헤드 없이 직접 호출. 결과는 레벨 순서대로 수집되어 결정론적입니다.

## 7. 배치 추론 투명 지원

**해결하는 문제**: GPU 배치 최적화가 파이프라인 코드를 오염

`Capability.BATCHING` 플래그가 있는 모듈은 프레임을 모아서 `process_batch()`로 호출됩니다. 없는 모듈은 자동으로 순차 fallback합니다.

```python
# GPU 모듈: 배치 지원 선언
@property
def capabilities(self):
    return ModuleCapabilities(flags=Capability.GPU | Capability.BATCHING)

def process_batch(self, frames, deps_list):
    return self.backend.detect_batch(frames)  # ONNX 배치 추론
```

같은 파이프라인에서 일부 모듈은 배치, 일부는 순차로 실행됩니다. 파이프라인 코드는 이를 모릅니다.

## 8. 플러그인 등록 + 스캐폴딩

**해결하는 문제**: 새 모듈 추가 시 루프 코드 수정

entry point 기반 자동 발견:

```toml
# vpx-face-detect/pyproject.toml
[project.entry-points."visualpath.modules"]
"face.detect" = "vpx.face_detect:FaceDetectionAnalyzer"
```

`vpx new` 명령으로 스캐폴딩:

```bash
vpx new scene.transition                    # vpx 플러그인 생성
vpx new face.landmark --depends face.detect # 의존 모듈 지정
vpx new scene.transition --internal         # 앱 내부 모듈
```

모듈 추가가 "파일 생성 + entry point 등록"으로 완료됩니다. 기존 코드 수정 없음.

## 9. 제로 코스트 Observability

**해결하는 문제**: 성능 측정 코드가 분석 코드에 산재

```python
# 꺼져 있으면 bool 체크 한 번
if hub.enabled:
    hub.emit(AnalyzerTimingRecord(...))
```

ObservabilityHub는 TraceLevel (OFF/MINIMAL/NORMAL/VERBOSE)로 제어됩니다. 싱크 에러는 조용히 무시 — 관측 실패가 파이프라인을 중단시키지 않습니다.

## 설계 원칙 요약

| 원칙 | 적용 |
|------|------|
| **선언 > 명령** | NodeSpec, ErrorPolicy, Capabilities, PortSchema |
| **경고 > 에러** | 호환성 검사, 스키마 검증이 실행을 막지 않음 |
| **투명 > 명시** | 격리, 배치가 Module 인터페이스 뒤에 숨음 |
| **조합 > 상속** | Capabilities, ErrorPolicy가 독립 메타데이터로 조합 |
| **관례 > 설정** | App 기본 라이프사이클, vpx new 네이밍 규칙 |
| **비용 0 > 항상** | Observability OFF 시 bool 체크만 |

---

# Part 2: 어떻게 풀건가

momentscan(구 momentscan)은 실시간 스트리밍에서 **배치 후처리**로 전환합니다. 같은 분석 모듈(vpx 플러그인)을 사용하지만, 실행 패턴이 달라집니다.

3-app 구조: `momentscan (분석/수집) → momentbank (저장/관리) → reportrait (AI 생성)`

현재 프레임워크의 준비 상태를 평가하고, 진화 방향을 정리합니다.

## 그대로 쓸 수 있는 것

### Module.process() + deps 시스템

프레임 단위 추출은 momentscan의 모든 Phase에서 동일합니다:
- Phase 1 (batch highlight): 프레임별 수치 feature 추출
- Phase 2 (embedding experiment): 프레임별 임베딩 추출
- Phase 3 (identity collection): 프레임별 QualityGate + 듀얼 임베딩 + 버킷 분류

`depends` 선언으로 vpx 플러그인 재사용도 그대로 동작합니다.

### App 라이프사이클

`setup()` → 리소스 초기화, `after_run()` → 후처리 export, `teardown()` → 정리. 배치 앱의 패턴과 정확히 일치합니다:

```python
class MomentscanApp(vp.App):
    def setup(self):
        self.records = []

    def on_frame(self, frame, results):
        # 프레임별 수치 feature + 임베딩 축적
        self.records.append(extract_signals(frame, results))

    def after_run(self, result):
        # 전체 비디오 기준 정규화 + peak detection + identity 수집
        df = normalize_per_video(pd.DataFrame(self.records))
        windows = find_highlight_windows(df)
        identity_sets = collect_identity(df)
        export(windows, identity_sets)
```

### FlowGraph DAG

momentscan의 Phase별 파이프라인도 FlowGraph로 표현 가능합니다:

```
Phase 1: face.detect → face.expression → body.pose → [scoring module]
Phase 2: face.detect → body.pose → vision.embed → [delta scoring module]
Phase 3: face.detect → face.expression → vision.embed → [selection module]
```

### 플러그인 재사용

momentscan의 Phase별로 기존 vpx 플러그인을 재사용합니다:

| 플러그인 | Phase 1 | Phase 2 | Phase 3 |
|---------|:-:|:-:|:-:|
| vpx-face-detect | O | O | O |
| vpx-face-expression | O | | O |
| vpx-body-pose | O | O | O |
| **vpx-vision-embed** (신규) | | O | O |

## 부족한 것

### 시간 축 집계 프리미티브

현재 Module은 **프레임 하나**만 봅니다. 하지만 배치 처리는 비디오 전체를 본 후에 판단해야 합니다:

| momentscan Phase | 필요한 비디오 레벨 연산 |
|---|---|
| Phase 1 (batch highlight) | per-video MAD 정규화, percentile 계산 |
| Phase 2 (embedding experiment) | EMA baseline, temporal delta, peak detection |
| Phase 3 (identity collection) | medoid prototype 계산, coverage 충족률 |

현재 우회 방법: `on_frame()`에서 누적 → `after_run()`에서 비디오 레벨 연산. 동작하지만, Phase별로 누적 로직을 반복 구현해야 합니다.

**진화 방향**: `after_run(result)`에 비디오 전체 Observation 접근 수단 제공. 또는 `FrameAccumulator` 유틸리티를 공용으로 제공하여 앱별 반복 제거.

### 비디오 간 상태 초기화

여러 비디오를 순차 처리할 때, 모듈의 내부 상태(EMA baseline, tracker 등)가 이전 비디오에서 이월됩니다. 현재 `module.reset()` 호출은 앱 책임입니다.

**진화 방향**: `App.run_batch(video_list)` 메서드 추가. 비디오 간 자동 리셋 + `on_video_start()` / `on_video_end()` 훅 제공.

```python
class App:
    def run_batch(self, video_list):
        for video in video_list:
            self._reset_modules()
            self.on_video_start(video)
            result = self.run(video)
            self.on_video_end(video, result)

    def on_video_start(self, video_path):
        """비디오별 초기화 (출력 디렉토리 생성 등)."""
        pass

    def on_video_end(self, video_path, result):
        """비디오별 마무리 (export, 메타 저장 등)."""
        pass
```

## 진화가 필요한 것

### 공용 유틸리티 → vpx-sdk 이동

momentscan의 Phase별 + momentbank가 공통으로 필요한 것들이 있습니다. 중복 구현하면 계약이 어긋나므로, `vpx-sdk` 또는 공용 위치에 한 번 구현해야 합니다:

| 유틸리티 | 용도 | 현재 위치 |
|---------|------|-----------|
| QualityGate (strict/loose) | 프레임 품질 판정 | 미구현 |
| blur / exposure 계산 | 공용 feature | momentscan quality analyzer에 산재 |
| L2 normalize, cosine similarity | 임베딩 연산 | 미구현 |
| 버킷 분류 (yaw/pitch/expression) | Phase 1 + Phase 3 | 미구현 |

### 신규 vpx 플러그인: vpx-vision-embed

DINOv2 / SigLIP / OpenCLIP 임베딩 추출. momentscan Phase 2와 Phase 3이 공유합니다:

```
libs/vpx/plugins/vision-embed/
├── src/vpx/vision_embed/
│   ├── analyzer.py           # VisionEmbedAnalyzer (Module)
│   ├── output.py             # VisionEmbedOutput
│   └── backends/
│       ├── dinov2.py          # DINOv2Backend
│       ├── siglip.py          # SigLIPBackend
│       └── openclip.py        # OpenCLIPBackend
```

`vpx new vision.embed`로 스캐폴딩 후 구현. 기존 인프라 변경 없음.

### momentbank의 위치

momentbank는 앱이라기보다 **라이브러리**(memory bank API)에 가깝습니다. momentscan이 `update()`를 호출하고, reportrait이 `select_refs()`를 호출합니다.

선택지:
- `apps/momentbank/` — 독립 앱 + CLI
- `libs/momentbank/` — 라이브러리, 다른 앱이 import

현재 계획: 앱으로 시작하되, API가 안정화되면 libs로 이동 가능.

## 현재 상태 요약

```
                 그대로 사용              진화 필요              신규 구현
                ─────────────         ─────────────         ─────────────
Module          process(frame, deps)  after_run 접근성       -
                depends/deps          run_batch()
                process_batch()       on_video_start/end

FlowGraph       DAG, validate()       -                     -

Backend         Simple, Worker        -                     -
                Batch mode

App             setup/teardown        run_batch()           -
                on_frame, after_run   on_video_start/end

Plugin          entry points          -                     vpx-vision-embed
                vpx new

공용 유틸        -                     -                     QualityGate
                                                            blur/exposure
                                                            embedding utils
                                                            bucket classifiers
```

기존 프레임워크의 핵심(Module, FlowGraph, Backend, App)은 그대로 사용할 수 있습니다. 진화 항목은 편의성 개선이지 차단 요소가 아닙니다 — 앱 레벨 우회로 당장 개발을 시작할 수 있습니다.
