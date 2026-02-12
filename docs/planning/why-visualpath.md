# 왜 visualpath인가

> "루프에서 ML 라이브러리 몇 개 돌리면 되는 거 아니야?"

이 질문은 자연스럽고, 처음에는 맞습니다. 이 문서는 그 단순한 루프가 어떻게 관리 불가능해지는지를 momentscan(구 momentscan)의 실제 개발 과정을 통해 설명합니다.

---

## 단순한 루프에서 시작

영상에서 하이라이트 순간을 감지하는 프로그램을 만든다고 합시다. 게임 루프처럼 작성하면 됩니다:

```python
for frame in video:
    faces = detect_faces(frame)
    poses = detect_poses(frame)
    if is_highlight(faces, poses):
        save_clip(frame)
```

5줄입니다. 동작합니다. 데모로 충분합니다.

그리고 요구사항이 하나씩 추가됩니다.

---

## 요구사항이 누적되는 과정

### 1단계: "표정도 분석해줘"

```python
for frame in video:
    faces = detect_faces(frame)
    expressions = analyze_expressions(frame, faces)  # faces 결과 필요
    poses = detect_poses(frame)
    if is_highlight(faces, expressions, poses):
        save_clip(frame)
```

`analyze_expressions`는 `detect_faces`의 결과가 필요합니다. 함수 호출 순서와 변수 전달을 직접 관리해야 합니다. 아직은 괜찮습니다.

### 2단계: "주탑승자만 분석해줘"

```python
for frame in video:
    faces = detect_faces(frame)
    expressions = analyze_expressions(frame, faces)
    roles = classify_faces(frame, faces)  # 누가 주탑승자인지
    poses = detect_poses(frame)
    main_face = find_main(roles)
    if main_face and is_highlight(faces, expressions, poses, main_face):
        save_clip(frame)
```

`classify_faces`도 `detect_faces` 결과가 필요합니다. 의존성 체인이 생깁니다. `is_highlight`의 인자가 늘어나고 있습니다.

### 3단계: "제스처도 감지해줘"

```python
for frame in video:
    faces = detect_faces(frame)
    expressions = analyze_expressions(frame, faces)
    roles = classify_faces(frame, faces)
    poses = detect_poses(frame)
    gestures = detect_gestures(frame)
    main_face = find_main(roles)
    if main_face and is_highlight(faces, expressions, poses, gestures, main_face):
        save_clip(frame)
```

분석 함수가 5개로 늘었습니다. 아직 관리할 수 있습니다.

### 4단계: "face와 pose를 같이 돌리면 GPU에서 죽어요"

여기서 처음으로 루프 바깥의 문제가 발생합니다.

```
ImportError: libc10_cuda.so: undefined symbol: cudaGetDriverEntryPointByVersion
```

onnxruntime-gpu(face)와 PyTorch(pose)가 같은 프로세스에서 CUDA를 초기화하면 심볼이 충돌합니다. 코드 품질과 무관한, 라이브러리 조합의 구조적 문제입니다.

루프 안에서는 해결할 수 없습니다. pose를 별도 subprocess에서 실행해야 합니다:

```python
pose_process = start_subprocess("pose_worker.py")

for frame in video:
    faces = detect_faces(frame)
    expressions = analyze_expressions(frame, faces)
    roles = classify_faces(frame, faces)
    poses = send_to_subprocess(pose_process, frame)  # IPC
    gestures = detect_gestures(frame)
    main_face = find_main(roles)
    if main_face and is_highlight(faces, expressions, poses, gestures, main_face):
        save_clip(frame)
```

이제 subprocess 관리, IPC 프로토콜, 프레임 직렬화, 타임아웃 처리가 루프 코드에 섞이기 시작합니다.

### 5단계: "얼굴 검출과 표정 분석도 onnxruntime이 충돌해요"

`insightface`(onnxruntime-gpu)와 `hsemotion-onnx`(onnxruntime CPU)를 같은 venv에 설치하면 .so 파일이 덮어써집니다. 이번에는 별도 venv가 필요합니다:

```python
face_venv_process = start_venv_subprocess("/opt/venvs/venv-face-detect", "face_worker.py")
expr_venv_process = start_venv_subprocess("/opt/venvs/venv-expression", "expr_worker.py")
pose_process = start_subprocess("pose_worker.py")

for frame in video:
    faces = send_to_venv(face_venv_process, frame)
    expressions = send_to_venv(expr_venv_process, frame, deps={"faces": faces})
    roles = classify_faces(frame, faces)  # 순수 Python이라 inline OK
    poses = send_to_subprocess(pose_process, frame)
    gestures = detect_gestures(frame)
    main_face = find_main(roles)
    if main_face and is_highlight(faces, expressions, poses, gestures, main_face):
        save_clip(frame)
```

루프 앞뒤에 3개의 프로세스 관리 코드가 붙었습니다. 각 프로세스마다 시작/종료, 에러 처리, 타임아웃, deps 직렬화가 필요합니다. "5줄의 단순한 루프"는 이미 없습니다.

### 6단계: "성능이 느려요. 어디서 병목인지 알고 싶어요"

```python
for frame in video:
    t0 = time.time()
    faces = send_to_venv(face_venv_process, frame)
    t1 = time.time()
    expressions = send_to_venv(expr_venv_process, frame, deps={"faces": faces})
    t2 = time.time()
    roles = classify_faces(frame, faces)
    t3 = time.time()
    poses = send_to_subprocess(pose_process, frame)
    t4 = time.time()
    gestures = detect_gestures(frame)
    t5 = time.time()
    # ... timing logging, trace output, P95 calculation ...
```

타이밍 코드가 분석 로직과 섞입니다.

### 7단계: "설정으로 바꿀 수 있게 해줘"

어떤 환경에서는 GPU가 없어서 pose를 빼야 하고, 어떤 환경에서는 venv 분리 없이 돌려야 합니다. 설정에 따라 프로세스 생성, inline 실행, venv 실행을 분기해야 합니다:

```python
if config.pose.isolation == "process":
    pose_worker = start_subprocess(...)
elif config.pose.isolation == "venv":
    pose_worker = start_venv_subprocess(config.pose.venv_path, ...)
else:
    pose_worker = InlinePoseWrapper(...)

if config.face.isolation == "venv":
    face_worker = start_venv_subprocess(config.face.venv_path, ...)
else:
    face_worker = InlineFaceWrapper(...)

# ... 각 extractor마다 같은 분기 반복 ...

for frame in video:
    # ... 각 worker 타입에 따라 다른 호출 방식 ...
```

### 8단계: "새로운 분석 앱도 만들어야 해요"

momentscan과 별개로, OCR 분석이나 씬 분류 앱이 필요해집니다. 각 앱에서 같은 문제를 다시 겪습니다:
- subprocess 관리
- IPC 프로토콜
- deps 전달
- 타이밍 측정
- 설정 기반 격리 전환
- 에러 핸들링

---

## 루프가 아닌 것들

위 과정을 돌아보면, **분석 로직 자체는 여전히 단순**합니다. `detect_faces(frame)`은 한 줄이고, 그 자체는 복잡하지 않습니다.

복잡해진 것은 분석 로직을 **조합하고 실행하는 인프라**입니다:

| 관심사 | 단순 루프에서 | 실제로 필요한 것 |
|--------|-------------|----------------|
| 실행 순서 | 직접 나열 | 의존성 그래프 자동 해석 |
| 데이터 전달 | 변수 전달 | deps 시스템 (프로세스 경계 포함) |
| 환경 격리 | 해당 없음 | Process/Venv 수준 격리 |
| IPC | 해당 없음 | visualbase.ipc 기반 직렬화 |
| 성능 측정 | print문 | 구조화된 Trace 시스템 |
| 설정 | 하드코딩 | 격리 수준/백엔드 설정 전환 |
| 에러 처리 | try/except | Worker 실패 복원, fallback |
| 재사용 | 복사-붙여넣기 | 플러그인 등록 |

이것들은 각각 따로 보면 "if문 몇 개"로 해결할 수 있습니다. 하지만 **동시에 조합**되면 단순한 루프로는 감당할 수 없는 복잡도가 됩니다.

---

## ML 생태계의 현실

위의 8단계는 의존성 충돌이라는 **한 가지 종류**의 문제만 다뤘습니다. 실제 ML/CV 개발에서는 더 근본적인 문제들이 있습니다.

### 실험적이고 불완전한 라이브러리

ML 생태계의 라이브러리들은 웹 프레임워크와 다릅니다. API가 자주 바뀌고, 문서가 부족하고, 특정 환경에서만 동작하는 코드가 흔합니다. 새로운 논문의 reference implementation을 가져다 쓰는 일도 빈번합니다 — 그 코드가 프로덕션 품질이 아닌 경우가 대부분입니다.

이런 상황에서 두 가지가 필요합니다:

**격리**: 불안정한 라이브러리가 전체 시스템을 오염시키지 않아야 합니다. 실험적인 표정 분석 모델이 segfault를 내더라도 얼굴 검출은 계속 돌아야 합니다.

**빠른 교체**: 모델 A가 기대에 못 미치면 모델 B로 교체해야 합니다. 이때 바꿔야 하는 것은 `analyze()` 내부뿐이어야 합니다. IPC 프로토콜을 바꾸거나 프로세스 관리 코드를 고치는 일이 되면 교체 비용이 급격히 올라갑니다.

단순한 루프에서는 이 두 가지가 모두 어렵습니다. 모든 코드가 같은 프로세스, 같은 루프 안에 있으니까요. 모델 하나를 교체하려면 루프 전체를 이해해야 하고, 격리하려면 루프를 분해해야 합니다.

### 리소스를 독점하는 프로세스

ML 워크로드는 경량 마이크로서비스와 다릅니다. 모델 하나가 GPU 메모리 수 GB를 점유하고, 추론 한 번에 CPU를 수십 ms 독점합니다. 여러 모델을 같은 프로세스에서 돌리면 메모리 경합, GPU context 전환, GIL 경쟁이 발생합니다.

프로세스 격리는 이 문제의 자연스러운 해결책이지만, 격리를 직접 구현하면 7단계의 코드가 됩니다 — 모듈마다 프로세스 생성, IPC, 설정 분기가 반복됩니다. 격리 자체가 목적이 아니라 분석 결과를 얻는 것이 목적인데, 인프라 코드가 분석 코드를 압도합니다.

### 파이프라인 구조와 실행 방식의 결합

가장 근본적이면서 잘 인식되지 않는 문제입니다.

"얼굴 검출 → 표정 분석 → 하이라이트 판정"이라는 **분석 흐름**은 동일한데, 이것을 **어떻게 실행하느냐**는 상황마다 다릅니다:

- 개발 중: 모든 모듈을 한 프로세스에서 순차 실행 (디버깅 편의)
- 스테이징: GPU 충돌 모듈만 프로세스 격리
- 프로덕션: 모듈별 독립 venv + 프로세스 격리
- 배치 처리: 프레임을 모아서 GPU 배치 추론

단순한 루프에서는 이 실행 방식이 루프 코드 자체에 박혀 있습니다. `for frame in video: faces = send_to_venv(...)` — 여기서 `send_to_venv`를 `direct_call`로 바꾸려면 루프를 고쳐야 합니다. 실행 방식을 바꿀 때마다 분석 흐름을 건드리게 되고, 분석 흐름을 바꿀 때마다 실행 방식의 제약을 고려해야 합니다.

이 둘이 결합되어 있으면, 같은 분석 로직을 다른 환경에서 돌리는 것이 사실상 별도 구현이 됩니다.

---

## visualpath가 흡수하는 것

visualpath는 위의 인프라 관심사를 플랫폼으로 분리합니다:

```
┌─────────────────────────────────────────────────────┐
│  vpx 플러그인 (분석 모듈)                             │
│                                                     │
│  class FaceDetectionAnalyzer(Module):               │
│      name = "face.detect"                           │
│      def analyze(self, frame, deps=None):           │
│          return detect_faces(frame)                 │
│                                                     │
│  class ExpressionAnalyzer(Module):                  │
│      name = "face.expression"                       │
│      depends = ["face.detect"]                      │
│      def analyze(self, frame, deps=None):           │
│          return analyze_expressions(frame, deps)    │
│                                                     │
│  → 분석 로직만 작성. 격리, IPC, 순서를 모름.          │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  visualpath (플랫폼)                                 │
│                                                     │
│  - depends 선언 → FlowGraph가 실행 순서 자동 결정    │
│  - deps 직렬화 → 프로세스 경계 넘어 자동 전달        │
│  - CUDA 충돌 감지 → WorkerModule 자동 격리           │
│  - onnxruntime 충돌 → venv 격리                     │
│  - 설정으로 격리 수준 전환 (IsolationConfig)          │
│  - TraceLevel로 성능 측정 on/off                     │
│  - entry point discovery로 모듈 자동 등록            │
│  - ErrorPolicy로 Worker 실패 시 retry/fallback      │
└─────────────────────────────────────────────────────┘
```

앱 개발자의 관점에서:

```python
# vpx 플러그인에서 하는 일의 전부
class PoseAnalyzer(Module):
    name = "body.pose"

    def analyze(self, frame, deps=None):
        results = self.model(frame.data)
        return Observation(source="body.pose", ...)
```

이 코드는 SimpleBackend에서 인라인 실행될 수도, WorkerBackend에서 별도 프로세스로 실행될 수도 있습니다. 모듈 코드는 동일합니다. 격리 결정은 `ModuleSpec.isolation` 설정에서 이루어집니다.

---

## 두 번째 앱이 나올 때

visualpath의 가치가 가장 명확해지는 시점은 **두 번째 분석 앱**이 만들어질 때입니다.

momentscan만 있을 때는 "과잉 설계"로 보일 수 있습니다. 하지만 portrait981의 전체 파이프라인이 가동되면:

```
momentscan (분석/수집)
  → visualpath의 Module, FlowGraph, WorkerBackend, ObservabilityHub 사용
  → Phase별로 같은 vpx 플러그인 재사용, 실행 패턴만 변경

appearance-vault (저장/관리)
  → momentscan의 출력을 memory bank로 저장, match/select API 제공

reportrait (AI 생성)
  → appearance-vault의 select_refs() → ComfyUI 연동
```

각 앱은 분석 로직(`analyze` 메서드)만 작성합니다. subprocess 관리, IPC, deps 전달, 성능 측정은 모두 visualpath가 제공합니다. 이것이 없으면 각 앱이 독자적으로 같은 인프라를 다시 만들어야 합니다.

---

## 개발자가 자기 일에 집중할 수 있는 구조

위의 8단계를 다시 봅시다. 단순한 루프가 복잡해진 원인은 **분석 알고리즘이 어려워서가 아닙니다**. CUDA 심볼 충돌, onnxruntime .so 덮어쓰기, IPC 직렬화, 프로세스 생명주기 관리 — 이것들은 ML/CV와 무관한 시스템 인프라 문제입니다.

문제는 이 두 종류의 작업이 같은 코드에 섞일 때 발생합니다:

| 작업 | ML/CV 전문성 | 시스템 인프라 전문성 |
|------|-------------|-------------------|
| 모델 선택 (SCRFD vs RetinaFace) | O | |
| 전처리/후처리 최적화 | O | |
| feature 설계, threshold 튜닝 | O | |
| ONNX 배치 추론 구현 | O | |
| CUDA 런타임 충돌 해결 | | O |
| 프로세스 격리 + IPC 설계 | | O |
| Worker 장애 복원, fallback | | O |
| 의존성 그래프 스케줄링 | | O |

visualpath가 없으면 ML/CV 개발자가 두 영역을 모두 다뤄야 합니다. 얼굴 검출 모델을 교체하려는데 IPC 프로토콜을 건드려야 하고, 표정 분석 정확도를 올리려는데 venv 격리 스크립트를 디버깅해야 합니다. 전문성의 경계가 무너집니다.

visualpath가 흡수하는 것은 오른쪽 열입니다. 분리 후:

```
ML/CV 개발자의 작업 범위:
  Module.analyze() 구현
  모델 선택, 전처리, 후처리
  threshold 튜닝, 정확도 개선
  → vpx 플러그인 하나가 경계

인프라의 작업 범위:
  FlowGraph, WorkerBackend, IsolationConfig
  IPC, 프로세스 관리, 에러 복원
  → visualpath가 경계
```

새로운 분석 모듈을 추가하는 개발자는 PyTorch나 ONNX Runtime을 알면 됩니다. `Module`을 상속하고 `analyze()`를 구현하면 끝입니다. 프로세스 격리가 필요하면 설정 한 줄을 바꾸지, Worker 코드를 작성하지 않습니다. 반대로, 인프라를 개선하는 개발자는 ML 모델의 내부를 알 필요 없이 Worker와 FlowGraph를 수정할 수 있습니다.

---

## momentscan(구 momentscan)에서 실증된 문제들

이론적인 주장이 아니라, 실제 개발에서 겪은 문제입니다:

| 문제 | 발생 시점 | 단순 루프의 한계 | visualpath의 해결 |
|------|----------|----------------|-------------------|
| onnxruntime GPU/CPU .so 덮어쓰기 | pip install | 같은 venv에서 해결 불가 | VenvWorker |
| onnxruntime-gpu + PyTorch CUDA 심볼 충돌 | runtime import | 같은 프로세스에서 해결 불가 | ProcessWorker |
| face.detect → face.expression → face.classify 의존성 | 모듈 분리 시 | 수동 변수 전달, 순서 관리 | depends/deps + FlowGraph |
| 프레임별 성능 병목 추적 | 최적화 시 | print문 산재 | TraceLevel + ObservabilityHub |
| 개발 환경 vs 프로덕션 격리 수준 차이 | 배포 시 | 코드 분기 | IsolationConfig로 전환 |
| 모듈 추가 시 루프 코드 수정 | 확장 시 | 루프에 코드 추가 | entry point 등록 (`vpx new`) |

자세한 기술 분석은 [isolation.md](../isolation.md)를 참조하세요.

---

## 정리

"루프에서 ML 라이브러리 몇 개 돌리면 되는 거 아니야?"

맞습니다, **프로토타입에서는**. 하지만 프로덕션에서는:

1. **환경 충돌**이 발생하고 (CUDA 런타임, pip 패키지)
2. **의존성 체인**이 복잡해지고 (face → expression → classifier)
3. **성능 측정**이 필요해지고 (어디서 느린지)
4. **격리 수준**이 환경마다 달라야 하고 (개발 vs 프로덕션)
5. **비슷한 앱**이 추가되면 인프라를 반복 구현하게 됩니다

visualpath는 이 인프라를 한 번 만들어서 모든 분석 앱이 공유하게 합니다. 앱 코드는 `analyze(frame, deps)` — 여전히 단순합니다.
