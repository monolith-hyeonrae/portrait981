# Portrait981 데모 시연 가이드

> 대상: 팀 내부 | 소요: ~25분 | 목표: 전체 파이프라인 흐름 이해 + 기반 기술 어필 + 개발 진행 상황 공유

---

## 1. 프로젝트 소개 (2분)

### 한 줄 요약

> "981파크 탑승 영상에서 **하이라이트 순간을 자동 감지**하고, **최적의 참조 프레임을 수집**한 뒤, **AI로 초상화를 생성**한다."

### 전체 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│  범용 레이어 (도메인 무관, 재사용 가능)                    │
│                                                          │
│  visualbase          visualpath           vpx-sdk        │
│  미디어 I/O          분석 프레임워크       플러그인 SDK    │
│  ┌────────┐         ┌──────────┐         ┌──────────┐   │
│  │ 소스    │  ──→   │ DAG 그래프│  ──→   │ 모듈 규약 │   │
│  │ 버퍼    │         │ 백엔드    │         │ 자동 등록 │   │
│  │ IPC     │         │ 격리      │         │ 시각화    │   │
│  └────────┘         └──────────┘         └──────────┘   │
└──────────────────────────────────────────────────────────┘
                          │
┌──────────────────────────────────────────────────────────┐
│  981파크 특화 레이어                                      │
│                                                          │
│  momentscan  ──→  momentbank  ──→  reportrait            │
│  분석/수집          저장/관리        AI 생성               │
└──────────────────────────────────────────────────────────┘
```

**핵심 설계 원칙**: 범용 레이어는 981파크를 모른다. 어떤 영상 분석 서비스에도 재사용 가능.

---

## 2. 범용 레이어 ① — visualbase (3분)

> 핵심 메시지: "어떤 영상 소스든 동일한 인터페이스로 다룬다"

### 통합 소스 추상화

```python
from visualbase import VisualBase, FileSource, CameraSource, RTSPSource

vb = VisualBase()
vb.connect(FileSource("video.mp4"))       # 파일
# vb.connect(CameraSource(0))             # USB 카메라
# vb.connect(RTSPSource("rtsp://..."))    # IP 카메라

for frame in vb.get_stream(fps=30):
    process(frame.data)   # 항상 동일한 Frame 객체
```

3가지 소스, 하나의 API. 소스 종류에 따라 내부 버퍼가 자동 선택됨:
- **파일** → FileBuffer (seek 기반)
- **스트리밍** → RingBuffer (tmpfs 순환 버퍼, 24시간 무중단 운영)

### Trigger 기반 클립 추출

```python
from visualbase import Trigger

# 특정 순간 ± 전후 구간 추출
trigger = Trigger.point(event_time_ns=10_000_000_000, pre_sec=3.0, post_sec=2.0)
result = vb.trigger(trigger)   # → 7초~12초 구간 MP4
```

FFmpeg stream copy로 빠른 추출, 파일/스트리밍 소스 모두 동작.

### IPC 인프라

모든 프로세스 간 통신의 **단일 인프라 레이어**:

| Transport | 프레임 전송 | 메시지 | RPC | 용도 |
|-----------|-----------|--------|-----|------|
| **FIFO** | O | - | - | 로컬 고속 스트리밍 |
| **UDS** | - | O | - | 로컬 메시지 (저오버헤드) |
| **ZMQ** | O | O | O | 분산/네트워크 배포 |

런타임에 전송 수단 교체 가능 (TransportFactory). visualpath-isolation이 이 위에 분석 프로토콜을 구현.

---

## 3. 범용 레이어 ② — visualpath (3분)

> 핵심 메시지: "파이프라인을 선언하면, 실행은 백엔드가 알아서 한다"

### 선언적 파이프라인 (AST/Interpreter 패턴)

```python
# 파이프라인 선언 (무엇을 할지)
graph = (FlowGraphBuilder()
    .source("frames")
    .path("analysis", modules=[face_detect, smile_trigger])
    .on_trigger(handle_trigger)
    .build())

# 실행은 백엔드에 위임 (어떻게 할지)
result = SimpleBackend().execute(frames, graph)
```

**FlowGraph** = 파이프라인의 AST (추상 구문 트리)
- 17종 NodeSpec으로 소스, 분석, 필터, 분기, 합류를 선언
- 모든 spec은 immutable dataclass

**Backend** = AST를 해석하는 인터프리터
- 동일한 그래프가 다른 백엔드에서 실행 가능

| 백엔드 | 특징 | 용도 |
|--------|------|------|
| **SimpleBackend** | 순차/병렬, 배치 처리 | 개발, 로컬 실행 |
| **WorkerBackend** | 모듈별 프로세스/venv 격리 | ML 의존성 충돌 해결 |
| **PathwayBackend** | Rust 스트리밍, 워터마크, 백프레셔 | 실시간 처리 |

### 4단계 격리 수준

코드 변경 없이 격리 수준만 바꿀 수 있다:

```
INLINE (0)  → 같은 프로세스, 같은 스레드 (가장 빠름)
THREAD (1)  → 같은 프로세스, 다른 스레드
PROCESS (2) → 같은 venv, 다른 프로세스
VENV (3)    → 다른 venv, 다른 프로세스 (onnxruntime GPU/CPU 공존)
```

워커 격리는 투명하게 동작 — 모듈 코드는 격리 수준을 모른다.

### 3-Level API

```python
# Level 0: 코어 (완전 제어)
graph = FlowGraph()
graph.add_node(source); graph.add_node(path); graph.add_edge(...)

# Level 1: 빌더 (간결한 선언)
graph = FlowGraphBuilder().source("s").path("p", modules=[...]).build()

# Level 2: 앱 컨벤션 (최소 코드)
class MyApp(vp.App):
    modules = ["face.detect", "face.expression"]
    def after_run(self, result): ...
result = MyApp().run("video.mp4")
```

---

## 4. 범용 레이어 ③ — vpx-sdk + 플러그인 시스템 (3분)

> 핵심 메시지: "분석 모듈을 만들고, 등록하고, 테스트하는 전체 워크플로우가 자동화되어 있다"

### 통합 모듈 인터페이스

```python
class FaceDetector(Module):
    depends = []                          # 의존성 선언

    @property
    def name(self) -> str:
        return "face.detect"              # 도메인.액션 네이밍

    def process(self, frame, deps=None) -> Observation:
        faces = self._detect(frame.data)
        return Observation(               # 통합 출력
            source=self.name,
            signals={"face_count": len(faces)},
            data=FaceDetectOutput(faces=faces),
        )
```

- 모든 모듈이 `Module` → `Observation` 단일 인터페이스
- `depends` 선언만으로 실행 순서 자동 결정 (토폴로지 정렬)
- `deps` dict로 이전 모듈 결과 접근

### 자동 스캐폴딩 — `vpx new` 시연

```bash
# 새 분석 모듈 자동 생성
vpx new face.landmark --depends face.detect
```

**한 줄로 생성되는 것들**:

```
libs/vpx/plugins/face-landmark/
├── pyproject.toml          # 패키지 메타데이터 + entry point 등록
├── src/vpx/face_landmark/
│   ├── __init__.py         # re-export
│   ├── analyzer.py         # FaceLandmarkAnalyzer 스텁
│   ├── output.py           # FaceLandmarkOutput 데이터클래스
│   └── backends/base.py    # 백엔드 Protocol
└── tests/
    └── test_face_landmark.py   # PluginTestHarness 포함
```

- root `pyproject.toml` workspace members에 **자동 등록**
- `vpx list`에 **자동 발견** (entry_points 기반, 수동 등록 불필요)
- `vpx run face.landmark --input video.mp4`로 **즉시 실행**

### 현재 플러그인 생태계

```
8개 vpx 플러그인 (외부 ML 모델):
  face.detect (InsightFace SCRFD)     face.expression (HSEmotion)
  face.parse  (BiSeNet)               face.au (LibreFace)
  head.pose   (6DRepNet)              portrait.score (CLIP)
  body.pose   (YOLO-Pose)             hand.gesture (MediaPipe)

6개 앱 내부 분석기 (ML 불필요):
  face.classify  face.quality  face.baseline
  face.gate      frame.quality  frame.scoring
```

### 시각화 마크 시스템

분석 모듈은 "무엇을 그릴지"만 선언, 렌더링은 별도:

```python
def annotate(self, obs) -> List[Mark]:
    return [
        BBoxMark(x1, y1, x2, y2, label="face:main"),   # 바운딩 박스
        AxisMark(cx, cy, yaw, pitch, roll),               # 3D 머리 축
        BarMark(x, y, value=0.8, label="smile"),          # 강도 바
    ]
```

---

## 5. 실시간 분석 시연 — momentscan debug (3분)

> 핵심 메시지: "앞에서 설명한 모든 것이 실제로 동작하는 모습"

```bash
# 모든 analyzer 활성화 — 실시간 오버레이 확인
momentscan debug video.mp4

# 얼굴 관련만 보기
momentscan debug video.mp4 -e face

# 분산 모드 (프로세스 격리 + 병렬 처리)
momentscan debug video.mp4 --distributed
```

### 시연 포인트

| 오버레이 요소 | 설명 | 관련 기술 |
|-------------|------|----------|
| 초록/주황 bbox | main/passenger 자동 분류 | face.classify |
| 표정 바 | 실시간 smile/surprise 강도 | face.expression |
| 3D 축 | 머리 회전 (yaw/pitch/roll) | head.pose (AxisMark) |
| Gate 상태 | PASS/FAIL + suitability | face.gate |
| 하단 패널 | 프레임 타이밍, 분석 요약 | observability |

**보여줄 것**: 키보드 1~8로 레이어 토글, 영상 끝나면 하이라이트 분석 자동 출력

---

## 6. 배치 처리 시연 — momentscan process (2분)

> 핵심 메시지: "한 줄 명령으로 영상 분석 → 하이라이트 탐지 → 리포트 생성까지 자동화"

```bash
momentscan process video.mp4 -o ./output
```

### 출력 결과

```
output/
├── highlight/
│   ├── 0/          # 하이라이트 구간 (프레임 이미지들)
│   └── ...
├── report.html     # 인터랙티브 분석 리포트 ← 브라우저에서 열기
├── windows.json    # 하이라이트 구간 메타데이터
└── collection/     # 수집된 참조 프레임 (pose × category 그리드)
```

**보여줄 것**: `report.html`에서 타임라인 + Score Decomposition 확인

### Scoring 공식

```
Final = 0.35 × Quality + 0.65 × Impact + 0.30 × Passenger Bonus

Quality: blur(0.30) + face_size(0.20) + identity/frontalness(0.30)
Impact:  smile(0.25) + head_yaw(0.15) + portrait_best(0.25)
```

---

## 7. 프레임 저장 + AI 생성 — momentbank → reportrait (3분)

> 핵심 메시지: "분석된 최적 프레임이 자동으로 저장되고, AI 초상화 생성까지 파이프라인이 연결된다"

### momentbank — 인물별 참조 프레임 축적

```python
from momentbank.ingest import lookup_frames

# pose별 조회
frames = lookup_frames("test_member", pose="frontal", top_k=3)

# category별 조회
frames = lookup_frames("test_member", category="warm_smile", top_k=5)
```

영상을 처리할수록 참조 프레임이 누적 → 품질이 점진적으로 향상.

### reportrait — ComfyUI 워크플로우 자동 생성

```bash
# dry-run: 워크플로우 주입 확인
reportrait generate test_member --pose frontal --dry-run

# 직접 이미지 + 커스텀 워크플로우
reportrait generate --ref face.jpg --workflow my_i2i.json --node 81

# 원격 GPU 서버 (RunPod)
reportrait generate --ref face.jpg \
  --comfy-url https://xxx.proxy.runpod.net \
  --api-key $RUNPOD_API_KEY
```

출력:
```
Generated 1 image(s) in 12.3s
  file:///home/user/output/person_0/ComfyUI_00001_.png
```

---

## 8. 기술 요약 + 규모 (2분)

### 범용 레이어가 주는 가치

| 레이어 | 핵심 가치 | 코드량 |
|--------|----------|--------|
| **visualbase** | 소스 추상화 + IPC 인프라 + 클립 추출 | ~4,000줄, 182 tests |
| **visualpath** | 선언적 DAG + 멀티 백엔드 + 격리 | ~16,000줄, 722 tests |
| **vpx-sdk** | 플러그인 규약 + 자동 스캐폴딩 + 테스트 하네스 | ~800줄, 130 tests |

→ 어떤 영상 분석 서비스에도 재사용 가능한 **범용 프레임워크**.

### 981파크 특화 레이어

| 앱 | 역할 | 테스트 |
|----|------|--------|
| **momentscan** | 14개 분석기 DAG + 하이라이트 탐지 + 수집 | 558 tests |
| **momentbank** | Identity bank + 프레임 저장/조회 | 61 tests |
| **reportrait** | ComfyUI 자동 생성 + RunPod 연동 | 41 tests |

### 전체 규모

```
19개 패키지  |  8개 ML 플러그인  |  14개 분석 모듈
~1,700 tests |  ~25,000줄 (범용) + ~15,000줄 (981파크)
```

---

## 시연 순서 요약

```
① 프로젝트 소개 (2분)
     전체 아키텍처: 범용 레이어 + 981파크 특화 레이어

② visualbase (3분)
     소스 추상화, RingBuffer, IPC Transport

③ visualpath (3분)
     선언적 DAG, 멀티 백엔드, 4단계 격리

④ vpx-sdk + 플러그인 (3분)
     통합 Module 인터페이스, vpx new 자동 스캐폴딩, 8개 플러그인

⑤ momentscan debug (3분)
     실시간 오버레이로 ①~④가 동작하는 모습

⑥ momentscan process (2분)
     배치 처리 → report.html 리포트

⑦ momentbank + reportrait (3분)
     프레임 저장 → AI 초상화 생성 데모

⑧ 기술 요약 (2분)
     코드 규모 + 범용성 어필
```

### 시연 전 체크리스트

- [ ] 테스트 영상 준비 (얼굴 2명 이상, 표정 변화 있는 것)
- [ ] `uv sync --all-packages --all-extras` 완료
- [ ] ComfyUI 서버 기동 (로컬 또는 RunPod)
- [ ] momentbank에 테스트 데이터 저장 (`momentscan process` 1회 실행)
- [ ] 브라우저 탭 열어놓기 (report.html 확인용)
