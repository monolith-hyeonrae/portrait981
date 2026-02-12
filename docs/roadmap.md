# Portrait981 개발 로드맵

> 최종 수정: 2026-02-11
> 현재 상태: **Phase 17 완료** (vpx new 스캐폴딩)

---

## 프로젝트 개요

### 아키텍처 계층

```
┌─────────────────────────────────────────────────────────┐
│  범용 레이어 (재사용 가능)                               │
│  visualbase (미디어 I/O) → visualpath (분석 프레임워크) │
│  vpx-sdk (공유 타입/프로토콜) + vpx-* (비전 분석 모듈) │
│                                                         │
│  • 다른 프로젝트에서 import 가능                        │
│  • 비즈니스 로직 없음                                   │
│  • Action 처리 안 함 (콜백만 제공)                      │
└─────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────┐
│  981파크 특화 레이어 (앱)                                │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────┐  │
│  │ momentscan  │→ │ appearance-vault │→ │ reportrait│  │
│  │ (분석 앱)   │  │ (저장)           │  │ (AI 변환) │  │
│  └─────────────┘  └──────────────────┘  └───────────┘  │
│                              │                          │
│                   portrait981 (통합 앱)                 │
│                                                         │
│  • 비즈니스 로직 포함                                   │
│  • on_trigger → 클립 저장 등 Action 처리               │
└─────────────────────────────────────────────────────────┘
```

### 패키지 정의

| 패키지 | 성격 | CLI | 역할 | 상태 |
|--------|------|-----|------|------|
| `visualbase` | 범용 라이브러리 | `visualbase` | 미디어 I/O | ✅ 완료 (151 tests) |
| `visualpath` | 범용 프레임워크 | - | FlowGraph, Backend, Isolation | ✅ 완료 (314 tests) |
| `visualpath-isolation` | 범용 프레임워크 | - | Worker 프로세스 격리 | ✅ 완료 (136 tests) |
| `visualpath-pathway` | 범용 프레임워크 | - | Pathway 스트리밍 백엔드 | ✅ 완료 (80 tests) |
| `visualpath-cli` | 범용 프레임워크 | `visualpath` | YAML 기반 파이프라인 CLI | ✅ 완료 (45 tests) |
| `vpx-sdk` | 범용 라이브러리 | - | 모듈 SDK, PluginTestHarness | ✅ 완료 (43 tests) |
| `vpx-runner` | 범용 도구 | `vpx` | Analyzer 러너 CLI + 스캐폴딩 | ✅ 완료 (55 tests) |
| `vpx-viz` | 범용 도구 | - | 시각화 오버레이 | ✅ 완료 |
| `vpx-face-detect` | 범용 플러그인 | - | InsightFace SCRFD | ✅ 완료 |
| `vpx-face-expression` | 범용 플러그인 | - | HSEmotion | ✅ 완료 |
| `vpx-body-pose` | 범용 플러그인 | - | YOLO-Pose | ✅ 완료 |
| `vpx-hand-gesture` | 범용 플러그인 | - | MediaPipe Hands | ✅ 완료 |
| `momentscan` | 981파크 앱 | `momentscan` | 얼굴/장면 분석, 하이라이트 | ✅ 완료 (327 tests) |
| `appearance-vault` | 981파크 앱 | `vault` | member_id별 저장/검색 | ⬜ 예정 |
| `reportrait` | 981파크 앱 | `reportrait` | I2I/I2V AI 재해석 | ⬜ 예정 |
| `portrait981` | 981파크 앱 | `p981` | 전체 파이프라인 통합 | ⬜ 예정 |

### 디렉토리 구조

```
portrait981/                        ← repo root (uv workspace)
├── libs/
│   ├── visualbase/                 # 미디어 소스 (범용)
│   ├── visualpath/
│   │   ├── core/                   # FlowGraph, Backend, Interpreter
│   │   ├── isolation/              # Worker 프로세스 격리
│   │   ├── pathway/                # Pathway 스트리밍 백엔드
│   │   └── cli/                    # YAML 기반 파이프라인 CLI
│   └── vpx/
│       ├── sdk/                    # 모듈 SDK
│       ├── runner/                 # Analyzer 러너 CLI
│       ├── viz/                    # 시각화 오버레이
│       └── plugins/                # Analyzer 플러그인
│           ├── face-detect/        # InsightFace SCRFD
│           ├── face-expression/    # HSEmotion
│           ├── body-pose/          # YOLO-Pose
│           └── hand-gesture/       # MediaPipe Hands
├── apps/
│   └── momentscan/                 # 얼굴/장면 분석 앱
├── docs/
│   ├── index.md                    # 문서 인덱스
│   ├── architecture.md             # 아키텍처
│   ├── stream-synchronization.md   # 스트림 동기화
│   ├── isolation.md                # ML 의존성 격리
│   ├── algorithms.md               # 하이라이트 알고리즘
│   ├── roadmap.md                  # 로드맵 (이 파일)
│   └── planning/                   # 기획/히스토리
├── models/
│   └── yolov8m-pose.pt
├── pyproject.toml                  # workspace root
└── CLAUDE.md
```

---

## Phase 로드맵

### Phase 1-7: 핵심 기능 구현 ✅ 완료

| Phase | 패키지 | 내용 | 상태 |
|-------|--------|------|------|
| 1 | visualbase | FileSource, Frame, Sampler | ✅ |
| 2 | visualbase | FileBuffer, Clipper, Trigger | ✅ |
| 3 | momentscan | MomentDetector 스켈레톤 | ✅ |
| 4 | momentscan | FaceExtractor, PoseExtractor, QualityExtractor | ✅ |
| 5 | momentscan | HighlightFusion, 클립 추출 파이프라인 | ✅ |
| 6 | momentscan | HSEmotionBackend, benchmark CLI | ✅ |
| 7 | visualbase | CameraSource, RTSPSource, RingBuffer, CLI | ✅ |

### Phase 7.5: 패키지 Rename 및 GitHub 업로드 ✅ 완료

- ✅ portrait981-moment → momentscan rename
- ✅ 테스트 검증
- ✅ Git 초기화 완료

### Phase 8: Worker 프로세스 격리 (IPC) ✅ 완료

| 단계 | 패키지 | 내용 | 상태 |
|------|--------|------|------|
| 8.0 | visualbase | IPC 인터페이스 추상화 (interfaces.py, factory.py) | ✅ |
| 8.1 | momentscan | ExtractorProcess를 인터페이스 의존으로 변경 | ✅ |
| 8.2 | momentscan | FusionProcess를 인터페이스 의존으로 변경 | ✅ |
| 8.3 | momentscan | ExtractorOrchestrator 구현 (스레드 병렬) | ✅ |
| 8.4 | visualbase | IngestProcess 정리 | ⬜ 리팩토링 필요 |
| 8.5 | both | CLI 확장 및 통합 테스트 | ⬜ 예정 |
| 8.6 | visualbase | ZeroMQ Transport | ✅ |
| 8.7 | visualbase | 데몬 모드 | ✅ |
| 8.8 | visualbase | WebRTC 출력 | ✅ |
| 8.9 | visualbase | GPU 가속 (nvdec/vaapi) | ✅ |

### Phase 9-14: GR차량 시나리오 및 플랫폼 분리 ✅ 완료

| 단계 | 패키지 | 내용 | 상태 |
|------|--------|------|------|
| 9.0 | momentscan | camera_gaze / passenger_interaction 트리거 | ✅ |
| 9.0 | momentscan | GestureExtractor (V사인, 엄지척) | ✅ |
| 9.0 | momentscan | CLI --gokart 플래그 | ✅ |
| 10 | momentscan | Observability 시스템 | ✅ |
| 11 | momentscan | 의존성 분리 (Worker별 venv) | ✅ |
| 12 | visualpath | 플랫폼 로직 분리 (범용) | ✅ |
| 13 | visualpath | IPC 프로세스 이동 | ✅ |
| 14 | momentscan | PipelineOrchestrator (독립 앱) | ✅ |

### Phase 15: 모노레포 구조화 + Analyzer 독립화 ✅ 완료 (2026-02-07 ~ 02-09)

portrait981 모노레포로 통합 및 analyzer 패키지 독립화.

| 단계 | 내용 | 커밋 |
|------|------|------|
| 15.1 | portrait981 모노레포 초기 구성 (libs/apps 계층) | `e4a62f0` |
| 15.2 | Extractor → vpx 네임스페이스 독립 패키지화 | `d64dd56` |
| 15.3 | --distributed 모드 프로세스 격리 수정 | `2abaffb` |
| 15.4 | Extractor → Analyzer 전체 리네이밍 | `8c89d7a` |
| 15.5 | distributed 모드 face.expression 누락 수정 | `6eb36ef` |
| 15.6 | Output/Type을 vpx 패키지로 이동 | `d80b6ab` |
| 15.7 | vpx 패키지명 analyzer name 일치화 + plugins/ 분리 | `dfa6377` |

### Phase 16: Module 인터페이스 안정화 + vpx infra ✅ 완료 (2026-02-09)

Module boundary 설계 및 vpx 인프라 패키지 추가.

| 단계 | 내용 | 커밋 |
|------|------|------|
| 16.1 | Capability, PortSchema, ErrorPolicy, ExecutionProfile | `a2bc565` |
| 16.2 | vpx-sdk, vpx-runner, vpx-viz 패키지 | `4a471dc` |

### Phase 17: vpx new 스캐폴딩 ✅ 완료 (2026-02-11)

새 vpx 모듈 생성 자동화. `vpx new` CLI 서브커맨드 추가.

| 단계 | 내용 |
|------|------|
| 17.1 | `scaffold.py` — 이름 파생, 템플릿 생성, workspace 등록 |
| 17.2 | `cli.py` — `new` 서브커맨드 파서 및 핸들러 |
| 17.3 | `test_scaffold.py` — 31 tests (dry-run, 파일 생성, workspace 등록, harness 검증) |

### Phase 9a-9c: 981파크 완성 ⬜ 예정

| 단계 | 패키지 | 내용 | 상태 |
|------|--------|------|------|
| 9a | appearance-vault | member_id 기반 asset 저장 | ⬜ 예정 |
| 9b | reportrait | I2I/I2V AI 변환 | ⬜ 예정 |
| 9c | portrait981 | 전체 파이프라인 통합 | ⬜ 예정 |

---

## 아키텍처 결정 사항 (참고)

### 범용/특화 레이어 분리
- **범용 레이어**: visualbase, visualpath, vpx-sdk, vpx-*
  - 다른 프로젝트에서 재사용 가능
  - 비즈니스 로직 없음
  - on_trigger 콜백만 제공 (Action 처리 안 함)
- **특화 레이어**: momentscan, appearance-vault, reportrait, portrait981
  - 981파크 비즈니스 로직 포함
  - on_trigger → 클립 저장 등 Action 처리

### 패키지 의존 방향
```
visualpath → vpx-sdk → vpx-* → momentscan (app)
```
- Analyzer는 `vpx-sdk`에 의존, `momentscan`에 의존하지 않음
- momentscan는 `vpx-*`를 optional extras로 참조

### Analyzer 이름 규칙
`domain.action` 점표기법 사용. 패키지명도 일치:

| Analyzer | 패키지 | Import |
|----------|--------|--------|
| `face.detect` | vpx-face-detect | `vpx.face_detect` |
| `face.expression` | vpx-face-expression | `vpx.face_expression` |
| `face.classify` | momentscan core | `momentscan.algorithm.analyzers.face_classifier` |
| `body.pose` | vpx-body-pose | `vpx.body_pose` |
| `hand.gesture` | vpx-hand-gesture | `vpx.hand_gesture` |
| `frame.quality` | momentscan core | `momentscan.algorithm.analyzers.quality` |
| `mock.dummy` | momentscan core | `momentscan.algorithm.analyzers.dummy` |

### IPC 방식
- FIFO/UDS 기본 지원
- ZeroMQ Transport 추가 완료
- 인터페이스 추상화로 교체 가능

### 실행 경로
- 단일 실행 경로: `ms.run()` → `build_graph(isolation_config)` → `Backend.execute()`
- `WorkerBackend`이 isolated 모듈을 `WorkerModule`로 래핑 후 `SimpleBackend`에 위임
- `MomentscanPipeline`, `PipelineOrchestrator`는 deprecated (unified path로 위임)

### Fanout 위치
- Library 모드: momentscan 내부 스레드 병렬
- 독립 프로세스 모드: visualbase가 FIFO×N 분배

---

## 테스트 현황

| 패키지 | 디렉토리 | 테스트 수 |
|--------|----------|----------|
| visualbase | `libs/visualbase/tests/` | 151 |
| visualpath core | `libs/visualpath/core/tests/` | 314 |
| visualpath-isolation | `libs/visualpath/isolation/tests/` | 136 |
| visualpath-pathway | `libs/visualpath/pathway/tests/` | 80 |
| visualpath-cli | `libs/visualpath/cli/tests/` | 45 |
| vpx-sdk | `libs/vpx/sdk/tests/` | 43 |
| vpx-runner | `libs/vpx/runner/tests/` | 55 |
| momentscan | `apps/momentscan/tests/` | 326 |
| **합계** | | **1,150** |

---

## 검증 명령어

```bash
cd ~/repo/monolith/portrait981

# 전체 workspace 동기화
uv sync --all-packages --all-extras

# 패키지별 테스트
uv run pytest libs/visualbase/tests/ -v
uv run pytest libs/visualpath/core/tests/ -v
uv run pytest libs/visualpath/isolation/tests/ -v
uv run pytest libs/visualpath/pathway/tests/ -v
uv run pytest libs/visualpath/cli/tests/ -v
uv run pytest libs/vpx/sdk/tests/ -v
uv run pytest libs/vpx/runner/tests/ -v
uv run pytest apps/momentscan/tests/ -v

# E2E 테스트
uv run momentscan process video.mp4 -o ./clips --fps 10

# GR차량 모드 테스트
uv run momentscan process video.mp4 --gokart -o ./clips
```
