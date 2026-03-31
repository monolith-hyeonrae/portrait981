# Portrait981 개발 로드맵

> 최종 수정: 2026-03-09
> 현재 상태: **Phase 21 완료** (코드베이스 리팩토링 + 데이터 계약)

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
│  │ momentscan  │→ │ personmemory │→ │ reportrait│  │
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
| `visualpath` | 범용 프레임워크 | - | FlowGraph, Backend, Interpreter | ✅ 완료 (411 tests) |
| `visualpath-isolation` | 범용 프레임워크 | - | Worker 프로세스 격리 | ✅ 완료 (136 tests) |
| `visualpath-pathway` | 범용 프레임워크 | - | Pathway 스트리밍 백엔드 | ✅ 완료 (80 tests) |
| `visualpath-cli` | 범용 프레임워크 | `visualpath` | YAML 기반 파이프라인 CLI | ✅ 완료 (45 tests) |
| `vpx-sdk` | 범용 라이브러리 | - | 모듈 SDK, PluginTestHarness | ✅ 완료 (66 tests) |
| `vpx-runner` | 범용 도구 | `vpx` | Analyzer 러너 CLI + 스캐폴딩 | ✅ 완료 (64 tests) |
| `vpx-viz` | 범용 도구 | - | 시각화 오버레이 | ✅ 완료 |
| `vpx-face-detect` | 범용 플러그인 | - | InsightFace SCRFD | ✅ 완료 |
| `vpx-face-expression` | 범용 플러그인 | - | HSEmotion | ✅ 완료 |
| `vpx-face-parse` | 범용 플러그인 | - | BiSeNet face segmentation | ✅ 완료 |
| `vpx-face-au` | 범용 플러그인 | - | ONNX Action Unit | ✅ 완료 |
| `vpx-head-pose` | 범용 플러그인 | - | 6DRepNet 6DoF | ✅ 완료 |
| `vpx-portrait-score` | backward-compat shim | - | → momentscan 내부로 이전 | ✅ shim 유지 |
| `vpx-body-pose` | 범용 플러그인 | - | YOLO-Pose | ✅ 완료 |
| `vpx-hand-gesture` | 범용 플러그인 | - | MediaPipe Hands | ✅ 완료 |
| `momentscan` | 981파크 앱 | `momentscan` | 얼굴/장면 분석, 하이라이트, 수집 | ✅ 완료 (584 tests) |
| `momentscan-report` | 981파크 앱 | - | HTML 리포트 생성 (Plotly) | ✅ 완료 (12 tests) |
| `personmemory` | 981파크 앱 | - | member_id별 저장/검색 | ✅ 완료 (61 tests) |
| `reportrait` | 981파크 앱 | `reportrait` | I2I/I2V AI 변환 (ComfyUI) | ✅ 완료 (41 tests) |
| `portrait981` | 981파크 앱 | `p981` | 전체 파이프라인 통합 | ✅ 완료 (47 tests) |

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
│           ├── face-parse/         # BiSeNet face segmentation
│           ├── face-au/            # ONNX Action Unit
│           ├── head-pose/          # 6DRepNet 6DoF
│           ├── portrait-score/     # backward-compat shim (→ momentscan)
│           ├── body-pose/          # YOLO-Pose
│           └── hand-gesture/       # MediaPipe Hands
├── apps/
│   ├── momentscan/                 # 얼굴/장면 분석 + 수집 앱
│   ├── momentscan-report/          # HTML 리포트 생성 (Plotly)
│   ├── personmemory/                 # Identity memory bank + 프레임 저장
│   ├── reportrait/                 # AI 초상화 생성 (ComfyUI)
│   └── portrait981/                # 통합 오케스트레이터 (E2E 파이프라인)
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

### Phase 18: 981파크 앱 파이프라인 구현 ✅ 완료 (2026-02-12 ~ 03-01)

Batch Highlight 위에 Identity Collection + personmemory + reportrait 전체 파이프라인 완성.

| 단계 | 패키지 | 내용 | 커밋 |
|------|--------|------|------|
| 18.1 | momentscan | ArcFace/DINOv2 임베딩 하이라이트 통합 | `884c4bf` |
| 18.2 | momentscan, personmemory | vpx-vision-embed + personmemory 앱 | `e0f5784` |
| 18.3 | momentscan | Phase 3 identity builder + 듀얼 임베딩 파이프라인 | `8c37341` |
| 18.4 | momentscan | Pivot 기반 identity 수집 시스템 | `031634c` |
| 18.5 | vpx | vpx-face-au (LibreFace AU) + vpx-head-pose (6DRepNet) | `d9be948` |
| 18.6 | momentscan | face.gate + face.quality seg 개선 + 디버그 오버레이 | `e973ba6` |
| 18.7 | momentscan | face.baseline (Welford online stats) + gate scoring 개선 | `70aa55b` |
| 18.8 | momentscan | passenger suitability score + 리포트 업데이트 | `3036c45` |
| 18.9 | momentscan | Reference-Guided Collection + 카탈로그 기반 Impact Score | `f9e6e5f` |
| 18.10 | reportrait | CLI + personmemory ingest + distributed 병렬 수정 | `960abed` |

**Phase 2 (Embedding Experiment) 결과**: DINOv2/SigLIP temporal delta를 시도했으나,
~2분 라이드 비디오 특성에서 범용 임베딩의 temporal delta는 수치 feature 대비 이점이 불충분.
대신 **CLIP 4축 portrait scoring** (disney_smile, charisma, wild_roar, playful_cute)과
**카탈로그 기반 유사도**로 도메인 특화 접근을 채택하여 더 나은 결과를 얻음.
→ Phase 2는 실험 완료, 별도 진행 불필요.

**Phase 3 (Identity Collection) 진화**: 초기 버킷 기반(yaw×pitch×expression) 설계에서
**Pose×Category 그리드 + 카탈로그 유사도 기반 수집**으로 발전.
portrait.score(CLIP)가 momentscan 내부 모듈로 정착.

### Phase 19: vpx-face-parse + quality gate 고도화 ✅ 완료 (2026-02-23 ~ 03-01)

BiSeNet 세그멘테이션 기반 quality gate 5단계 체인 완성.

| 단계 | 내용 |
|------|------|
| 19.1 | vpx-face-parse (BiSeNet) 플러그인 |
| 19.2 | face.gate — 5단계 gate chain (confidence → blur → parsing → exposure → seg) |
| 19.3 | face.quality — head crop blur/exposure + BiSeNet seg ratios (face/eye/mouth/hair) |
| 19.4 | 3-level exposure fallback: contrast(mask) → brightness(absolute) → frame-level |
| 19.5 | vision-embed → portrait-score 분리 + head_aesthetic scoring 제거 |

### Phase 20: momentscan → reportrait E2E CLI ✅ 완료 (2026-03-01 ~ 03-05)

3-app 파이프라인 CLI 연동 및 distributed 모드 안정화.

| 단계 | 내용 |
|------|------|
| 20.1 | reportrait CLI (`reportrait generate`) — lookup_frames 연동, `--ref`/`--dry-run` |
| 20.2 | reportrait ComfyClient — urllib REST, Bearer 토큰, URL 스킴 자동 보정 |
| 20.3 | reportrait workflow injection — `_meta.role` 기반 LoadImage/prompt 자동 주입 |
| 20.4 | personmemory ingest — debug CLI에서 자동 bank 저장 |
| 20.5 | distributed 모드 — parallel 기본 활성화, ZMQ tuple 직렬화 버그 수정 |

### Phase 21: 코드베이스 리팩토링 + 데이터 계약 ✅ 완료 (2026-03-08 ~ 03-09)

아키텍처 경계 정리 및 모듈 표준화.

| 단계 | 내용 | 커밋 |
|------|------|------|
| 21.1 | momentscan-report 패키지 분리 (3,052줄 → 독립 패키지, 12 tests) | `916d759` |
| 21.2 | debug.py 중복 제거 (`_run_batch_and_export` 추출, 66줄 감소) | `916d759` |
| 21.3 | vpx-portrait-score → momentscan 내부 마이그레이션 (1,232줄) | `8d059d3` |
| 21.4 | 내부 analyzer Module 포맷 정규화 (depends, capabilities, lifecycle) | `7647d45` |
| 21.5 | Protocol 기반 데이터 계약 (contracts.py: FIELD_SOURCES 57필드, CONSUMER_DEPS) | `0933caf` |
| 21.6 | annotate() 메서드 추가 (quality, frame_scoring, face_baseline) + 검증 함수 | `060a4d3` |

### Phase 22: portrait981 통합 오케스트레이터 ✅ 완료 (2026-03-09)

scan → bank → generate E2E 파이프라인 통합 및 CLI.

| 단계 | 내용 |
|------|------|
| 22.1 | `Portrait981Pipeline` — JobSpec → scan → lookup → generate → JobResult |
| 22.2 | 2-Phase 배치 실행 — scan 순차(GPU+SIGINT) + generate 병렬(I/O) |
| 22.3 | StepEvent 콜백 — 프레임 단위 실시간 진행 추적 |
| 22.4 | Rich 라이브 테이블 — 배치 진행 현황 (스피너 + FPS + 프레임 카운터) |
| 22.5 | PARTIAL 상태 — scan 성공 + generate 실패 시 생성만 재시도 가능 |
| 22.6 | CLI (`p981`) — run, batch, scan, generate, status 서브커맨드 |
| 22.7 | Ctrl+C 배치 중단 — `interrupt()` → 현재 스캔 완료 후 나머지 스킵 |

### Phase 23: 서비스 인프라 연동 ⬜ 예정

사내 인프라(Kafka, S3)와의 연동 레이어 구축.
상세 설계: [portrait981-serve.md](../apps/portrait981-serve.md)

| 단계 | 패키지 | 내용 | 상태 |
|------|--------|------|------|
| 23.1 | visualbase | S3 미디어 소스 — fetch + 로컬 캐시, optional extra (`visualbase[s3]`) | ⬜ |
| 23.2 | portrait981-serve | Kafka consumer — 신규 이벤트 구독 (상위 팀 생성), E2E 자동 처리 | ⬜ Kafka 미팅 후 |
| 23.3 | portrait981-serve | REST API — scan-only, generate-only, test, 상태 조회 | ⬜ |
| 23.4 | portrait981-serve | Kafka producer — `portraitCreatedEvent` 발행, 결과 S3 업로드 | ⬜ |
| 23.5 | portrait981-serve | ComfyUI 노드풀 라우팅 (가용 노드 선택) | ⬜ |

```
cju-activity-status-api
    │  [Kafka] 신규 이벤트 (Avro, 상위 팀 생성)
    ▼
portrait981-serve (scan 노드, GPU 중급)
    │  S3 fetch → scan (로컬) → bank
    │  generate → ComfyUI 노드풀 (GPU 고급, 별도 장비)
    │  결과 S3 업로드
    │  [Kafka] portraitCreatedEvent
    ▼
상위 서비스
```

**아키텍처 원칙 확립**:
- vpx = "what do you see?" (범용, 도메인 비종속)
- momentscan = "what does it mean for 981park?" (해석, 도메인 특화)
- portrait.score (CLIP scoring)는 981파크 도메인 특화 → momentscan 내부가 적합

---

## 981파크 앱 현황 요약

### momentscan (분석/수집)

**3-Phase 진화 완료**:

| Phase | 설명 | 상태 | 비고 |
|-------|------|------|------|
| Phase 1 | Batch Highlight (MAD z-score + peak detection) | ✅ 완료 | 9채널 scoring, HTML report |
| Phase 2 | Embedding Experiment (DINOv2/SigLIP temporal delta) | ✅ 실험 완료 | CLIP portrait scoring으로 대체 |
| Phase 3 | Identity Collection (Pose×Category 그리드) | ✅ 완료 | 카탈로그 기반 수집 |

**14개 Analyzer DAG**:
```
face.detect → face.classify → face.baseline (stateful)
     │ → face.expression, face.quality, face.parse, portrait.score, face.au, head.pose
     └→ face.gate (depends: detect+classify, optional: quality+frame.quality+head.pose)
body.pose, hand.gesture, frame.quality (independent)
frame.scoring (depends: face.detect, optional: expression+classify+pose+quality)
```

### personmemory (저장/관리)

- `ingest_collection()`: CollectionResult → member 디렉토리 저장
- `lookup_frames(member_id, pose=, category=, top_k=)`: 참조 이미지 조회
- JSON persistence (members.json)

### reportrait (AI 변환)

- `PortraitGenerator`: template load → ref inject → prompt inject → queue → wait → download
- `ComfyClient`: urllib REST, Bearer 토큰 인증, RunPod proxy 지원
- CLI: `reportrait generate` — `--ref`, `--pose`, `--category`, `--dry-run`, `--node`

---

## 남은 작업

| 항목 | 우선순위 | 설명 |
|------|---------|------|
| visualbase S3 소스 | 높 | 사내 S3에서 미디어 소스 fetch + 로컬 캐시 |
| portrait981-serve | 높 | Kafka(E2E) + REST(scan/generate/test/status), Avro 직렬화 |
| ComfyUI 노드풀 라우팅 | 중 | 가용 generate 노드 선택 (GPU 고급, 별도 장비) |
| Kafka 미팅 | 높 | 수신 토픽 확정 (상위 팀 생성), 발행 토픽 등록, consumer group 규약 |
| backward-compat shim 정리 | 낮 | vpx-portrait-score, export_report shim 제거 (외부 사용자 확인 후) |

---

## 아키텍처 결정 사항 (참고)

### 범용/특화 레이어 분리
- **범용 레이어**: visualbase, visualpath, vpx-sdk, vpx-*
  - 다른 프로젝트에서 재사용 가능
  - 비즈니스 로직 없음
  - on_trigger 콜백만 제공 (Action 처리 안 함)
- **특화 레이어**: momentscan, personmemory, reportrait, portrait981
  - 981파크 비즈니스 로직 포함
  - on_trigger → 클립 저장 등 Action 처리

### 패키지 의존 방향
```
visualpath → vpx-sdk → vpx-* → momentscan (app)
                                    ↓
                              personmemory → reportrait
```
- Analyzer는 `vpx-sdk`에 의존, `momentscan`에 의존하지 않음
- momentscan는 `vpx-*`를 optional extras로 참조
- portrait.score는 도메인 특화 모듈이므로 momentscan 내부에 위치

### Analyzer 이름 규칙
`domain.action` 점표기법 사용. 패키지명도 일치:

| Analyzer | 패키지 | Import |
|----------|--------|--------|
| `face.detect` | vpx-face-detect | `vpx.face_detect` |
| `face.expression` | vpx-face-expression | `vpx.face_expression` |
| `face.parse` | vpx-face-parse | `vpx.face_parse` |
| `face.au` | vpx-face-au | `vpx.face_au` |
| `head.pose` | vpx-head-pose | `vpx.head_pose` |
| `body.pose` | vpx-body-pose | `vpx.body_pose` |
| `hand.gesture` | vpx-hand-gesture | `vpx.hand_gesture` |
| `face.classify` | momentscan core | `momentscan.algorithm.analyzers.face_classifier` |
| `face.quality` | momentscan core | `momentscan.algorithm.analyzers.face_quality` |
| `face.gate` | momentscan core | `momentscan.algorithm.analyzers.frame_gate` |
| `face.baseline` | momentscan core | `momentscan.algorithm.analyzers.face_baseline` |
| `portrait.score` | momentscan core | `momentscan.algorithm.analyzers.portrait_score` |
| `frame.quality` | momentscan core | `momentscan.algorithm.analyzers.quality` |
| `frame.scoring` | momentscan core | `momentscan.algorithm.analyzers.frame_scoring` |

### IPC 방식
- FIFO/UDS 기본 지원
- ZeroMQ Transport 추가 완료
- 인터페이스 추상화로 교체 가능

### 실행 경로
- 단일 실행 경로: `ms.run()` → `build_graph(isolation_config)` → `Backend.execute()`
- `WorkerBackend`이 isolated 모듈을 `WorkerModule`로 래핑 후 `SimpleBackend`에 위임
- `--distributed` 모드: 프로세스 격리 + 병렬 실행 동시 활성화

### Fanout 위치
- Library 모드: momentscan 내부 스레드 병렬
- 독립 프로세스 모드: visualbase가 FIFO×N 분배

---

## 테스트 현황

| 패키지 | 디렉토리 | 테스트 수 |
|--------|----------|----------|
| visualbase | `libs/visualbase/tests/` | 151 |
| visualpath core | `libs/visualpath/core/tests/` | 411 |
| visualpath-isolation | `libs/visualpath/isolation/tests/` | 136 |
| visualpath-pathway | `libs/visualpath/pathway/tests/` | 80 |
| visualpath-cli | `libs/visualpath/cli/tests/` | 45 |
| vpx-sdk | `libs/vpx/sdk/tests/` | 66 |
| vpx-runner | `libs/vpx/runner/tests/` | 64 |
| momentscan | `apps/momentscan/tests/` | 584 |
| momentscan-report | `apps/momentscan-report/tests/` | 12 |
| personmemory | `apps/personmemory/tests/` | 61 |
| reportrait | `apps/reportrait/tests/` | 41 |
| portrait981 | `apps/portrait981/tests/` | 47 |
| **합계** | | **1,698** |

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
uv run pytest apps/momentscan-report/tests/ -v
uv run pytest apps/personmemory/tests/ -v
uv run pytest apps/reportrait/tests/ -v

# portrait981 통합 테스트
uv run pytest apps/portrait981/tests/ -v
uv run p981 run video.mp4 --member-id test_1
uv run p981 batch /path/to/videos/ --scan-only

# E2E 테스트
uv run momentscan process video.mp4 -o ./clips --fps 10

# GR차량 모드 테스트
uv run momentscan process video.mp4 --gokart -o ./clips

# 디버그 모드
uv run momentscan debug video.mp4 -e face,pose --distributed
```
