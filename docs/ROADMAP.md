# Portrait981 개발 로드맵

> 최종 수정: 2026-02-02
> 현재 상태: **Phase 14 완료** (PipelineOrchestrator)

---

## 프로젝트 개요

### 아키텍처 계층

```
┌─────────────────────────────────────────────────────────┐
│  범용 레이어 (재사용 가능)                               │
│  visualbase (미디어 I/O) → visualpath (분석 프레임워크) │
│                                                         │
│  • 다른 프로젝트에서 import 가능                        │
│  • 비즈니스 로직 없음                                   │
│  • Action 처리 안 함 (콜백만 제공)                      │
└─────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────┐
│  981파크 특화 레이어 (앱)                                │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────┐  │
│  │ facemoment  │→ │ appearance-vault │→ │ reportrait│  │
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
| `visualbase` | 범용 라이브러리 | `visualbase` | 미디어 I/O | ✅ Phase 8 완료 (151 tests) |
| `visualpath` | 범용 프레임워크 | - | 분석 프레임워크 | ✅ 완료 |
| `facemoment` | 981파크 앱 | `facemoment` | 얼굴 분석, 이벤트 트리거 | ✅ Phase 14 완료 (160 tests) |
| `appearance-vault` | 981파크 앱 | `vault` | member_id별 저장/검색 | ⬜ Phase 9a |
| `reportrait` | 981파크 앱 | `reportrait` | I2I/I2V AI 재해석 | ⬜ Phase 9b |
| `portrait981` | 981파크 앱 | `p981` | 전체 파이프라인 통합 | ⬜ Phase 9c |

### 디렉토리 구조

```
~/repo/monolith/
├── visualbase/           # ✅ Phase 8 완료 (범용)
├── visualpath/           # ✅ 완료 (범용)
├── facemoment/           # ✅ Phase 14 완료 (981파크 앱)
├── appearance-vault/     # ⬜ Phase 9a (981파크 앱)
├── reportrait/           # ⬜ Phase 9b (981파크 앱)
└── portrait981/          # ⬜ Phase 9c (981파크 앱 - orchestrator)
```

---

## Phase 로드맵

### Phase 1-7: 핵심 기능 구현 ✅ 완료

| Phase | 패키지 | 내용 | 상태 |
|-------|--------|------|------|
| 1 | visualbase | FileSource, Frame, Sampler | ✅ |
| 2 | visualbase | FileBuffer, Clipper, Trigger | ✅ |
| 3 | facemoment | MomentDetector 스켈레톤 | ✅ |
| 4 | facemoment | FaceExtractor, PoseExtractor, QualityExtractor | ✅ |
| 5 | facemoment | HighlightFusion, 클립 추출 파이프라인 | ✅ |
| 6 | facemoment | HSEmotionBackend, benchmark CLI | ✅ |
| 7 | visualbase | CameraSource, RTSPSource, RingBuffer, CLI | ✅ |

### Phase 7.5: 패키지 Rename 및 GitHub 업로드 ✅ 완료

- ✅ portrait981-moment → facemoment rename
- ✅ 테스트 검증
- ✅ Git 초기화 완료

### Phase 8: A-B*-C 프로세스 분리 (IPC) ✅ 완료

| 단계 | 패키지 | 내용 | 상태 |
|------|--------|------|------|
| 8.0 | visualbase | IPC 인터페이스 추상화 (interfaces.py, factory.py) | ✅ |
| 8.1 | facemoment | ExtractorProcess를 인터페이스 의존으로 변경 | ✅ |
| 8.2 | facemoment | FusionProcess를 인터페이스 의존으로 변경 | ✅ |
| 8.3 | facemoment | ExtractorOrchestrator 구현 (스레드 병렬) | ✅ |
| 8.4 | visualbase | IngestProcess 정리 | ⬜ 리팩토링 필요 |
| 8.5 | both | CLI 확장 및 통합 테스트 | ⬜ 예정 |
| 8.6 | visualbase | ZeroMQ Transport | ✅ |
| 8.7 | visualbase | 데몬 모드 | ✅ |
| 8.8 | visualbase | WebRTC 출력 | ✅ |
| 8.9 | visualbase | GPU 가속 (nvdec/vaapi) | ✅ |

### Phase 9-14: GR차량 시나리오 및 플랫폼 분리 ✅ 완료

| 단계 | 패키지 | 내용 | 상태 |
|------|--------|------|------|
| 9.0 | facemoment | camera_gaze 트리거 | ✅ |
| 9.0 | facemoment | passenger_interaction 트리거 | ✅ |
| 9.0 | facemoment | GestureExtractor (V사인, 엄지척) | ✅ |
| 9.0 | facemoment | CLI --gokart 플래그 | ✅ |
| 10 | facemoment | Observability 시스템 | ✅ |
| 11 | facemoment | 의존성 분리 (Worker별 venv) | ✅ |
| 12 | visualpath | 플랫폼 로직 분리 (범용) | ✅ |
| 13 | visualpath | IPC 프로세스 이동 | ✅ |
| 14 | facemoment | PipelineOrchestrator (독립 앱) | ✅ |

### Phase 9a-9c: 981파크 완성 ⬜ 예정

| 단계 | 패키지 | 내용 | 상태 |
|------|--------|------|------|
| 9a | appearance-vault | member_id 기반 asset 저장 | ⬜ 예정 |
| 9b | reportrait | I2I/I2V AI 변환 | ⬜ 예정 |
| 9c | portrait981 | 전체 파이프라인 통합 | ⬜ 예정 |

---

## 아키텍처 결정 사항 (참고)

### 범용/특화 레이어 분리
- **범용 레이어**: visualbase, visualpath
  - 다른 프로젝트에서 재사용 가능
  - 비즈니스 로직 없음
  - on_trigger 콜백만 제공 (Action 처리 안 함)
- **특화 레이어**: facemoment, appearance-vault, reportrait, portrait981
  - 981파크 비즈니스 로직 포함
  - on_trigger → 클립 저장 등 Action 처리

### IPC 방식
- FIFO/UDS 기본 지원
- ZeroMQ Transport 추가 완료
- 인터페이스 추상화로 교체 가능

### 패키지 관계
- **Library 모드**: facemoment가 visualbase import (개발/단일 카메라)
- **독립 프로세스 모드**: portrait981 orchestrator가 프로세스 관리 (프로덕션/다중 카메라)

### Fanout 위치
- Library 모드: facemoment 내부 스레드 병렬
- 독립 프로세스 모드: visualbase가 FIFO×N 분배

---

## 테스트 현황

| 패키지 | 테스트 수 | 상태 |
|--------|----------|------|
| visualbase | 151 tests | ✅ |
| visualpath | - | ✅ |
| facemoment | 160 tests | ✅ |

---

## 검증 명령어

```bash
# visualbase 테스트
cd ~/repo/monolith/visualbase && uv run pytest tests/ -v

# facemoment 테스트
cd ~/repo/monolith/facemoment && uv run pytest tests/ -v

# E2E 테스트
uv run facemoment process video.mp4 -o ./clips --fps 10

# GR차량 모드 테스트
uv run facemoment process video.mp4 --gokart -o ./clips
```

---

## 세션별 컨텍스트 문서

각 패키지의 독립적인 세션 개발을 위한 CLAUDE.md 문서:

- `visualbase/CLAUDE.md`: 미디어 I/O (범용 라이브러리)
- `visualpath/CLAUDE.md`: 분석 프레임워크 (범용 프레임워크)
- `facemoment/CLAUDE.md`: 얼굴/장면 분석 (981파크 앱)
- `portrait981/CLAUDE.md`: 통합 오케스트레이터 (981파크 앱)
