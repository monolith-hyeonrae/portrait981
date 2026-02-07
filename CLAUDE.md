# Portrait981 Monorepo

981파크 고객 경험 서비스. uv workspace 기반 모노레포.
각 패키지별 CLAUDE.md에 상세 내용.

## 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│  범용 레이어 (재사용 가능)                               │
│  visualbase (미디어 I/O) → visualpath (분석 프레임워크) │
└─────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────┐
│  981파크 특화 레이어                                     │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────┐  │
│  │ facemoment  │→ │ appearance-vault │→ │ reportrait│  │
│  │ (분석 앱)   │  │ (저장)           │  │ (AI 변환) │  │
│  └─────────────┘  └──────────────────┘  └───────────┘  │
│                              │                          │
│                   portrait981 (통합 앱)                 │
└─────────────────────────────────────────────────────────┘
```

## 패키지 관계

```
visualbase (미디어 I/O)
  → visualpath (분석 프레임워크)
      → facemoment (core: CLI, pipeline, fusion, classifier, quality, dummy)
          → facemoment-face-detect (InsightFace SCRFD, onnxruntime-gpu)
          → facemoment-expression  (HSEmotion, onnxruntime CPU)
          → facemoment-face        (legacy composite = face-detect + expression)
          → facemoment-pose        (YOLO-Pose, ultralytics)
          → facemoment-gesture     (MediaPipe Hands)
      → portrait981 (통합 오케스트레이터, 미구현)
```

## 디렉토리 구조

```
portrait981/                    ← repo root
├── libs/
│   ├── visualbase/             # 미디어 소스 (범용)
│   └── visualpath/
│       ├── core/               # 분석 프레임워크
│       ├── cli/                # CLI 도구
│       ├── isolation/          # Worker 격리
│       └── pathway/            # Pathway 백엔드
├── apps/
│   └── facemoment/
│       ├── core/               # 얼굴/장면 분석 core
│       └── extractors/
│           ├── face-detect/    # InsightFace SCRFD
│           ├── expression/     # HSEmotion
│           ├── face/           # legacy composite
│           ├── pose/           # YOLO-Pose
│           └── gesture/        # MediaPipe Hands
├── docs/
│   ├── ROADMAP.md
│   └── planning/
├── models/
│   └── yolov8m-pose.pt
├── pyproject.toml              # workspace root
└── CLAUDE.md
```

## Workspace 멤버

| 패키지 | 디렉토리 | 설명 |
|--------|----------|------|
| visualbase | `libs/visualbase/` | 미디어 소스 (범용) |
| visualpath | `libs/visualpath/core/` | 분석 프레임워크 (범용) |
| visualpath-isolation | `libs/visualpath/isolation/` | Worker 격리 |
| visualpath-cli | `libs/visualpath/cli/` | CLI 도구 |
| visualpath-pathway | `libs/visualpath/pathway/` | Pathway 백엔드 |
| facemoment | `apps/facemoment/core/` | 얼굴/장면 분석 core |
| facemoment-face-detect | `apps/facemoment/extractors/face-detect/` | 얼굴 검출 |
| facemoment-expression | `apps/facemoment/extractors/expression/` | 표정 분석 |
| facemoment-face | `apps/facemoment/extractors/face/` | 복합 (legacy) |
| facemoment-pose | `apps/facemoment/extractors/pose/` | 포즈 추정 |
| facemoment-gesture | `apps/facemoment/extractors/gesture/` | 제스처 감지 |

## Namespace Package 패턴

`facemoment` 네임스페이스를 여러 패키지가 공유. 4개 `__init__.py`에 `pkgutil.extend_path` 사용:
- `facemoment/__init__.py`
- `facemoment/moment_detector/__init__.py`
- `facemoment/moment_detector/extractors/__init__.py`
- `facemoment/moment_detector/extractors/backends/__init__.py`

import 경로는 변경 없이 네임스페이스 머징이 투명하게 동작.

## 개발 워크플로우

```bash
cd /home/hyeonrae/repo/monolith/portrait981
uv sync --all-packages --all-extras   # 전체 workspace 동기화
uv run pytest apps/facemoment/core/tests/ -v    # facemoment 테스트
```

## Production venv 격리

ML 의존성 충돌 방지를 위해 extractor별 독립 venv:

```bash
# face_detect: onnxruntime-gpu
uv venv venv-face-detect && uv pip install -e "apps/facemoment/extractors/face-detect"

# expression: onnxruntime CPU (hsemotion-onnx)
uv venv venv-expression && uv pip install -e "apps/facemoment/extractors/expression"

# pose / gesture: 각각 독립 venv
uv venv venv-pose && uv pip install -e "apps/facemoment/extractors/pose"
uv venv venv-gesture && uv pip install -e "apps/facemoment/extractors/gesture"
```

### onnxruntime GPU/CPU 충돌 방지

로컬 개발 시 workspace root `pyproject.toml`의 `override-dependencies`로 CPU onnxruntime 차단:
```toml
override-dependencies = ["onnxruntime ; sys_platform == 'never'"]
```
Production에서는 venv 격리로 해결 (override 불필요).
