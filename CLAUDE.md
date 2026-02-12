# Portrait981 Monorepo

981파크 고객 경험 서비스. uv workspace 기반 모노레포.
각 패키지별 CLAUDE.md에 상세 내용.

## 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│  범용 레이어 (재사용 가능)                               │
│  visualbase (미디어 I/O) → visualpath (분석 프레임워크) │
│  vpx-sdk (공유 타입/프로토콜) + vpx-* (비전 분석 모듈) │
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
visualbase (미디어 I/O + IPC 인프라)
  → visualpath (분석 프레임워크)
      → visualpath-isolation (Worker 격리, visualbase.ipc 기반)
      → vpx-sdk (공유 타입: Observation, Module, protocols)
          → vpx-face-detect      (InsightFace SCRFD, onnxruntime-gpu)
          → vpx-face-expression  (HSEmotion, onnxruntime CPU)
          → vpx-body-pose        (YOLO-Pose, ultralytics)
          → vpx-hand-gesture     (MediaPipe Hands)
      → facemoment (core: CLI, analyzers, monitoring)
      → portrait981 (통합 오케스트레이터, 미구현)
```

## 디렉토리 구조

```
portrait981/                    ← repo root
├── libs/
│   ├── visualbase/             # 미디어 소스 (범용)
│   ├── visualpath/
│   │   ├── core/               # 분석 프레임워크
│   │   ├── cli/                # CLI 도구
│   │   ├── isolation/          # Worker 격리
│   │   └── pathway/            # Pathway 백엔드
│   └── vpx/
│       ├── sdk/                # vpx-sdk (모듈 SDK)
│       ├── runner/             # vpx-runner (Analyzer 러너)
│       ├── viz/                # vpx-viz (시각화)
│       └── plugins/            # Analyzer 플러그인
│           ├── face-detect/    # InsightFace SCRFD
│           ├── face-expression/# HSEmotion
│           ├── body-pose/      # YOLO-Pose
│           └── hand-gesture/   # MediaPipe Hands
├── apps/
│   └── facemoment/             # 얼굴/장면 분석 core
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
| vpx-face-detect | `libs/vpx/plugins/face-detect/` | 얼굴 검출 |
| vpx-face-expression | `libs/vpx/plugins/face-expression/` | 표정 분석 |
| vpx-body-pose | `libs/vpx/plugins/body-pose/` | 포즈 추정 |
| vpx-hand-gesture | `libs/vpx/plugins/hand-gesture/` | 제스처 감지 |
| vpx-sdk | `libs/vpx/sdk/` | 모듈 SDK |
| vpx-runner | `libs/vpx/runner/` | Analyzer 러너 |
| facemoment | `apps/facemoment/` | 얼굴/장면 분석 core |

## Namespace Package 패턴

두 개의 namespace를 여러 패키지가 공유:

**`vpx` namespace** (4개 analyzer + sdk + runner + viz 패키지):
- 각 `libs/vpx/*/src/vpx/__init__.py`에 `pkgutil.extend_path` 사용

**`facemoment` namespace** (core 패키지):
- `facemoment/__init__.py`
- `facemoment/algorithm/__init__.py`
- `facemoment/algorithm/analyzers/__init__.py` (vpx에서 re-import)

## Import 경로

```python
# 공유 타입 (vpx-sdk)
from vpx.sdk import Module, Observation
from vpx.face_detect.types import FaceObservation
from vpx.face_detect.output import FaceDetectOutput
from vpx.body_pose.output import PoseOutput
from vpx.body_pose.types import KeypointIndex
from vpx.hand_gesture.types import GestureType

# Analyzer (vpx 패키지)
from vpx.face_detect import FaceDetectionAnalyzer
from vpx.face_expression import ExpressionAnalyzer
from vpx.body_pose import PoseAnalyzer
from vpx.hand_gesture import GestureAnalyzer

# facemoment-specific (vpx 플러그인 구조)
from facemoment.algorithm.analyzers.face_classifier import FaceClassifierAnalyzer
from facemoment.algorithm.analyzers.face_classifier import ClassifiedFace, FaceClassifierOutput
from facemoment.algorithm.analyzers.quality import QualityAnalyzer
from facemoment.algorithm.analyzers.quality import QualityOutput

# DummyAnalyzer (visualpath core, 테스트 전용)
from visualpath.core import DummyAnalyzer
```

## 개발 워크플로우

```bash
cd /home/hyeonrae/repo/monolith/portrait981
uv sync --all-packages --all-extras   # 전체 workspace 동기화
uv run pytest apps/facemoment/tests/ -v    # facemoment 테스트
```

## vpx CLI

```bash
# 등록된 analyzer 목록
vpx list

# analyzer 실행
vpx run face.detect --input video.mp4 --max-frames 100
vpx run face.detect,face.expression --input video.mp4 --fps 5

# 새 모듈 스캐폴딩
vpx new face.landmark                          # vpx 플러그인 생성
vpx new face.landmark --depends face.detect    # 의존 모듈 지정
vpx new face.landmark --no-backend             # backends/ 생략
vpx new scene.transition --internal            # facemoment 내부 모듈
vpx new face.landmark --dry-run                # 미리보기
```

`vpx new`는 파일 생성 후 root `pyproject.toml`의 workspace members에 자동 등록한다.

## Production venv 격리

ML 의존성 충돌 방지를 위해 analyzer별 독립 venv:

```bash
# face_detect: onnxruntime-gpu
uv venv venv-face-detect && uv pip install -e "libs/vpx/plugins/face-detect"

# face_expression: onnxruntime CPU (hsemotion-onnx)
uv venv venv-face-expression && uv pip install -e "libs/vpx/plugins/face-expression"

# body_pose / hand_gesture: 각각 독립 venv
uv venv venv-body-pose && uv pip install -e "libs/vpx/plugins/body-pose"
uv venv venv-hand-gesture && uv pip install -e "libs/vpx/plugins/hand-gesture"
```

### onnxruntime GPU/CPU 충돌 방지

로컬 개발 시 workspace root `pyproject.toml`의 `override-dependencies`로 CPU onnxruntime 차단:
```toml
override-dependencies = ["onnxruntime ; sys_platform == 'never'"]
```
Production에서는 venv 격리로 해결 (override 불필요).
