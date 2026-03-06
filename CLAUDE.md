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
│  │ momentscan  │→ │ momentbank │→ │ reportrait│  │
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
          → vpx-face-parse       (BiSeNet, face segmentation)
          → vpx-face-au          (ONNX, Action Unit)
          → vpx-head-pose        (6DRepNet, 6DoF)
          → vpx-portrait-score   (backward-compat shim → momentscan)
          → vpx-body-pose        (YOLO-Pose, ultralytics)
          → vpx-hand-gesture     (MediaPipe Hands)
      → momentscan (분석/수집 앱)
      → momentbank (저장/관리)
      → reportrait (AI 초상화 생성, ComfyUI 브릿지)
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
│           ├── face-parse/     # BiSeNet face segmentation
│           ├── face-au/        # ONNX Action Unit
│           ├── head-pose/      # 6DRepNet 6DoF
│           ├── portrait-score/ # backward-compat shim (→ momentscan)
│           ├── body-pose/      # YOLO-Pose
│           └── hand-gesture/   # MediaPipe Hands
├── apps/
│   ├── momentscan/             # 얼굴/장면 분석 + 수집
│   ├── momentbank/             # Identity memory bank + 프레임 저장
│   ├── reportrait/             # AI 초상화 생성 (ComfyUI 브릿지)
│   └── momentscan-report/      # HTML 리포트 생성 (Plotly)
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
| vpx-face-detect | `libs/vpx/plugins/face-detect/` | 얼굴 검출 (InsightFace SCRFD) |
| vpx-face-expression | `libs/vpx/plugins/face-expression/` | 표정 분석 (HSEmotion) |
| vpx-face-parse | `libs/vpx/plugins/face-parse/` | 얼굴 세그멘테이션 (BiSeNet) |
| vpx-face-au | `libs/vpx/plugins/face-au/` | Action Unit 분석 (ONNX) |
| vpx-head-pose | `libs/vpx/plugins/head-pose/` | 6DoF 머리 포즈 (6DRepNet) |
| vpx-portrait-score | `libs/vpx/plugins/portrait-score/` | backward-compat shim (→ momentscan) |
| vpx-body-pose | `libs/vpx/plugins/body-pose/` | 포즈 추정 (YOLO-Pose) |
| vpx-hand-gesture | `libs/vpx/plugins/hand-gesture/` | 제스처 감지 (MediaPipe) |
| vpx-sdk | `libs/vpx/sdk/` | 모듈 SDK |
| vpx-runner | `libs/vpx/runner/` | Analyzer 러너 |
| vpx-viz | `libs/vpx/viz/` | 시각화 도구 |
| momentscan | `apps/momentscan/` | 얼굴/장면 분석 + 수집 |
| momentbank | `apps/momentbank/` | Identity memory bank + 프레임 저장 |
| reportrait | `apps/reportrait/` | AI 초상화 생성 (ComfyUI) |
| momentscan-report | `apps/momentscan-report/` | HTML 리포트 생성 (Plotly) |

## Namespace Package 패턴

두 개의 namespace를 여러 패키지가 공유:

**`vpx` namespace** (4개 analyzer + sdk + runner + viz 패키지):
- 각 `libs/vpx/*/src/vpx/__init__.py`에 `pkgutil.extend_path` 사용

**`momentscan` namespace** (core 패키지):
- `momentscan/__init__.py`
- `momentscan/algorithm/__init__.py`
- `momentscan/algorithm/analyzers/__init__.py` (vpx에서 re-import)

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

# momentscan-specific (vpx 플러그인 구조)
from momentscan.algorithm.analyzers.face_classifier import FaceClassifierAnalyzer
from momentscan.algorithm.analyzers.face_classifier import ClassifiedFace, FaceClassifierOutput
from momentscan.algorithm.analyzers.quality import QualityAnalyzer
from momentscan.algorithm.analyzers.quality import QualityOutput

# DummyAnalyzer (visualpath core, 테스트 전용)
from visualpath.core import DummyAnalyzer
```

## 개발 워크플로우

```bash
cd /home/hyeonrae/repo/monolith/portrait981
uv sync --all-packages --all-extras   # 전체 workspace 동기화
uv run pytest apps/momentscan/tests/ -v    # momentscan 테스트 (558)
uv run pytest apps/momentbank/tests/ -v    # momentbank 테스트 (61)
uv run pytest apps/reportrait/tests/ -v    # reportrait 테스트 (41)
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
vpx new scene.transition --internal            # momentscan 내부 모듈
vpx new face.landmark --dry-run                # 미리보기
```

`vpx new`는 파일 생성 후 root `pyproject.toml`의 workspace members에 자동 등록한다.

## reportrait CLI

```bash
# lookup_frames로 참조 이미지 조회 후 생성
reportrait generate test_3 --pose left30 --dry-run
reportrait generate test_3 --category warm_smile --prompt "portrait"

# 직접 참조 이미지 지정 (lookup 건너뜀)
reportrait generate --ref photo1.jpg photo2.jpg --prompt "portrait"

# 워크플로우 파일 직접 지정 (I2I/I2V)
reportrait generate --ref face.jpg --workflow /path/to/i2v.json

# 특정 노드에만 이미지 주입
reportrait generate --ref face.jpg --workflow workflow.json --node 81

# 원격 ComfyUI 서버 (RunPod)
reportrait generate --ref face.jpg --comfy-url https://xxx.proxy.runpod.net --api-key $RUNPOD_API_KEY
```

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
