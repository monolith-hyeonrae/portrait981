# Portrait981 Monorepo

981파크 고객 경험 서비스. uv workspace 기반 모노레포.
각 패키지별 CLAUDE.md에 상세 내용.

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│  범용 레이어 (visual* Platform)                                  │
│  visualbase (미디어 I/O + 원본 관리 + ROI + Trigger)            │
│  → visualpath (분석 프레임워크, FlowGraph, warm executor)       │
│  → vpx-sdk (Module 프로토콜: initialize/process/reset/release)  │
│  → vpx-* (비전 분석 모듈 7개)                                    │
│  → visualbind (65D signal binding + 4단 judge)                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  981파크 앱 레이어                                               │
│  momentscan (분석) → personmemory (기억) → reportrait (생성)    │
│                              │                                   │
│  portrait981 (오케스트레이터, warm scanner 배치)                 │
│  portrait981-serve (REST API, FastAPI + S3 + ComfyUI)           │
│  portrait981-gateway (향후, Kafka/SQS adapter)                  │
│  annotator (라벨링: expression + pose + lighting)               │
└─────────────────────────────────────────────────────────────────┘
```

## 핵심 설계 원칙

### Module Lifecycle (warm executor)
```
initialize()     ← 모델 로딩 1회 (GPU 모델, XGBoost)
├── scan(video1) ← module.reset()만 (모델 유지)
├── scan(video2)
└── scan(videoN)
shutdown()       ← module.release() (GPU 메모리 해제)
```
`Module.cleanup() = reset() + release()`. warm 모드에서는 reset()만 호출.

### 4단 Judge (VisualBind)
```
gate (HeuristicStrategy)  → 물리적 품질 (blur, exposure, yaw 극단)
quality (TreeStrategy)    → shoot/cut binary (quality_v1.pkl)
expression (TreeStrategy) → 5-class: cheese/chill/edge/goofy/hype (bind_v12.pkl)
pose (TreeStrategy)       → 3-class: front/angle/side (pose_v10.pkl)
```

### 65D Signal
AU(12) + Emotion(8) + Pose(3) + Detection(4) + FaceQuality(5) + FrameQuality(3) + Seg(4) + DerivedSeg(6) + Lighting(20)

### q×e×z 프레임 선정
`score = quality_conf × expression_conf × max(z_score, 0.1)`
- z_score: per-video 상대적 특별함 (AU+Emotion 20D, Fast-Slow)

### ROI + 좌표계 (설계 진행중)
```
ROI 계층: Global ⊃ Body ⊃ Portrait ⊃ Head ⊃ Face
좌표 태그: Coord(value, Space.NORM, roi="portrait") — naked float 금지
ROISpec: 공식 정의 (name, expand, size, aspect_ratio)
ROICrop: 실현된 crop + to_frame()/from_frame() 좌표 변환
```
face.detect가 ROI 생성, face.lighting이 첫 번째 마이그레이션 완료.

### Trigger Flow (설계 완료, 구현 예정)
```
visualbase (원본 보유) → 분석용 경량 프레임 → analyzers → SHOOT → Trigger
                       ← Trigger 수신 → 원본 품질 프레임/클립 추출
```
SourceProfile: 소스 미디어 특성 (codec, bit_depth, color_space) 보존.

### 다인 탑승 (설계 완료, 구현 예정)
- ride_type: 비디오 전체 맥락에서 SOLO/DUO/GROUP 결정 (프레임 단위 판단 금지)
- per-face signal + per-face judge
- scene_score: 둘 다 SHOOT일 때 min(person_scores)

## 디렉토리 구조

```
portrait981/                    ← repo root
├── libs/
│   ├── visualbase/             # 미디어 I/O + ROI + Trigger + SourceProfile
│   ├── visualpath/
│   │   ├── core/               # 분석 프레임워크 + warm executor
│   │   ├── cli/                # CLI 도구
│   │   ├── isolation/          # Worker 격리
│   │   └── pathway/            # Pathway 백엔드
│   ├── visualbind/             # 65D signal binding + 4단 judge
│   └── vpx/
│       ├── sdk/                # Module SDK + ROI specs
│       ├── runner/             # Analyzer 러너 (LiteRunner)
│       ├── viz/                # 시각화
│       └── plugins/            # 7개 vpx analyzer
├── apps/
│   ├── momentscan/             # 분석 앱 (app/ + cli/)
│   ├── momentscan-plugins/     # 3개 내부 analyzer
│   │   ├── face-quality/       # 얼굴 품질 (blur/exposure + seg)
│   │   ├── face-lighting/      # 조명 분석 (DPR SH + 9-sector skin)
│   │   └── frame-quality/      # 프레임 blur/brightness
│   ├── personmemory/           # Identity memory bank
│   ├── reportrait/             # AI 초상화 생성 (ComfyUI)
│   ├── portrait981/            # 오케스트레이터 (scan → lookup → generate)
│   ├── portrait981-serve/      # REST API (FastAPI + S3)
│   ├── portrait981-docs/       # 문서 사이트 (mkdocs)
│   └── annotator/              # 라벨링/리뷰/병합
├── data/datasets/portrait-v1/  # images/ + labels.csv
├── models/                     # XGBoost .pkl + DPR .t7
├── scripts/                    # 분석/비교 스크립트
└── pyproject.toml              # workspace root
```

## Import 경로

```python
# visualbase
from visualbase import Frame, FileSource, ImageSource, open_video
from visualbase.core.roi import ROISpec, ROICrop
from visualbase.core.coordinate import Coord, Coord3D, Space, Space3D
from visualbase.sources.profile import SourceProfile

# visualpath
from visualpath.core.module import Module
from visualpath.backends.simple.executor import GraphExecutor

# vpx-sdk
from vpx.sdk import Module, Observation
from vpx.sdk.roi_specs import FACE, PORTRAIT, HEAD, ROI_REGISTRY

# vpx analyzers
from vpx.face_detect import FaceDetectionAnalyzer
from vpx.face_detect.output import FaceDetectOutput  # .roi_crops 포함

# visualbind
from visualbind import VisualBind, HeuristicStrategy, TreeStrategy, bind_observations
from visualbind.judge import JudgmentResult
from visualbind.signals import SIGNAL_FIELDS, normalize_signal

# momentscan
from momentscan.app import Momentscan, FrameResult, SignalSummary
from momentscan.app.debug import MomentscanDebug
from momentscan.app.report import export_report
import momentscan as ms
results = ms.run("video.mp4")
result = ms.extract_signals(image_bgr)

# portrait981
from portrait981 import Portrait981Pipeline, JobSpec, PipelineConfig
```

## 개발 워크플로우

```bash
cd /home/hyeonrae/repo/monolith/portrait981
uv sync --all-packages --all-extras   # 전체 workspace 동기화
uv run pytest libs/visualbase/tests/ -v    # visualbase (182)
uv run pytest libs/visualpath/core/tests/ -v  # visualpath (411)
uv run pytest libs/visualbind/tests/ -v    # visualbind (59)
uv run pytest apps/momentscan/tests/ -v    # momentscan
uv run pytest apps/portrait981/tests/ -v   # portrait981 (55)
uv run pytest apps/portrait981-serve/tests/ -v  # serve (16)

# 문서 사이트
cd apps/portrait981-docs
uv run --package portrait981-docs --extra serve mkdocs serve
```

## CLI

```bash
# momentscan
momentscan run video.mp4                    # 기본 분석
momentscan run video.mp4 --debug            # cv2 디버그 오버레이
momentscan run video.mp4 --report out.html  # HTML 리포트
momentscan run video.mp4 --ingest test_3    # personmemory 저장
momentscan info                             # analyzer 상태

# portrait981
p981 run video.mp4 --member-id test_3       # E2E 파이프라인
p981 batch videos/ --scan-only --ingest     # 배치 scan + ingest
p981 batch videos/ --ingest                 # 배치 전체 (scan + generate)
p981 generate test_3 --pose frontal         # 생성만
p981 status test_3                          # personmemory 현황

# annotator
annotator label video.mp4 --fps 2 --output labels.html
annotator review data/datasets/portrait-v1 --output review.html

# vpx
vpx list                                    # 등록 analyzer
vpx run face.detect --input video.mp4       # analyzer 단독 실행
vpx new face.landmark --depends face.detect # 새 모듈 스캐폴딩

# visualbind
visualbind train --data signals.parquet --labels labels.json --strategy xgboost
visualbind eval --data signals.parquet --labels labels.json --model model.pkl
```

## 데이터셋

```
data/datasets/portrait-v1/
├── images/         ← 모든 이미지 (gitignore, 로컬만)
├── labels.csv      ← 다축 라벨 (filename, member_id, expression, pose, lighting, source)
└── dataset.yaml

Expression: cheese 🧀 / chill 🧊 / edge 🗡️ / goofy 🤪 / hype 🔥
Pose:       front / angle / side
Lighting:   dramatic 🔦 / natural ☀️ / backlit 🌅
```

## 현재 진행 상태

### 완료
- momentscan v2 (v1 전면 제거, app/ 구조, lifecycle)
- VisualBind 4단 judge + q×e×z 프레임 선정
- warm executor (Module reset/release, GraphExecutor keep_warm)
- p981 batch --ingest (warm scanner 재사용)
- portrait981-serve v2 (scan 응답, personmemory, ingest)
- visualbase: Frame 유틸, ImageSource, open_video, SourceProfile
- ROI + 좌표계 Phase 1~3 + face.lighting 마이그레이션
- DPR 조명 (SfSNet 폐기), annotator lighting 3-class

### 설계 완료, 구현 예정
- ROI Phase 4~5: FlowGraph routing + 나머지 analyzer 마이그레이션
- Trigger Flow: SHOOT → visualbase 원본 추출 + SourceProfile 기반 코덱
- 다인 탑승: ride_type 앵커 + per-face signal + scene scoring
- 데일리 운영: staging + 실험 리포트 + multi-annotator shoot 선택
- portrait981-gateway: Kafka/SQS adapter (운영 배포 시점)
