# MomentScan

981파크 Portrait981 파이프라인의 **얼굴/장면 분석 앱**.
visualpath 프레임워크(vp.App) + visualbind 4단 judge 기반으로 얼굴 감지, 표정 분석, 포즈 추정, 조명 분석, portrait moment 판별을 수행.

## 디렉토리 구조

```
src/momentscan/
├── __init__.py                # ms.run() re-export
├── main.py                    # run() — MomentscanV2 위임
├── v2.py                      # MomentscanV2(vp.App) — 핵심 분석 앱
├── v2_debug.py                # DebugV2 — cv2 디버그 오버레이 (3D cube, SH sphere, AU wireframe)
├── v2_report.py               # HTML 리포트 (Plotly: expression/gate/q×e×z timeline, AU heatmap)
├── cli/
│   ├── __init__.py            # main(), argparse (run + info)
│   ├── utils.py               # 노이즈 억제, 로그 설정
│   └── commands/
│       └── info.py            # momentscan info
├── algorithm/                 # (v1 legacy — 제거 예정)
├── signals/                   # (v1 legacy — 제거 예정)
└── visualize/                 # (v1 legacy — 제거 예정)
```

내부 analyzer는 `apps/momentscan-plugins/`로 분리됨 (momentscan namespace 공유):
```
apps/momentscan-plugins/
├── face-classify/       # FaceClassifierAnalyzer (역할 분류)
├── face-quality/        # FaceQualityAnalyzer (head crop blur/exposure + seg)
├── face-baseline/       # FaceBaselineAnalyzer (Welford online stats)
├── face-gate/           # FrameGateAnalyzer (per-face quality gate)
├── face-lighting/       # FaceLightingAnalyzer (DPR SH + 9-sector skin analysis)
├── frame-quality/       # QualityAnalyzer (프레임 전체 blur/brightness)
├── frame-scoring/       # FrameScoringAnalyzer (프레임 스코어링)
└── portrait-score/      # PortraitScoreAnalyzer (CLIP 4축 aesthetic)
```

ML 의존성이 필요한 외부 analyzer는 vpx 플러그인으로 분리됨 (vpx-face-detect, vpx-face-expression, vpx-body-pose, vpx-hand-gesture, vpx-face-au, vpx-head-pose, vpx-face-parse).

## High-Level API

```python
import momentscan as ms

results = ms.run("video.mp4")              # list[FrameResult]
results = ms.run("video.mp4", fps=2)

from momentscan.v2 import MomentscanV2
app = MomentscanV2(
    quality_model="models/quality_v1.pkl",
    expression_model="models/bind_v12.pkl",
    pose_model="models/pose_v10.pkl",
)
results = app.run("video.mp4", fps=2)
summary = app.summary(results)
selected = app.select_frames(results, top_k=10)
```

설정 상수: `DEFAULT_FPS=2`, `DEFAULT_BACKEND="simple"`

## MomentscanV2 아키텍처

```
Video Source (vp.App.run)
     │ Frame (per-frame)
     ▼
Analyzers (8개 모듈, DAG):
  face.detect → face.expression, face.au, head.pose
              → face.parse → face.quality, face.lighting
              → frame.quality
     │ Observations
     ▼
on_frame():
  bind_observations() → 65D signal dict
  VisualBind 4단 judge:
    1. gate (HeuristicStrategy) — 물리적 품질
    2. quality (TreeStrategy binary) — shoot/cut
    3. expression (TreeStrategy 5-class) — cheese/chill/edge/goofy/hype
    4. pose (TreeStrategy 3-class) — front/angle/side
  → FrameResult 축적
     ▼
after_run():
  1. Temporal smoothing (65D, moving average window=3)
  2. Fast-Slow z_score (20D expression only, per-video 정규화)
  → list[FrameResult]
```

## 65D Signal

visualbind가 observer 출력을 결합하여 생성하는 65차원 signal vector:

| 그룹 | 차원 | 필드 |
|------|------|------|
| AU (Action Unit) | 12D | au1~au26 (12개) |
| Emotion | 8D | em_happy~em_sad |
| Pose | 3D | head_yaw_dev, head_pitch, head_roll |
| Detection | 4D | face_confidence, area_ratio, center_distance, aspect_ratio |
| Face Quality | 5D | blur, exposure, contrast, clipped, crushed |
| Frame Quality | 3D | blur_score, brightness, contrast |
| Segmentation | 4D | seg_face, seg_eye, seg_mouth, seg_hair |
| Derived Seg | 6D | eye_visible, mouth_open, glasses, backlight, nose_x, nose_y |
| Lighting | 20D | 9-sector(8) + DPR summary(3) + DPR SH raw(9) |

## 4단 Judge (VisualBind)

| 단계 | 전략 | 모델 | 클래스 | 역할 |
|------|------|------|--------|------|
| gate | HeuristicStrategy | 없음 | pass/fail | 물리적 품질 (blur, exposure, yaw/pitch/roll 극단값, signal validity) |
| quality | TreeStrategy | quality_v1.pkl | shoot/cut | 객관적 사용 가능성 (binary) |
| expression | TreeStrategy | bind_v12.pkl | cheese/chill/edge/goofy/hype | shoot 프레임 표정 분류 (5-class) |
| pose | TreeStrategy | pose_v10.pkl | front/angle/side | shoot 프레임 포즈 분류 (3-class) |

`is_shoot`: gate 통과 + quality="shoot"

## q×e×z 프레임 선정

```
score = quality_conf × expression_conf × max(z_score, 0.1)
```

- **q** (quality_conf): shoot 확률 — "품질이 좋은가"
- **e** (expression_conf): 표정 확신도 — "확실한 표정인가"
- **z** (z_score): 비디오 내 상대적 특별함 — "평소 대비 특별한가"

`select_frames()`: expression×pose 버킷별 best q×e×z → top_k

## Fast-Slow z_score

Expression signal (AU 12D + Emotion 8D = 20D)만으로 per-video 정규화:
```python
z = (sig - mu) / sqrt(var)    # per-dimension z-score
z_score = sqrt(mean(z²))      # RMS → 스칼라
```

"같은 비디오 안에서 이 프레임이 얼마나 특별한가"를 수치화.

## CLI

```bash
momentscan run video.mp4                              # 기본 분석
momentscan run video.mp4 --fps 5                      # FPS 변경
momentscan run video.mp4 --debug                      # cv2 디버그 오버레이
momentscan run video.mp4 --debug --no-window -o out.mp4  # 파일 저장
momentscan run video.mp4 --report out.html            # HTML 리포트
momentscan run video.mp4 --ingest test_3              # personmemory에 SHOOT 프레임 저장

momentscan info                                       # analyzer/backend 상태
momentscan info --graph                               # FlowGraph 표시
```

### 기본 모델

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--quality-model` | `models/quality_v1.pkl` | shoot/cut binary |
| `--bind-model` | `models/bind_v12.pkl` | expression 5-class |
| `--pose-model` | `models/pose_v10.pkl` | pose 3-class |

## 디버그 오버레이 (v2_debug)

DebugV2(MomentscanV2) — cv2 창 or 파일 출력:
- **Panel**: Gate signals, Quality/Expression/Pose 확률바, AU face wireframe, SH lighting sphere
- **Video overlay**: face bbox (gate 색상), lighting skin contour, pose 3D cube, nose position
- **Timeline**: AU heatmap, gate severity, expression confidence 히스토리
- **Coverage grid**: expression×pose 버킷 커버리지

## MOMENTSCAN_MODULES

```python
MOMENTSCAN_MODULES = [
    "face.detect", "face.au", "face.expression", "head.pose",
    "face.parse", "face.quality", "face.lighting", "frame.quality",
]
```

## Analyzer 전체 목록

| Analyzer | name | depends | 패키지 |
|----------|------|---------|--------|
| FaceDetectionAnalyzer | `face.detect` | - | vpx-face-detect |
| ExpressionAnalyzer | `face.expression` | face.detect | vpx-face-expression |
| FaceParseAnalyzer | `face.parse` | face.detect | vpx-face-parse |
| FaceAUAnalyzer | `face.au` | face.detect | vpx-face-au |
| HeadPoseAnalyzer | `head.pose` | face.detect | vpx-head-pose |
| FaceQualityAnalyzer | `face.quality` | face.detect | momentscan-face-quality |
| FaceLightingAnalyzer | `face.lighting` | face.detect, face.parse | momentscan-face-lighting |
| QualityAnalyzer | `frame.quality` | - | momentscan-frame-quality |

## 테스트

```bash
cd /home/hyeonrae/repo/monolith/portrait981
uv sync --all-packages --all-extras
uv run pytest apps/momentscan/tests/ -v
```
