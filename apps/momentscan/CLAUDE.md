# MomentScan

981파크 Portrait981 파이프라인의 **얼굴/장면 분석 앱**.
visualpath 프레임워크(vp.App) + visualbind 4단 judge 기반.

## 디렉토리 구조

```
src/momentscan/
├── __init__.py          # ms.run(), ms.extract_signals(), Momentscan re-export
├── __main__.py          # python -m momentscan → cli.main()
├── app/
│   ├── __init__.py      # Momentscan, FrameResult, SignalSummary re-export
│   ├── core.py          # Momentscan(vp.App) — 핵심 분석 앱
│   ├── debug.py         # MomentscanDebug — cv2 디버그 오버레이
│   └── report.py        # export_report() — Plotly HTML 리포트
└── cli/
    ├── __init__.py      # main() — argparse, app으로 위임
    ├── utils.py         # 노이즈 억제, 로그 설정
    └── commands/
        ├── __init__.py
        └── info.py      # momentscan info
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

## Lifecycle

```
initialize()          ← 모델 로딩 (1회, 무거움)
├── scan(video1)      ← 프레임 처리 (가벼움, 상태만 리셋)
├── scan(video2)
├── scan(video3)
└── ...
shutdown()            ← 리소스 해제
```

`setup()`은 vp.App lifecycle hook으로, 첫 호출 시 `initialize()`를 lazy 실행.
배치 실행 시 `initialize()`를 명시적으로 호출하면 모델을 재사용.

## API

```python
# 단발 실행
import momentscan as ms
results = ms.run("video.mp4")
result = ms.extract_signals(image_bgr)   # 단일 이미지 (FlowGraph 경로)

# 배치 실행 (모델 1회 로딩)
from momentscan.app import Momentscan

app = Momentscan(
    quality_model="models/quality_v1.pkl",
    expression_model="models/bind_v12.pkl",
    pose_model="models/pose_v10.pkl",
)
app.initialize()
for video in videos:
    results = app.scan(video)
    selected = app.select_frames(results, top_k=10)
    summary = app.summary(results)
app.shutdown()
```

## 진입점

```
momentscan run video.mp4          → pyproject.toml [scripts] → cli.main() → app.scan()
python -m momentscan run video.mp4 → __main__.py → cli.main() → app.scan()
import momentscan as ms; ms.run()  → __init__.py → app.scan()
```

CLI는 argparse 파싱만 담당. 앱 조립과 실행은 app이 중심.

## 아키텍처

```
Video Source (vp.App.run)
     │ Frame (per-frame)
     ▼
Analyzers (8개 모듈, FlowGraph DAG):
  Level 0: face.detect, frame.quality
  Level 1: face.au, face.expression, head.pose, face.parse
  Level 2: face.quality, face.lighting
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
| gate | HeuristicStrategy | 없음 | pass/fail | 물리적 품질 |
| quality | TreeStrategy | quality_v1.pkl | shoot/cut | 객관적 사용 가능성 |
| expression | TreeStrategy | bind_v12.pkl | cheese/chill/edge/goofy/hype | 표정 분류 (shoot-only) |
| pose | TreeStrategy | pose_v10.pkl | front/angle/side | 포즈 분류 (shoot-only) |

## q×e×z 프레임 선정

```
score = quality_conf × expression_conf × max(z_score, 0.1)
```

`select_frames()`: expression×pose 버킷별 best q×e×z → top_k

## CLI

```bash
momentscan run video.mp4                              # 기본 분석
momentscan run video.mp4 --debug                      # cv2 디버그 오버레이
momentscan run video.mp4 --report out.html            # HTML 리포트
momentscan run video.mp4 --ingest test_3              # personmemory에 저장
momentscan info                                       # analyzer/backend 상태
momentscan info --graph                               # FlowGraph 표시
```

## 디버그 오버레이

MomentscanDebug(Momentscan) — cv2 창 or 파일 출력:
- **Panel**: Gate signals, Quality/Expression/Pose 확률바, AU face wireframe, SH lighting sphere
- **Video overlay**: face bbox, lighting skin contour, pose 3D cube, nose position
- **Timeline**: AU heatmap, gate severity, expression confidence
- **Coverage grid**: expression×pose 버킷 커버리지

## Analyzer (MOMENTSCAN_MODULES)

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
uv run pytest apps/momentscan/tests/ -v    # 161 tests
```
