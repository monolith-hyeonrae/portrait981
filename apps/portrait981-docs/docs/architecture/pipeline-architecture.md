# Pipeline Architecture — Signal Production → Interpretation → Adaptation

> 최종 업데이트: 2026-03-26 (MomentscanV2, 43D signal, VisualBind judge 이후)

## 설계 철학

visualpath → visualbind → visualgrow의 3단계는
**단일 모델을 연구할 리소스가 없는 조직이 시스템을 단계적으로 도입하는 구조**다.

```
Day 1:  frozen model을 가져다 쓴다 (visualpath)
        → 즉시 동작, 수동 임계값으로 시작

Week 2: 데이터가 쌓이면 결합을 학습한다 (visualbind)
        → frozen model은 그대로, 판단만 개선

Month 3: 시스템이 스스로 성장한다 (visualgrow)
        → 도메인 적응, 자율 개선
```

각 단계는 이전 단계를 대체하지 않고 위에 올라간다.
visualpath만으로 동작 가능, visualbind는 선택적 개선, visualgrow는 미래 확장.
**애자일: 각 단계에서 가치를 전달, 완벽한 모델을 기다리지 않는다.**

## 두 가지 축

### 범용 프레임워크 (파이프라인)

```
visualbase  → 미디어 I/O (카메라, 비디오, 프레임)
visualpath  → DAG 기반 분석 프레임워크 (FlowGraph, App, Backend)
vpx         → 플러그인 분석 모듈 (7개 frozen model)
visualbind  → signal 결합 + 학습 기반 판단 (XGBoost, HeuristicStrategy)
visualgrow  → 지속 적응 + 자율 성장 (계획)
```

```
visualbase (미디어 소스)
    ↓ Frame
visualpath (FlowGraph) + vpx (플러그인 분석 모듈)
    ↓ Observations
visualbind (bind_observations → 43D signals → VisualBind judge)
    ↓ JudgmentResult (gate + expression + pose)
visualgrow (자율 성장, 계획)
```

### 응용 프로그램 (981파크 특화)

```
momentscan          = visualpath + vpx + visualbind를 조합한 분석 앱
                      v1 (BatchHighlightEngine, legacy)
                      v2 (vp.App + VisualBind judge, 현재 기본)
momentscan-plugins  = 도메인 특화 분석 플러그인 (face.quality, face.classify 등)
personmemory          = 고객 기억 시스템 (member 단위 장기 저장)
reportrait          = AI 초상화 생성 (ComfyUI bridge)
portrait981         = 통합 오케스트레이터 (scan → bank → generate)
annotator           = 라벨링/리뷰/데이터셋 관리 도구
```

```
범용 모듈 (vpx plugins):        도메인 모듈 (momentscan plugins):
  face.detect (InsightFace)       face.quality (마스크 기반 측정)
  face.expression (HSEmotion)     face.classify (역할 분류)
  face.au (LibreFace)             frame.quality (프레임 품질)
  head.pose (6DRepNet)            frame.scoring (프레임 점수)
  face.parse (BiSeNet)            face.baseline (Welford stats)
  body.pose (YOLO-Pose)           face.gate (legacy, HeuristicStrategy로 이동)
  hand.gesture (MediaPipe)
```

## 3-Layer Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: visualpath + vpx (Signal Production)           │
│                                                         │
│ "각자 한 가지 문제만 푸는 작은 분석기들의 DAG"          │
│                                                         │
│ visualbase: 미디어 소스                                  │
│ visualpath: FlowGraph (의존성 관리, 실행 제어)          │
│ vpx plugins: 범용 frozen model (face, pose, gesture)    │
│ momentscan plugins: 도메인 특화 (quality, classify)     │
│                                                         │
│ frozen models → per-frame Observations                  │
│ 가치: 즉시 동작, 추가 학습 불필요                       │
├─────────────────────────────────────────────────────────┤
│ Layer 2: visualbind (Signal Interpretation)              │
│                                                         │
│ "다중 분석기 출력을 결합하여 통합 판단"                  │
│                                                         │
│ bind_observations() → 43D signals                       │
│ VisualBind judge:                                       │
│   HeuristicStrategy (3단 gate)                          │
│   TreeStrategy (XGBoost expression/pose)                │
│ Cross-model agreement로 오탐 감지                       │
│                                                         │
│ 가치: 같은 frozen model로 정확도 대폭 향상              │
├─────────────────────────────────────────────────────────┤
│ Layer 3: visualgrow (Continuous Adaptation) — 계획      │
│                                                         │
│ personmemory.consolidate() (단기→장기 기억 통합)          │
│ pseudo-label → 자율 재학습                              │
│ Person-conditioned distribution                         │
│ Face State Embedding                                    │
│                                                         │
│ 가치: 서비스가 돌수록 자동 개선                         │
└─────────────────────────────────────────────────────────┘
```

## Layer 간 관계

```
Layer 1 (visualpath):
  입력: 비디오 프레임 / 이미지
  출력: Observations (각 analyzer의 출력)
  역할: signal을 생산한다

Layer 2 (visualbind):
  입력: Observations → bind_observations() → 43D signals
  출력: JudgmentResult (gate + expression + pose)
  역할: signal을 해석한다
  구현: VisualBind(HeuristicStrategy, TreeStrategy × 2)

Layer 3 (visualgrow, 계획):
  입력: Layer 1 signal + Layer 2 판단 + 운영 피드백
  출력: 개선된 모델, per-member distribution
  역할: 시스템을 성장시킨다

응용 (momentscan v2):
  Layer 1 + Layer 2를 vp.App 위에서 조합
  on_frame()에서 bind → judge → FrameResult 축적
```

**핵심 원칙: Layer 2는 Layer 1의 출력만 소비한다.**
visualbind는 visualpath가 생산한 signal을 받아서 해석하지,
자체적으로 signal을 추출하지 않는다.

## 43D Signal Vector

frozen 모듈의 실제 출력만 포함. 수동 composites와 CLIP placeholder 제거 (v5에서 정리).

```
AU 12D:         AU1-AU26 (LibreFace DISFA, 0-5 scale)
Emotion 8D:     happy, neutral, surprise, angry, contempt, disgust, fear, sad
Pose 3D:        yaw_dev, pitch, roll (6DRepNet)
Detection 4D:   confidence, face_area_ratio, face_center_distance, face_aspect_ratio
Face QA 5D:     blur, exposure, contrast, clipped_ratio, crushed_ratio
Frame QA 3D:    blur_score, brightness, contrast
Segmentation 4D: seg_face, seg_eye, seg_mouth, seg_hair (BiSeNet)
Derived Seg 4D: eye_visible_ratio, mouth_open_ratio, glasses_ratio, backlight_score
```

모든 signal이 frozen model 출력에 기반. XGBoost가 feature interaction을 자동 학습.

## VisualBind 3단 Gate

```
1단: 물리적 품질 (heuristic)
  exposure 50~200, contrast > 0.10, blur > 5, clipped/crushed < 0.15

2단: 포즈 극단값 (heuristic)
  |yaw| < 55°, |pitch| < 35°, |roll| < 35°
  combined √(yaw² + pitch² + roll²) < 55°

3단: Signal validity (cross-model agreement)
  seg_face > 0.01 (BiSeNet이 얼굴을 찾아야)
  AU 합계 > 0.05 (근육 활동이 있어야)
  → 포즈 추정기가 실패해도 다른 모델이 감지
```

Gate fail → expression/pose 판단 건너뜀 (최적화).
Gate pass → XGBoost가 expression + pose 분류.
XGBoost CUT = gate는 통과했지만 미적으로 적합하지 않은 프레임.

## Gate → XGBoost 점진적 흡수

gate의 품질 판단은 본질적으로 XGBoost의 결정 트리에 포함되어야 한다.

```
Phase 0 (초기):
  face.gate = 유일한 품질 필터 (하드코딩)
  XGBoost = 표정/포즈만 판단

Phase 1 (현재):
  HeuristicStrategy = 3단 gate (물리 품질 + 포즈 + signal validity)
  XGBoost = 표정 + 포즈, face_aspect_ratio 등으로 미묘한 품질 패턴도 학습
  → #23 프레임: gate 통과했지만 XGBoost가 face_aspect_ratio 불일치로 CUT 판정 (실증)

Phase 2 (다음):
  XGBoost가 품질 판단의 주체
  heuristic gate = 극단적 불량만 (최후 방어선)

Phase 3 (장기):
  gate = 최소한 하드웨어 체크 (no face, 프레임 깨짐)
  학습 모델 = 품질 + 표정 + 포즈 통합 판단
```

## 모델 현황

| 모델 | 파일 | 차원 | 데이터 | CV Accuracy |
|------|------|------|--------|-------------|
| Expression | bind_v6.pkl | 43D | 337건, 6 classes | 75.7% |
| Pose | pose_v4.pkl | 43D | 329건, 3 classes | 87.5% |

Expression classes: cheese, chill, cut, edge, goofy, hype
Pose classes: front, angle, side

## 핵심 원칙

1. **Signal 추출은 한 곳** — bind_observations() (v2) 또는 SignalExtractor (이미지 모드)
2. **Layer 1 → Layer 2 단방향** — visualbind는 visualpath 출력만 소비
3. **Gate는 안전망** — XGBoost가 개선되어도 heuristic gate 유지
4. **단계적 도입** — visualpath만으로 동작, visualbind는 선택적
5. **서비스 = 실험** — 같은 signal 경로, 같은 gate, 같은 품질
6. **Cross-model agreement** — 모델 간 불일치로 오탐 감지
