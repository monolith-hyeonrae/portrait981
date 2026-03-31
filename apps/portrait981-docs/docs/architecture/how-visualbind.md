# visualbind — 다중 관측 결합 판단 시스템

> 최종 업데이트: 2026-03-26 (43D signal, VisualBind judge, 3단 gate, cross-model agreement)

`why-visualbind.md`에서 다룬 문제의식을 VisualBind가 어떤 설계로 해결하는지 정리.
동적 적응(crowds pseudo-label, 재학습, drift)은 `how-visualgrow.md` 참조.

---

## 핵심 아이디어

```
visualbind  = 지금 이 순간의 판단 (정적)
              frozen 모듈 출력(43D)을 결합하여 통합 판단
              VisualBind judge: gate + expression + pose

visualgrow  = 매일 데이터로 성장하는 적응 (동적)
              crowds pseudo-label → 재학습 → drift 적응
              (별도 패키지, how-visualgrow.md 참조)
```

## 전체 흐름

```
                         ┌─────────────────────────────────┐
                         │       Raw Video Frame           │
                         └───────────────┬─────────────────┘
                                         │
                            [7 Frozen Teachers (vpx)]
                          face.detect, face.au, face.expr,
                          head.pose, face.parse, body.pose,
                          hand.gesture
                            + [momentscan plugins]
                          face.quality, frame.quality
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │  bind_observations()    │
                            │  Observations → 43D     │
                            │  signal dict             │
                            └────────────┬────────────┘
                                         │
                              ┌──────────┼──────────┐
                              ▼          ▼          ▼
                         [Gate]    [Expression]  [Pose]
                        Heuristic   XGBoost     XGBoost
                        3-stage    (bind_v6)   (pose_v4)
                              │          │          │
                              └──────────┼──────────┘
                                         ▼
                                  JudgmentResult
                              (gate + expr + pose)
```

## Phased Complexity

```
Level 0 (완료): N_eff 분석 + XGBoost baseline 검증
                N_eff = 9.95, XGBoost >> Catalog
                292 human labels 수집

Level 1 (현재): VisualBind judge (gate + XGBoost)
                43D signal, 3단 gate, 337 labels
                MomentscanV2에서 per-frame judgment
                Cross-model agreement로 오탐 감지
```

Level 2 이상 (crowds pseudo-label, Vision Student, drift 적응)은
visualgrow의 범위 → `how-visualgrow.md` 참조.

---

## observer_bind — 다중 관측 결합

visualbind의 핵심: 여러 analyzer의 Observation을 하나의 signal dict로 결합.

```python
from visualbind import bind_observations

observations = [face_detect_obs, face_au_obs, expression_obs, ...]
signals = bind_observations(observations)
# → {"au1_inner_brow": 0.07, "em_happy": 0.44, "head_yaw_dev": 15.2, ...}
```

vpx analyzer들이 각자의 이름 규칙으로 출력하는 signal을 visualbind 표준 43D로 변환:
- face.au: `au_au1` → `au1_inner_brow`
- face.expression: `expression_happy` → `em_happy`
- face.detect: bbox → `face_aspect_ratio` (w/h 비율) 파생

---

## VisualBind Judge — 통합 판단

gate + expression + pose를 하나의 호출로:

```python
from visualbind import VisualBind, HeuristicStrategy, TreeStrategy

judge = VisualBind(
    gate=HeuristicStrategy(),
    expression=TreeStrategy.load("models/bind_v6.pkl"),
    pose=TreeStrategy.load("models/pose_v4.pkl"),
)

result = judge(signals)
# result.gate_passed: bool
# result.gate_reasons: ["gate.pose.extreme_yaw", ...]
# result.expression: "cheese"
# result.expression_conf: 0.95
# result.expression_scores: {"cheese": 0.95, "chill": 0.03, ...}
# result.pose: "front"
# result.is_shoot: True/False
```

Gate fail → expression/pose 건너뜀 (최적화).

---

## HeuristicStrategy — 3단 Gate

학습 없이 threshold로 "이 프레임의 signal을 신뢰할 수 있는가"를 판단.

### 1단: 물리적 품질

| 체크 | Threshold | 의미 |
|------|-----------|------|
| exposure | 50~200 | 너무 어둡거나 밝음 |
| contrast | > 0.10 | 대비 부족 |
| clipped_ratio | < 0.15 | 과노출 픽셀 비율 |
| crushed_ratio | < 0.15 | 저노출 픽셀 비율 |
| face_blur | > 5.0 | 흔들림 |
| confidence | > 0.7 | 얼굴 검출 신뢰도 |

### 2단: 포즈 극단값

| 체크 | Threshold | 의미 |
|------|-----------|------|
| \|yaw\| | < 55° | 과도한 측면 |
| \|pitch\| | < 35° | 과도한 위/아래 (6DRepNet 포화) |
| \|roll\| | < 35° | 과도한 기울임 |
| combined | < 55° | √(yaw² + pitch² + roll²), 다축 열화 |

### 3단: Signal validity (cross-model agreement)

| 체크 | Threshold | 의미 |
|------|-----------|------|
| seg_face | > 0.01 | BiSeNet이 얼굴을 찾아야 |
| AU 합계 | > 0.05 | 근육 활동이 있어야 |

3단이 핵심 — head pose 추정기가 거짓말을 해도 다른 모델이 감지.

**실증 사례**: test_0 #23 — yaw=9.5°(정면 주장), seg_face=0.021(거의 0), AU=0.15(미미).
6DRepNet이 실패했지만 BiSeNet + AU가 "이 얼굴을 제대로 볼 수 없다"는 증거를 제공.
→ heuristic gate에서 못 잡더라도 XGBoost가 face_aspect_ratio=0.33으로 CUT 판정.

---

## TreeStrategy — XGBoost 분류

XGBoost multiclass classifier on 43D signal vectors.

```python
strategy = TreeStrategy.load("models/bind_v6.pkl")
scores = strategy.predict(signals)  # dict를 직접 받음
# {"cheese": 0.95, "chill": 0.03, "cut": 0.01, "edge": 0.005, ...}
```

- `predict(signals: dict)` — raw signals dict 직접 전달
- `predict(vec: np.ndarray)` — normalized vector도 지원
- 내부적으로 model의 feature_names에 맞춰 자동 정렬

`use_xgboost=False`로 logistic regression fallback도 지원.

### bundled pkl 포맷

```python
# 학습 시 저장
joblib.dump({"model": xgb_model, "meta": {
    "feature_names": [...],
    "classes": ["cheese", "chill", "cut", ...],
}}, "models/bind_v6.pkl")

# 로딩 시 자동 인식
strategy = TreeStrategy.load("models/bind_v6.pkl")
```

별도 .json 메타파일 또는 bundled {model, meta} pkl 모두 지원.

---

## 43D Signal Vector

frozen 모듈의 실제 출력만 포함. CLIP placeholder와 수동 composites는 v6에서 제거.

```
AU 12D:         AU1-AU26 (LibreFace)
Emotion 8D:     happy, neutral, surprise, angry, contempt, disgust, fear, sad (HSEmotion)
Pose 3D:        yaw_dev, pitch, roll (6DRepNet)
Detection 4D:   confidence, area_ratio, center_distance, aspect_ratio (InsightFace)
Face QA 5D:     blur, exposure, contrast, clipped, crushed (FaceQuality + BiSeNet mask)
Frame QA 3D:    blur_score, brightness, contrast (OpenCV)
Segmentation 4D: seg_face, seg_eye, seg_mouth, seg_hair (BiSeNet)
Derived Seg 4D: eye_visible, mouth_open, glasses, backlight (BiSeNet 파생)
```

XGBoost가 feature interaction(AU6×AU12 → Duchenne smile 등)을 tree split으로 자동 학습.
수동 composite 불필요.

---

## 패키지 구조

```
libs/visualbind/src/visualbind/
├── __init__.py          # public API: VisualBind, bind_observations, etc.
├── observer_bind.py     # Observations → 43D signal dict
├── judge.py             # VisualBind, JudgmentResult
├── signals.py           # 43D signal fields, normalization, ranges
├── strategies/
│   ├── __init__.py      # BindingStrategy Protocol
│   ├── heuristic.py     # HeuristicStrategy (3단 gate)
│   └── tree.py          # TreeStrategy (XGBoost / logistic regression)
├── selector.py          # select_frames (diversity selection)
├── profile.py           # CategoryProfile (legacy, 비교용)
├── analyzer.py          # correlation matrix, N_eff computation
├── evaluator.py         # compare_strategies, metrics
└── cli.py               # visualbind analyze / train / eval
```

### legacy (코드에 남아있으나 v2에서 미사용)

- `strategies/catalog.py` — CatalogStrategy (centroid matching baseline)
- `strategies/two_stage.py` — TwoStageStrategy (폐기, VisualBind judge로 대체)
- `profile.py` — catalog profile 관리

---

## CLI

```bash
# Day 0: 독립성 분석 + N_eff
visualbind analyze --data ./signals.parquet

# 학습 (XGBoost)
visualbind train --data ./signals.parquet --labels ./labels.json --strategy xgboost --output ./model.pkl

# 평가
visualbind eval --data ./signals.parquet --labels ./labels.json --model ./model.pkl
```

---

## MVP 결과 (Day 0)

### N_eff 분석

```
N_eff = 9.95 (14개 observer 중)
→ Go/No-Go 기준 (>= 3) 충족
→ 충분한 독립 정보가 존재
```

Face-crop 기반 모듈 간 상관이 존재하나,
pose/quality/segmentation 등 이종 observer가 독립 정보를 제공.

### XGBoost 채택 근거

292건 human labels로 평가:

```
XGBoost >> Catalog (유의미한 차이)
```

- XGBoost: 43D joint space에서 비선형 boundary 학습
- Catalog: 축 정렬 직사각형 경계 (각 축 독립 threshold의 centroid matching)

`why-visualbind.md`에서 예측한 "축별 독립 근사의 한계"가 실증적으로 확인.

### 현재 모델

| 모델 | 파일 | 차원 | 데이터 | CV Accuracy |
|------|------|------|--------|-------------|
| Expression | bind_v6.pkl | 43D | 337건, 6 classes | 75.7% ± 13.4% |
| Pose | pose_v4.pkl | 43D | 329건, 3 classes | 87.5% ± 5.9% |

Expression classes: cheese, chill, cut, edge, goofy, hype
Pose classes: front, angle, side

---

## 설계 문서 안내

| 문서 | 범위 |
|------|------|
| `why-visualbind.md` | 이론적 기반 (축별 독립 근사, crowds consensus) |
| `how-visualbind.md` | 현재 구현 (VisualBind judge, 43D, gate) — 이 문서 |
| `how-visualgrow.md` | 동적 적응 (crowds pseudo-label, 재학습, drift) |
| `projected-crowds.md` | Projected Crowds 상세 이론 |
| `person-conditioned-distribution.md` | 개인별 표현 분포, 단기/장기 기억 |
