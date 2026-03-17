# visualbind — 정적 판단 시스템

> `why-visualbind.md`에서 다룬 문제의식을 VisualBind가 어떤 설계로 해결하는지 정리.
> 동적 적응(crowds pseudo-label, 재학습, drift)은 `how-visualgrow.md` 참조.

---

# Part 0: 아키텍처 개요

## 핵심 아이디어

```
visualbind  = 지금 이 순간의 판단 (정적)
              frozen 모듈 출력(45D)을 조합하여 "이 프레임이 어떤 버킷에 적합한가" 판단
              strategies: Catalog (centroid matching) / XGBoost (tree) / TwoStage (gate+classify)

visualgrow  = 매일 데이터로 성장하는 적응 (동적)
              crowds pseudo-label → 재학습 → drift 적응
              (별도 패키지, how-visualgrow.md 참조)
```

## 전체 흐름

```
                         ┌─────────────────────────────────────────────────┐
                         │              Raw Video Frame                    │
                         └───────────────────────┬───────────────────────┘
                                                 │
                                    [14 Frozen Teachers]
                                  face.detect, face.au,
                                  face.expr, head.pose, ...
                                                 │
                                                 ▼
                                    ┌────────────────────┐
                                    │  45D Signal Vector │
                                    │  (AU 12 + Emotion 8│
                                    │   + Pose 3 + Det 3 │
                                    │   + FQ 5 + FrameQ 3│
                                    │   + Seg 4 + Comp 3 │
                                    │   + CLIP 4)        │
                                    └────────┬───────────┘
                                             │
                              ┌──────────────┼──────────────┐
                              ▼              ▼              ▼
                         [Catalog]      [XGBoost]     [TwoStage]
                         centroid       tree-based    gate+classify
                         matching       classifier    two-phase
                              │              │              │
                              └──────────────┼──────────────┘
                                             ▼
                                    per-bucket scores
                                             │
                                             ▼
                                    [Selector]
                                    top-k frame selection
```

## Phased Complexity

```
Level 0 (완료): N_eff 분석 + XGBoost baseline 검증
                N_eff = 9.95, XGBoost >> Catalog
                292 human labels 수집

Level 1 (현재): XGBoost default strategy + TwoStage (gate+classify)
                45D signal, 3개 strategy, selector, CLI
                catalog_scoring 대비 유의미한 개선 확인
```

Level 2 이상 (crowds pseudo-label, Vision Student, drift 적응)은
visualgrow의 범위 → `how-visualgrow.md` 참조.

---

# Part 1: 아키텍처

## Strategies

### CatalogStrategy (baseline)

기존 catalog_scoring의 직접 대응. 참조 이미지로 카테고리 centroid를 구성하고
Fisher-weighted cosine distance로 매칭.

```python
strategy = CatalogStrategy(profiles=load_profiles(catalog_path))
scores = strategy.predict(frame_vec)  # {"warm_smile": 0.82, "cool_gaze": 0.65, ...}
```

### TreeStrategy (XGBoost, default)

XGBoost multiclass classifier on 45D signal vectors.
292건 human labels로 학습 시 Catalog 대비 큰 폭의 개선.

```python
strategy = TreeStrategy(use_xgboost=True)  # default
strategy.fit({"warm_smile": ref_vectors, "cool_gaze": ref_vectors2})
scores = strategy.predict(frame_vec)
```

`use_xgboost=False`로 logistic regression fallback도 지원.

### TwoStageStrategy (gate + classify)

1단계 gate (binary: 이 프레임이 어떤 버킷에든 적합한가?)
→ 2단계 classify (적합하다면 어느 버킷인가?)

```python
strategy = TwoStageStrategy(gate_strategy=gate, classify_strategy=classify)
```

### BindingStrategy Protocol

모든 strategy는 동일 Protocol을 구현:

```python
class BindingStrategy(Protocol):
    def fit(self, vectors: dict[str, np.ndarray], **kwargs) -> None: ...
    def predict(self, frame_vec: np.ndarray) -> dict[str, float]: ...
```

## Selector

Strategy의 per-bucket scores를 받아 top-k frame selection 수행.

```python
result = select_frames(scores_list, frame_ids, top_k=10)
# SelectionResult with SelectedFrame entries
```

## Signal Vector: 45D (+ 4D CLIP = 49D)

frozen 모듈의 출력만 사용. 규칙 기반 계산도 포함 (blur, brightness 등).
불필요한 차원은 strategy가 자동으로 무시 (XGBoost feature importance).

```
AU 12D:        AU1-AU26 (LibreFace DISFA, 0-5 scale)
Emotion 8D:    happy, neutral, surprise, angry, contempt, disgust, fear, sad
Pose 3D:       yaw_dev, pitch, roll (6DRepNet)
Detection 3D:  confidence, face_area_ratio, face_center_distance
Face QA 5D:    blur, exposure, contrast, clipped_ratio, crushed_ratio
Frame QA 3D:   blur_score, brightness, contrast
Segmentation 4D: seg_face, seg_eye, seg_mouth, seg_hair (BiSeNet)
Composites 3D: duchenne_smile, wild_intensity, chill_score
CLIP Mood 4D:  warm_smile, cool_gaze, playful_face, wild_energy (dynamic axes)
```

변경 근거 (Round 4-5 전문가 리뷰):
- AU 10→12: AU17(Chin Raiser), AU20(Lip Stretcher) 추가. 이미 LibreFace가 출력.
- Emotion 4→8: contempt은 cool_expression 버킷과 직접 관련.
- |yaw| → signed yaw_dev: 좌/우 portrait 구분 정보 보존.
- Face quality, frame quality, segmentation: 품질 gate 정보 포함.
- Composites: 도메인 특화 파생 신호 (duchenne = AU6×AU12 등).

---

# Part 2: 패키지 구조

## 현재 구조 (v0.2.0)

```
libs/visualbind/src/visualbind/
├── __init__.py        # public API exports
├── signals.py         # 45D signal fields, normalization, extraction
├── profile.py         # CategoryProfile, load/save profiles
├── analyzer.py        # correlation matrix, N_eff computation
├── strategies/
│   ├── __init__.py    # BindingStrategy Protocol
│   ├── catalog.py     # CatalogStrategy (centroid matching)
│   ├── tree.py        # TreeStrategy (XGBoost / logistic regression)
│   └── two_stage.py   # TwoStageStrategy (gate + classify)
├── selector.py        # select_frames, SelectionResult, SelectedFrame
├── evaluator.py       # compare_strategies, metrics
└── cli.py             # visualbind analyze / train / eval
```

## 의존성

```toml
[project]
dependencies = ["numpy", "pyarrow"]

[project.optional-dependencies]
train = ["torch", "torchvision", "crowd-kit", "scikit-learn", "pandas", "plotly", "Pillow"]
dev = ["pytest>=7.0.0"]
```

기본 의존성은 수집(signals, profile)만 사용 시 가볍게 유지.
`train` extras는 학습/평가 시에만 필요.

---

# Part 3: CLI

## 서브커맨드

```bash
# Day 0: 독립성 분석 + N_eff
visualbind analyze --data ./signals.parquet --output ./report.html

# 학습 (XGBoost default)
visualbind train --data ./signals.parquet \
    --labels ./labels.json \
    --strategy xgboost \
    --output ./model.json

# Catalog 기반 pseudo-label로 학습
visualbind train --data ./signals.parquet \
    --catalog ./catalog/ \
    --strategy catalog \
    --output ./model.json

# 평가 (strategy 비교)
visualbind eval --data ./signals.parquet \
    --labels ./labels.json \
    --catalog ./catalog/ \
    --model ./model.pkl \
    --output ./eval_report.html
```

## Strategy 옵션

| Strategy | CLI flag | 설명 |
|----------|----------|------|
| `xgboost` | `--strategy xgboost` (default) | XGBoost multiclass |
| `logistic` | `--strategy logistic` | Logistic regression fallback |
| `catalog` | `--strategy catalog` | Centroid matching (baseline) |

## Label 소스

| 옵션 | 설명 |
|------|------|
| `--labels` | JSON 파일 (human labels 또는 외부 pseudo-labels) |
| `--catalog` | Catalog 디렉토리 → nearest centroid으로 pseudo-label 할당 |

---

# Part 4: MVP 결과

## Day 0 결과 (완료)

### N_eff 분석

```
N_eff = 9.95 (14개 observer 중)
→ Go/No-Go 기준 (>= 3) 충족
→ 충분한 독립 정보가 존재
```

Face-crop 기반 모듈 간 상관이 예상대로 존재하나,
pose/quality/segmentation 등 이종 observer가 독립 정보를 제공.

### XGBoost vs Catalog 비교

292건 human labels로 평가:

```
XGBoost >> Catalog (유의미한 차이)
```

- XGBoost: 45D joint space에서 비선형 boundary 학습
- Catalog: 축 정렬 직사각형 경계 (각 축 독립 threshold의 centroid matching)

why-visualbind.md에서 예측한 "축별 독립 근사의 한계"가 실증적으로 확인됨.

### 수집된 데이터

- 292건 human labels (Teacher vote blinding 적용)
- 3개 버킷: warm_smile, cool_expression, lateral

## catalog_scoring과의 관계

catalog_scoring은 visualbind CatalogStrategy의 원형:

| catalog_scoring | visualbind 대응 |
|----------------|----------------|
| `SIGNAL_FIELDS`, `SIGNAL_RANGES` | `signals.py` (45D, 정규화 포함) |
| Fisher ratio weights | CatalogStrategy의 importance_weights |
| Category centroids | CategoryProfile의 mean_signals |
| `compute_importance_weights()` | `profile.py` Fisher ratio 계산 |
| — | TreeStrategy (XGBoost, 비선형 boundary) |
| — | TwoStageStrategy (gate + classify) |
| — | Selector (top-k selection) |

## 열린 질문

| 질문 | 현재 방향 |
|------|----------|
| XGBoost feature importance로 45D 중 유의미한 차원 식별 | 분석 예정 |
| TwoStage의 gate threshold 최적화 | grid search 예정 |
| 추가 human labels 수집 (목표 500건) | 진행 중 |
| Vision Student (raw image → 판단) 전환 시점 | visualgrow 범위 |

---

# 설계 문서 안내

| 문서 | 범위 |
|------|------|
| `why-visualbind.md` | 이론적 기반 (축별 독립 근사, Berkson's paradox, Projected Crowds) |
| `how-visualbind.md` | 정적 판단 (strategies, selector, CLI) — 이 문서 |
| `how-visualgrow.md` | 동적 적응 (crowds pseudo-label, 재학습, drift) |
| `projected-crowds.md` | Projected Crowds 상세 이론 |
