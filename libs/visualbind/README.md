# visualbind

Observer 출력 결합 프레임워크. 여러 frozen 모델의 출력을 학습 기반으로 결합하여 판단한다.

## 핵심 개념

```
momentscan  = observer 실행 (DAG 기반, frozen 모델 조합)
              "무엇을 관찰할지"

visualbind  = observer 출력 결합 (전략 교체 가능)
              "관찰 결과를 어떻게 판단할지"
```

## 전략

| 전략 | 설명 | 용도 |
|------|------|------|
| `CatalogStrategy` | Fisher-weighted centroid matching | baseline/fallback |
| `TreeStrategy` | XGBoost (logistic regression fallback) | **기본 전략** |

## 작업 흐름

### 1. 비디오 분석 → 21D signal 수집

```bash
uv run python scripts/day0_analysis.py ~/Videos/*.mp4 \
    --fps 2 --output signals.parquet
```

momentscan이 비디오를 처리하고 21D signal vector를 parquet로 저장.
N_eff 분석 리포트도 함께 출력.

### 2. 시각 비교

```bash
uv run python scripts/visual_compare.py ~/Videos/test.mp4 \
    --fps 2 --output compare.html
```

브라우저에서 Catalog / LR / XGBoost의 프레임별 판단 차이를 확인.
Disagreement 프레임이 빨간 테두리로 하이라이트.

### 3. Anchor Set 라벨링

```bash
uv run python scripts/label_tool.py ~/Videos/test.mp4 \
    --fps 2 --output labels.html --max-frames 500
```

브라우저에서 프레임별 정답을 클릭. Export 버튼으로 `labels.json` 다운로드.
Disagreement 프레임 우선 표시 (가장 정보량이 높은 프레임).

### 4. XGBoost 학습

```bash
# catalog pseudo-label로 학습 (라벨 없이 바로 가능)
visualbind train --data signals.parquet \
    --catalog data/catalogs/portrait-v1 \
    --strategy xgboost --output models/bind_v1.json

# human label로 학습 (Step 3 완료 후)
visualbind train --data signals.parquet \
    --labels labels.json \
    --strategy xgboost --output models/bind_v1.json
```

### 5. 평가

```bash
visualbind eval --data signals.parquet \
    --catalog data/catalogs/portrait-v1 \
    --model models/bind_v1.json
```

Catalog vs XGBoost AUC 비교.

### 6. momentscan에 적용

```bash
# 기존 방식 (catalog only)
momentscan process video.mp4 -o output/

# XGBoost 추가 (catalog과 병행)
momentscan process video.mp4 -o output/ --bind-model models/bind_v1.json
```

`--bind-model` 지정 시 bind_scores가 highlight scoring에 반영.
catalog_scores와 공존 (replacing이 아닌 blending).

## 프레임 선택 (selector)

기존 BatchHighlightEngine (7단계: delta→normalize→gate→score→smooth→peak→window)을
3단계로 대체:

```python
from visualbind import select_frames, TreeStrategy

strategy = TreeStrategy.load("models/bind_v1")
result = select_frames(
    vectors,           # (N, 21) signal matrix
    strategy,
    top_k=5,           # 카테고리당 상위 5프레임
    gate_mask=gate,    # quality gate (boolean)
)

for cat, frames in result.per_category.items():
    print(f"{cat}: {[f.index for f in frames]}")
```

peak detection은 "변화가 큰 순간"을 찾지만,
최적 portrait은 "안정적으로 좋은 순간"의 한가운데일 수 있다.
직접 concept score로 선택하는 것이 더 정확하다.

## CLI

```bash
visualbind analyze --data signals.parquet              # Day 0: N_eff, 상관행렬
visualbind train --data signals.parquet --strategy xgboost --output model.json
visualbind eval --data signals.parquet --model model.json
```

## 21D Signal Vector

```
AU 10D:        AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU25, AU26
               (face.au, LibreFace DISFA 0-5)
Emotion 4D:    em_happy, em_neutral, em_surprise, em_angry
               (face.expression, HSEmotion)
Pose 3D:       head_yaw_dev, head_pitch, head_roll
               (head.pose, 6DRepNet)
CLIP 4D:       warm_smile, cool_gaze, playful_face, wild_energy
               (portrait.score, CLIP text axes)
```

visualbind의 확장 23D는 위 21D + detect_confidence(1D) + face_size_ratio(1D).
기존 catalog과 호환을 위해 momentscan 연동 시 21D를 사용.

## 설계 문서

- `docs/planning/why-visualbind.md` — 왜 필요한가 (이론적 배경)
- `docs/planning/how-visualbind.md` — 어떻게 구현하는가 (아키텍처, MVP 경로)
- `docs/planning/projected-crowds.md` — Projected Crowds 이론 (상세)
