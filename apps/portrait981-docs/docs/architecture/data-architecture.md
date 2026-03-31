# Data Architecture — 수동/자동 라벨 분리 + 3단계 데이터 흐름

## 핵심 원칙

```
predictions.csv = 모델의 "의견" (휘발성, 언제든 폐기 가능)
labels.csv      = 확정된 "사실" (ground truth, 영구)
personmemory      = 서비스 "기억" (member 단위, 검증된 것만)
```

수동 라벨과 자동 예측은 같은 포맷이지만 **절대 섞이지 않는다**.
모델이 하루 2000건을 생산해도 labels.csv는 오염되지 않는다.

## 저장 구조

```
data/datasets/portrait-v1/
├── images/               ← 프레임 이미지 (불변)
├── videos/               ← 압축 비디오
├── labels.csv            ← ground truth (수동 라벨만)
├── predictions.csv       ← 모델 예측 (자동, 버전 관리)
├── signals.parquet       ← 43D signal 벡터
├── videos.csv            ← 워크플로우 메타
└── dataset.yaml

data/personmemory/          ← 고객 기억 (member_id 필수)
└── members/

data/gallery/             ← 생산물 (고객 전달)
└── {member_id}/
```

## 파일 스키마

### labels.csv — Ground Truth
```csv
filename, workflow_id, expression, pose, moment, source
test_3_0015.jpg, test_3, cheese, front, , manual
cap_1_0023.jpg, cap_1, hype, front, yes, manual
```
- `source`: manual (수동), verified (자동→사람 승인)
- 사람이 직접 확인한 것만 기록
- 학습 데이터의 유일한 소스

### predictions.csv — 모델 의견
```csv
filename, workflow_id, expression, pose, model, confidence, timestamp
test_3_0015.jpg, test_3, cheese, front, xgboost_v4, 0.87, 2026-03-23T17:00
test_3_0015.jpg, test_3, cheese, front, xgboost_v5, 0.91, 2026-03-25T10:00
test_3_0042.jpg, test_3, hype, front, xgboost_v4, 0.72, 2026-03-23T17:00
```
- `model`: 모델 버전 (일괄 폐기의 키)
- `confidence`: 예측 확신도
- `timestamp`: 예측 시점
- 같은 프레임에 여러 모델 버전 공존 가능
- 언제든 특정 모델 버전 행 전체 삭제 가능

### signals.parquet — Signal 벡터
```
filename | au1_inner_brow | ... | backlight_score (43D)
```
- 모델 독립적 (signal은 frozen analyzer 출력)
- 모델이 바뀌어도 재추출 불필요 (signal 추가 시에만)

## 데이터 흐름

### 수동 라벨링 (현재)
```
비디오 → annotator label → 사람이 SHOOT/expression/pose 판단
→ Confirm Merge → labels.csv + images/
→ extract_signals.py → signals.parquet
→ visualbind train → XGBoost 모델
```

### 자동 예측 (다음 단계)
```
비디오 → momentscan → 43D signal
→ XGBoost predict → predictions.csv (model=xgboost_v4)
→ 리뷰 서버에서 labels.csv와 비교
```

### 모델 교체 시
```
xgboost_v4 → v5 업그레이드
→ predictions.csv에서 model=xgboost_v4 행 삭제 (또는 보존)
→ 전체 비디오 재예측 → model=xgboost_v5 행 추가
→ labels.csv 영향 없음
```

### 승격 (미래, 자동화 시)
```
predictions.csv (자동)
    ↓ 사람이 리뷰에서 승인
labels.csv (source=verified)
    ↓ member_id 있으면
personmemory (서비스 기억)
```

현재는 승격 자동화 단계가 아님. 수동 라벨링과 자동 예측은 분리 유지.

## 리뷰 서버 비교 뷰

labels.csv + predictions.csv를 조인하여 상태 표시:

| 상태 | 수동 | 자동 | 표시 | 의미 |
|------|------|------|------|------|
| 일치 | cheese | cheese | 초록 | 모델과 사람이 동의 |
| 불일치 | cheese | hype | 주황 | 검토 필요 |
| 자동만 | — | cheese | 파랑 | 모델만 판단 (미검증) |
| 수동만 | cheese | — | 흰색 | 예측 미실행 |

## 학습 데이터 정책

```
XGBoost 학습 시:
  입력: labels.csv (source=manual OR source=verified) + signals.parquet
  predictions.csv는 절대 학습에 사용하지 않음

미래 (VisualGrow):
  Phase 1: manual만으로 학습 (현재)
  Phase 2: manual + high-confidence verified로 학습
  Phase 3: pseudo-label 파이프라인 (crowds supervision)
```

## personmemory 축적 정책

```
personmemory.ingest 우선순위:
  1순위: labels.csv source=manual (사람이 확인)
  2순위: labels.csv source=verified (사람이 승인한 자동)
  3순위: 없음 (predictions.csv에서 직접 축적하지 않음)

predictions.csv는 personmemory에 직접 들어가지 않는다.
모델의 "의견"이 고객의 "기억"이 되면 안 된다.
```

## 볼륨 예상

```
일일:
  자동 예측: ~2000 workflows × ~200 frames = ~400,000 predictions
  수동 라벨: ~50-100 frames (검토/수정)

predictions.csv: 빠르게 커짐 → 모델 버전별 관리 필요
labels.csv: 천천히 성장 → 품질 유지
signals.parquet: workflow당 1회 추출 → 재추출 드묾
```
