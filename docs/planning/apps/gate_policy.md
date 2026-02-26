# gate_policy — 프레임 품질 Gate 방법론

> **face.gate analyzer** 설계 문서. 마스터 플랜: [momentscan.md](momentscan.md), scoring 통합: [highlight_rules.md](highlight_rules.md)
>
> 주탑승자(main)와 동승자(passenger)에 서로 다른 gate 전략을 적용하는 이유와 구체 정책.

## 1. 목적

촬영 품질이 **복구 불가능하게 나쁜 프레임을 배제**한다.
블러, 노출 불량, 얼굴 미검출 등 후보정으로도 고객용 사진을 만들 수 없는 프레임만 걸러낸다.

핵심 원칙:
- **Gate는 보수적이어야 한다** — 의심스러우면 통과. 품질 차이는 scoring에서 반영.
- **역할별 전략이 다르다** — 주탑승자는 binary gate, 동승자는 soft scoring.
- **미측정 = 통과** — 0 값(analyzer 미실행)은 penalty 없이 통과.

## 2. 역할 분류 (face.classify → face.gate 입력)

| 역할 | 기준 | gate 전략 |
|------|------|-----------|
| `main` | 가장 크고 안정적인 얼굴 (정확히 1명) | **Binary gate** — PASS/FAIL |
| `passenger` | 두 번째 안정 얼굴 (0~1명) | **Suitability score** — 0.0~1.0 |
| `transient` | 짧은 등장 / 위치 불안정 | 자동 거부 |
| `noise` | 작은 얼굴 / 낮은 confidence | 자동 거부 |

## 3. 주탑승자 (main) — Binary Gate

### 3.1 설계 근거

주탑승자 사진은 고객에게 **직접 전달되는 최종 산출물**이다.
블러, 과/저노출, 오검출 프레임이 best frame으로 선택되면 고객 경험이 치명적으로 손상된다.
따라서 복구 불가 조건은 **hard filter로 완전 배제**한다.

### 3.2 Gate 체인

`FaceGateConfig` 기본 임계값 기준. 순서대로 체크하며 **모든 실패 조건을 누적** (early exit 아님).

```
1. Detection
   confidence < 0.7             → gate.detect.confidence

2. Sharpness (3-level fallback)
   face_blur > 0 AND < 5.0      → gate.blur.face      (face crop Laplacian)
   face_blur = 0 AND
     frame_blur > 0 AND < 50.0  → gate.blur.frame      (frame-level fallback)

3. Parsing Coverage
   parsing > 0 AND < 0.50       → gate.parsing.coverage (BiSeNet 마스크)
   parsing = 0                  → skip (미측정)

4. Exposure (3-level fallback, 상호 배타)
   a. face_contrast > 0:                           ← face.quality mask-based
      contrast < 0.05           → gate.exposure.contrast   (CV flat/washed)
      clipped_ratio > 0.30      → gate.exposure.white      (과포화)
      crushed_ratio > 0.30      → gate.exposure.black      (암전)
   b. face_exposure > 0:                           ← face.quality absolute
      exposure ∉ [40, 220]      → gate.exposure.brightness
   c. frame_bright > 0:                            ← frame.quality fallback
      brightness ∉ [40, 220]    → gate.exposure.brightness

5. Seg-based (parsing ≥ 0.50일 때만 활성)
   seg_mouth < 0.01             → 마스크 착용 판정 → seg_face 체크 skip
   seg_mouth ≥ 0.01 AND
     seg_face < 0.10            → gate.exposure.seg_face   (노출 붕괴)
```

### 3.3 Fail Reasons

모든 실패 사유는 `gate.` prefix + dot notation:

| fail reason | 조건 | 근거 |
|---|---|---|
| `gate.detect.missing` | 얼굴 미검출 | 인물 사진 불가 |
| `gate.detect.confidence` | confidence < 0.7 | 오검출 위험 |
| `gate.role.rejected` | noise/transient | 지나가는 사람 |
| `gate.role.no_main` | main face 미분류 | 주탑승자 없음 |
| `gate.blur.face` | face crop Laplacian < 5.0 | 얼굴 블러 복구 불가 |
| `gate.blur.frame` | frame Laplacian < 50.0 | 전체 모션 블러 |
| `gate.parsing.coverage` | BiSeNet coverage < 50% | 노출 불량 proxy |
| `gate.exposure.contrast` | CV < 0.05 | 얼굴 flat/washed out |
| `gate.exposure.white` | clipped > 30% | 과노출 복구 불가 |
| `gate.exposure.black` | crushed > 30% | 암전 복구 불가 |
| `gate.exposure.brightness` | 밝기 ∉ [40, 220] | 절대 밝기 범위 초과 |
| `gate.exposure.seg_face` | seg_face < 10% | 과노출로 얼굴 seg 붕괴 |

### 3.4 Scoring에서의 사용

```python
gate_mask = [r.gate_passed for r in records]  # bool 배열
```

- **Peak detection**: gate-pass 프레임만 peak 후보
- **Best frame selection**: gate-fail 프레임은 후보에서 완전 배제 (score = -1)
- **Final score 계산**: gate와 무관하게 전 프레임 산출, gate는 peak/best 선택에서만 필터

### 3.5 Exposure 판정 우선순위

face.quality mask-based 메트릭 → absolute brightness → frame-level brightness.

Local contrast(CV = std/mean)는 피부톤에 무관한 노출 품질 지표.
CV가 측정 가능하면 absolute brightness보다 우선하여 사용.

### 3.6 마스크 착용 감지

BiSeNet parsing이 충분할 때(coverage ≥ 50%):
- `seg_mouth < 1%` → 마스크 착용으로 판정
- 마스크 착용 시 `seg_face` 체크 skip (seg 바운더리가 마스크 경계에서 부정확)
- 마스크 미착용 시에만 `seg_face < 10%` → 과노출로 인한 seg 붕괴 감지

## 4. 동승자 (passenger) — Suitability Score

### 4.1 설계 근거

동승자 촬영 적합성은 **보너스 요소**이다:
- 동승자가 잘 나온 프레임은 고객 만족도가 높으므로 **강력히 우선**
- 동승자가 잘 안 보인다고 프레임을 **차단하면 안 됨** — 주탑승자가 잘 나온 사진이 버려짐
- 따라서 binary gate(PASS/FAIL)가 아닌 **continuous scoring(0.0~1.0)**으로 전환

### 4.2 Suitability 계산

confidence와 parsing_coverage 두 축의 soft threshold:

```python
conf_score  = min(confidence / passenger_confidence_min, 1.0)
                # passenger_confidence_min = 0.5 (main의 0.7보다 관대)
                # 임계값 이상 → 1.0, 미만 → 비례 감소

parse_score = min(parsing / parsing_coverage_min, 1.0)
                # parsing_coverage_min = 0.50 (main과 동일 임계값)
                # parsing = 0 (미측정) → 1.0 (패널티 없음)

suitability = conf_score × parse_score   # 0.0 ~ 1.0
```

| 시나리오 | confidence | parsing | suitability |
|---------|-----------|---------|-------------|
| 선명한 동승자 | 0.8 | 0.7 | **1.0** |
| parsing 미측정 | 0.6 | 0 | **1.0** |
| 낮은 confidence | 0.25 | 0.7 | **0.5** |
| 동승자 미감지 | — | — | **0.0** |

### 4.3 Gate를 적용하지 않는 항목

| 항목 | 이유 |
|------|------|
| blur | 동승자는 주탑승자보다 배경에 가까워 blur가 자연스러움. 차단 근거 부족 |
| exposure | 조명 위치 차이로 동승자에게 불리한 노출은 빈번. 주탑승자 노출이 OK면 충분 |
| contrast, clipped, crushed | 동일 이유 — 주탑승자 gate가 전체 프레임 품질을 보장 |
| seg_face, seg_mouth | 동승자 얼굴은 작고 가려져 있을 가능성이 높아 seg 신뢰도 낮음 |
| pose (yaw/pitch) | 동승자가 정면이 아닌 것은 자연스러움 |

### 4.4 Scoring 통합

```python
# 기존
final = 0.35 × quality + 0.65 × impact           # 범위 [0, 1.0]

# 추가
final = 위 + passenger_bonus_weight × suitability  # 범위 [0, ~1.3]
#       passenger_bonus_weight = 0.30 (기본값)
```

- 동승자 적합(suitability=1.0) → **+0.30 가산** → 기존 최대 1.0에서 1.3까지
- 동승자 부적합/미감지(suitability=0.0) → 보너스 없음 (**차감도 없음**)
- Peak detection은 상대적(prominence 기반)이므로 절대 스케일 변화는 무관
- 동승자가 잘 나온 구간의 프레임이 상대적으로 강력히 우선됨

### 4.5 Weight 선정 근거

`passenger_bonus_weight = 0.30`:
- 기존 final score 범위 [0, 1.0]에서 suitability=1.0일 때 +30%
- quality(0.35) + impact(0.65) 합산의 약 절반에 해당
- 동승자 적합 여부가 "미소 강도"보다 강하고 "전체 품질"보다 약한 수준의 영향력
- `HighlightConfig.passenger_bonus_weight`로 조정 가능

## 5. Config 참조

```python
@dataclass(frozen=True)
class FaceGateConfig:
    # Main face thresholds
    face_confidence_min: float = 0.7
    face_blur_min: float = 5.0
    frame_blur_min: float = 50.0
    exposure_min: float = 40.0
    exposure_max: float = 220.0

    # Passenger suitability (soft scoring)
    passenger_confidence_min: float = 0.5

    # Local contrast
    contrast_min: float = 0.05
    clipped_max: float = 0.3
    crushed_max: float = 0.3

    # Parsing coverage
    parsing_coverage_min: float = 0.50

    # Seg-based
    seg_mouth_min: float = 0.01
    seg_face_min: float = 0.10

@dataclass
class HighlightConfig:
    passenger_bonus_weight: float = 0.30
```

## 6. 데이터 흐름

```
face.detect → face.classify → face.gate
                                  │
                  ┌───────────────┼───────────────┐
                  │               │               │
              noise/transient   main           passenger
              gate_passed=F   gate chain     suitability
              role.rejected   PASS/FAIL      0.0~1.0
                                │               │
                          ┌─────┘               │
                          ▼                     ▼
                    FrameRecord            FrameRecord
                    .gate_passed           .passenger_suitability
                    .gate_fail_reasons     .passenger_confidence
                          │                     │
                          ▼                     ▼
                  BatchHighlightEngine
                    gate_mask (peak/best 필터)
                    + passenger_bonus (additive)
                          │
                          ▼
                    final = 0.35×Q + 0.65×I + 0.30×suitability
```

## 7. 변경 이력

| 날짜 | 변경 | 근거 |
|------|------|------|
| 2025-05 | face.gate 신설 | frame.gate에서 per-face gate로 전환 |
| 2025-06 | parsing_coverage gate 추가 | BiSeNet coverage 낮을 때 노출 불량 proxy |
| 2025-06 | seg-based gate 추가 | 마스크 착용 감지 + 과노출 seg 붕괴 감지 |
| 2025-06 | parsing_coverage_min 0.15→0.50 상향 | 실제 데이터에서 50% 미만이면 노출 불량 빈도 높음 |
| 2026-02 | 동승자 binary gate → suitability score | 동승자 부적합으로 인한 프레임 차단 방지. additive bonus로 전환 |
