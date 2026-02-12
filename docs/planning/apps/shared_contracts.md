# Shared Contracts — 앱 간 공통 정의

> momentscan, appearance-vault, reportrait이 공유하는 인터페이스, 타입, 기준값 정의.
> 이 문서의 정의를 바꾸면 전체 파이프라인에 영향이 있으므로 변경 시 전체 리뷰 필수.

## 1. QualityGate

프레임 품질 판정. 두 단계(strict / loose)로 나뉜다.

### Strict Gate (anchor, prototype, memory update용)

| 항목 | 메트릭 | 기준 | 비고 |
|------|--------|------|------|
| face_confidence | InsightFace det_score | >= 0.7 | |
| face_size | bbox area / frame area | >= 0.01 | |
| blur | Laplacian variance | <= tau_blur (TBD) | 높을수록 선명 — threshold는 데이터 기반 튜닝 |
| exposure | pixel mean | 40 <= mean <= 220 | |
| occlusion | keypoint confidence drop | <= 0.3 (비율) | optional, 초기엔 생략 가능 |

### Loose Gate (coverage 후보용)

strict보다 완화:
- face_confidence >= 0.5
- blur threshold 2배 허용
- occlusion <= 0.5

### 구현 위치

`libs/vpx/sdk/` 또는 공용 패키지에 `QualityGate` 클래스로 구현.
각 앱에서 import하여 사용.

```python
from vpx.sdk.quality import QualityGate, GateLevel

gate = QualityGate()
result = gate.check(frame, face_bbox, level=GateLevel.STRICT)
# result.passed, result.blur, result.exposure, result.reasons
```

## 2. Embedding Standard

앱 간 임베딩 호환성 보장을 위한 고정 규격.

| 항목 | 값 | 비고 |
|------|---|------|
| Face-ID model | InsightFace ArcFace (recognition model) | 연관(same-person) 판정 전용 |
| General vision model | **TBD** (DINOv2 / SigLIP / OpenCLIP 중 선택) | 다양성/노벨티 판정용 |
| Input crop — face | 112x112 aligned (InsightFace 기본) | Face-ID용 |
| Input crop — general | 224x224 center-crop (face bbox 1.5x 확장) | General vision용 |
| Upper-body crop | 224x224 (shoulder 기준 상반신) | body novelty용 (optional) |
| L2 normalize | Yes | 모든 임베딩 저장/비교 전 L2 정규화 |
| Dimensionality | model 의존 (512 or 768) | 변경 시 memory bank 재구축 필요 |

### 모델 변경 정책

임베딩 모델 버전이 바뀌면:
1. 기존 memory bank 무효화 → 재구축
2. novelty / stable threshold 재튜닝 필요
3. 모든 출력에 `embed_model_version` 태그 포함

## 3. Meta Schema

모든 앱이 출력하는 프레임/이미지 메타데이터의 공통 필드.

```json
{
  "frame_idx": 1234,
  "timestamp_ms": 41133.3,
  "face_bbox": [x1, y1, x2, y2],
  "crop_box": [x1, y1, x2, y2],
  "quality": {
    "blur": 150.3,
    "exposure_mean": 128.5,
    "face_confidence": 0.92,
    "face_area_ratio": 0.045,
    "occlusion_ratio": 0.0
  },
  "gate_level": "strict",
  "embed_model_version": "dinov2-vits14-v1",
  "gate_version": "v1",
  "person_id": 0
}
```

### 좌표계

- bbox: `[x1, y1, x2, y2]` pixel 좌표 (정수)
- 정규화 좌표 사용 시: `[x1/W, y1/H, x2/W, y2/H]` float, 필드명에 `_norm` 접미사

### Timestamp

- 기준: 비디오 시작 시점 = 0
- 단위: milliseconds (float)
- frame_idx: 0-based, 디코딩 순서

## 4. 공용 Feature 계산

아래 계산은 **모든 앱에서 동일한 구현**을 사용해야 한다.

| Feature | 계산 방법 | 구현 위치 |
|---------|----------|-----------|
| blur | `cv2.Laplacian(gray, cv2.CV_64F).var()` | vpx-sdk 또는 공용 utils |
| exposure | `gray.mean()`, `np.percentile(gray, [5, 95])` | 동일 |
| face_bbox | InsightFace SCRFD 출력 그대로 | vpx-face-detect |
| face_area_ratio | `(w*h) / (frame_W * frame_H)` | 동일 |
| head_pose (yaw/pitch/roll) | InsightFace 5-point landmark 기반 | vpx-face-detect |
| upper_body_crop | YOLO-Pose keypoint 기반 shoulder 영역 | vpx-body-pose |

## 5. 앱 간 의존 관계

```
vpx plugins ───► momentscan
                    │
                    ├── Phase 1: highlight (수치 feature + peak detection)
                    ├── Phase 2: embedding experiment (DINOv2/SigLIP delta)
                    └── Phase 3: identity collection (듀얼 임베딩 + 버킷 다양성)
                    │
                    │ (identity set: anchor/coverage/challenge + meta)
                    ▼
              appearance-vault
                    │ (select_refs → image paths)
                    ▼
               reportrait
                    │ (ComfyUI workflow injection)
                    ▼
              ComfyUI (외부)
```

### 연결 정책

1. **momentscan → appearance-vault**: identity set 이미지 + 임베딩 전달, match/update API 호출
2. **appearance-vault → reportrait**: select_refs() → reference image 경로 전달
3. **reportrait → ComfyUI**: workflow JSON에 image path 주입, REST API로 생성 요청
4. momentscan 내부: Phase 1/2 highlight window → Phase 3 sampling priority로 활용

## 6. 출력 디렉토리 규격

```
output/
├── {video_id}/
│   ├── momentscan/
│   │   ├── highlight/
│   │   │   ├── windows.json
│   │   │   ├── timeseries.csv
│   │   │   └── frames/
│   │   ├── identity/
│   │   │   ├── person_0/
│   │   │   │   ├── anchors/
│   │   │   │   ├── coverage/
│   │   │   │   ├── challenge/
│   │   │   │   └── meta.json
│   │   │   └── person_1/
│   │   └── debug/
│   ├── appearance-vault/
│   │   ├── person_0/
│   │   │   └── memory_bank.json
│   │   └── person_1/
│   └── reportrait/
│       ├── generated/
│       ├── workflow_used.json
│       └── _version.json
```

## 7. 버전 태깅

모든 출력 파일에 아래 버전 정보 포함:

```json
{
  "_version": {
    "app": "identity_builder",
    "app_version": "0.1.0",
    "embed_model": "dinov2-vits14",
    "gate_version": "v1",
    "created_at": "2026-02-12T15:30:00Z"
  }
}
```

## 8. 개발 순서

| 순서 | 앱 / Phase | 이유 |
|------|-----------|------|
| 1 | momentscan 리네임 + 배치 전환 | 기반 정비 |
| 2 | momentscan Phase 1 (batch highlight) | 기존 로직 확장, 빠른 베이스라인 |
| 3 | momentscan Phase 2 (embedding experiment) | Phase 1과 병렬 비교, 효과 검증 |
| 4 | momentscan Phase 3 (identity collection) | 진짜 목표, 데이터 수집 |
| 5 | appearance-vault | Phase 3 출력을 저장/관리하는 핵심 인프라 |
| 6 | reportrait | memory bank API 확정 후 ComfyUI 연동 |

## 9. 평가 기준

### momentscan — Highlight (Phase 1 & 2)

- 구간 겹침률 (Phase 1 vs Phase 2)
- 선택 프레임 품질 (blur, exposure, face_size 평균)
- 구매/저장 상관성 (운영 데이터 확보 후)

### momentscan — Identity Collection (Phase 3)

- 버킷 커버리지 충족률 (yaw 6/7, pitch 4/5, expression 3/4)
- Anchor 품질 (avg quality_score, frontalness)
- 중복도 (saved set의 avg novelty)

### appearance-vault

- 노드 수 안정성 (비디오 진행에 따른 k 변화)
- stable_score 분포 (동일 인물 매칭 정확도)
- reference 선택 적합성 (query-to-selected coherence)

### reportrait

- 인물 일관성 (생성 이미지의 Face-ID similarity)
- 생성 품질 (FID, LPIPS 등)
- 사용자 선호도 (A/B 테스트)

### 배포 전략

```
momentscan Phase 1 → Phase 2 shadow → appearance-vault → reportrait
```

- Phase 2는 Phase 1과 병렬 shadow mode로 비교 후 전환
- 메트릭 하락 시 자동 배포 차단
