# momentscan — 인물/장면 분석 수집 앱

> 라이드 비디오에서 의미 있는 순간(moment)을 탐지하고,
> 인물의 다양한 상태를 체계적으로 수집하는 분석 앱.
> portrait981 파이프라인의 첫 번째 단계.

## 위치

```
momentscan (분석/수집) → momentbank (저장/관리) → reportrait (AI 생성)
```

## 현재 상태 (2026-03-09)

3-Phase 진화 완료. 584 tests 통과.

- FlowGraph 기반 14개 Analyzer DAG 파이프라인
- Batch Highlight (Phase 1): MAD z-score 정규화 + peak detection
- Reference-Guided Collection (Phase 3): Pose×Category 그리드 + 카탈로그 유사도
- CLIP portrait scoring (portrait.score): 4축 미학 점수 + catalog 매칭
- face.gate 5단계 quality chain + passenger suitability score
- debug/process CLI + HTML report (momentscan-report 분리)
- distributed 모드: 프로세스 격리 + 병렬 실행

## 진화 경로 (3 Phase)

### Phase 1: Batch Highlight (수치/통계) ✅ 완료

기존 momentscan의 실시간 trigger 방식을 **배치 후처리**로 전환.
~2분 라이드 비디오 특성상 배치가 유리한 이유:

- **Per-video 정규화**: 전체 비디오의 통계(median, MAD)를 먼저 계산, 상대적 peak 탐지
- **Global context**: 영상 전체를 본 후 최적 구간 선택 (실시간은 미래 정보 없음)
- **단순한 아키텍처**: on_frame() 축적 → after_run() 분석

핵심 요소:
- 수치 feature delta (smile_intensity, head_yaw, portrait_best 등 9채널)
- Per-video MAD z-score 정규화
- scipy.signal.find_peaks 기반 peak detection
- Explainability log (timeseries.csv + HTML report)

상세: [highlight_rules.md](highlight_rules.md)

### Phase 2: Embedding Experiment ✅ 실험 완료 → portrait.score로 대체

DINOv2/SigLIP temporal delta로 semantic saliency를 감지하는 실험적 접근을 시도.

**결과**: 범용 임베딩(DINOv2/SigLIP)의 temporal delta는 ~2분 라이드 비디오 특성에서
수치 feature 대비 충분한 이점을 보이지 못함. 대신 도메인 특화 접근이 더 효과적:

- **CLIP 4축 portrait scoring** (disney_smile, charisma, wild_roar, playful_cute)
  → 포트레이트 미학 기준에 특화된 점수 체계
- **카탈로그 기반 유사도** (catalog_scores)
  → 목표 카테고리와의 직접 유사도로 Impact 점수 계산
- **ArcFace face identity** (face_identity)
  → 동일인 판정 및 수집 anchor

이 모듈들은 `portrait.score` analyzer로 momentscan 내부에 정착.
`vpx-vision-embed` → `vpx-portrait-score` 경로를 거쳐 최종적으로 momentscan 내부로 마이그레이션.

히스토리: [highlight_vector.md](highlight_vector.md)

### Phase 3: Identity-aware Collection ✅ 완료

portrait981의 핵심 목표. 인물의 **다양한 상태를 체계적으로 수집**하여
디퓨전 모델의 reference image로 사용.

초기 설계 (버킷 기반):
- 듀얼 임베딩 분리: Face-ID(ArcFace)는 동일인 판정, General Vision(DINOv2)은 다양성 측정
- 버킷: yaw(7) × pitch(5) × expression(4) 조합별 best 이미지
- 3-Set 출력: Anchor + Coverage + Challenge

**최종 구현** (Pose×Category 그리드 + 카탈로그):
- **5 pose pivot**: frontal, three-quarter, side-profile, looking-up, three-quarter-up
- **카탈로그 카테고리**: warm_smile, gentle_smile, neutral 등 (CLIP 유사도 기반)
- **Grid selection**: pose×category 셀별 top-K, 품질 gate 통과 필수
- **Soft radius**: r_accept = 15° (euclidean, left/right symmetry)
- **ArcFace identity**: 동일인 anchor 기준 수집

상세: [identity_builder.md](identity_builder.md), [portrait_pivots.md](portrait_pivots.md)

### Highlight ↔ Identity Builder 연결

Phase 1의 highlight 결과를 Phase 3 identity collection에 활용:

| 단계 | 방식 | 상태 |
|------|------|------|
| **1. Sampling Priority** | highlight window 내 프레임 우선 수집 | ✅ 적용 |
| **2. QualityGate 공유** | shared_contracts의 gate 기준 재사용 | ✅ 적용 |
| **3. Learning Data** | highlight 판정 데이터를 identity 학습에 활용 | ⬜ 운영 데이터 확보 후 |

## Analyzer DAG (14개)

```
face.detect → face.classify → face.baseline (stateful, Welford online stats)
     │ → face.expression     (HSEmotion)
     │ → face.quality        (head crop blur/exposure + BiSeNet seg)
     │ → face.parse          (BiSeNet 19-class)
     │ → portrait.score      (CLIP 4축 + catalog)
     │ → face.au             (Action Unit, LibreFace)
     │ → head.pose           (6DRepNet 6DoF)
     └→ face.gate (depends: detect+classify, optional: quality+frame.quality+head.pose)
body.pose (YOLO-Pose), hand.gesture (MediaPipe), frame.quality (OpenCV)
frame.scoring (depends: face.detect, optional: expression+classify+pose+quality)
```

| Analyzer | Name | 패키지 | Stateful |
|----------|------|--------|----------|
| FaceDetectionAnalyzer | `face.detect` | vpx-face-detect | ✅ (tracker) |
| ExpressionAnalyzer | `face.expression` | vpx-face-expression | |
| FaceClassifierAnalyzer | `face.classify` | momentscan core | ✅ (role lock) |
| FaceParseAnalyzer | `face.parse` | vpx-face-parse | |
| FaceQualityAnalyzer | `face.quality` | momentscan core | |
| FrameGateAnalyzer | `face.gate` | momentscan core | |
| FaceBaselineAnalyzer | `face.baseline` | momentscan core | ✅ (Welford) |
| FaceAUAnalyzer | `face.au` | vpx-face-au | |
| HeadPoseAnalyzer | `head.pose` | vpx-head-pose | |
| PortraitScoreAnalyzer | `portrait.score` | momentscan core | |
| PoseAnalyzer | `body.pose` | vpx-body-pose | |
| GestureAnalyzer | `hand.gesture` | vpx-hand-gesture | |
| QualityAnalyzer | `frame.quality` | momentscan core | |
| FrameScoringAnalyzer | `frame.scoring` | momentscan core | |

## App Lifecycle (배치 패턴)

```python
class MomentscanApp(vp.App):
    def setup(self):
        self._frame_records: list[FrameRecord] = []
        self._collection_records: list[CollectionRecord] = []

    def on_frame(self, frame, results):
        # 축적: FlowData → FrameRecord/CollectionRecord 변환
        self._frame_records.append(extract_frame_record(frame, results))
        self._collection_records.append(extract_collection_record(frame, results))

    def after_run(self, result):
        # Batch Highlight
        highlight = BatchHighlightEngine().analyze(self._frame_records)
        # Collection
        collection = CollectionEngine().analyze(self._collection_records)
        return Result(highlights=highlight, collection=collection)
```

## Scoring Formula

```
Final = 0.35 × Quality + 0.65 × Impact + 0.30 × passenger_suitability

Quality = 0.30×head_blur + 0.20×face_size + 0.30×face_identity + 0.25×frontalness
Impact = top-3 weighted (smile_intensity, head_yaw, portrait_best)
```

Gate-pass only EMA: gate_fail 프레임은 이전 smoothed 값 유지.
상세: [highlight_rules.md](highlight_rules.md), [gate_policy.md](gate_policy.md)

## vpx 의존성

| 패키지 | 용도 |
|--------|------|
| vpx-face-detect | 얼굴 검출, 추적, ArcFace embedding |
| vpx-face-expression | 표정 (smile, eye_open, emotion) |
| vpx-face-parse | BiSeNet 19-class 세그멘테이션 |
| vpx-face-au | Action Unit (LibreFace ONNX) |
| vpx-head-pose | 6DoF head pose (6DRepNet) |
| vpx-body-pose | 상반신 포즈 (YOLO-Pose) |
| vpx-hand-gesture | 제스처 (MediaPipe Hands, optional) |
| vpx-sdk | Module, Observation, marks |
| visualbase | 비디오 디코딩 |

## 출력 구조

```
output/{video_id}/
├── highlight/
│   ├── windows.json           # 하이라이트 구간 목록
│   ├── timeseries.csv         # 프레임별 feature 덤프
│   ├── report.html            # 인터랙티브 타임라인 (Plotly)
│   └── frames/                # 구간별 best frame crops
├── collection/
│   ├── person_{id}/
│   │   ├── {pose}_{category}/  # 셀별 크롭
│   │   └── meta.json
│   └── collection_report.html  # 갤러리 + scatter (Plotly)
├── report.html                 # 통합 탭 리포트
└── debug/
    └── score_curve.json
```
