# momentscan — 인물/장면 분석 수집 앱

> 라이드 비디오에서 의미 있는 순간(moment)을 탐지하고,
> 인물의 다양한 상태를 체계적으로 수집하는 분석 앱.
> momentscan의 진화형이자 portrait981 파이프라인의 첫 번째 단계.

## 위치

```
momentscan (분석/수집) → appearance-vault (저장/관리) → reportrait (AI 생성)
```

## 현재 상태

기존 momentscan가 momentscan으로 발전. 현재 구현:
- FlowGraph 기반 실시간 스트리밍 분석
- highlight analyzer (expression spike, head turn 등 trigger 기반)
- face_classifier, quality, frame_scoring analyzers
- CLI + 시각화

## 진화 경로 (3 Phase)

### Phase 1: Batch Highlight (수치/통계)

기존 momentscan의 실시간 trigger 방식을 **배치 후처리**로 전환.
~2분 라이드 비디오 특성상 배치가 유리한 이유:

- **Per-video 정규화**: 전체 비디오의 통계(median, MAD)를 먼저 계산, 상대적 peak 탐지
- **Global context**: 영상 전체를 본 후 최적 구간 선택 (실시간은 미래 정보 없음)
- **단순한 아키텍처**: on_frame() 축적 → after_run() 분석

핵심 요소:
- 수치 feature delta (mouth_open, head_velocity, wrist_raise 등)
- Per-video MAD z-score 정규화
- scipy.signal.find_peaks 기반 peak detection
- Explainability log (timeseries.csv)

상세: [highlight_rules.md](highlight_rules.md)

### Phase 2: Embedding Experiment (검증 단계)

DINOv2/SigLIP temporal delta로 semantic saliency를 감지하는 실험적 접근.
Phase 1과 **병렬 운영**하며 효과를 비교.

핵심 질문: "임베딩 기반 temporal delta가 수치 feature보다 나은가?"

- face/body crop → DINOv2/SigLIP embedding
- EMA baseline 대비 cosine distance (temporal delta)
- Phase 1과 구간 겹침률(IoU) 비교
- 선택적 GRU-lite/TCN-lite reranker (shadow mode부터)

상세: [highlight_vector.md](highlight_vector.md)

### Phase 3: Identity-aware Collection (최종 목표)

portrait981의 진짜 목표. 인물의 **다양한 상태를 체계적으로 수집**하여
디퓨전 모델의 reference image로 사용.

핵심 개념:
- **듀얼 임베딩 분리**: Face-ID(ArcFace)는 동일인 판정, General Vision(DINOv2)은 다양성 측정
- **버킷 기반 다양성**: yaw(7) x pitch(5) x expression(4) 조합별 best 이미지
- **3-Set 출력**: Anchor(정면 고품질) + Coverage(각도/표정 다양성) + Challenge(극단 조건)
- **Coverage-driven 중단**: 수량이 아닌 커버리지 충족 기준

Phase 1/2의 highlight window는 sampling priority로 활용:
highlight 구간 내 프레임에 수집 우선순위 부여.

상세: [identity_builder.md](identity_builder.md)

## App Lifecycle (배치 패턴)

visualpath의 Spec/Interpreter 분리 덕분에 FlowGraph 변경 없이 배치 지원:

```python
class MomentscanApp(vp.App):
    def setup(self):
        self.records: list[FrameRecord] = []

    def on_frame(self, frame, results):
        # 축적: 프레임별 수치 feature + 임베딩 추출 결과
        self.records.append(extract_signals(frame, results))

    def after_run(self, result):
        # 분석: 전체 비디오 기준 정규화 + peak detection
        df = pd.DataFrame(self.records)
        df = normalize_per_video(df)            # Phase 1
        embeddings = stack_embeddings(df)        # Phase 2
        peaks = find_peaks(df.impact_score)
        windows = generate_windows(peaks)
        identity_sets = collect_identity(df)     # Phase 3
        export(windows, identity_sets)
```

## vpx 의존성

| 패키지 | Phase | 용도 |
|--------|-------|------|
| vpx-face-detect | 1,2,3 | 얼굴 검출, 추적, head pose, Face-ID embedding |
| vpx-face-expression | 1,3 | 표정 proxy (mouth_open, eye_open, smile) |
| vpx-body-pose | 1,3 | 상반신 포즈 (wrist, elbow, torso) |
| vpx-hand-gesture | 1 | 제스처 (optional) |
| vpx-sdk | all | QualityGate, Observation |
| visualbase | all | 비디오 디코딩 |
| **vpx-vision-embed** | 2,3 | DINOv2/SigLIP 임베딩 추출 (신규) |

## 출력 구조

```
output/{video_id}/momentscan/
├── highlight/
│   ├── windows.json           # 하이라이트 구간 목록
│   ├── timeseries.csv         # 프레임별 feature 덤프
│   └── frames/                # 구간별 best frame
├── identity/                  # Phase 3
│   ├── person_0/
│   │   ├── anchors/
│   │   ├── coverage/
│   │   ├── challenge/
│   │   └── meta.json
│   └── person_1/
└── debug/
    ├── score_curve.json
    └── embedding_delta.json   # Phase 2
```

## 개발 순서

| 순서 | 작업 | 의존 |
|------|------|------|
| 1 | momentscan → momentscan 리네임 | - |
| 2 | 배치 모드 전환 (on_frame 축적 + after_run 분석) | - |
| 3 | per-video 정규화 + peak detection (Phase 1) | 2 |
| 4 | timeseries.csv + explainability log | 3 |
| 5 | vpx-vision-embed 플러그인 (Phase 2) | - |
| 6 | embedding temporal delta + Phase 1 비교 | 3, 5 |
| 7 | 듀얼 임베딩 + 버킷 수집 (Phase 3) | 5 |
| 8 | appearance-vault 연동 | 7 |
