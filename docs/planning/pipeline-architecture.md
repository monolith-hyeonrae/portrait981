# Pipeline Architecture — Signal Production → Interpretation → Adaptation

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
visualbase  → 미디어 I/O + IPC 인프라 (카메라, 비디오, 프레임)
visualpath  → DAG 기반 분석 프레임워크 (frozen model orchestration)
vpx         → 플러그인 분석 모듈 (face.detect, face.expression, head.pose, ...)
             → vpx-sdk: Module/Observation 프로토콜
             → vpx-runner: analyzer 등록 + DAG 실행
             → vpx plugins: 각 frozen model 래핑 (7개)
visualbind  → signal 결합 + 학습 기반 판단 (XGBoost)
visualgrow  → 지속 적응 + 자율 성장 (pseudo-label, embedding)
```

```
visualbase (미디어 소스)
    ↓ Frame
visualpath (DAG 프레임워크) + vpx (플러그인 분석 모듈)
    ↓ per-frame signals
visualbind (학습 기반 판단)
    ↓ decisions
visualgrow (자율 성장)
```

도메인에 독립적. 어떤 비전 분석 문제에도 적용 가능한 범용 계층.
vpx는 visualpath 위에서 실행되는 분석 모듈 생태계.

### 응용 프로그램 (981파크 특화)

```
momentscan          = visualpath + vpx + visualbind를 조합한 분석 앱
momentscan-plugins  = 도메인 특화 분석 플러그인 (face.quality, face.gate, face.classify 등)
                      → vpx와 같은 Module 프로토콜, momentscan namespace
momentbank          = 고객 기억 시스템 (member 단위 장기 저장)
reportrait          = AI 초상화 생성 (ComfyUI bridge)
portrait981         = 통합 오케스트레이터 (E2E 파이프라인)
annotator           = 라벨링/리뷰/데이터셋 관리 도구
```

```
범용 모듈 (vpx plugins):        도메인 모듈 (momentscan plugins):
  face.detect (InsightFace)       face.quality (마스크 기반 측정)
  face.expression (HSEmotion)     face.gate (품질 게이트)
  face.au (LibreFace)             face.classify (역할 분류)
  head.pose (6DRepNet)            frame.quality (프레임 품질)
  face.parse (BiSeNet)            frame.scoring (프레임 점수)
  body.pose (YOLO-Pose)           portrait.score (CLIP 4축)
  hand.gesture (MediaPipe)        face.baseline (Welford stats)
```

범용 프레임워크 위에 도메인 로직을 얹은 응용.
vpx plugins는 범용 (어떤 프로젝트에서도 재사용),
momentscan plugins는 981파크 특화 (portrait 도메인 로직).

## 3-Layer Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: visualpath + vpx (Signal Production)            │
│                                                          │
│ "각자 한 가지 문제만 푸는 작은 분석기들의 DAG"           │
│                                                          │
│ visualbase: 미디어 소스 (카메라, 비디오, 이미지)         │
│ visualpath: DAG 프레임워크 (의존성 관리, 실행 제어)      │
│ vpx plugins: 범용 frozen model (face, pose, gesture)     │
│ momentscan plugins: 도메인 특화 (quality, gate, score)   │
│                                                          │
│ frozen models → per-frame signals                       │
│ 하드코딩 임계값 → 품질 gate                              │
│ 숙련된 엔지니어가 데이터 기반으로 튜닝                    │
│                                                          │
│ 한계: 모델 드리프트, 새 환경에서 재튜닝 필요             │
│ 가치: 즉시 동작, 추가 학습 불필요                        │
├─────────────────────────────────────────────────────────┤
│ Layer 2: visualbind (Signal Interpretation)              │
│                                                          │
│ "단위 분석기의 결과를 유기적으로 해석하는 결정 트리"      │
│                                                          │
│ 49D signals → XGBoost → expression / pose / quality     │
│ 데이터에서 학습 → 하드코딩 임계값 보완                   │
│ 도메인 환경에 적응적                                     │
│                                                          │
│ 한계: 라벨 데이터 필요, 학습 기반 판단                   │
│ 가치: 같은 frozen model로 정확도 대폭 향상               │
├─────────────────────────────────────────────────────────┤
│ Layer 3: visualgrow (Continuous Adaptation)              │
│                                                          │
│ "서비스가 돌수록 데이터가 쌓이고 모델이 진화"            │
│                                                          │
│ pseudo-label → 자율 재학습                               │
│ Face State Embedding → Moment Embedding                 │
│ 도메인 특화 통합 모델                                    │
│                                                          │
│ 한계: 대규모 데이터 + 운영 경험 필요                     │
│ 가치: 완전한 도메인 적응 시스템                           │
└─────────────────────────────────────────────────────────┘
```

## Layer 간 관계

```
Layer 1 (visualpath):
  입력: 비디오 프레임 / 이미지
  출력: 49D signal vector + 품질 gate 판단
  역할: signal을 생산한다
  구현: vpx analyzers + momentscan face.quality/face.gate

Layer 2 (visualbind):
  입력: Layer 1이 생산한 signal
  출력: expression / pose / quality 판단
  역할: signal을 해석한다
  구현: XGBoost TreeStrategy

Layer 3 (visualgrow):
  입력: Layer 1 signal + Layer 2 판단 + 운영 피드백
  출력: 개선된 모델, pseudo-label
  역할: 시스템을 성장시킨다

응용 (momentscan):
  Layer 1 + Layer 2를 조합하여 비디오 분석 수행
  DAG 실행, 프레임 수집, 배치 하이라이트
```

**핵심 원칙: Layer 2는 Layer 1의 출력만 소비한다.**
visualbind는 visualpath가 생산한 signal을 받아서 해석하지,
자체적으로 signal을 추출하지 않는다.

## 현재 문제: Layer 1 우회

### 발견 경위
품질 게이트가 predict_report에서 작동하지 않는 문제를 추적하다가,
scripts/가 visualpath 파이프라인을 완전히 우회하고 있음을 발견.

### 문제 구조
```
올바른 흐름 (설계 의도):
  비디오 → visualpath DAG → 49D signal + face.gate
                                    ↓
                              visualbind XGBoost → 판단

실제 구현 (scripts/):
  비디오 → extract_signals.py (analyzer 직접 호출, DAG 무시)
         → predict_report.py (하드코딩 gate 별도 구현)
         → 판단

  visualpath의 DAG, 의존성 관리, face.quality 마스크 기반 측정,
  face.gate 다중 조건이 전부 무시됨.
```

### 구체적 누락 사항
| visualpath 파이프라인에 있는 것 | scripts에 없었던 것 |
|-------------------------------|-------------------|
| BiSeNet 마스크 기반 face.quality | 단순 crop 전체 픽셀 평균 |
| 3-level fallback (parsing → landmark → center) | fallback 없음 |
| face.gate (contrast, clipped, crushed, parsing_coverage, seg_face, seg_mouth) | 하드코딩 exposure threshold만 |
| face.classify (main/passenger 구분) | 최대 얼굴만 사용 |
| BBoxSmoother (temporal smoothing) | 없음 |
| frame.quality (프레임 전체 품질) | 수동 계산 |

## 리팩토링 방향

### 원칙
1. **Signal 추출은 한 곳에만 존재** — momentscan의 analyzer + face.quality + face.gate
2. **visualbind는 signal만 소비** — 자체 signal 추출 금지
3. **실험/학습도 같은 경로** — 서비스와 실험의 signal이 동일해야 함
4. **이미지 모드 지원** — 비디오 없이 이미지에서도 동일 signal 추출 가능

### 방안 A: visualpath에 이미지 모드 추가

```python
# visualpath가 비디오와 이미지 모두 처리
results = visualpath.run(input="video.mp4", fps=2)     # 기존
results = visualpath.run(input="images/*.jpg")          # 신규

# 내부적으로 같은 DAG, 같은 analyzer, 같은 gate
# visualbind는 results에서 signal만 꺼내서 학습/예측
```

장점: 단일 진입점, 완전한 DAG 활용
단점: visualpath가 비디오 스트리밍에 결합되어 있어 수정 범위 큼

### 방안 B: SignalExtractor 공유 모듈

```python
# momentscan 내부에 signal 추출 함수를 공개 API로 노출
from momentscan.signals import SignalExtractor

extractor = SignalExtractor()  # analyzer 로딩 + DAG 구성
signals = extractor.extract(image)  # 단일 이미지
# → face.detect → face.quality (마스크 기반) → face.gate → 49D signal

# momentscan 비디오 파이프라인도 내부적으로 이걸 사용
# scripts/도 이걸 사용 → 동일 signal 보장
```

장점: 최소 수정, 이미지/비디오 모두 지원, DAG 로직 공유
단점: momentscan 앱 내부 구조 노출

### 방안 C: vpx-runner 확장

```python
# vpx run이 이미 DAG 실행을 지원
# 이미지 입력 모드 + momentscan 플러그인 통합
signals = vpx.run(
    analyzers=["face.detect", "face.au", "face.quality", "face.gate", ...],
    input="image.jpg",
    output="signals",
)
```

장점: 기존 인프라 활용, 범용 프레임워크에 위치
단점: momentscan 플러그인이 vpx namespace에 혼합

### 추천: 방안 B

momentscan은 visualpath + visualbind의 응용 앱이므로,
signal 추출 공유 모듈이 momentscan 안에 있는 것이 자연스럽다.
(향후 범용화가 필요하면 visualpath로 올릴 수 있다.)

```
momentscan/
├── app.py              # 비디오 파이프라인 (기존)
├── signals/
│   └── extractor.py    # SignalExtractor (신규, 공유)
└── algorithm/
    └── batch/          # BatchHighlightEngine (기존)

scripts/
├── extract_signals.py  → from momentscan.signals import SignalExtractor
├── predict_report.py   → SignalExtractor + visualbind
└── batch_report.py     → SignalExtractor + visualbind
```

## SignalExtractor 인터페이스 (방안 B)

```python
class SignalExtractor:
    """momentscan DAG 기반 signal 추출 — 비디오/이미지 공용."""

    def __init__(self, config: dict | None = None):
        """analyzer 로딩 + DAG 구성.

        로딩되는 analyzer:
          vpx: face.detect, face.au, face.expression, head.pose, face.parse
          momentscan: face.quality, face.gate, frame.quality
        """

    def extract_from_image(self, image: np.ndarray) -> SignalResult:
        """단일 이미지에서 signal 추출.

        Returns:
            SignalResult:
                signals: dict[str, float]   # 49D raw signal
                gate_passed: bool           # face.gate 판단
                gate_reasons: list[str]     # gate 실패 사유
                face_count: int
                main_face_id: int
        """

    def extract_from_video(self, video_path: Path, fps: int = 2) -> list[SignalResult]:
        """비디오에서 프레임별 signal 추출."""

    def get_signal_fields(self) -> tuple[str, ...]:
        """현재 설정의 signal field 목록."""
```

## visualbind 통합

```python
# 현재 (문제):
# predict_report.py가 자체적으로 analyzer 호출 + XGBoost 실행

# 목표:
from momentscan.signals import SignalExtractor
from visualbind import TreeStrategy  # XGBoost

extractor = SignalExtractor()
xgb = TreeStrategy.load("models/bind_v4.pkl")

for result in extractor.extract_from_video(video_path):
    if not result.gate_passed:
        prediction = "cut"  # Layer 1 gate에서 이미 거부
    else:
        prediction = xgb.predict(result.signals)  # Layer 2 해석
```

**Layer 1 gate + Layer 2 XGBoost가 자연스럽게 결합.**
gate는 최소 품질 보장, XGBoost는 표정/포즈 판단.
둘 다 같은 signal을 사용하지만 역할이 다름.

## face.gate → XGBoost 흡수 (visualpath → visualbind 진화)

gate의 품질 판단은 본질적으로 XGBoost의 결정 트리에 포함되어야 한다.
"사진으로 쓸 수 있는가?"와 "어떤 표정인가?"는 같은 signal에서 나오는 판단이다.

```
현재 (두 시스템 분리):
  face.gate: exposure > 200 → CUT (하드코딩)
  XGBoost:   em_happy + mouth_open → cheese (학습)

목표 (통합):
  XGBoost:   exposure 높고 contrast 낮으면 → CUT  (품질 판단)
             em_happy + mouth_open → cheese        (표정 판단)
             같은 결정 트리 안에서 품질과 표정을 함께 판단
```

이것이 visualpath → visualbind 진화의 핵심:
**하드코딩 임계값이 학습된 결정 경계로 대체된다.**

gate의 모든 signal은 이미 49D에 포함되어 있다:
- face_exposure, face_contrast, clipped_ratio, crushed_ratio → 49D 안에 있음
- XGBoost가 충분한 품질-CUT 예시를 보면 자연스럽게 학습

### 전환 단계

```
Phase 0 (현재):
  face.gate = 유일한 품질 필터 (하드코딩)
  XGBoost = 표정/포즈만 판단
  → 품질-CUT 라벨 데이터 부족

Phase 1 (단기):
  face.gate = 안전망으로 유지
  XGBoost = 표정 + 품질 함께 학습 시작
  → 품질 나쁜 프레임을 CUT으로 라벨링하여 학습 데이터 축적
  → gate와 XGBoost가 이중으로 필터링

Phase 2 (중기):
  face.gate = 극단적 불량만 (매우 보수적)
  XGBoost = 품질 판단의 주체
  → XGBoost가 gate보다 정교하게 품질 판단
  → gate는 XGBoost 실패 시 최후 방어선

Phase 3 (장기):
  face.gate = 최소한의 하드웨어 레벨 체크 (no face, 프레임 깨짐 등)
  XGBoost (또는 후속 모델) = 품질 + 표정 + 포즈 통합 판단
  → gate의 역할이 거의 소멸
```

### 현재 부족한 것

XGBoost가 품질 gating을 학습하려면 **품질-CUT 라벨**이 필요:
- 현재 CUT: 대부분 "표정이 안 좋아서" (표정 CUT)
- 필요: "과노출이라서", "흔들려서", "어두워서" (품질 CUT)
- 방법: 품질 나쁜 비디오(test_12 등)에서 의도적으로 품질-CUT 라벨링

## 단계적 진화

```
현재 (Phase 0):
  momentscan → 하드코딩 gate → 수동 임계값
  결과: 즉시 동작, but 새 카메라/환경에 취약

Phase 1 (지금 하는 것):
  momentscan → signal → visualbind XGBoost
  결과: 데이터 기반 판단, gate + XGBoost 이중 필터

Phase 2 (다음):
  momentscan → signal → XGBoost + face.gate 임계값도 학습
  결과: gate threshold도 데이터에서 최적화

Phase 3 (미래):
  momentscan → signal → 통합 임베딩 모델
  결과: frozen model crowds 기반 도메인 특화 모델

Phase 4 (장기):
  VisualGrow → pseudo-label → 자율 재학습
  결과: 서비스가 돌수록 자동 개선
```

## 리팩토링 범위

### 즉시 (이번 주)
1. `SignalExtractor` 클래스 구현 (momentscan 패키지 안)
2. `extract_signals.py` → SignalExtractor 사용으로 전환
3. `predict_report.py` → SignalExtractor + face.gate 활용, 하드코딩 gate 제거
4. `batch_report.py` → 동일

### 이후
5. momentscan 비디오 파이프라인도 SignalExtractor 내부 사용
6. face.gate 설정을 visualbind에서 학습 가능하게
7. visualbind를 momentscan의 선택적 post-processor로 통합

## 핵심 원칙

1. **Signal 추출은 한 곳** — SignalExtractor가 유일한 signal 소스
2. **Layer 1 → Layer 2 단방향** — visualbind는 momentscan 출력만 소비
3. **Gate는 안전망** — XGBoost가 아무리 좋아져도 gate 유지
4. **단계적 도입** — momentscan만으로 동작, visualbind는 선택적
5. **서비스 = 실험** — 같은 signal 경로, 같은 gate, 같은 품질
