# MomentScan v2

> 최종 업데이트: 2026-03-26

## 핵심

vp.App 상속 + VisualBind per-frame judgment. ~130줄.
visual* 프레임워크를 정석대로 활용하는 간결한 구현.

```python
class MomentscanV2(vp.App):
    modules = ["face.detect", "face.au", "face.expression", "head.pose",
               "face.parse", "face.quality", "frame.quality"]
    fps = 2

    setup()     → VisualBind(gate=Heuristic, expression=XGBoost, pose=XGBoost)
    on_frame()  → bind_observations(obs) → judge(signals) → FrameResult
    after_run() → return list[FrameResult]
```

## 파이프라인

```
Video → vp.App.run() → FlowGraph (DAG)
  │
  ├─ face.detect → face.au → face.expression → head.pose
  ├─ face.parse → face.quality
  └─ frame.quality
  │
  ▼ on_frame()
  bind_observations(observations) → 43D signals
  VisualBind judge:
    1. HeuristicStrategy (3단 gate)
    2. TreeStrategy (expression: bind_v6.pkl)
    3. TreeStrategy (pose: pose_v4.pkl)
  → JudgmentResult → FrameResult 축적
  │
  ▼ select_frames()
  expression × pose 버킷별 best → 다양성 기반 선택
```

## FrameResult

```python
@dataclass
class FrameResult:
    frame_idx: int
    timestamp_ms: float
    signals: dict[str, float]       # 43D raw signals
    judgment: JudgmentResult        # gate + expression + pose
    face_detected: bool
    face_count: int

    # shortcuts
    gate_passed → judgment.gate_passed
    expression → judgment.expression
    expression_conf → judgment.expression_conf
    pose → judgment.pose
    is_shoot → judgment.is_shoot
```

## 사용법

```python
# Python API
import momentscan as ms

results = ms.run("video.mp4")  # list[FrameResult]
selected = MomentscanV2.select_frames(results, top_k=10)

# CLI
momentscan v2 video.mp4
momentscan v2 video.mp4 --debug                 # cv2 시각화
momentscan v2 video.mp4 --debug --output out.mp4  # 디버그 영상 저장
momentscan v2 video.mp4 --bind-model models/bind_v6.pkl --pose-model models/pose_v4.pkl
```

## 디버그 시각화

`--debug` 플래그로 실시간 시각화:

```
┌──────────────────────┬──────────┐
│  Video Frame         │ Gate     │
│  (bbox + head pose   │ Expression│
│   axes overlay)      │ Pose     │
│                      │ AU Face  │
│                      │ Coverage │
├──────────────────────┴──────────┤
│  AU Heatmap (12 AU × time)      │
├─────────────────────────────────┤
│  Expression Timeline            │
│  (per-category confidence)      │
│  + Gate severity square wave    │
└─────────────────────────────────┘
```

- **영상**: face bbox (gate 색상) + 3축 head pose 화살표
- **우측 패널**: gate 이유 + signal 바 + expression/pose 확률 분포 + AU 얼굴 와이어프레임 + coverage 그리드
- **AU 히트맵**: 12 AU × 시간, 강도 색상 (검→초→노→빨)
- **타임라인**: 카테고리별 confidence 곡선 + gate severity 연속 wave + SHOOT 마커
- gate fail 구간도 expression curve 표시 (어두운 색, 끊기지 않음)
- 키보드: `q` 종료, `space` 일시정지

## v1과의 차이

| 관점 | v1 | v2 |
|------|-----|-----|
| 상속 | vp.App | vp.App |
| 판단 시점 | after_run (batch temporal) | on_frame (per-frame) |
| 엔진 | BatchHighlightEngine | VisualBind judge |
| Signal | v1 자체 FrameRecord 추출 | bind_observations() 43D |
| 정규화 | MAD z-score (per-video) | 없음 (모델 내장) |
| 출력 | HighlightWindow[] | FrameResult[] |
| Temporal | EMA + peak detect | 없음 (향후 고려) |
| 복잡도 | ~400줄 | ~130줄 |
| 디버그 | 14 analyzer 오버레이 | 판단 근거 중심 시각화 |

## 배치 실행 + 서버 모드 (TODO)

### 현재 문제

vp.App.run()이 매 호출마다 setup()/teardown()을 실행:
- MomentscanV2.setup(): VisualBind judge 로딩 (TreeStrategy.load)
- FlowGraph: analyzer initialize/cleanup (GPU 모델 로딩/해제)
- InsightFace, LibreFace, 6DRepNet, BiSeNet 등 GPU 모델 재로딩 ~10초/회

```
현재 (비효율):
  video_1 → setup(모델 7개 로딩 10초) → 분석 30초 → teardown
  video_2 → setup(모델 7개 로딩 10초) → 분석 30초 → teardown
  → 2000건 × 10초 = 모델 로딩만 5.5시간

필요:
  setup(모델 7개 로딩 10초, 1회) → video_1 → video_2 → ... → teardown
  → 모델 로딩 0
```

### 해결 방향

**1단계: 모델 캐시 (App 수준)**
- setup()에서 이미 로딩된 모델이면 재로딩 건너뜀
- VisualBind judge: 한번 로딩 후 self에 캐시
- FlowGraph analyzer: initialize된 상태 유지, video 전환 시 reset만

**2단계: 배치 CLI**
```bash
# 디렉토리 일괄 처리 (모델 1회 로딩)
momentscan batch /videos/*.mp4 --output /results/

# 또는 목록 파일
momentscan batch --list jobs.txt --output /results/
```

**3단계: 서버 모드 (REST API)**
```bash
# 모델 상주 + HTTP 요청 처리
momentscan serve --port 8080

# 클라이언트
curl -X POST http://localhost:8080/analyze -F video=@test.mp4
```

서버 모드의 이점:
- 모델 GPU 메모리에 상주 → 요청별 로딩 0
- 비동기 큐 처리 가능 → 피크 시간 대응
- portrait981-serve와 통합 가능
- 이후 Redis/Kafka 큐 기반으로 확장

### 운영 요구사항
- 일 2000건 비디오 처리
- 탑승 완료 후 고객 체험 가치 유효 시간 내 콘텐츠 전달
- GPU 자원 효율적 활용

## CLI 정리 (TODO)

현재 v1 명령어(debug, process, collect, bank, catalog-build)와 v2가 혼재.
v1 제거하고 v2 기반으로 단순화:

```bash
# 현재 (복잡)
momentscan debug video.mp4          # v1 debug
momentscan process video.mp4        # v1 process
momentscan v2 video.mp4 --debug     # v2 debug

# 목표 (단순)
momentscan run video.mp4            # 분석 (기본)
momentscan run video.mp4 --debug    # 디버그 시각화
momentscan batch /videos/           # 배치 처리
momentscan serve --port 8080        # 서버 모드
momentscan info                     # 시스템 정보
```

## 재활용하는 컴포넌트

| 컴포넌트 | 패키지 | 역할 |
|----------|--------|------|
| vpx plugins (7개) | libs/vpx/plugins/ | frozen model 분석 |
| face.quality | momentscan-plugins/ | BiSeNet 마스크 기반 품질 측정 |
| frame.quality | momentscan-plugins/ | 프레임 전체 품질 |
| bind_observations | visualbind | Observations → 43D signal dict |
| VisualBind | visualbind | gate + expression + pose 통합 판단 |
| HeuristicStrategy | visualbind | 3단 gate (물리 품질 + 포즈 + signal validity) |
| TreeStrategy | visualbind | XGBoost 분류 |
| vp.App + FlowGraph | visualpath | DAG 실행, 비디오 순회, lifecycle |
