# Momentscan v2 — 간결한 재설계

## 철학

본질만 남기고 나머지는 버린다.
200줄 수준의 핵심 로직. plugins과 visualbind가 무거운 일을 하고, app은 조합만.

## 핵심 로직 (본질)

```python
for frame in video:
    signals = extract(frame)           # plugins → 49D
    if not gate_ok(signals):           # visualbind HeuristicStrategy
        continue
    expression = predict(signals)      # visualbind TreeStrategy
    pose = predict_pose(signals)       # visualbind TreeStrategy
    results.append(frame, expression, pose, signals)
```

## 재활용 (기존 코드 그대로)

| 컴포넌트 | 패키지 | 역할 |
|----------|--------|------|
| vpx plugins (7개) | libs/vpx/plugins/ | frozen model 분석 |
| momentscan-face-quality | apps/momentscan-plugins/ | BiSeNet 마스크 기반 품질 측정 |
| momentscan-frame-quality | apps/momentscan-plugins/ | 프레임 전체 품질 |
| momentscan-portrait-score | apps/momentscan-plugins/ | CLIP 4축 |
| SignalExtractor | momentscan.signals | DAG 기반 signal 추출 |
| HeuristicStrategy | visualbind | threshold 기반 gate |
| TreeStrategy | visualbind | XGBoost 판단 |
| visualbase | libs/visualbase/ | 미디어 I/O |

## 새로 만드는 것

```
apps/momentscan/src/momentscan/
├── app_v2.py              # 핵심 앱 (~200줄)
└── cli_v2.py              # CLI 진입점 (간결)
```

## app_v2.py 설계

```python
@dataclass
class FrameResult:
    """프레임별 분석 결과 — 단일 레코드."""
    frame_idx: int
    timestamp_ms: float
    signals: dict[str, float]       # 49D raw signals
    gate_passed: bool
    gate_reasons: list[str]
    expression: str                  # XGBoost 예측 (cheese/chill/...)
    expression_conf: float
    pose: str                        # XGBoost 예측 (front/angle/side)
    pose_conf: float
    face_detected: bool
    face_count: int


class MomentscanV2:
    """간결한 momentscan — plugins + visualbind 조합."""

    def __init__(
        self,
        bind_model: str | Path | None = None,
        pose_model: str | Path | None = None,
        gate_config: GateConfig | None = None,
    ):
        self.extractor = SignalExtractor()
        self.gate = HeuristicStrategy(gate_config or GateConfig())

        if bind_model:
            self.bind = TreeStrategy.load(bind_model)
        else:
            self.bind = None

        if pose_model:
            self.pose = TreeStrategy.load(pose_model)
        else:
            self.pose = None

    def initialize(self):
        self.extractor.initialize()

    def analyze_image(self, image: np.ndarray, frame_id: int = 0) -> FrameResult:
        """단일 이미지 분석."""
        sig = self.extractor.extract(image, frame_id)

        # Gate (visualbind HeuristicStrategy)
        gate_fails = self.gate.check_gate_from_signals(sig.signals)

        # Expression (visualbind TreeStrategy)
        expression, expr_conf = "", 0.0
        if self.bind and sig.face_detected:
            vec = self._to_vector(sig.signals)
            scores = self.bind.predict(vec)
            if scores:
                expression = max(scores, key=scores.get)
                expr_conf = scores[expression]

        # Pose
        pose, pose_conf = "", 0.0
        if self.pose and sig.face_detected:
            vec = self._to_vector(sig.signals)
            scores = self.pose.predict(vec)
            if scores:
                pose = max(scores, key=scores.get)
                pose_conf = scores[pose]

        return FrameResult(
            frame_idx=frame_id,
            timestamp_ms=0.0,
            signals=sig.signals,
            gate_passed=len(gate_fails) == 0,
            gate_reasons=gate_fails,
            expression=expression,
            expression_conf=expr_conf,
            pose=pose,
            pose_conf=pose_conf,
            face_detected=sig.face_detected,
            face_count=sig.face_count,
        )

    def analyze_video(self, video_path, fps=2, max_frames=500):
        """비디오 분석 — 프레임별 결과 리스트."""
        results = []
        for idx, image, sig in self.extractor.extract_from_video(video_path, fps, max_frames):
            result = self.analyze_image(image, idx)
            result.timestamp_ms = idx / fps * 1000
            results.append(result)
        return results

    def select_frames(self, results, top_k=10):
        """SHOOT 프레임 중 다양성 기반 선택."""
        shoot = [r for r in results if r.gate_passed and r.expression != 'cut']
        if not shoot:
            return []

        # expression × pose 버킷별 best
        buckets = {}
        for r in shoot:
            key = f"{r.expression}|{r.pose}"
            if key not in buckets or r.expression_conf > buckets[key].expression_conf:
                buckets[key] = r

        selected = sorted(buckets.values(), key=lambda r: -r.expression_conf)
        return selected[:top_k]

    def _to_vector(self, signals):
        from visualbind.signals import SIGNAL_FIELDS, normalize_signal
        return np.array([normalize_signal(signals.get(f, 0.0), f)
                        for f in SIGNAL_FIELDS])
```

## 기존 v1과의 관계

```
v1 (기존): 유지, 점진적 deprecation
  - BatchHighlightEngine, CollectionEngine, FrameRecord 등
  - 기존 테스트 558개 유지
  - 디버그 오버레이, 리포트 등 부수 기능

v2 (신규): 핵심 로직만
  - SignalExtractor + visualbind
  - 단일 FrameResult 레코드
  - 간결한 select_frames
  - scripts/predict_report, batch_report가 v2를 사용

전환:
  v2가 안정되면 v1의 기능을 v2로 점진적 이전
  v1 고유 기능 (debug overlay, highlight windows) → 별도 유틸로 분리
```

## 구현 순서

1. `app_v2.py` 작성 (~100줄)
2. `cli_v2.py` 작성 (debug + process)
3. scripts/predict_report.py → MomentscanV2 사용으로 전환
4. scripts/batch_report.py → 동일
5. 테스트 작성
6. 안정화 후 v1 deprecation
