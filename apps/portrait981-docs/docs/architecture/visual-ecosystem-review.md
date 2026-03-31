# Visual* 생태계 리뷰 — 간결함을 되찾자

> 최종 업데이트: 2026-03-26 (MomentscanV2 + VisualBind judge 이후)

## 원래 목적

visual* 생태계는 문제를 간단하게 바라볼 수 있게 하는 것이 목적이다.
각 레이어가 하나의 관심사만 깔끔하게 처리하고, 앱은 조합만 하면 된다.

```
visualbase:  "프레임을 줘"
visualpath:  "분석기를 돌려줘"
vpx:         "이 문제를 풀어줘"
visualbind:  "결과를 합쳐서 판단해줘"
```

## 현재 상태

MomentscanV2 구현을 통해 6개 복잡도 전이 지점 중 4개가 해결되었다.
앱 코드가 ~130줄로 줄었고, 프레임워크 위에서 조합만 하는 형태에 가까워졌다.

---

## 해결된 문제

### ✅ observer_bind (P0, visualbind)

**이전:** 앱에서 Observation → signals dict 수동 변환
**현재:** `bind_observations(observations)` → 50D signals dict 자동 변환

```python
# v2: 한 줄
signals = bind_observations(observations)
```

### ✅ predict_from_signals (P0, visualbind)

**이전:** signals dict → numpy vector 수동 변환 → predict
**현재:** `TreeStrategy.predict(signals)` — dict를 직접 받음

```python
# v2: dict 직접 전달
scores = strategy.predict(signals)
```

### ✅ VisualBind Judge end-to-end (P3→P0, visualbind)

**이전:** gate, expression, pose를 앱에서 개별 호출
**현재:** `VisualBind(gate, expression, pose)` — 한 번의 호출로 통합 판단

```python
judge = VisualBind(
    gate=HeuristicStrategy(),
    expression=TreeStrategy.load("models/bind_v5.pkl"),
    pose=TreeStrategy.load("models/pose_v3.pkl"),
)
result = judge(signals)  # gate + expression + pose 한번에
```

### ✅ 비디오 순회 (P1, visualbase/visualpath)

**이전:** 앱에서 cv2.VideoCapture 직접 사용
**현재:** `vp.App.run("video.mp4")` — FlowGraph가 프레임 순회 + DAG 실행

```python
class MomentscanV2(vp.App):
    def on_frame(self, frame, terminal_results):
        # frame과 결과가 자동으로 전달됨
```

### ✅ 모듈 로딩 (P1, visualpath) — 부분 해결

**이전:** 앱에서 try/except로 7개 analyzer 수동 로딩
**현재:** `Path.from_plugins()` — entry_points 기반 자동 발견

```python
# 사용 가능하지만 v2에서는 modules 리스트 선언 방식 사용 중
modules = ["face.detect", "face.au", "face.expression", ...]
```

### ✅ HeuristicStrategy — gate를 visualbind로 흡수

**이전:** momentscan-plugins/face-gate에서 gate 판단
**현재:** visualbind HeuristicStrategy로 이동, 3단 gate 구조

```
1단: 물리적 품질 (blur, exposure, contrast, clipping)
2단: 포즈 극단값 (yaw, pitch, roll, combined)
3단: Signal validity (seg_face=0, AU 합계≈0 → 포즈 추정기 실패 감지)
```

---

## 남은 문제

### ⬜ Frame.from_image() 없음 (P0, visualbase)

SignalExtractor에서 이미지 분석 시 Frame 수동 생성이 여전히 필요:

```python
# 현재: 5개 파라미터
frame = Frame(data=image, frame_id=0, t_src_ns=0, width=w, height=h)

# 필요: 간편 생성자
frame = Frame.from_image(image)
```

**v2에서는 vp.App이 Frame 생성을 처리하므로 앱 코드에서는 문제 없음.**
SignalExtractor(이미지 모드)에서만 불편.

### ⬜ Path 이름 충돌 (P0, visualpath)

```python
from visualpath.core.path import Path  # pathlib.Path와 충돌
```

논의에서 `VisualPath`로 rename 방향이 나왔지만 아직 미실행.
v2에서는 `modules` 리스트를 선언하고 vp.App이 내부적으로 처리하므로
앱 코드에서 Path를 직접 다루지 않음 — 긴급도 낮아짐.

### ⬜ Signal 이름 비표준 (P2, vpx)

```python
# vpx analyzer 출력: "au_au1", "expression_happy"
# visualbind 표준:   "au1_inner_brow", "em_happy"
# → observer_bind.py가 번역
```

방안 C(observer_bind 유지) 채택 — 기존 analyzer 수정 없이 번역 레이어로 해결.
장기적으로 vpx-sdk에 signal 이름 레지스트리 도입 검토 가능.

### ⬜ analyzer 수동 초기화 (P1, vpx-sdk)

```python
analyzer = FaceDetectionAnalyzer()
analyzer.initialize()  # 별도 호출 필요
```

FlowGraph가 lifecycle을 관리하므로 v2에서는 문제 없음.
SignalExtractor에서만 수동 initialize() 호출.

### ⬜ BindingStrategy 인터페이스 (P2, visualbind)

```python
class BindingStrategy(Protocol):
    def fit(self, vectors, **kwargs): ...  # HeuristicStrategy는 no-op
    def predict(self, frame_vec): ...
```

HeuristicStrategy에 fit()이 의미 없음. 인터페이스 분리 검토 필요:
- `Predictor` (predict만) vs `Learner` (fit + predict)
- 또는 fit을 Optional로

---

## MomentscanV2 — 이상적 코드에 근접

### 리뷰 당시 이상적 앱 코드 (목표)

```python
pipeline = vp.Pipeline.load_all()
judge = vb.Judge(expression_model="...", pose_model="...")

for frame in vb.iter_video("video.mp4", fps=2):
    result = judge(pipeline.analyze(frame))
    if result.is_shoot:
        save(result)
```

### 현재 MomentscanV2 (~130줄)

```python
class MomentscanV2(vp.App):
    modules = ["face.detect", "face.au", "face.expression", ...]
    fps = 2

    def setup(self):
        self.judge = VisualBind(
            gate=HeuristicStrategy(),
            expression=TreeStrategy.load("models/bind_v5.pkl"),
            pose=TreeStrategy.load("models/pose_v3.pkl"),
        )

    def on_frame(self, frame, terminal_results):
        observations = [obs for fd in terminal_results for obs in fd.observations]
        signals = bind_observations(observations)
        judgment = self.judge(signals)
        self._results.append(FrameResult(...))
```

**6개 복잡도 전이 지점 중 4개 해결, 2개는 긴급도 낮음 (앱에서 노출 안 됨).**

---

## 진행 현황 요약

| 우선순위 | 문제 | 상태 | 비고 |
|---------|------|:----:|------|
| P0 | observer_bind | ✅ | visualbind/observer_bind.py |
| P0 | predict_from_signals | ✅ | TreeStrategy.predict(dict) |
| P0 | Frame.from_image() | ⬜ | SignalExtractor에서만 불편 |
| P0 | Path 이름 충돌 | ⬜ | v2에서 직접 노출 안 됨 |
| P1 | 모듈 auto-load | ✅ | Path.from_plugins() |
| P1 | 비디오 순회 | ✅ | vp.App.run() |
| P1 | analyzer 수동 초기화 | ⬜ | FlowGraph가 관리 |
| P2 | signal 이름 비표준 | ⬜ | observer_bind 번역 유지 |
| P2 | BindingStrategy 인터페이스 | ⬜ | fit-less 지원 검토 |
| P3 | VisualBind Judge e2e | ✅ | VisualBind(gate, expr, pose) |

**해결: 5/10 | 미해결(긴급도 낮음): 5/10**

---

## 원칙 (유지)

1. **앱은 조합만** — 프레임워크가 복잡도를 숨겨야 한다
2. **한 줄이면 한 줄** — Frame.from_image(img), strategy.predict(signals)
3. **네이밍은 명확하게** — Path ≠ pathlib.Path
4. **표준은 한 곳에서** — signal 이름, 정규화 범위, 매핑 규칙 (observer_bind)
5. **코드가 문학이 되도록** — 읽으면 의도가 보여야 한다
