# Visual* 생태계 리뷰 — 간결함을 되찾자

## 원래 목적

visual* 생태계는 문제를 간단하게 바라볼 수 있게 하는 것이 목적이다.
각 레이어가 하나의 관심사만 깔끔하게 처리하고, 앱은 조합만 하면 된다.

```
visualbase:  "프레임을 줘"
visualpath:  "분석기를 돌려줘"
vpx:         "이 문제를 풀어줘"
visualbind:  "결과를 합쳐서 판단해줘"
```

## 현재 상태: 복잡도가 목적을 압도

앱(momentscan)이 프레임워크의 복잡도를 직접 다루고 있다.
프레임워크가 문제를 숨겨야 하는데, 오히려 문제를 노출하고 있다.

---

## visualbase 문제

### Frame 생성이 장황
```python
# 현재: 5개 파라미터 수동 지정
Frame(data=image, frame_id=0, t_src_ns=0, width=640, height=480)

# width/height는 image.shape에서 자동 추출 가능
# t_src_ns는 대부분 0 (이미지 모드)
# frame_id는 순차 자동 부여 가능
```

**개선:**
```python
# 이미지에서 자동 생성
frame = Frame.from_image(image)
frame = Frame.from_image(image, frame_id=42)

# 비디오 프레임은 기존 방식 유지 (타임스탬프 필요)
```

### 비디오 순회가 불편
```python
# 현재: 앱에서 cv2 직접 사용
cap = cv2.VideoCapture(str(video_path))
while True:
    ret, frame = cap.read()
    ...

# visualbase에 비디오 소스가 있지만 앱에서 안 쓰는 이유:
# - 콜백 기반 (on_frame) → 단순 for 루프 불가
# - 설정이 복잡
```

**개선:**
```python
# 단순한 제너레이터
for frame in visualbase.iter_video("video.mp4", fps=2):
    result = analyze(frame)
```

---

## visualpath 문제

### Path 이름 충돌
```python
from visualpath.core.path import Path as AnalysisPath  # 매번 별명 필요
from pathlib import Path                                # 진짜 Path
self._path = AnalysisPath(...)                          # 파일 경로? 분석 경로?
```

**개선:** `Path` → `Pipeline` 또는 `AnalyzerDAG` 또는 `Graph`

### 모듈 로딩이 수동
```python
# 현재: 앱에서 import + try/except 반복
_try("face.detect", "vpx.face_detect", "FaceDetectionAnalyzer")
_try("face.au", "vpx.face_au", "FaceAUAnalyzer")
_try("face.expression", "vpx.face_expression", "ExpressionAnalyzer")
_try("head.pose", "vpx.head_pose", "HeadPoseAnalyzer")
_try("face.parse", "vpx.face_parse", "FaceParseAnalyzer")
_try("face.quality", "momentscan.face_quality", "FaceQualityAnalyzer")
_try("frame.quality", "momentscan.frame_quality", "QualityAnalyzer")

# entry_points 기반 plugin discovery가 있지만:
# - "전부 로드해줘"가 안 됨
# - 어떤 analyzer가 필요한지 앱이 알아야 함
```

**개선:**
```python
# vpx list로 확인 가능한 것을 코드에서도
pipeline = visualpath.Pipeline.from_registry(["face.*", "frame.quality"])
# 또는
pipeline = visualpath.Pipeline.load_all()  # 설치된 모든 analyzer
```

### analyze_all() 결과가 Observation 리스트 — 앱이 파싱해야
```python
# 현재: source별로 직접 파싱
observations = pipeline.analyze_all(frame)
for obs in observations:
    if obs.source == "face.detect": ...
    elif obs.source == "face.au": ...

# 프레임워크가 이걸 해줘야 함
```

**개선:**
```python
# dict로 반환
result = pipeline.analyze(frame)
# result["face.detect"].signals → {face_count: 2, ...}
# result["face.au"].signals → {au_au1: 0.07, ...}

# 또는 직접 signals dict로
signals = pipeline.extract_signals(frame)  # → 49D dict
```

---

## vpx 문제

### Signal 이름 비표준
```python
# face.au 출력
{"au_au1": 0.07, "au_au12": 0.28, ...}

# face.expression 출력
{"expression_happy": 0.44, "expression_neutral": 0.03, ...}

# visualbind가 기대하는 이름
{"au1_inner_brow": 0.07, "au12_lip_corner": 0.28, ...}
{"em_happy": 0.44, "em_neutral": 0.03, ...}

# → 번역 레이어(observer_bind)가 필요해짐
```

**원인:** vpx analyzer가 각자의 백엔드 라이브러리 이름을 그대로 노출.
InsightFace는 "AU1", HSEmotion은 "happy", LibreFace는 "au_au1".

**개선 방향:**
- 방안 A: vpx analyzer가 표준 이름으로 출력 → observer_bind 불필요
- 방안 B: vpx-sdk에 signal 이름 레지스트리 → analyzer가 등록 시 매핑 선언
- 방안 C: observer_bind 유지 (현실적, 기존 analyzer 수정 없음)

### analyzer 초기화가 분리
```python
analyzer = FaceDetectionAnalyzer()
analyzer.initialize()  # 별도 호출 필요
```

**개선:** `__init__`에서 자동 초기화 또는 lazy 초기화

---

## visualbind 문제

### predict가 numpy vector만 받음
```python
# 현재: signals dict → 수동 변환 → predict
vec = np.array([normalize_signal(signals.get(f, 0.0), f) for f in SIGNAL_FIELDS])
scores = strategy.predict(vec)

# 앱에서 매번 이 변환을 해야 함
```

**개선:**
```python
# signals dict를 직접 받는 API
scores = strategy.predict_from_signals(signals)

# 또는 bind 자체가 end-to-end
result = visualbind.judge(observations)
# → bind_observations + normalize + predict 한번에
```

### observer_bind가 visualbind에 있어야 할 핵심 기능인데 최근에야 추가
```
visualbind = "여러 observer를 결합(bind)"
→ observer 출력을 결합하는 것이 핵심 기능
→ 그런데 이 기능이 원래 없었고 앱에서 수동으로 하고 있었음
```

### 전략 인터페이스가 학습(fit) 중심
```python
class BindingStrategy(Protocol):
    def fit(self, vectors, **kwargs): ...
    def predict(self, frame_vec): ...
```

HeuristicStrategy는 fit이 no-op. 인터페이스가 맞지 않음.

---

## 앱(momentscan) 문제

### 이상적 앱 코드

프레임워크가 제대로 동작하면 앱은 이렇게 간결해야 함:

```python
import visualpath as vp
import visualbind as vb

# 선언적 구성
pipeline = vp.Pipeline.load_all()
judge = vb.Judge(
    expression_model="models/bind_v4.pkl",
    pose_model="models/pose_v2.pkl",
)

# 이미지 분석
result = judge(pipeline.analyze(frame))

# 비디오 분석
for frame in vb.iter_video("video.mp4", fps=2):
    result = judge(pipeline.analyze(frame))
    if result.is_shoot:
        save(result)
```

### 현재 앱 코드가 장황한 이유

1. Frame 수동 생성 (visualbase)
2. 모듈 수동 로딩 (visualpath)
3. Observation → signals 수동 변환 (visualbind)
4. signals → vector 수동 변환 (visualbind)
5. gate + predict 개별 호출 (visualbind)
6. cv2 비디오 순회 직접 (visualbase)

**6개 지점에서 프레임워크가 앱에 복잡도를 전가하고 있음.**

---

## 개선 우선순위

### P0 (즉시, 앱 영향 큼)

| # | 문제 | 위치 | 개선 |
|---|------|------|------|
| 1 | Frame.from_image() 없음 | visualbase | 간편 생성자 추가 |
| 2 | observer_bind가 없었음 | visualbind | 완료 (observer_bind.py) |
| 3 | predict_from_signals 없음 | visualbind | strategy에 dict 입력 메서드 추가 |
| 4 | Path 이름 충돌 | visualpath | 이름 변경 검토 |

### P1 (단기, 개발자 경험)

| # | 문제 | 위치 | 개선 |
|---|------|------|------|
| 5 | 모듈 수동 로딩 | visualpath | plugin auto-load API |
| 6 | 비디오 순회 불편 | visualbase | iter_video 제너레이터 |
| 7 | analyzer 수동 초기화 | vpx-sdk | lazy init 또는 auto-init |

### P2 (중기, 아키텍처)

| # | 문제 | 위치 | 개선 |
|---|------|------|------|
| 8 | signal 이름 비표준 | vpx | 표준화 또는 레지스트리 |
| 9 | BindingStrategy 인터페이스 | visualbind | fit-less strategy 지원 |
| 10 | analyze_all → dict | visualpath | 결과 접근 방식 개선 |

### P3 (장기)

| # | 문제 | 위치 | 개선 |
|---|------|------|------|
| 11 | visualbind.Judge end-to-end | visualbind | observations → result 한번에 |
| 12 | 선언적 파이프라인 구성 | visualpath | YAML/config 기반 |

---

## 원칙

1. **앱은 조합만** — 프레임워크가 복잡도를 숨겨야 한다
2. **한 줄이면 한 줄** — Frame.from_image(img), strategy.predict_from_signals(sig)
3. **네이밍은 명확하게** — Path ≠ pathlib.Path, _path ≠ file_path
4. **표준은 한 곳에서** — signal 이름, 정규화 범위, 매핑 규칙
5. **코드가 문학이 되도록** — 읽으면 의도가 보여야 한다
