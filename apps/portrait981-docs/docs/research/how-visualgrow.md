# visualgrow — 매일 데이터로 성장하는 적응 시스템

> visualbind가 "지금 어떻게 판단할지"를 다룬다면,
> visualgrow는 "어떻게 계속 개선할지"를 다룬다.

---

## 핵심 아이디어

매일 쏟아지는 unlabeled data를 crowds consensus로 자동 활용하여
visualbind의 strategy를 재학습. 수동 라벨링 병목을 제거.

```
Day 1:  14 frozen teachers + 수동 threshold → 운용
             ↓
        2-3000건/일 × 30fps → 수만 프레임/일
             ↓
        다수 observer 고신뢰 합의 → pseudo ground truth
             ↓
        strategy 자동 재학습 (XGBoost → 더 나은 XGBoost)
             ↓
Day 30: 도메인 특화된 판단 모델
```

**annotation-free training, annotation-efficient validation**:
학습에는 annotation이 불필요하고, 검증에만 소량의 Anchor Set을 사용한다.

---

## Phased Complexity

### Level 1: Majority Vote + XGBoost 재학습 (batch)

```
43D signals → binary votes (threshold 적용) → majority vote → pseudo-label
pseudo-label + 43D → XGBoost 재학습 (배치, 수동 트리거)
```

가장 단순한 형태. visualbind의 현재 XGBoost strategy를 새 데이터로 재학습.
Anchor Set (500건 human labels)으로 재학습 전후 성능 비교.

### Level 2: Dawid-Skene + Anchor Set 모니터링

```
majority vote → Dawid-Skene weighted pseudo-label (teacher reliability 가중)
+ Anchor Set 대비 주간 성능 추적
+ Tier 분류 (DS posterior 기반):
    Tier 1 (posterior > 0.95): 학습 데이터 (고신뢰)
    Tier 2 (0.05-0.95):       경계선, hard negative 후보
    Tier 3 (posterior ~ 0.5): blind spot 후보
```

### Level 3: Vision Student + Crowds Supervision (미래)

```
Raw Image → Vision backbone (MobileNetV3) → bucket 예측
+ crowds pseudo-label → BCE loss
+ teacher 출력 재구성 → soft loss (Kendall uncertainty weighting)
→ 14-model inference를 1-model inference로 교체
```

### Level 4: Video-LLM + RL (장기 비전)

```
Crowds consensus를 reward signal로 활용하여
end-to-end vision model 학습
```

**원칙: 실패 모드가 구체적으로 확인된 후에만 복잡도를 올린다.**
Level 3-4는 Level 1-2가 불충분함을 증명할 때까지 추측.

---

## 책임

1. **Crowds pseudo-label 생성** — threshold + majority vote / Dawid-Skene
2. **Strategy 자동 재학습 트리거** — 데이터 축적량 기반 스케줄
3. **Drift 감지 및 대응** — 5종 drift 자동 감지
4. **Few-shot 새 버킷 추가** — 50-100건으로 새 버킷 Head 학습
5. **개념 시드 자동화** — 참조 이미지 → frozen 모델 관찰 → threshold 자동 도출
6. **Anchor Set 관리 + 모니터링** — 주간 human agreement check
7. **Rollback / 안전장치** — 성능 하락 시 이전 모델로 복원

---

## Drift 적응 (5종)

### Data Drift (입력 분포 변화)

- 원인: 계절, 시간대, 고객층, 조명 환경 변화
- 감지: teacher 출력 분포의 일별 mean/std 추적
- 대응: replay buffer coverage 비율 자동 조정, 재학습

### Concept Drift (기준 변화) — 의도적 활용

- 원인: 디자인팀 요구사항 변경, 시즌별 수집 기준 변경
- 대응: 새 버킷 추가 (few-shot) 또는 기존 threshold 변경 → pseudo-label 자동 갱신
- **방어가 아니라 핵심 기능** — 빠른 기준 변경이 시스템의 가치

```
시즌 1: Π₁ = {warm_smile, cool_expression, lateral}
시즌 2: + eyes_closed 버킷 추가 (50건 few-shot)
시즌 3: warm_smile의 threshold 변경 → 개념 재정의 → 자동 재학습
```

### Teacher Drift (모듈 교체)

- 원인: upstream 모듈 버전 업그레이드
- 감지: teacher 모듈 버전을 학습 데이터에 태깅
- 대응: 버전 변경 감지 시 해당 모듈의 기존 데이터 무효화 + 재학습

### Preprocessing Drift (전처리 변경)

- 원인: face detection threshold/NMS 파라미터 변경, crop 로직 수정
- 위험성: 개별 teacher 모델 변경 없이 모든 face-crop 기반 모듈 출력이 동시 변화
- 감지: face crop 통계(종횡비, crop 크기, confidence 분포) 독립 모니터링

### Self-Reinforcing Drift — 구조적 차단

가장 위험한 drift. Student의 편향이 자기 학습을 오염시키는 루프.

**핵심 원칙:**
> Student는 pseudo-label 생성에 참여하지 않는다.
> Teacher(frozen)만이 유일한 pseudo-label 소스다.

```
Teacher (frozen) → vote → pseudo-label → strategy 재학습
                                              ↓
                                         버킷 판단 (운용)
                                              ✗ pseudo-label로 되돌아가지 않음
```

---

## 안전장치

### 핵심 모니터링: 주간 Human Agreement Check

> **주 1회, Anchor Set에서 100건 샘플링 → strategy 판단 vs 사람 판단 일치율 측정**

이것이 떨어지면 모든 것이 문제. 이것이 유지되면 나머지는 부차적.

### Rollback 기준

```
[자동 롤백]
- 주간 human agreement가 이전 주 대비 5%p 이상 하락 → 이전 모델로 롤백

[경보 + 수동 조사]
- 주간 human agreement가 이전 주 대비 2%p 이상 하락 → 알림
- Tier 3 비율 전체의 30% 초과

[학습 중단]
- 재학습 모델이 이전 모델 대비 Anchor Set에서 유의미하게 열위
```

### Blind Spot Registry

Tier 3 (합의 실패) 프레임을 별도 저장.
모든 teacher가 불확실한 영역 = 시스템의 맹점.
주기적 분석으로 새 모듈 추가나 threshold 조정의 근거로 활용.

### Replay Buffer (Level 1+)

학습 시 데이터 편향 방지:

```
50% = 최근 7일 데이터 (recency)
30% = 전체 버퍼에서 조건별 균등 추출 (coverage)
10% = Tier 2 (경계선) 샘플 (hard negatives)
10% = 고신뢰 합의 샘플 (anchor, 안정성 유지)
```

---

## 패키지 구조 (계획)

```
libs/visualgrow/src/visualgrow/
├── collector.py      # 매일 signal 수집 → parquet
├── pseudo_label.py   # crowds consensus → labels (majority vote / Dawid-Skene)
├── scheduler.py      # 재학습 스케줄 관리
├── monitor.py        # Anchor Set 대비 성능 추적, drift 감지
├── drift.py          # 5종 drift 감지 로직
├── replay.py         # replay buffer 관리
└── cli.py            # visualgrow collect / retrain / monitor
```

### CLI (계획)

```bash
# 일별 signal 수집
visualgrow collect --input ./daily_signals/ --output ./buffer/

# pseudo-label 생성 + 재학습
visualgrow retrain --buffer ./buffer/ --strategy xgboost --output ./models/

# Anchor Set 대비 모니터링
visualgrow monitor --model ./models/latest.pkl --anchor ./anchor_set/
```

### 학습 빈도 (단계적 전환)

```
초기:     수동 트리거 — visualgrow retrain CLI로 필요 시 실행
검증 후:  주간 배치 — 주 1회 재학습
안정화 후: 야간 배치 — 매일 재학습
```

---

## 궁극적 비전: 도메인 특화 모델 생산

visualgrow의 최종 목표는 strategy 재학습에 머무르지 않는다.
**14개 frozen 모델의 지식을 통합한 도메인 특화 모델을 자율적으로 생산**하는 것이다.

### 자기 강화 루프

```
시작:    14개 외부 frozen 모델 → visualpath에서 실행
              ↓
         visualgrow가 매일 데이터 + crowds consensus로 학습
              ↓
생산:    도메인 특화 모델 생산 (Student)
              ↓
장착:    생산된 모델이 visualpath의 frozen 모델을 대체
              ↓
강화:    더 나은 teacher → 더 나은 pseudo-label → 더 나은 Student
              ↓
성숙:    단일 도메인 모델이 14개 모듈 전체를 대체
         visualpath 퇴역
```

### 생산 가능한 모델

**1. 얼굴 상태 임베딩 모델**

14개 teacher의 지식이 합쳐진 feature space.
표정, AU, 포즈, 품질, 감정을 모두 인코딩한 단일 벡터.

```
Teacher 14개 (각각 부분적 관측)
  → crowds supervision → Student 학습
  → Student의 중간 feature = 얼굴 상태의 rich embedding
  → face retrieval: "이 표정과 비슷한 프레임 찾기"
  → face clustering: 비슷한 상태끼리 자동 그룹핑
```

**2. 도메인 특화 얼굴 검출기**

범용 InsightFace SCRFD는 우리 환경(차량 내부, 특정 카메라, 조명)에서 miss 발생.
매일 데이터에서 "face.detect + face.parse + face.quality" crowds 합의로
이 환경에 특화된 검출기를 자동 학습.

```
학습 데이터: 매일 10M+ 프레임에서 자동 생성
→ 범용 모델보다 우리 도메인에서 정확
→ fine-tuning 없이 crowds supervision만으로 달성
```

**3. 야외 강인 얼굴 인식기**

981파크의 데이터는 야외/차량 환경이라 조명 변화, 역광, 그림자, 진동이 극심.
이 환경에서 매일 축적되는 데이터로 학습된 얼굴 인식기는
**야외 환경에 강인한 범용 모델**로도 가치가 있다.

```
실내 학습 모델 (ArcFace 등):
  → 실내 데이터로 학습 → 야외 성능 저하 (domain gap)

우리 데이터:
  → 야외/차량 환경 매일 3000건 × 365일
  → 자연스러운 조명 변화, 다양한 고객
  → 이 환경에서 crowds supervision으로 학습된 모델
  → 야외 face recognition benchmark에서 경쟁력 있을 수 있음
```

이것은 981파크 내부 도구를 넘어서 **범용적으로 활용될 수 있는 자산**이다.

**4. 통합 모델 (궁극)**

검출 + 인식 + 표정 + 포즈 + 품질 → 하나의 모델.
Florence-2가 대규모 인프라로 달성한 것을 도메인 스케일로.

```
Florence-2:  수백 개 모델 → FLD-5B (54억 annotation) → unified model (대규모 compute)
visualgrow:  14개 모델 → crowds pseudo-label (일 10M+) → domain Student (GPU 1개)
```

### visualpath와의 관계 변화

```
현재:    visualpath = 14개 frozen 모델 실행 (inference 필수)
         visualbind = 판단 (strategy)
         visualgrow = 적응 (재학습)

성숙:    visualpath = annotation 공장 (학습 시에만)
         visualgrow = 도메인 모델 생산
         Student = inference 전담 (14개 모듈 → 단일 모델)

궁극:    visualpath = 퇴역 (새 도메인 bootstrapping 용도만)
         visualgrow가 생산한 모델이 시스템의 핵심
```

---

## 설계 문서

| 문서 | 범위 |
|------|------|
| `why-visualbind.md` | 이론적 기반 (crowds, Projected Crowds, Berkson's paradox) |
| `how-visualbind.md` | 정적 판단 (strategies, selector, CLI) |
| `how-visualgrow.md` | 동적 적응 (이 문서) |
| `projected-crowds.md` | Projected Crowds 상세 이론 |
