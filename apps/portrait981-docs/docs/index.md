# Portrait981

> 981파크 고객 경험 AI — 어트랙션 영상에서 고객의 순간을 포착하고, 이해하고, 창작한다.

---

## 이 프로젝트는 무엇인가

탑승 중 고객의 얼굴에는 속도의 흥분, 코너를 돌 때의 긴장, 동승자와 눈이 마주치는 순간의 웃음 등
짧지만 강렬한 순간들이 새겨진다. 고객 스스로도 인식하지 못한 채 지나가는, 그러나 가장 생생한 체험의 흔적.

portrait981은 이 순간들을 AI가 자동으로 감지·수집하고, 탑승 맥락과 결합하여
고객마다 고유하게 재해석된 콘텐츠를 생성하는 시스템이다.

> 981파크의 시선으로 포착하고, 981파크의 감각으로 재구성한
> **"파크가 고객에게 선물하는 또 다른 자신의 모습"**

```
영상 → 분석 → 저장 → 생성
       momentscan   personmemory   reportrait
```

단순한 사진 판매가 아니라, **매 방문마다 고객을 더 깊이 이해하는 AI 시스템**.

---

## 핵심 관점

### "사진을 찍는 것"이 아니라 "사람을 이해하는 것"

기존 테마파크 사진 서비스 (디즈니 PhotoPass 등):

> 카메라 → 사진 → 판매 → 끝

Portrait981:

> 카메라 → 14개 AI 모델로 분석 → 개인별 표현 분포 축적 → 재방문할수록 정교해지는 이해

같은 "미소"도 사람마다 다르다.
A씨는 눈으로 웃고, B씨는 입으로 웃는다.
이 차이를 **수학적으로 표현하고, 시간에 걸쳐 축적**하는 것이 portrait981의 핵심.

→ [Person-Conditioned Distribution](research/person-conditioned-distribution.md)

---

## 기술 스택

### visual* 생태계 — 범용 프레임워크

portrait981의 모든 기술은 **범용 프레임워크** 위에 구축.
portrait981은 이 프레임워크의 첫 번째 응용 프로그램.

```
visualbase   — Read    감각, 미디어 I/O
visualpath   — Process 경로, 분석 파이프라인 (FlowGraph + DAG)
vpx          — Measure 측정, 개별 비전 분석 모듈 (14개)
visualbind   — Bind    결합, 다중 관측 → 통합 판단
visualgrow   — Grow    성장, 매일 데이터로 적응 (계획)
```

→ [파이프라인 3-Layer 아키텍처](architecture/pipeline-architecture.md)
→ [visual* 생태계 리뷰](architecture/visual-ecosystem-review.md)

### 3-App 파이프라인 — 981파크 특화

| 앱 | 역할 | 핵심 |
|----|------|------|
| **momentscan** | 분석 | 영상 → 프레임별 표정/포즈/품질 판단 |
| **personmemory** | 저장 | 고객별 기억 — identity, 프레임, signal 분포 |
| **reportrait** | 생성 | 참조 이미지 → ComfyUI → AI 초상화 |

```
momentscan → personmemory → reportrait
  분석         기억         창작
```

→ [MomentScan v2](apps/momentscan-v2.md) |
  [MomentBank](apps/personmemory.md) |
  [Reportrait](apps/reportrait.md)

---

## 연구 방향

### Person-Conditioned Distribution (핵심 연구)

기존 AI 모델은 얼굴의 **일관성**(같은 사람 = 같은 벡터)만 학습.
portrait981은 같은 사람 안에서의 **다양성**을 표현해야 한다.

- 개인별 표현 분포 (μ, Σ) 축적
- 새 프레임의 다양성 가치 = Mahalanobis distance 기반
- 테마파크 어트랙션 = 자연스러운 다양한 표정을 반복 관찰할 수 있는 세계 유일의 환경
- visual* frozen specialist crowds → annotation-free dataset 자동 구축

→ [전체 설계 문서](research/person-conditioned-distribution.md)

### VisualBind — 다중 관측 결합

14개 frozen 비전 모델의 출력을 crowds consensus로 결합하여
단일 판단(expression, pose, quality gate)을 내리는 시스템.

→ [왜 VisualBind인가](architecture/why-visualbind.md) |
  [설계와 구현](architecture/how-visualbind.md)

### Face State Embedding — 장기 비전

identity + expression + pose + quality를 하나의 모델이 담는 통합 임베딩.
14개 frozen specialist의 crowds consensus → Student 모델 학습.

→ [Face State Embedding](research/vision-face-state-embedding.md)

---

## 사업적 맥락

981파크는 테마파크 프랜차이즈화를 준비 중.
지점이 늘수록 데이터 자산이 **초선형**으로 성장하는 구조.

- 기존 매출 (티켓): 선형 성장
- 데이터 자산: 네트워크 효과 + 복리 축적
- 경쟁 해자: 기술 + 인프라 + 관점 — 빅테크도 쉽게 따라할 수 없는 사업 구조

→ [프랜차이즈와 데이터 해자](business/data-moat.md)

---

## 패키지 구조

```
portrait981/                     ← 모노레포 루트
├── libs/
│   ├── visualbase/              # 미디어 I/O (범용)
│   ├── visualpath/              # 분석 프레임워크 (범용)
│   │   ├── core/                #   FlowGraph, App, Backend
│   │   ├── isolation/           #   Worker 프로세스 격리
│   │   ├── cli/                 #   CLI 도구
│   │   └── pathway/             #   Pathway 스트리밍 백엔드
│   ├── vpx/                     # 비전 분석 모듈 (범용)
│   │   ├── sdk/                 #   Module, Observation 타입
│   │   ├── runner/              #   vpx CLI
│   │   ├── viz/                 #   시각화
│   │   └── plugins/ (7개)       #   face-detect, expression, au, parse, head-pose, body-pose, hand-gesture
│   └── visualbind/              # 다중 관측 결합 (XGBoost/Catalog/Heuristic)
├── apps/
│   ├── momentscan/              # 분석 앱 (v1 legacy + v2 simplified)
│   ├── momentscan-plugins/ (7개) # 내부 analyzer 플러그인
│   ├── personmemory/              # 고객 기억 (identity + frames + signals)
│   ├── reportrait/              # AI 초상화 생성 (ComfyUI)
│   ├── portrait981/             # 통합 오케스트레이터
│   ├── portrait981-serve/       # REST API 서빙
│   ├── portrait981-docs/        # 이 문서 사이트
│   └── annotator/               # 라벨링/리뷰 도구
├── data/                        # 데이터셋, 카탈로그
├── models/                      # 학습된 모델 (.pkl)
└── scripts/                     # 실험 스크립트
```

---

## 시작하기

```bash
# 전체 빌드
cd /home/hyeonrae/repo/monolith/portrait981
uv sync --all-packages --all-extras

# 문서 사이트 로컬 실행
uv run --package portrait981-docs --extra serve mkdocs serve

# 주요 테스트
uv run pytest apps/momentscan/tests/ -v      # momentscan (558)
uv run pytest apps/personmemory/tests/ -v      # personmemory (61)
uv run pytest apps/reportrait/tests/ -v      # reportrait (41)
uv run pytest libs/visualbind/tests/ -v      # visualbind (58)
```

---

*최종 업데이트: 2026-03-25*
