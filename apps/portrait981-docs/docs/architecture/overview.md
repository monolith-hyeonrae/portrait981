# 아키텍처 개요

## 두 개의 레이어

Portrait981은 **범용 프레임워크**와 **도메인 특화 앱**으로 구성된다.

```
┌─────────────────────────────────────────────────────────┐
│  범용 레이어 (재사용 가능)                               │
│  visualbase (미디어 I/O) → visualpath (분석 프레임워크) │
│  vpx-sdk (공유 타입/프로토콜) + vpx-* (비전 분석 모듈) │
│  visualbind (다중 관측 결합)                             │
└─────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────┐
│  981파크 특화 레이어                                     │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────┐  │
│  │ momentscan  │→ │ personmemory       │→ │ reportrait│  │
│  │ (분석 앱)   │  │ (고객 기억)      │  │ (AI 생성) │  │
│  └─────────────┘  └──────────────────┘  └───────────┘  │
│                              │                          │
│                   portrait981 (통합 앱)                 │
└─────────────────────────────────────────────────────────┘
```

## visual* 생태계

visual* 라이브러리는 portrait981의 핵심 인프라이자, 독립적으로 재사용 가능한 범용 프레임워크.

| 라이브러리 | 역할 | 비유 |
|-----------|------|------|
| **visualbase** | 미디어 소스 추상화 (파일, 스트림, 카메라) | 눈 (감각) |
| **visualpath** | 분석 파이프라인 프레임워크 (FlowGraph, App, Backend) | 신경 경로 |
| **vpx** | 개별 비전 분석 모듈 14개 (얼굴, 포즈, 제스처 등) | 전문 뉴런 |
| **visualbind** | 다중 모듈 출력 결합 → 통합 판단 | 연합 영역 |
| **visualgrow** | 매일 데이터로 적응하는 학습 시스템 (계획) | 시냅스 성장 |

```
영상 → visualbase (I/O)
     → visualpath (FlowGraph)
       → vpx modules (측정)
     → visualbind (결합 + 판단)
     → visualgrow (적응, 계획)
```

## 3-App 파이프라인

| 단계 | 앱 | 입력 | 출력 |
|------|-----|------|------|
| 분석 | momentscan | 영상 | 프레임별 expression/pose/quality 판단 |
| 저장 | personmemory | 판단 결과 | member별 프레임 + signal 분포 축적 |
| 생성 | reportrait | 참조 이미지 + 프롬프트 | AI 초상화 |

```
momentscan (분석) → personmemory (기억) → reportrait (창작)
```

- **momentscan v1** (legacy): BatchHighlightEngine, temporal 분석
- **momentscan v2** (현재): vp.App + VisualBind per-frame judgment, ~130줄
- **portrait981**: 세 앱을 하나의 호출로 연결하는 통합 오케스트레이터

## 상세 문서

- [파이프라인 3-Layer 아키텍처](pipeline-architecture.md) — visualpath → visualbind → visualgrow
- [visual* 생태계 리뷰](visual-ecosystem-review.md) — 복잡도 전이 지점 분석
- [데이터 아키텍처](data-architecture.md) — labels.csv / predictions.csv / personmemory 분리
