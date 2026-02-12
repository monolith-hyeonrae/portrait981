# Portrait981 Documentation

> 최종 업데이트: 2026-02-10

## 아키텍처 & 설계

| 문서 | 설명 |
|------|------|
| [architecture.md](architecture.md) | visualpath 3계층 아키텍처, 플러그인 시스템, Worker 격리, 실행 백엔드 |
| [stream-synchronization.md](stream-synchronization.md) | 분산 실행 스트림 동기화, Pathway 백엔드, Observability 연동 |
| [isolation.md](isolation.md) | ML 의존성 충돌 사례 (onnxruntime GPU/CPU, CUDA 런타임) 및 프로세스 격리 해결 |

## 도메인 & 알고리즘

| 문서 | 설명 |
|------|------|
| [algorithms.md](algorithms.md) | momentscan 하이라이트 감지 알고리즘 (표정 급변, 품질 게이트, 프레임 스코어링 등 9종) |

## 프로젝트 관리

| 문서 | 설명 |
|------|------|
| [roadmap.md](roadmap.md) | Phase 1~16 개발 로드맵, 패키지 정의, 테스트 현황 |

## 기획 & 히스토리

| 문서 | 설명 |
|------|------|
| [planning/plan0.md](planning/plan0.md) | 초기 기획 |
| [planning/plan1.md](planning/plan1.md) | 추가 기획 |
| [planning/why-visualpath.md](planning/why-visualpath.md) | 단순 루프에서 플랫폼이 필요해지는 과정 (동기 문서) |
| [planning/phase-11-summary.md](planning/phase-11-summary.md) | Phase 11 의존성 분리 구조 요약 |

## 패키지별 문서 (CLAUDE.md)

각 패키지의 상세 정보는 패키지 디렉토리의 `CLAUDE.md`를 참조:

| 패키지 | CLAUDE.md |
|--------|-----------|
| portrait981 (root) | [CLAUDE.md](../CLAUDE.md) |
| visualpath core | [libs/visualpath/core/CLAUDE.md](../libs/visualpath/core/CLAUDE.md) |
| momentscan | [apps/momentscan/CLAUDE.md](../apps/momentscan/CLAUDE.md) |
