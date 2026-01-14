# ARCH.md
Portrait981 Backend Architecture

## 문서의 목적

이 문서는 `portrait981` 프로젝트가 **무엇을 만드는 시스템인지**를 설명한다.

- 본 문서는 아키텍처의 현재 합의 상태를 기록한다.
- 구현 상세나 알고리즘은 이 문서의 범위가 아니다.
- 구조적 변경은 대화를 통해 합의되며,
  합의된 내용은 사용자의 허락 하에 문서로 업데이트된다.

본 문서는 `AGENTS.md`에서 정의한 행동 원칙을 전제로 한다.

---

## 시스템 개요

Portrait981은  
**고객 비디오를 분석하여 의미 있는 순간(moment)을 누적하고,  
요청 시 해당 moment를 기반으로 시네마틱 포트레이트 결과물을 생성하는 시스템**이다.

시스템의 핵심 목적은 다음과 같다.

1. 고객 비디오로부터 의미 있는 moment를 지속적으로 발견하고 축적
2. 축적된 moment를 기반으로 다양한 스타일의 결과물을 온디맨드로 생성

---

## 핵심 정의 요약

- moment: 의미 있는 변화/피크가 나타나는 시간 구간(time window) + 대표 프레임
- 결과물: Asset Bundle 기반 결과물 (MVP 필수는 cinematic mp4 1개)
- 입력 경로 우선순위(개발 단계): CLI → REST/Job → Event Watcher
- Asset 최소 스키마: asset_ref/asset_type/member_id(optional)/created_at/source/blob_ref/meta
- MVP 범위: Discover 실작동 + Synthesize cinematic 산출

## 상세 문서

- moment 정의/선정: `specs/core-moment.md`
- 결과물 및 스타일: `specs/core-output.md`
- 실행 경로 우선순위: `specs/contracts-entrypoints.md`
- Asset 최소 스키마: `specs/core-asset-schema.md`
- Core MVP 범위: `specs/core-mvp-scope.md`
- 도메인 모듈 책임: `specs/core-domain-modules.md`
- Stage 계약: `specs/core-stage-contracts.md`
- Runtime API: `specs/runtime-api.md`
- Runtime 운영: `specs/runtime-ops.md`
- 상태 타임라인: `specs/core-state-timeline.md`
- 공통 규칙: `specs/contracts-common-rules.md`
- Synthesis 백엔드: `specs/core-synthesis-backend.md`
- 포트/어댑터: `specs/contracts-ports-and-adapters.md`
- Runtime 포트: `specs/runtime-ports.md`
- 개발/배포: `specs/contracts-dev-and-deploy.md`

---

## 아키텍처의 큰 축

시스템은 두 개의 독립적인 축으로 구성된다.

### p981-core

- 비즈니스 로직의 중심
- Stage 정의와 Domain 로직을 포함
- 외부 시스템에 직접 의존하지 않음
- Mock Adapter를 통해 단독 실행 가능

### p981-runtime

- 실행과 운영을 책임지는 레이어
- 외부 요청을 Job으로 변환
- 재시도, DLQ, 모니터링, 알림을 통해 실행을 보장
- Core를 호출하여 동일한 비즈니스 로직을 실행

---

## Core 개념 구성

Core는 다음 개념으로 구성된다.

- Stage  
  작업의 구성과 순서를 정의

- Domain  
  실제 처리 로직을 담당

- Executor  
  Stage 실행의 단일 진입점

- Ports  
  외부 시스템과의 추상 인터페이스

---

## Runtime 개념 구성

Runtime은 다음을 책임진다.

- 외부 요청 수신
- Job 상태 관리
- 실패 처리 및 재시도
- 운영 가시성 제공

Runtime은 비즈니스 판단을 하지 않는다.

---

## 데이터 및 원본 비디오 원칙

- 원본 비디오는 일시적인 입력이다.
- 원본 비디오는 7일 뒤 삭제되며 이후 단계는 원본에 의존하지 않는다.
- 분석 이후 필요한 정보만 Asset 형태로 저장된다.

---

## 아키텍처 다이어그램

아키텍처 구조는 Mermaid 다이어그램으로 관리한다.

- Core 중심 구조: `specs/diagrams/core.mmd`
- Runtime 중심 구조: `specs/diagrams/runtime.mmd`
- 전체 개요 구조: `specs/diagrams/overview.mmd`

다이어그램은 구조 변경 시 반드시 함께 업데이트되어야 한다.

---

## 문서의 진화에 대한 합의

- 본 문서는 완성 문서가 아니다.
- 문서의 모호함은 대화를 통해 해소한다.
- 합의된 내용은 사용자의 허락 하에 문서에 반영한다.
- 문서를 기준으로 구현을 진행한다.

---

## 요약

- Portrait981은 비디오 기반 moment 분석 및 생성 시스템이다.
- Core는 비즈니스 판단을 담당한다.
- Runtime은 실행을 보장한다.
- 구조의 정본은 diagrams에 있다.
