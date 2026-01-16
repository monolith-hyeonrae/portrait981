# Entry Points and Priority

## 문서의 목적

이 문서는 실행 경로(Entry Points)와 MVP 단계별 우선순위를 정리한다.

---

## 우선순위 (MVP 단계)

1. MVP 1: CLI + Core 단독 실행 (Job 없음)
2. MVP 2: REST API + JobStore + Worker
3. MVP 3: Event Watcher (S3 등) → Discover 자동 트리거

- 위 순서는 개발 단계 우선순위이며, 런타임 요청 경합 우선순위 규칙은 별도 정의한다.

---

## 실행 트리거

- Discover: CLI 또는 영상 업로드 이벤트(Watcher)
- Synthesize: 사용자/시스템 요청 (API/CLI)

---

## CLI 역할 분리

- Core CLI (`p981.core.cli`): 엔진 스테이지를 단독으로 실행하는 디버그/검증용 진입점.
- Runtime CLI (`p981.runtime.cli`): 서비스 실행, 워커 구동 등 운영 환경 진입점.

---

## 원칙

개발 속도와 운영 연결성을 함께 고려한다.
