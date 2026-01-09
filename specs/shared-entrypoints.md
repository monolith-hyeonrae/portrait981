# Entry Points and Priority

## 문서의 목적

이 문서는 실행 경로(Entry Points)와 MVP 단계별 우선순위를 정리한다.

---

## 우선순위 (MVP 단계)

1. MVP 1: CLI + Core 단독 실행 (Job 없음)
2. MVP 2: REST API + JobStore + Worker
3. MVP 3: Event Watcher (S3 등) → Discover 자동 트리거

---

## 실행 트리거

- Discover: CLI 또는 영상 업로드 이벤트(Watcher)
- Synthesize: 사용자/시스템 요청 (API/CLI)

---

## 원칙

개발 속도와 운영 연결성을 함께 고려한다.
