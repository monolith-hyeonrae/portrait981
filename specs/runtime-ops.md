# Runtime Operations

## 문서의 목적

이 문서는 Runtime 레이어의 실행/운영 책임과 구성 요소를 정의한다.

---

## 구성 요소

- API: Stage 실행 요청 및 상태 조회
- Job Store: Job 생성 및 상태 저장
- Scheduler: 우선순위/동시성 기반 작업 할당
- Worker Pool: Stage 실행
- Watcher: 이벤트 기반 Discover 트리거
- Retry Handler: 재시도 정책 적용
- DLQ + DLQ Monitor: 실패 Job 보관 및 알림
- Notification Service: 완료/실패 알림
- Monitoring/Logging: 메트릭 및 로그 수집

---

## 실행 모델

- Job Store/Queue는 멀티프로세스/멀티호스트에서 공유 가능해야 한다.
- in-memory Job Store는 로컬 개발용으로만 사용한다.
- Worker Pool은 멀티프로세스 기준으로 설계한다.
- CPU/GPU 풀을 분리하고 resource_class(cpu/gpu)는 runtime 정책으로 결정한다.
- resource_class는 서버 상태에 따라 동적으로 설정될 수 있으며 기본값은 GPU일 수 있다.

---

## Job 상태

- PENDING: 대기 중
- RUNNING: 실행 중
- SUCCEEDED: 완료
- FAILED: 실패
- CANCELED: 취소

---

## Scheduler (MVP)

- 우선순위 기반 작업 할당
- 동시 실행 수 제어 (고객별, 전체)
- resource_class에 따라 CPU/GPU 풀로 라우팅한다.
- 기본값은 config로 정의하고 환경변수로 변경 가능해야 한다.
  - discover_concurrency (예: 4)
  - synthesize_concurrency (예: 1~2)

---

## Retry / DLQ (MVP)

- 일시적 실패 시 자동 재시도
- 지수 백오프(Exponential Backoff) + jitter
- 기본값: 최대 3회 재시도 (예: 5s, 30s, 120s)
- retryable vs non-retryable 분류는 Runtime 책임

---

## Watcher (MVP3)

- Object Storage 이벤트(ObjectCreated) 감지
- 이벤트 → Discover Job 생성

---

## 알림

- Job 완료/실패 시 알림 전송
- 기본 채널: Webhook (Slack Webhook 포함)
- 폭주 방지를 위해 DLQ 알림은 집계 후 발송

---

## 모니터링/로깅

- 메트릭: 실행 시간(스테이지/모듈), 성공/실패율, 재시도 빈도, 리소스 사용량, API 응답 시간
- 로그 레벨: INFO/WARN/ERROR/DEBUG
- 관찰성 스택은 구현 전에 확정한다.
