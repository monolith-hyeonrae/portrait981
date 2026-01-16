# Runtime Protocols

## 문서의 목적

이 문서는 runtime 전용 프로토콜(Job/Queue/Notification 등)을 정의한다.

---

## 프로토콜 정의 (Runtime)

- JobStore: Job 상태/결과 저장
- JobQueue: Job enqueue/dequeue (멀티프로세스/멀티호스트 지원)
- Notification: 완료/실패 알림 발송 (선택)
- Metrics: 메트릭 수집/발행 (선택)

---

## 백엔드 방향 (초기)

- JobStore/JobQueue: in-memory (dev) → Redis/Postgres/Message Queue
- Notification: Webhook (dev) → Webhook/이메일/슬랙
- Metrics: stdout (dev) → Prometheus/OTel

---

## 원칙

- Core는 runtime 프로토콜에 의존하지 않는다.
- runtime 프로토콜은 운영 환경 요구에 따라 교체 가능해야 한다.
- 프로토콜 목록과 세부 인터페이스는 추후 확정하며 스켈레톤 단계에서는 최소 프로토콜로 시작한다.
