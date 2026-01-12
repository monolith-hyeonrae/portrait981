# Runtime Ports

## 문서의 목적

이 문서는 runtime 전용 포트(Job/Queue/Notification 등)를 정의한다.

---

## 포트 정의 (Runtime)

- JobStore: Job 상태/결과 저장
- JobQueue: Job enqueue/dequeue (멀티프로세스/멀티호스트 지원)
- Notification: 완료/실패 알림 발송 (선택)
- Metrics: 메트릭 수집/발행 (선택)

---

## 어댑터 방향 (초기)

- JobStore/JobQueue: in-memory (dev) → Redis/Postgres/Message Queue
- Notification: Webhook (dev) → Webhook/이메일/슬랙
- Metrics: stdout (dev) → Prometheus/OTel

---

## 원칙

- Core는 runtime 포트에 의존하지 않는다.
- runtime 포트는 운영 환경 요구에 따라 교체 가능해야 한다.
- 포트 목록과 세부 인터페이스는 추후 확정하며 스켈레톤 단계에서는 최소 포트로 시작한다.
