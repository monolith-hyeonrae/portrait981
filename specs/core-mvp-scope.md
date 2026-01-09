# MVP Scope (Core)

## 결정 사항

- Discover는 실작동으로 moment/clip/keyframe까지 저장한다.
- Discover는 상태 타임라인과 Moment 선정까지 포함한다.
- Synthesize는 base/closeup/fullbody/cinematic 요청을 지원한다.
  - cinematic은 closeup + fullbody 선행 생성에 의존한다.
  - 필요한 부분은 코드로 모듈화하고, 부족한 단계는 ComfyUI 백엔드를 사용한다.

---

## 진행 방식

- 전체 실행 경로는 스켈레톤으로 동작해야 한다.
- 도메인 모듈은 stub부터 시작해 실제 구현으로 순차 교체한다.
- stub은 Stage 입출력 계약을 준수하며 asset 저장/조회까지 보장한다.

---

## 목표

요청 → 실행 → asset 적재까지의 end-to-end 흐름을 검증한다.
