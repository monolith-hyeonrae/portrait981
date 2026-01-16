# Development and Deployment

## 문서의 목적

이 문서는 개발/실행/배포 방식의 기본 합의를 기록한다.

---

## 패키지 구조

- p981.core: 비즈니스 로직
- p981.runtime: API/Job/Worker 등 실행 레이어

---

## 개발 실행 예시

- core 단독 실행(개발/검증)
  - `uv run python -m p981.core.cli --mode dev --observer log --observer frames discover --video-ref ...`
  - `uv run python -m p981.core.cli --mode dev --observer opencv discover --video-ref ...`
  - `uv run python -m p981.runtime.cli run`
- runtime API
  - `uv run uvicorn p981.runtime.api.main:app --host 0.0.0.0 --port 8000`
- worker
  - `uv run python -m p981.runtime.jobs.worker`

---

## 패키지 관리

- Python 패키지 관리는 uv를 사용한다.

---

## 배포 방향

- Docker 기반 배포를 기본으로 고려한다.
- 컨테이너 구성/오케스트레이션 상세는 추후 확정한다.
