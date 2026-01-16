# Portrait981

## 문서

- `AGENTS.md`: 작업 원칙/경계
- `specs/ARCH.md`: 프로젝트 개요

## 실행 (p981-core)

- `p981.core.cli`는 엔진 스테이지를 로컬에서 빠르게 확인하는 디버그용 CLI다.
- 서비스용 실행 진입점은 `p981.runtime.cli`로 분리한다.

```bash
uv run python -m p981.core.cli --mode dev --observer log discover --video-ref ./sample.mp4 --progress
```

```bash
uv run python -m p981.core.cli --mode dev --observer log synthesize --style base --moment-ref <moment_ref> --progress

```bash
uv run python -m p981.runtime.cli run
```
```
