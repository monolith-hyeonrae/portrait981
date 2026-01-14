# Portrait981

## 문서

- `AGENTS.md`: 작업 원칙/경계
- `specs/ARCH.md`: 프로젝트 개요

## 실행 (p981-core)

```bash
uv run python -m p981.core.launcher --mode dev --observer log discover --video-ref ./sample.mp4 --progress
```

```bash
uv run python -m p981.core.launcher --mode dev --observer log synthesize --style base --moment-ref <moment_ref> --progress
```
