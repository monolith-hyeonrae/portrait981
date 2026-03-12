# Portrait981-Serve

portrait981 파이프라인의 서빙 레이어. REST API + S3 + ComfyUI 노드풀.

## 디렉토리 구조

```
src/portrait981_serve/
├── __init__.py
├── app.py           # FastAPI 앱 팩토리 + entry point
├── config.py        # ServeConfig (환경변수 → dataclass)
├── routes.py        # REST 엔드포인트 4개
├── s3.py            # S3 다운로드/업로드
├── schemas.py       # Pydantic request/response 모델
└── node_pool.py     # ComfyUI 노드풀 (라운드로빈 + 장애 cooldown)
```

## API 엔드포인트

| 경로 | 메서드 | 용도 |
|------|--------|------|
| `/portrait/scan` | POST | scan + bank 저장 |
| `/portrait/generate` | POST | bank에서 참조 → 생성 |
| `/portrait/test` | POST | 파이프라인 검증 (bank 비저장) |
| `/portrait/status/{member_id}` | GET | 프레임 현황 조회 |
| `/health` | GET | 헬스체크 |

## 설정 (환경변수)

`P981_*` prefix. `config.py` 참조.

## 실행

```bash
p981-serve                                    # uvicorn on 0.0.0.0:8000
P981_COMFY_URLS=http://gpu1:8188,http://gpu2:8188 p981-serve
```

## 의존성

- `portrait981` — 파이프라인 오케스트레이터
- `fastapi` + `uvicorn` — REST 서버
- `boto3` — S3 클라이언트

## 테스트

```bash
uv run pytest apps/portrait981-serve/tests/ -v
```
