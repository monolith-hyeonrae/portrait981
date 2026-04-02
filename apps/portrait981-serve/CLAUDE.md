# Portrait981-Serve

portrait981 파이프라인의 서빙 레이어. REST API + S3 + ComfyUI 노드풀.
GPU 노드에서 warm scanner를 유지하며 HTTP 요청을 처리.

## 디렉토리 구조

```
src/portrait981_serve/
├── __init__.py
├── app.py           # FastAPI 앱 팩토리 + uvicorn entry point
├── config.py        # ServeConfig (P981_* 환경변수 → dataclass)
├── routes.py        # REST 엔드포인트 (scan/generate/test/status)
├── s3.py            # S3 다운로드(비디오) / 업로드(생성 결과)
└── schemas.py       # Pydantic request/response 모델
```

## API 엔드포인트

| 경로 | 메서드 | 용도 |
|------|--------|------|
| `POST /portrait/scan` | scan + personmemory 저장 |
| `POST /portrait/generate` | personmemory lookup → ComfyUI 생성 → S3 업로드 |
| `POST /portrait/test` | 파이프라인 검증 (personmemory 미저장) |
| `GET /portrait/status/{member_id}` | personmemory 프레임 현황 조회 |
| `GET /health` | 헬스체크 |

## 요청/응답 흐름

```
POST /portrait/scan { member_id, workflow_id, video_uri }
  1. S3에서 비디오 다운로드 (video_uri)
  2. Portrait981Pipeline.run_one(scan_only=True)
     → Momentscan.scan() (warm scanner 재사용)
  3. personmemory에 SHOOT 프레임 저장
  4. 임시 비디오 파일 정리
  → ScanResponse { status, frame_count, shoot_count, timing_sec }

POST /portrait/generate { member_id, generate: { workflow, prompt } }
  1. personmemory.lookup_frames(member_id)
  2. PortraitGenerator.generate() → ComfyUI
  3. 결과 이미지 S3 업로드
  → GenerateResponse { status, output_urls, ref_count, timing_sec }
```

## 설정 (환경변수)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `P981_COMFY_URLS` | `http://127.0.0.1:8188` | ComfyUI 서버 (쉼표 구분) |
| `P981_COMFY_API_KEY` | - | ComfyUI API 키 |
| `P981_SCAN_FPS` | `10` | scan FPS |
| `P981_SCAN_BACKEND` | `simple` | visualpath 백엔드 |
| `P981_S3_OUTPUT_BUCKET` | - | 생성 결과 업로드 버킷 |
| `P981_S3_REGION` | `ap-northeast-2` | AWS 리전 |
| `P981_DEFAULT_WORKFLOW` | `default` | 기본 ComfyUI 워크플로우 |

## 실행

```bash
p981-serve                                    # uvicorn on 0.0.0.0:8000
P981_COMFY_URLS=http://gpu1:8188 p981-serve   # GPU 노드 지정
```

## 배포 구조 (향후)

```
실험 에이전트 / 회사 시스템
     │ HTTP
     ▼
portrait981-serve (GPU 노드, warm scanner)
     │
     ├── S3 (비디오 다운로드, 결과 업로드)
     └── ComfyUI (이미지 생성)

운영 확장 시:
회사 브로커(Kafka/SQS)
     │
     ▼
portrait981-gateway (CPU, 수평확장, adapter)
     │ HTTP
     ▼
portrait981-serve (GPU 노드)
```

gateway는 serve와 분리 — serve는 REST API만, gateway는 브로커 프로토콜 adapter.

## 업데이트 필요 (TODO)

- [ ] scan 응답 v2 반영 (highlight_count → shoot_count, list[FrameResult] 기반)
- [ ] `momentbank` → `personmemory` import 수정 (status 엔드포인트)
- [ ] scan에 ingest 옵션 추가
- [ ] warm scanner 직접 활용 (현재 pipeline 경유)

## 의존성

- `portrait981` — 파이프라인 오케스트레이터
- `fastapi` + `uvicorn` — REST 서버
- `boto3` — S3 클라이언트

## 테스트

```bash
uv run pytest apps/portrait981-serve/tests/ -v
```
