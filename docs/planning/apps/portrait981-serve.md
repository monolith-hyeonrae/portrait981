# portrait981-serve — 서빙 레이어 설계

> 최종 수정: 2026-03-10
> 상태: 사내 인프라 조사 완료, Kafka 연동 미팅 대기

## 개요

상위 서비스와 p981 파이프라인 사이의 시스템 인터페이스.
Kafka 이벤트로 E2E 작업을 수신하고, REST API로 수동/테스트 요청을 처리한다.

```
cju-activity-status-api
    │
    │  [Kafka] activityVideoCreatedEvent
    │  {workflowId, memberId, videoUrl}
    │
    ▼
portrait981-serve (scan 노드, GPU 중급)
    │  S3 fetch → scan (로컬 GPU) → bank 저장
    │
    │  generate 요청 → ComfyUI 노드풀 (GPU 고급, 별도 장비)
    │  결과 수신 → S3 업로드
    │
    │  [Kafka] portraitCreatedEvent
    ▼
상위 서비스
```

### 인터페이스 구분

| 인터페이스 | 대상 | 프로토콜 | 용도 |
|-----------|------|----------|------|
| CLI (`p981`) | 개발자 | 터미널 | 완료 |
| Kafka consumer | 시스템 (이벤트) | Avro + Schema Registry | E2E 자동 처리 |
| REST API | 시스템 (수동) | JSON | scan-only, generate-only, test, 상태 조회 |

사내 패턴: Kafka는 이벤트 드리븐 흐름, REST는 수동/개별 요청 용도
(cju-activity-video-process와 동일한 패턴)

## 사내 인프라 현황

cju-activity-* 코드베이스 조사 결과:

| 항목 | 사내 표준 |
|------|----------|
| 메시징 | Kafka (SASL_SSL + PLAIN) |
| 직렬화 | **Avro** (Confluent Schema Registry) |
| 토픽 네이밍 | `eda-{platform-id}-{domain}-{event}` (camelCase) |
| 프레임워크 | `mommos-event` (Java, Python 버전 없음) |
| 에러 처리 | try-catch + 로깅 (DLQ 미사용) |
| S3 | ap-northeast-2, IAM access key |
| 엣지↔서버 | MQTT (AWS IoT Core) → Kafka 브릿지 |

### 사내 네이밍 규약 적용

참조: [Naming Rules](https://981park.atlassian.net/wiki/spaces/SP/pages/2429255930/Naming+Rules)

platform-id: `cju` (제주파크)

| 항목 | 규약 | portrait981-serve 적용 |
|------|------|----------------------|
| Git repo | `{park}-{도메인}-{기능}` | `cju-portrait981-serve` |
| Kafka 토픽 | `eda-{platform-id}-{domain}-{event}` | `eda-cju-portrait981-portraitCreatedEvent` |
| Avro 객체 | `{event}Avro` | `portraitCreatedEventAvro` |
| Consumer group | `{app-name}-{env}-group` | `cju-portrait981-serve-prd-group` |
| Producer client-id | `{app-name}-{env}-group` | `cju-portrait981-serve-prd-group` |
| API Gateway | `api.{platform-id}.981park.com/{도메인}` | `api.cju.981park.com/portrait981` |
| 로컬 개발 group | `{app-name}-{본인id}-group` | `cju-portrait981-serve-{id}-group` |

### 이벤트 흐름

```
차량 에이전트 (엣지)
    │  MQTT: UPLOAD_FINISH {workflowId, pathMap.s3Video}
    ▼
cju-activity-status-api (서버)
    │  상태 저장 + 다운스트림 이벤트 발행
    │
    ├→ [Kafka] 신규 이벤트 (상위 팀에서 생성)
    │   {workflowId, memberId, videoUrl, ...}
    │
    └→ portrait981-serve 구독
```

### 수신 이벤트 — 필요 필드 (상위 팀에 전달)

상위 팀에서 신규 이벤트를 만들어주기로 함. p981이 필요한 필드:

**필수**

| 필드 | 타입 | 설명 |
|------|------|------|
| `workflowId` | long | 탑승 식별자 |
| `memberId` | long | 고객 식별자 |
| `videoUrl` | String | S3 비디오 경로 |

**선택 (있으면 활용)**

| 필드 | 타입 | 설명 |
|------|------|------|
| `activityType` | String | 어트랙션 종류 (생성 스타일 분기에 활용 가능) |
| `ticketNumber` | long | 응답에 echo back 용도 |
| `priority` | int | 처리 우선순위 (기본: 0, 높을수록 우선, 특별 이벤트/VIP 대응) |
| `workflow` | String | 생성 템플릿 override (경기 기록/파크 이벤트별 스타일) |
| `prompt` | String | 스타일 프롬프트 override |

## S3 비디오 fetch

visualbase에서 담당. `visualbase[s3]` optional extra로 분리.

### fetch 방식

이벤트/요청에 포함된 `videoUrl`을 받아 1건씩 다운로드.
별도 S3 조회 API 불필요 — 항상 URL이 주어진다.

```
Kafka 이벤트 {videoUrl: "s3://..."}  →  visualbase.resolve(videoUrl)  →  로컬 경로
REST 요청   {video_uri: "s3://..."}  →  visualbase.resolve(video_uri) →  로컬 경로
```

### 캐시 정책

| 항목 | 정책 |
|------|------|
| 비디오 크기 | 30~120MB/건 (평균 ~75MB) |
| 캐시 방식 | **처리 완료 후 즉시 삭제** |
| 이유 | 수천 건/일 × 75MB = ~220GB/일, 전량 캐시 비현실적 |
| retry 시 | partial(generate 실패)는 비디오 불필요 (bank에 scan 데이터 보존) |
| 재스캔 필요 시 | S3에서 다시 다운로드 |

### 구현 위치

```python
from visualbase import resolve_source

local_path = resolve_source("s3://981park-raw/.../video.mp4")  # → /tmp/p981_cache/video.mp4
# scan 실행
# 처리 완료 후 로컬 파일 삭제
```

Kafka 미팅 전에 착수 가능.

## 노드 아키텍처

scan(GPU 중급)과 generate(GPU 고급)는 리소스 요구사항이 다르므로 반드시 별도 장비에서 실행.
generate는 대규모 모델이 상시 메모리에 상주해야 하며, GPU 등급에 따라 처리 시간 편차가 크다.

```
portrait981-serve (scan 노드, GPU 중급)
    │  scan ~1분/건
    │
    └→ ComfyUI 노드풀 (generate 노드, GPU 고급)
        generate ~1-3분/건 (고급 GPU 최적화 목표)
```

### GPU 등급별 generate 처리 시간

| GPU 등급 | generate 시간 | 비고 |
|----------|-------------|------|
| 고급 (목표) | ~1-3분 | 최적화 예정, 프로덕션 기준 |
| 중급 | 3-40분 | 편차 큼, 프로덕션 부적합 |

고급 GPU에서 최적화 완료 시 scan:generate ≈ 1:1~3 비율.

### 스케일 비율

```
수천 건/일 기준 (고급 GPU 최적화 후):
    serve (scan) 노드 2~3대
        │
        └→ ComfyUI (generate) 노드 3~5대
```

### 스케일링 방식

**scan 노드**: Kafka consumer group 파티션 분배로 수평 확장

```
Kafka consumer group: portrait981-serve
    ├→ serve 노드 A (GPU 중급) ──→ ComfyUI 노드풀
    ├→ serve 노드 B (GPU 중급) ──→   (공유)
    └→ serve 노드 C (GPU 중급) ──┘
```

**ComfyUI 노드풀**: serve에서 가용 노드에 라우팅
- 사내 패턴: Eureka + REST로 부하 인식 라우팅 (video-process 참조)
- 단순 대안: 라운드로빈 or 큐 기반 분배

## 메시지 포맷

### Kafka: E2E 이벤트 (자동)

#### 수신 — 신규 이벤트 구독

상위 팀에서 생성하는 신규 이벤트를 구독하여 E2E 자동 처리.
필요 필드는 위의 "수신 이벤트 — 필요 필드" 참조.

#### 발행 — portraitCreatedEvent (신규 토픽)

```
토픽: eda-cju-portrait981-portraitCreatedEvent
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `workflowId` | long | 탑승 식별자 (echo back) |
| `memberId` | long | 고객 식별자 (echo back) |
| `activityType` | String | 어트랙션 종류 (echo back) |
| `status` | String | `"DONE"` / `"PARTIAL"` / `"FAILED"` |
| `scanFrameCount` | int | 분석된 프레임 수 |
| `scanHighlightCount` | int | 감지된 하이라이트 수 |
| `outputUrls` | list[String] | 생성된 AI 초상화 S3 경로 |
| `errorMessage` | String | (nullable) 실패 시 사유 |
| `timingSec` | float | 총 처리 시간 |
| `publishDt` | String | 발행 시각 |

### REST API: 수동/개별 요청

| 엔드포인트 | 메서드 | 용도 |
|-----------|--------|------|
| `/portrait/scan` | POST | scan + bank 저장만 |
| `/portrait/generate` | POST | bank에서 참조 이미지 → 생성만 |
| `/portrait/test` | POST | 파이프라인 검증 (bank 저장 안 함) |
| `/portrait/status/{member_id}` | GET | 프레임 현황 조회 |

#### POST /portrait/scan — Scan-only

```json
{
  "member_id": 1423,
  "workflow_id": 20260309001423,
  "video_uri": "s3://981park-raw/2026/03/09/car-07/wf-20260309-1423.mp4"
}
```

비디오 분석 + bank 저장만 수행. 생성은 나중에 별도 요청.

#### POST /portrait/generate — Generate-only

```json
{
  "member_id": 1423,
  "generate": {
    "purpose": "avatar",
    "workflow": "avatar-v1",
    "prompt": "3D cartoon style avatar"
  }
}
```

member_id의 bank 데이터에서 참조 이미지를 선택하여 생성.

#### POST /portrait/test — 파이프라인 검증

```json
{
  "video_uri": "s3://981park-raw/test/sample.mp4",
  "generate": {
    "workflow": "2026-spring-v2",
    "ref_images": [
      "s3://981park-raw/test/face_01.jpg",
      "s3://981park-raw/test/face_02.jpg"
    ]
  }
}
```

조합:
- `video_uri`만 → scan만 검증
- `ref_images`만 → generate만 검증
- 둘 다 → E2E 검증
- bank에 저장하지 않음, 결과물은 임시 경로

#### GET /portrait/status/{member_id} — 상태 조회

bank에 저장된 member의 프레임 현황 조회.

### 생성 옵션

E2E, generate, test 모두 생성 스타일을 지정할 수 있다.
시즌/분기별로 `workflow` 값이 변경된다.

| 필드 | 타입 | 설명 |
|------|------|------|
| `workflow` | string | 생성 템플릿 (기본값: `"default"`, 예: `"2026-spring-v2"`) |
| `prompt` | string | 스타일 프롬프트 |
| `purpose` | string | 생성 목적 (`"portrait"`, `"avatar"`, `"event"`) |
| `ref_images` | list[string] | 직접 지정 참조 이미지 (bank 조회 건너뜀) |

생성 스타일 결정 우선순위:

1. **이벤트에 포함된 값** — workflow/prompt가 있으면 우선 적용 (경기 기록, 파크 이벤트 등)
2. **서버 기본 설정** — 이벤트에 없으면 `P981_DEFAULT_WORKFLOW` / `P981_DEFAULT_PROMPT` 사용

시즌별 기본 스타일은 서버 설정으로 관리하되, 상위 서비스가 특정 상황에 맞게 override할 수 있다.

## 서버 설정

portrait981-serve가 관리하는 설정값. 환경변수 또는 설정 파일로 주입.

### 생성 설정

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `P981_DEFAULT_WORKFLOW` | `"default"` | E2E 시 적용할 생성 템플릿 (시즌별 변경) |
| `P981_DEFAULT_PROMPT` | `""` | E2E 시 적용할 기본 스타일 프롬프트 |
| `P981_TOP_K` | `3` | bank에서 선택할 최대 참조 이미지 수 |

### ComfyUI 노드풀

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `P981_COMFY_URLS` | `"http://127.0.0.1:8188"` | ComfyUI 노드 URL 목록 (쉼표 구분) |
| `P981_COMFY_API_KEY` | - | ComfyUI 인증 키 |

### Scan 설정

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `P981_SCAN_FPS` | `10` | 분석 프레임 레이트 |
| `P981_SCAN_BACKEND` | `"simple"` | momentscan 실행 백엔드 |
| `P981_COLLECTION_PATH` | - | 카탈로그 경로 (포즈/카테고리 정의) |

### S3

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `P981_S3_OUTPUT_BUCKET` | - | 결과물 업로드 버킷 |
| `P981_S3_OUTPUT_PREFIX` | `"portrait/"` | 결과물 경로 prefix |
| `P981_S3_REGION` | `"ap-northeast-2"` | AWS 리전 |
| `P981_S3_ACCESS_KEY` | - | AWS access key |
| `P981_S3_SECRET_KEY` | - | AWS secret key |

### Kafka

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `P981_KAFKA_BOOTSTRAP` | - | Kafka 브로커 주소 |
| `P981_KAFKA_CONSUMER_GROUP` | `"cju-portrait981-serve-{env}-group"` | consumer group ID |
| `P981_KAFKA_TOPIC_IN` | - | 수신 토픽 (상위 팀에서 결정) |
| `P981_KAFKA_TOPIC_OUT` | `"eda-cju-portrait981-portraitCreatedEvent"` | 발행 토픽 |
| `P981_KAFKA_API_KEY` | - | SASL 인증 키 |
| `P981_KAFKA_API_SECRET` | - | SASL 인증 시크릿 |
| `P981_KAFKA_SCHEMA_REGISTRY_URL` | - | Confluent Schema Registry URL |

## 도메인 모델

```
member_id (고객, 앱 회원가입 시 발급, 영구)
    └── workflow_id (탑승, 어트랙션 체크인마다 발급, 1회성)
            └── video (1:1, S3 저장)
```

- 고객이 여러 번 탑승 → 같은 member_id에 여러 workflow_id
- E2E 결과물은 workflow_id별 격리 (해당 탑승 프레임만 사용)
- generate-only는 member_id의 전체 bank 이력 활용 가능

## 미팅 체크리스트

### Kafka

- [x] 직렬화: Avro (Confluent Schema Registry)
- [x] 인증: SASL_SSL + PLAIN
- [x] 토픽 네이밍: `eda-{platform-id}-{domain}-{event}` (camelCase)
- [x] consumer group 규약: `{app-name}-{env}-group`
- [x] `mommos-event` Python 버전 없음 → `confluent-kafka[avro]`로 직접 구현

**미팅에서 확인할 것 (2건):**

- [ ] 수신 토픽 — 상위 팀이 생성할 토픽 이름 + 필수 필드(workflowId, memberId, videoUrl) 반영 확인
- [ ] 발행 토픽 — `eda-cju-portrait981-portraitCreatedEvent` 등록 절차 + Schema Registry에 Avro 스키마 등록

### ComfyUI 노드풀

- [x] 노드 관리 방식: 수동 URL 목록으로 시작, 필요 시 Eureka로 전환 가능 (노드 선택 로직만 교체)
- [ ] 가용 노드 선택 로직 (라운드로빈 → 필요 시 부하 인식)
- [ ] 노드 장애 시 재시도 정책

### S3

- [ ] 입력 버킷/경로 규약 (`video/original/{activityType}/{YYYYMMDD}/`)
- [ ] 출력 버킷/경로 규약
- [ ] 인증 방식 (IAM role, access key)

### 비즈니스

- [x] 처리 우선순위: 수신 이벤트에 `priority` 선택 필드로 지원 (기본 0, 특별 이벤트/VIP 대응)
- [ ] 실패 시 재시도 정책 (자동/수동, 최대 횟수)
- [ ] 결과물 보존 기간
- [x] E2E 생성 스타일: 서버 기본 설정 + 이벤트에서 override 가능

### p981 제공 정보

- 비디오 1개 처리 시간: scan ~1분, generate ~1-3분 (고급 GPU 최적화 목표)
- scan: GPU 중급 (YOLO, InsightFace, CLIP 추론)
- generate: GPU 고급 필수 (Stable Diffusion 등, 모델 상시 메모리 상주, 중급 시 3-40분)
- scan과 generate는 반드시 별도 장비
- partial 상태 = scan 데이터 보존, generate만 재요청 가능
- 시즌별 스타일 변경은 서버 설정의 `workflow` 값 교체

---

## 개발 계획

### 설계 결정 요약

| 항목 | 결정 | 비고 |
|------|------|------|
| 서비스 이름 | `portrait981-serve` (Git: `cju-portrait981-serve`) | |
| E2E 트리거 | Kafka — 상위 팀이 신규 이벤트 생성 | |
| 수동/테스트 | REST API (scan, generate, test, status) | |
| S3 fetch | visualbase에서 담당, 처리 후 즉시 삭제 | |
| S3 업로드 | portrait981-serve에서 담당 | |
| 생성 스타일 | 서버 기본 설정 + 이벤트 override | |
| 처리 우선순위 | 이벤트 `priority` 필드 (선택, 기본 0) | |
| Kafka 직렬화 | Avro + Confluent Schema Registry | |
| Kafka 구현 | `confluent-kafka[avro]` 직접 (mommos-event Python 없음) | |
| ComfyUI 노드풀 | 수동 URL 목록, 필요 시 Eureka 전환 | |
| scan/generate 장비 | 반드시 별도 (GPU 중급 / GPU 고급) | |

### 개발 순서

**Step 1: 미팅 전 착수 가능**

| 순서 | 작업 | 패키지 | 내용 |
|------|------|--------|------|
| 1-1 | visualbase S3 소스 | `visualbase` | `resolve_source("s3://...")` → 로컬 경로, `visualbase[s3]` optional extra |
| 1-2 | REST API 서버 | `portrait981-serve` | FastAPI 앱, 4개 엔드포인트 (scan, generate, test, status) |
| 1-3 | ComfyUI 노드풀 | `portrait981-serve` | URL 목록 라운드로빈, 장애 시 다음 노드 fallback |
| 1-4 | S3 결과 업로드 | `portrait981-serve` | 생성 결과물 → S3 업로드 + 경로 반환 |

**Step 2: 미팅 후 착수**

| 순서 | 작업 | 패키지 | 내용 |
|------|------|--------|------|
| 2-1 | Kafka consumer | `portrait981-serve` | 수신 토픽 구독, Avro 역직렬화, E2E 자동 처리 |
| 2-2 | Kafka producer | `portrait981-serve` | `portraitCreatedEvent` 발행, Avro 직렬화 |
| 2-3 | Avro 스키마 등록 | - | Schema Registry에 `portraitCreatedEventAvro` 등록 |

### 미팅 전 확인 필요 (S3)

REST API + S3 업로드 개발에 필요:

- [ ] S3 접근 정보 (access key, secret key, 또는 IAM role)
- [ ] 입력 버킷 이름 + 경로 패턴
- [ ] 출력 버킷 이름 + 경로 패턴

### 미팅 시 확인 (Kafka, 2건)

- [ ] 수신 토픽 이름 + 필수 필드(workflowId, memberId, videoUrl) 반영 확인
- [ ] 발행 토픽 `eda-cju-portrait981-portraitCreatedEvent` 등록 절차 + Avro 스키마 등록
