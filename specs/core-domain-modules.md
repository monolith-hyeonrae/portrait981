# Domain Modules

## 문서의 목적

이 문서는 p981-core 도메인 모듈의 책임과 경계를 정의한다.

---

## DDD 레이어 정리

- Domain: 순수 모델/정책/알고리즘 (외부 의존 없음)
- Application: 서비스 인터페이스 및 stage/executor 워크플로 (`core/application`)
- Infra: 외부 의존 구현체 (`core/infra`)

---

## p981-media

- 역할: 미디어 등록 및 프레임/클립 추출
- 책임: 원본 비디오 접근 및 미디어 단위 추출만 수행
- 비책임: 판단, 선택, 분석 로직 없음
- 캐시: 동일 미디어 추출 요청 시 캐시 결과 반환 가능
- video_ref 처리: 로컬 경로, file://, http(s)://, s3://, blob_ref (skeleton 기본)
- media_handle: 정규화된 video_ref에 대한 경량 참조 + 기본 메타 + 캐시 키
- frame_source: media_handle에서 요청 fps/구간에 대한 지연 생성 스트림
- 샘플링 정책: stage/도메인 요구가 결정, media가 실행
- 스켈레톤 출력: keyframe_pack=JPEG zip, moment_clip=MP4(H.264/AAC)
- ObserverProtocol: 프레임 관측 이벤트 출력 (frame_index, timestamp_ms, avg_luma)

---

## p981-state

- 역할: 고객 상태 타임라인 생성
- 책임: emotion / hand / quality 등 시간 기반 상태 관측
- 비책임: moment 판단 또는 선택 로직 없음
- 입력: stage가 전달하는 media_handle 또는 frame_source
- 출력: state_timeline_ref (`specs/core-state-timeline.md`)

---

## p981-moment

- 역할: Moment 분석 및 결정
- 책임: 후보 생성, 중복 제거, 다양성 조건을 만족하는 N개 선정
- 비책임: 미디어 추출 직접 수행하지 않음

---

## p981-asset

- 역할: 자산 및 히스토리 관리
- 책임: 모든 중간/최종 산출물 저장 및 조회, 고객 Moment 히스토리 누적 (member_id 제공 시)
- 저장소: Object Storage(미디어), Metadata DB(메타/인덱스)
- 인덱싱: member_id, timestamp, style, quality_score 등 검색 최적화

---

## p981-synthesis

- 역할: 결과물 생성
- 책임: 요청된 스타일에 따른 생성 수행
- 제약: Moment 선택 또는 분석 로직 없음
- 의존성: cinematic은 closeup + fullbody 결과에 의존
- 구현 방식:
  - 가능한 단계는 코드 모듈로 구현한다.
  - 부족한 단계는 ComfyUI 백엔드 호출로 대체한다.
  - 외부 호출은 backend/protocol을 통해 추상화한다.

---

## p981-stage

- 역할: 도메인 스테이지 정의
- 책임: Stage 구성 모듈, 입력/출력 계약, 실행 순서 정의
- 비책임: 실행 방식(API, Job, Worker)에는 관여하지 않음
- 원칙: orchestration 외 알고리즘/정책은 도메인 모듈로 위임
- 확장: Stage 고유 로직은 policy/planner로 분리하고 Stage는 실행만 담당
- 구성: `stage/<stage_name>/` 하위에 입력/출력 계약과 스텝 구성(steps)을 둔다.
- executor: 공용 실행 유틸(예: StageExecutor/StepRunner)만 제공하고, stage 고유 로직은 포함하지 않는다.
