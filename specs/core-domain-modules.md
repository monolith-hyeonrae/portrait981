# Domain Modules

## 문서의 목적

이 문서는 p981-core 도메인 모듈의 책임과 경계를 정의한다.

---

## p981-media

- 역할: 미디어 등록 및 프레임/클립 추출
- 책임: 원본 비디오 접근 및 미디어 단위 추출만 수행
- 비책임: 판단, 선택, 분석 로직 없음
- 캐시: 동일 미디어 추출 요청 시 캐시 결과 반환 가능
- video_ref 처리: 로컬 경로, file://, http(s)://, s3://, blob_ref (skeleton 기본)
- 스켈레톤 출력: keyframe_pack=JPEG zip, moment_clip=MP4(H.264/AAC)
- ObservationPort: 프레임 관측 이벤트 출력 (frame_index, timestamp_ms, avg_luma)

---

## p981-state

- 역할: 고객 상태 타임라인 생성
- 책임: emotion / hand / quality 등 시간 기반 상태 관측
- 비책임: moment 판단 또는 선택 로직 없음
- 출력: state_timeline_ref (`specs/core-state-timeline.md`)

---

## p981-moment

- 역할: Moment 분석 및 결정
- 책임: 후보 생성, 중복 제거, 다양성 조건을 만족하는 N개 선정
- 비책임: 미디어 추출 직접 수행하지 않음

---

## p981-asset

- 역할: 자산 및 히스토리 관리
- 책임: 모든 중간/최종 산출물 저장 및 조회, 고객 Moment 히스토리 누적 (customer_id 제공 시)
- 저장소: Object Storage(미디어), Metadata DB(메타/인덱스)
- 인덱싱: customer_id, timestamp, style, quality_score 등 검색 최적화

---

## p981-synthesis

- 역할: 결과물 생성
- 책임: 요청된 스타일에 따른 생성 수행
- 제약: Moment 선택 또는 분석 로직 없음
- 의존성: cinematic은 closeup + fullbody 결과에 의존
- 구현 방식:
  - 가능한 단계는 코드 모듈로 구현한다.
  - 부족한 단계는 ComfyUI 백엔드 호출로 대체한다.
  - 외부 호출은 adapter/port를 통해 추상화한다.

---

## p981-stage

- 역할: 도메인 스테이지 정의
- 책임: Stage 구성 모듈, 입력/출력 계약, 실행 순서 정의
- 비책임: 실행 방식(API, Job, Worker)에는 관여하지 않음
