# Protocols and Backends

## 문서의 목적

이 문서는 외부 의존성을 교체 가능하게 하기 위한 프로토콜/백엔드 경계를 정의한다.

---

## 프로토콜 정의 (Core)

- BlobStore: 미디어 파일 저장/조회 (clip, keyframe, video 등)
- MetaStore: asset 메타데이터 저장/조회
- AssetIndex: 검색/중복판별/벡터 인덱스 (선택)
- Cache: 재처리 방지 및 성능 최적화 (선택)
- ObserverProtocol: 분석/시각화용 이벤트 출력 (선택)

Runtime 전용 프로토콜(JobStore/JobQueue 등)은 `specs/runtime-ports.md`에 정의한다.

---

## 백엔드 방향 (초기)

- BlobStore: 로컬 FS (dev) → S3/객체 스토리지
- MetaStore: JSON 파일/SQLite (dev) → Postgres
- AssetIndex: none (dev) → VectorDB (선택)
- ObserverProtocol: log/noop/opencv (dev) → Pixeltable(frames)/Rerun (선택)

### ObserverProtocol (스켈레톤)

- 기본 이벤트: frame_index, timestamp_ms, avg_luma, video_ref
- frames 백엔드는 Pixeltable을 사용하며 `p981_observations` 테이블에 저장한다.
- 스켈레톤 단계에서는 실행마다 `p981_observations`를 초기화(replace)한다.
- Image는 별도 저장하지 않고 frame_path만 기록한다.
- opencv 백엔드는 frame_path가 있는 이벤트를 즉시 시각화하고, 간단한 로그를 출력한다.

---

## 원칙

- Core는 프로토콜 인터페이스만 의존한다.
- 백엔드 구현체는 `core/infra/backends`에 두고 `core/application/protocols`는 인터페이스만 유지한다.
- 백엔드 선택은 config/environment로 제어한다.
- dev 기본값은 단순 구현을 사용하되, 교체 비용이 낮아야 한다.
- 프로토콜 목록과 세부 인터페이스는 추후 확정하며 스켈레톤 단계에서는 최소 프로토콜로 시작한다.
