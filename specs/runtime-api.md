# Runtime API (MVP2)

## 문서의 목적

이 문서는 Runtime 레이어의 외부 API 계약을 정의한다.

---

## 공통 규칙

- API는 blob을 직접 반환하지 않고 asset_ref만 반환한다.
- metadata 스키마/노출 방식은 추후 정의하며, 스켈레톤 단계에서는 임의 필드가 임시로 포함될 수 있다.
- POST 요청은 Job을 생성하고 202 Accepted로 응답한다.
- status 값은 아래 enum으로 고정한다.
  - PENDING | RUNNING | SUCCEEDED | FAILED | CANCELED
- resource_class(cpu/gpu)는 API에서 받지 않고 runtime 내부 정책으로 결정한다.

---

## POST /api/v1/discover

### Request

```
{
  "video_ref": "string",
  "member_id": "string"
}
```

- member_id는 optional이며, 누락 시 고객 히스토리 기반 중복 제거/누적을 생략할 수 있다.

### Response (202)

```
{
  "job_id": "string",
  "status": "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED" | "CANCELED"
}
```

---

## POST /api/v1/synthesize

### Request

```
{
  "style": "base" | "closeup" | "fullbody" | "cinematic",
  "moment_ref": "string",
  "base_portrait_ref": "string",
  "closeup_image_ref": "string",
  "fullbody_image_ref": "string"
}
```

#### 스타일별 입력 조건

- base: moment_ref 필요
- closeup/fullbody: base_portrait_ref 필요
- cinematic: closeup_image_ref + fullbody_image_ref 필요
- 필수 외 필드의 검증/무시 정책은 추후 정의하며, 스켈레톤 단계에서는 무시할 수 있다.

### Response (202)

```
{
  "job_id": "string",
  "status": "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED" | "CANCELED"
}
```

---

## GET /api/v1/jobs/{job_id}

### Response

```
{
  "job_id": "string",
  "job_type": "DISCOVER" | "SYNTHESIZE",
  "status": "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED" | "CANCELED",
  "result": {}
}
```

#### DISCOVER 결과 예시

```
{
  "moment_pack": {
    "moments": [
      {
        "moment_ref": "string",
        "moment_clip_ref": "string",
        "keyframe_pack_ref": "string",
        "moment_meta": {}
      }
    ]
  },
  "history_updated": true
}
```

- member_id가 없는 요청은 history_updated가 false로 유지될 수 있다.
- moment_meta 스키마는 `specs/core-asset-schema.md`를 따른다.

#### SYNTHESIZE 결과 예시

```
{
  "generated_asset_ref": "string",
  "reused_existing": true
}
```

---

## GET /api/v1/jobs

- 필터링 지원: member_id, job_type, status
- 상세 필터는 구현 전에 확정한다.

---

## DELETE /api/v1/jobs/{job_id}

- Job 취소 요청
- 취소 성공 시 status는 CANCELED로 전환한다.

---

## 비고

- result는 status가 SUCCEEDED일 때만 포함한다.
- 필터링 필드와 동기 응답 규칙은 구현 전에 확정한다.
