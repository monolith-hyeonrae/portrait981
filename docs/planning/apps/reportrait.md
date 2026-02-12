# reportrait — AI 초상화 생성 앱

> momentbank가 선택한 reference image를 디퓨전 모델에 전달하여
> 인물 일관성을 유지한 AI 초상화를 생성하는 시스템.

## 위치

```
momentscan (분석/수집) → momentbank (저장/관리) → reportrait (AI 생성)
```

## 핵심 역할

1. momentbank의 `select_refs()` → reference image 경로 수신
2. ComfyUI workflow에 reference image 주입
3. 디퓨전 모델로 초상화 생성 (InstantID / IP-Adapter / PuLID)

## ComfyUI Bridge

### 연동 방식

```
momentbank.select_refs(query)
        │ paths_for_comfy = [img1.jpg, img2.jpg, ...]
        ▼
comfy_bridge
        │ workflow.json 내 reference image 경로 교체
        ▼
ComfyUI API (REST)
        │ POST /prompt — workflow JSON 전송
        ▼
ComfyUI (InstantID / IP-Adapter / PuLID 노드)
        │
        ▼
생성된 이미지
```

### ComfyUI API 활용

- ComfyUI는 REST API로 workflow JSON을 받아 실행
- `comfy_bridge`가 workflow template의 reference image 경로를 동적으로 교체
- 생성 완료 후 결과 이미지를 수신

### Reference 조합 비율

```
Anchor : Coverage : Challenge = 20 : 70 : 10 (기본값)
```

| 상황 | 조정 |
|------|------|
| ID drift 잦음 | Anchor 비율 상향 (30:60:10) |
| 표정/각도 다양성 부족 | Coverage 비율 상향 (15:75:10) |
| 어두운 조건 빈번 | Challenge 비율 상향 (20:65:15) |

간접 가중: ComfyUI가 weight tensor를 직접 받지 못하는 경우,
이미지 수/반복으로 가중 (예: anchor 2장, coverage 1장).

## 디퓨전 모델 옵션

| 모델/기법 | 특징 | 상태 |
|-----------|------|------|
| InstantID | Face-ID 기반 identity 유지 | ComfyUI 노드 존재, 실험 완료 |
| IP-Adapter | 이미지 prompt로 스타일/구도 제어 | ComfyUI 노드 존재 |
| PuLID | Pure/clean identity embedding | ComfyUI 노드 존재 |

### 자체 구현 필요 영역 (TBD)

- ComfyUI 노드로 해결되지 않는 커스텀 conditioning
- 다중 reference image의 가중 합성
- 배치 생성 파이프라인

## 출력 구조

```
output/{video_id}/reportrait/
├── generated/
│   ├── portrait_001.jpg
│   ├── portrait_001_meta.json
│   └── ...
├── workflow_used.json       # 실제 사용된 workflow 기록
└── _version.json
```

## 의존성

| 항목 | 역할 |
|------|------|
| momentbank | select_refs API → reference image 경로 |
| ComfyUI | 디퓨전 생성 엔진 (외부 서비스) |

## 개발 순서

| 순서 | 작업 | 의존 |
|------|------|------|
| 1 | ComfyUI API 연동 모듈 (workflow JSON 전송/수신) | ComfyUI 설치 |
| 2 | comfy_bridge (workflow template + image path 교체) | 1 |
| 3 | momentbank.select_refs() 연동 | momentbank |
| 4 | 배치 생성 파이프라인 | 2, 3 |
| 5 | 품질 평가 + 피드백 루프 | 4 |
