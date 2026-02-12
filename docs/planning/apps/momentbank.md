# momentbank — 인물 외형 저장/관리 시스템

> momentscan이 수집한 인물별 이미지와 임베딩을 저장하고,
> reportrait이 필요로 하는 reference image를 맥락 기반으로 선택하는 시스템.
> 단일 prototype의 한계를 극복하여 **상태별 다중 centroid**를 유지.

## 위치

```
momentscan (분석/수집) → momentbank (저장/관리) → reportrait (AI 생성)
```

## 핵심 역할

1. **저장**: momentscan(identity_builder)이 수집한 이미지/임베딩을 인물별로 축적
2. **매칭**: 새 프레임의 동일인 판정 (Face-ID embedding 기반)
3. **선택**: reportrait의 디퓨전 모델에 전달할 reference image를 맥락 기반으로 선택

## Memory Bank 설계

인물당 최대 k개(기본 10) memory node. 각 노드는 인물의 특정 상태 영역을 대표
(예: "정면 나", "어두운 조명 나", "옆얼굴 나").

### 핵심 API (3개)

| API | 역할 | 호출자 |
|-----|------|--------|
| `update(e_id, quality, meta, image_path)` | 새 관측 반영 (EMA merge / new node) | momentscan |
| `match(e_id)` → MatchResult | 동일인 판정 (stable_score) | momentscan |
| `select_refs(query)` → RefSelection | reference image 선택 | reportrait |

### reportrait 연동

원칙: **이미지 경로 전달** (임베딩 직접 주입 금지)

- Memory bank 임베딩(ArcFace)과 ComfyUI 내부 임베딩(CLIP-Vision)은 공간/차원이 다름
- `select_refs()` → `paths_for_comfy = [img1.jpg, img2.jpg, ...]`
- Reference 조합: Anchor(20%) + Coverage(70%) + Challenge(10%)

상세 설계: [identity_memory.md](identity_memory.md)

## 출력 구조

```
output/{video_id}/momentbank/
├── person_0/
│   └── memory_bank.json
├── person_1/
│   └── memory_bank.json
└── _version.json
```

## vpx 의존성

| 패키지 | 용도 |
|--------|------|
| vpx-sdk | 공유 타입 |
| vpx-face-detect | Face-ID 임베딩 (ArcFace recognition model) |

momentbank 자체는 임베딩 모델을 직접 실행하지 않음.
momentscan이 추출한 임베딩을 받아서 저장/검색만 수행.

## 개발 순서

| 순서 | 작업 | 의존 |
|------|------|------|
| 1 | MemoryNode, NodeMeta 데이터클래스 | - |
| 2 | MemoryBank (load/save JSON) + update() | 1 |
| 3 | match() API + momentscan 연동 | 2 |
| 4 | select_refs() API | 2 |
| 5 | reportrait 연동 (image path 전달) | 4 |
