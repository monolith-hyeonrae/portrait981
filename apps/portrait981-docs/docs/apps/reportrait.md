# reportrait — AI 초상화 생성 앱

> 최종 업데이트: 2026-03-30

personmemory가 선택한 reference image를 디퓨전 모델에 전달하여
인물 일관성을 유지한 AI 초상화를 생성하는 시스템.

## 위치

```
momentscan (분석/수집) → personmemory (저장/관리) → reportrait (AI 생성)
```

## 두 가지 생성 모드

### 1. 어트랙션 추억 (per-workflow, 즉시)

```
탑승 완료 → momentscan run → SHOOT 프레임
  → 이번 workflow의 best frame 선택
  → reportrait → 어트랙션 스타일 AI portrait
  → 고객 앱 전송
```

빠른 피드백이 핵심. 이번 탑승의 프레임만 사용.

### 2. Base-Portrait (per-member, 축적)

```
personmemory → portrait quality score 최고 프레임 선택
  → IC-Light 조명 보정 (optional)
  → 업스케일 + 손/결함 제거
  → base-portrait 완성
  → 앱 프로필 / 경기 대시보드 디스플레이
```

이 사람을 대표하는 프로필 사진. 재방문할수록 더 좋은 base-portrait 갱신 가능.

## Base-Portrait 파이프라인

### 프레임 선정

personmemory에서 portrait quality score 기준 best:

```
portrait_score = lighting_quality × expression_confidence × sharpness

lighting_quality = lighting_ratio × face_brightness_std / 100
  → dramatic lighting (Rembrandt, split) = 높은 점수
  → flat lighting = 낮은 점수

sharpness = min(face_blur / 50, 1.0)
  → 선명한 프레임 우대
```

### AI 보정 파이프라인 (ComfyUI)

```
base frame (좋은 표정 + 좋은 조명)
  ↓
[IC-Light] 조명 보정 (optional)
  → Rembrandt / golden hour 등 이상적 portrait lighting 적용
  → 조명이 아쉬운 좋은 표정을 조명까지 완벽하게 보정
  ↓
[Upscale] 해상도 향상
  → 4x upscale (ESRGAN / SwinIR)
  ↓
[Inpaint] 결함 제거
  → 손/팔 가림 제거, 배경 정리
  ↓
base-portrait 완성
```

### IC-Light 활용

[IC-Light](https://github.com/lllyasviel/IC-Light) — 조명 조건을 제어하는 relighting 모델.

```
IC-Light의 핵심:
  "Lighting Preferences are just initial latents"
  → 조명을 latent space 초기값으로 표현
  → 물리적으로 일관된 light transport (3D-aware)
```

두 가지 활용 방향:

**분석 도구 (face.lighting 고도화, Phase 3):**
```
image → IC-Light encoder → lighting latent descriptor
  → 현재 pixel 통계(ratio, std) 대비 훨씬 풍부한 조명 표현
  → 연속 공간에서 조명 유사도/품질 측정
  → personmemory에 lighting latent 저장
```

**생성 도구 (reportrait, Phase 2):**
```
base frame (좋은 표정, 조명 보통)
  → IC-Light로 이상적 portrait lighting 적용
  → Rembrandt, golden hour, studio 등 선택 가능
  → 어트랙션에서 조명 통제 불가 → 후처리로 보정
```

ComfyUI에 IC-Light 노드가 이미 존재 → reportrait 워크플로우에 통합 가능.

## ComfyUI Bridge

```
personmemory.get_reference() 또는 get_profile_reference()
        │ image paths
        ▼
reportrait workflow
        │ reference injection + prompt + IC-Light conditioning
        ▼
ComfyUI API (REST) — RunPod 또는 로컬
        │ POST /prompt
        ▼
ComfyUI (InstantID / IP-Adapter / PuLID + IC-Light + Upscale)
        │
        ▼
생성된 이미지 → gallery
```

## 디퓨전 모델 옵션

| 모델/기법 | 특징 | 용도 |
|-----------|------|------|
| InstantID | Face-ID 기반 identity 유지 | 기본 portrait 생성 |
| IP-Adapter | 이미지 prompt로 스타일/구도 제어 | 스타일 변환 |
| PuLID | Pure/clean identity embedding | 고품질 identity |
| IC-Light | 조명 relighting | base-portrait 조명 보정 |
| ESRGAN/SwinIR | 업스케일 | 해상도 향상 |

## 구현 Phase

```
Phase 1 (현재):
  ComfyUI API 연동 완료 (comfy_bridge)
  InstantID 워크플로우 테스트 완료
  RunPod 원격 실행 지원

Phase 2 (다음):
  base-portrait 파이프라인 (선정 → IC-Light → 업스케일 → 결함 제거)
  personmemory 연동 (get_profile_reference)
  배치 생성 파이프라인

Phase 3 (중기):
  IC-Light latent → face.lighting 분석기 고도화
  스타일 다양화 (어트랙션별 테마)
  고객 반응 피드백 → creative memory (personmemory)
```

## 의존성

| 항목 | 역할 |
|------|------|
| personmemory | get_reference / get_profile_reference → 참조 이미지 |
| ComfyUI | 디퓨전 생성 엔진 (RunPod 또는 로컬) |
| IC-Light | 조명 relighting (ComfyUI 노드) |
