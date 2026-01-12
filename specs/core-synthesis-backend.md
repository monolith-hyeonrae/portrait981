# Synthesis Backend Integration

## 문서의 목적

이 문서는 p981-synthesis가 외부 생성 백엔드(ComfyUI)를 활용하는 원칙을 정의한다.

---

## 기본 원칙

- 가능한 처리 단계는 코드 모듈로 구현한다.
- 코드로 구현하기 어려운 단계는 ComfyUI 백엔드 호출로 대체한다.
- 외부 호출은 adapter/port를 통해 추상화한다.
- 입출력은 asset_ref 기반으로 수행한다.
- 합성 전용 포트 정의는 추후 결정한다. 스켈레톤 단계에서는 core 공통 포트를 사용한다.

---

## MVP 적용

- closeup/fullbody 결과를 선행 생성해야 cinematic 제작이 가능하다.
- MVP에서는 cinematic_video_ref 산출을 목표로 한다.
- 필요 시 ComfyUI 워크플로를 호출하여 closeup/fullbody/cinematic을 생성한다.
- 생성된 중간 결과는 asset으로 저장하여 재사용 가능하게 한다.

---

## 리소스 고려

- ComfyUI 기반 생성은 GPU 리소스를 요구할 수 있다.
- 해당 작업은 GPU worker pool에서 실행되도록 resource_class=GPU로 분리한다.

---

## 확정 필요 사항

- ComfyUI 엔드포인트/인증 방식
- 워크플로 식별자 및 버전 관리 방식
- 입력/출력 payload 포맷
