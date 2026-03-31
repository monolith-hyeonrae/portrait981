# Reportrait

AI 초상화 생성 앱. ComfyUI 워크플로우 기반 이미지/비디오 생성 브릿지.
personmemory의 `lookup_frames()`로 참조 이미지를 조회하거나, 직접 이미지를 주입하여 생성.

## 디렉토리 구조

```
src/reportrait/
├── __init__.py
├── cli.py              # argparse CLI (reportrait generate)
├── comfy_client.py     # ComfyUI REST API 클라이언트
├── generator.py        # PortraitGenerator 오케스트레이터
├── types.py            # GenerationConfig, GenerationRequest, GenerationResult
├── workflow.py         # 워크플로우 템플릿 로딩 + injection
└── templates/
    └── default.json    # 기본 워크플로우 템플릿
```

## CLI

```bash
# member_id로 lookup_frames 조회 후 생성
reportrait generate <member_id> --pose left30 --category warm_smile --prompt "portrait"

# 직접 참조 이미지 지정 (lookup 건너뜀)
reportrait generate --ref photo1.jpg photo2.jpg --prompt "portrait"

# 워크플로우 파일 직접 지정 (I2I/I2V)
reportrait generate --ref face.jpg --workflow /path/to/workflow.json

# 특정 LoadImage 노드에만 주입
reportrait generate --ref face.jpg --workflow workflow.json --node 81

# dry-run: ComfyUI 미호출, 주입된 workflow JSON 출력
reportrait generate test_3 --pose left30 --dry-run

# 원격 ComfyUI (RunPod proxy)
reportrait generate --ref face.jpg --comfy-url https://xxx.proxy.runpod.net --api-key $RUNPOD_API_KEY
```

### CLI 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `member_id` | 프레임 조회용 멤버 ID (--ref 없을 때 필수) | - |
| `--ref` | 참조 이미지 직접 지정 (nargs=+) | - |
| `--node` | 주입 대상 LoadImage 노드 ID (nargs=+) | auto-detect by `_meta.role` |
| `--pose` | pose_name 필터 (e.g. left30, frontal) | None |
| `--category` | category 필터 (e.g. warm_smile) | None |
| `--top` | 최대 참조 이미지 수 | 3 |
| `--prompt` | 스타일 프롬프트 | "" |
| `--workflow` | 템플릿 이름 또는 .json 파일 경로 | "default" |
| `--comfy-url` | ComfyUI 서버 URL | http://127.0.0.1:8188 |
| `--api-key` | API 키 (RUNPOD_API_KEY env fallback) | None |
| `--output` | 출력 디렉토리 | auto |
| `--dry-run` | workflow JSON만 출력, 생성 미실행 | false |

## 핵심 모듈

### ComfyClient (`comfy_client.py`)

urllib 기반 ComfyUI REST API 클라이언트. 외부 HTTP 의존성 없음.

- `queue_prompt(workflow)` → prompt_id
- `wait_for_completion(prompt_id)` → history entry (polling)
- `download_images(history, output_dir)` → List[Path]
- Bearer 토큰 인증 지원 (`api_key` 파라미터)
- URL 스킴 자동 보정 (`localhost:8188` → `http://localhost:8188`)

### PortraitGenerator (`generator.py`)

생성 오케스트레이터. load template → inject refs → inject prompt → queue → wait → download.

- `generate(request)` — 기본 생성 (GenerationRequest)
- `generate_from_bank(bank_path)` — MemoryBank에서 select_refs 후 생성
- `generate_from_lookup(member_id)` — lookup_frames() 조회 후 생성

### Workflow Injection (`workflow.py`)

- `load_template(name)` — 템플릿 로딩 (직접 .json 경로 / CWD / 패키지 templates/)
- `inject_references(workflow, ref_paths, node_ids=)` — LoadImage 노드에 이미지 경로 주입
  - `node_ids` 지정 시 해당 노드만 타겟
  - 미지정 시 `_meta.role="reference"` 노드 자동 감지
- `inject_prompt(workflow, prompt)` — `_meta.role="positive"` CLIPTextEncode에 텍스트 주입

## 의존성

- `personmemory` — lookup_frames(), load_bank() 등
- 외부: 없음 (urllib만 사용)
- 선택: ComfyUI 서버 (로컬 또는 RunPod)

## 테스트

```bash
cd /home/hyeonrae/repo/monolith/portrait981
uv run pytest apps/reportrait/tests/ -v    # 41 tests
```

## RunPod 연동 참고

- ComfyUI 서버에 `--enable-cors-header` 플래그 필요 (CORS 호스트 검증 우회)
- RunPod proxy URL: `https://xxx.proxy.runpod.net`
- API 키: `RUNPOD_API_KEY` 환경변수 또는 `--api-key` 옵션
