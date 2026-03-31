# Portrait981

통합 오케스트레이터 앱. momentscan → personmemory → reportrait 파이프라인을 단일 호출로 자동화.

## 디렉토리 구조

```
src/portrait981/
├── __init__.py       # Pipeline, JobSpec 등 re-export
├── types.py          # JobSpec, JobStatus, JobResult, PipelineConfig, JobHandle, StepEvent
├── pipeline.py       # Portrait981Pipeline (core orchestrator)
├── node_pool.py      # NodePool — ComfyUI 멀티노드 관리 (acquire/release/health)
├── progress.py       # 진행률 콜백 유틸리티
└── cli.py            # p981 CLI (argparse, thin)
```

## API

```python
from portrait981 import Portrait981Pipeline, JobSpec, PipelineConfig

pipeline = Portrait981Pipeline()
result = pipeline.run_one(JobSpec(video_path="v.mp4", member_id="test_3"))
pipeline.shutdown()

# 배치 (scan 순차, generate 병렬)
results = pipeline.run_batch([
    JobSpec(video_path="v1.mp4", member_id="test_3"),
    JobSpec(video_path="v2.mp4", member_id="test_5"),
])

# 비동기
handle = pipeline.submit(JobSpec(video_path="v.mp4", member_id="test_3"))
result = handle.result()
```

## 실행 흐름

```
run_one / run_batch / submit
     │
     ▼
_execute_job:
  1. SCAN: ms.run(video) → scan_result (member_lock 보호)
  2. LOOKUP: lookup_frames(member_id, pose, category, top_k) → ref_paths
  3. GENERATE: PortraitGenerator(config).generate(request) → generation_result
     │
     ▼
JobResult (status: DONE / PARTIAL / FAILED)
```

## 동시성

| 관심사 | 전략 |
|--------|------|
| scan | 호출 스레드에서 순차 실행 (GPU-bound) |
| generate | ThreadPoolExecutor (I/O-bound, ComfyUI 자체 큐) |
| member_id 충돌 | defaultdict(threading.Lock) — 같은 member는 scan 직렬화 |
| batch | scan 순차 → generate 병렬 dispatching |
| interrupt | Ctrl+C → _interrupted flag → 남은 job skip |

## StepEvent 콜백

```python
pipeline = Portrait981Pipeline(on_step=lambda e: print(f"[{e.step}] {e.status}: {e.detail}"))
```

이벤트: scan/started, scan/progress, scan/completed, lookup/completed, generate/started, generate/completed

## CLI

```bash
p981 run video.mp4 --member-id test_3 [--pose frontal] [--scan-only]
p981 batch videos/ --member-id-from filename [--workers 2]
p981 scan video.mp4 --member-id test_3
p981 generate test_3 [--pose frontal] [--prompt "portrait"]
p981 status test_3
```

## 의존성

- `momentscan` — ms.run() 분석
- `personmemory` — lookup_frames() 프레임 조회
- `reportrait` — PortraitGenerator 생성

## 테스트

```bash
uv run pytest apps/portrait981/tests/ -v    # 55 tests
```
