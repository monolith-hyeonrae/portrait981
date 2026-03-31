# Portrait981

통합 오케스트레이터 앱. momentscan → personmemory → reportrait 파이프라인을 단일 호출로 자동화.

## 디렉토리 구조

```
src/portrait981/
├── __init__.py       # Pipeline, JobSpec 등 re-export
├── types.py          # JobSpec, JobStatus, JobResult, PipelineConfig, JobHandle
├── pipeline.py       # Portrait981Pipeline (core orchestrator)
└── cli.py            # p981 CLI (argparse, thin)
```

## API

```python
from portrait981 import Portrait981Pipeline, JobSpec, PipelineConfig

pipeline = Portrait981Pipeline()
result = pipeline.run_one(JobSpec(video_path="v.mp4", member_id="test_3"))
pipeline.shutdown()
```

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
uv run pytest apps/portrait981/tests/ -v
```
