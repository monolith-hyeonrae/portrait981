# momentscan

Face moment detection and highlight clip extraction.

## Quick Start

```python
import momentscan as ms

# Process a video (one-liner)
triggers = ms.run("video.mp4")
print(f"Found {len(triggers)} highlights")
```

### With Options

```python
# Adjust settings
triggers = ms.run(
    "video.mp4",
    fps=10,           # frames per second
    cooldown=3.0,     # seconds between triggers
)

# With callback
ms.run("video.mp4", on_trigger=lambda t: print(f"Trigger: {t.label}"))
```

### With Clip Extraction

```python
# Analyze and extract clips
result = fm.analyze("video.mp4", output_dir="./clips")
print(f"{len(result.triggers)} triggers, {result.clips_extracted} clips")
```

## Installation

```bash
# Basic
pip install momentscan

# With ML backends (recommended)
pip install momentscan[all]

# Or with uv
uv pip install -e ".[all]"
```

## API Reference

| Function | Description |
|----------|-------------|
| `ms.run(video)` | Process video, return triggers |
| `fm.analyze(video, output_dir)` | Process video with stats and clip extraction |

### Parameters

```python
ms.run(
    video,                    # Path to video file
    extractors=None,          # ["face", "pose", "gesture", "quality"] or None for auto
    fps=10,                   # Frames per second to process
    cooldown=2.0,             # Seconds between triggers
    use_ml=True,              # Use ML extractors (False for dummy)
    on_trigger=None,          # Callback function
)
```

---

## Advanced Usage

For complex use cases, use the class-based API:

```python
from momentscan import MomentDetector
from momentscan.algorithm.extractors import FaceExtractor, PoseExtractor
from momentscan.algorithm.analyzers.highlight import HighlightFusion

detector = MomentDetector(
    extractors=[FaceExtractor(), PoseExtractor()],
    fusion=HighlightFusion(cooldown_sec=3.0),
    clip_output_dir="./clips",
)

clips = detector.process_file("video.mp4", fps=10)
```

## CLI Commands

```bash
# Process video
momentscan process video.mp4 -o ./clips

# Debug with visualization
momentscan debug video.mp4 -e face      # face only
momentscan debug video.mp4 -e face,pose # multiple
momentscan debug video.mp4 --no-ml      # dummy mode

# System info
momentscan info -v
```

## Architecture

Part of the Portrait981 system (FlowGraph + Backend architecture):

```
┌─────────────────────────────────────────────────────────────┐
│  ms.run("video.mp4")                                        │
│                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│  │   Source    │ ──► │   Modules   │ ──► │   Fusion    │   │
│  │  (frames)   │     │ face, pose  │     │ (triggers)  │   │
│  └─────────────┘     └─────────────┘     └─────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Modules

| Module | Backend | Description |
|--------|---------|-------------|
| `face.detect` | InsightFace SCRFD | Face detection + landmarks |
| `face.expression` | HSEmotion | Facial expression analysis |
| `body.pose` | YOLO-Pose | Body keypoints + gestures |
| `hand.gesture` | MediaPipe | Hand gestures (V-sign, thumbsup) |
| `frame.quality` | OpenCV | Blur, brightness, contrast |

## Trigger Types

| Trigger | Signal | Description |
|---------|--------|-------------|
| expression_spike | Adaptive EWMA | Happy emotion spike (sustained) |
| head_turn | Yaw velocity | Fast head rotation |
| hand_wave | Pose detection | Waving gesture |
| gesture_vsign | MediaPipe | V-sign hand pose |

## Dependencies

Core:
- Python >= 3.10
- NumPy, OpenCV
- visualbase, visualpath

ML (optional extras):
- `[face]`: InsightFace, HSEmotion, ONNX Runtime
- `[pose]`: Ultralytics (YOLO)
- `[gesture]`: MediaPipe

## Related Packages

- **visualbase**: Media streaming and clip extraction
- **visualpath**: Video analysis pipeline platform

## License

MIT
