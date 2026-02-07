# visualbase

Media platform for video frame streaming and clip extraction.

## Features

- **Multi-source support**: File (MP4), USB camera, RTSP streams
- **Frame streaming**: Iterator-based frame access with optional FPS/resolution sampling
- **Ring buffer**: Memory-efficient tmpfs-based buffer for live streams (120s default retention)
- **Clip extraction**: FFmpeg-based clip extraction with trigger-based timing
- **Headless operation**: Run without GUI for server deployments

## Installation

```bash
pip install visualbase
```

Or with uv:

```bash
uv add visualbase
```

## Quick Start

### File playback

```python
from visualbase import VisualBase, FileSource

with VisualBase() as vb:
    vb.connect(FileSource("video.mp4"))

    for frame in vb.get_stream(fps=30):
        # frame.data is numpy array (H, W, C)
        # frame.t_src_ns is timestamp in nanoseconds
        process(frame)
```

### Camera capture with clip extraction

```python
from visualbase import VisualBase, CameraSource, Trigger

vb = VisualBase(clip_output_dir="./clips")
vb.connect(CameraSource(device_id=0), ring_buffer_retention_sec=120.0)

for frame in vb.get_stream(fps=30):
    if detect_event(frame):
        trigger = Trigger.point(
            event_time_ns=frame.t_src_ns,
            pre_sec=3.0,
            post_sec=1.0,
            label="event"
        )
        result = vb.trigger(trigger)
        print(f"Saved: {result.output_path}")
```

## CLI Commands

```bash
# Play video file
visualbase play video.mp4

# Show video info
visualbase info video.mp4

# Extract clip from video
visualbase clip video.mp4 --time 10.5 --pre 3.0 --post 2.0

# Test USB camera
visualbase camera 0 --fps 30

# Test RTSP stream
visualbase rtsp rtsp://localhost:8554/stream
```

## Architecture

```
visualbase/
├── sources/          # Video input sources
│   ├── file.py       # FileSource (MP4, etc.)
│   ├── camera.py     # CameraSource (USB/local)
│   └── rtsp.py       # RTSPSource (IP cameras)
├── core/
│   ├── frame.py      # Frame dataclass
│   ├── sampler.py    # FPS/resolution sampling
│   ├── buffer.py     # FileBuffer
│   └── ring_buffer.py # RingBuffer (tmpfs)
├── packaging/
│   ├── trigger.py    # Trigger definitions
│   └── clipper.py    # FFmpeg clip extraction
└── api.py            # VisualBase main class
```

## Requirements

- Python >= 3.10
- OpenCV >= 4.8.0
- NumPy >= 1.24.0
- FFmpeg (for clip extraction)

## License

MIT
