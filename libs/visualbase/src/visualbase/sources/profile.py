"""SourceProfile — media source characteristics.

Preserves original media properties so downstream consumers (Trigger-based
clip extraction, ROI crop saving, etc.) can make codec/quality decisions.

Usage:
    from visualbase.sources.profile import SourceProfile

    profile = SourceProfile.from_file("video.mov")
    print(profile.codec, profile.bit_depth, profile.color_space)
"""

from __future__ import annotations

import logging
import subprocess
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceProfile:
    """Media source characteristics.

    Attributes:
        uri: Source identifier (file path, RTSP URL, device ID).
        codec: Video codec ("prores", "h264", "h265", "vp9", "rawvideo", "unknown").
        bit_depth: Bits per channel (8, 10, 12).
        chroma: Chroma subsampling ("4:2:0", "4:2:2", "4:4:4").
        color_space: Color space ("rec709", "rec2020", "srgb", "bt470bg", "unknown").
        resolution: Original resolution (width, height).
        fps: Original frame rate.
        is_raw: RAW source (debayering needed).
        is_log: Log gamma curve (linear conversion needed for display).
        hdr: HDR content.
        pixel_format: FFmpeg pixel format string (e.g. "yuv420p", "yuv422p10le").
        container: Container format ("mov", "mp4", "mkv", "avi", "unknown").
    """
    uri: str = ""
    codec: str = "unknown"
    bit_depth: int = 8
    chroma: str = "4:2:0"
    color_space: str = "unknown"
    resolution: Tuple[int, int] = (0, 0)
    fps: float = 0.0
    is_raw: bool = False
    is_log: bool = False
    hdr: bool = False
    pixel_format: str = ""
    container: str = "unknown"

    @classmethod
    def from_file(cls, path) -> SourceProfile:
        """Probe a video file with ffprobe and build SourceProfile.

        Falls back to minimal profile if ffprobe is unavailable.
        """
        path = str(path)
        try:
            info = _ffprobe(path)
            if info is None:
                return cls(uri=path)
            return _parse_ffprobe(path, info)
        except Exception as e:
            logger.debug("ffprobe failed for %s: %s", path, e)
            return cls(uri=path)

    @classmethod
    def unknown(cls, uri: str = "") -> SourceProfile:
        """Create a minimal profile when source characteristics are unknown."""
        return cls(uri=uri)


def _ffprobe(path: str) -> Optional[dict]:
    """Run ffprobe and return parsed JSON."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", "-show_format",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def _parse_ffprobe(path: str, info: dict) -> SourceProfile:
    """Parse ffprobe JSON into SourceProfile."""
    # Find first video stream
    video = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            video = stream
            break

    if video is None:
        return SourceProfile(uri=path)

    codec = video.get("codec_name", "unknown")
    pix_fmt = video.get("pix_fmt", "")
    width = int(video.get("width", 0))
    height = int(video.get("height", 0))

    # FPS
    fps = 0.0
    r_frame_rate = video.get("r_frame_rate", "")
    if "/" in r_frame_rate:
        num, den = r_frame_rate.split("/")
        if int(den) > 0:
            fps = int(num) / int(den)

    # Bit depth from pixel format
    bit_depth = _parse_bit_depth(pix_fmt)

    # Chroma subsampling from pixel format
    chroma = _parse_chroma(pix_fmt)

    # Color space
    color_space = video.get("color_space", "unknown")
    if color_space in ("", "unknown"):
        color_space = _infer_color_space(codec, bit_depth)

    # Container
    fmt = info.get("format", {})
    container = fmt.get("format_name", "unknown").split(",")[0]

    # Log / RAW / HDR hints
    color_trc = video.get("color_transfer", "")
    is_log = color_trc in ("arib-std-b67", "smpte2084", "log", "log_sqrt")
    hdr = color_trc in ("arib-std-b67", "smpte2084") or bit_depth > 8
    is_raw = codec in ("rawvideo", "bayer_bggr8", "bayer_rggb8")

    return SourceProfile(
        uri=path,
        codec=codec,
        bit_depth=bit_depth,
        chroma=chroma,
        color_space=color_space,
        resolution=(width, height),
        fps=round(fps, 3),
        is_raw=is_raw,
        is_log=is_log,
        hdr=hdr,
        pixel_format=pix_fmt,
        container=container,
    )


def _parse_bit_depth(pix_fmt: str) -> int:
    """Infer bit depth from pixel format string."""
    if not pix_fmt:
        return 8
    if "10" in pix_fmt:
        return 10
    if "12" in pix_fmt:
        return 12
    if "16" in pix_fmt:
        return 16
    return 8


def _parse_chroma(pix_fmt: str) -> str:
    """Infer chroma subsampling from pixel format string."""
    if not pix_fmt:
        return "4:2:0"
    if "444" in pix_fmt:
        return "4:4:4"
    if "422" in pix_fmt:
        return "4:2:2"
    return "4:2:0"


def _infer_color_space(codec: str, bit_depth: int) -> str:
    """Best-guess color space from codec + bit depth."""
    if codec in ("prores", "dnxhd", "dnxhr"):
        return "rec709"
    if bit_depth >= 10:
        return "rec709"  # conservative default
    return "srgb"
