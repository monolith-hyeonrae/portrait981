"""Clip extraction using ffmpeg."""

import subprocess
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from datetime import datetime

from visualbase.packaging.trigger import Trigger


@dataclass
class ClipResult:
    """Result of clip extraction.

    Attributes:
        success: Whether extraction succeeded.
        output_path: Path to the output clip file.
        trigger: The trigger that initiated this clip.
        duration_sec: Actual duration of the extracted clip.
        error: Error message if extraction failed.
    """

    success: bool
    output_path: Optional[Path]
    trigger: Trigger
    duration_sec: float = 0.0
    error: str = ""


class Clipper:
    """Extracts clips from video files using ffmpeg.

    Args:
        output_dir: Directory to save extracted clips.
        codec: Output codec ("copy" for stream copy, "h264" for re-encode).

    Example:
        >>> clipper = Clipper(output_dir=Path("/data/clips"))
        >>> trigger = Trigger.point(event_time_ns=10_000_000_000, pre_sec=3.0, post_sec=2.0)
        >>> result = clipper.extract(Path("video.mp4"), trigger)
        >>> if result.success:
        ...     print(f"Clip saved to: {result.output_path}")
    """

    def __init__(
        self,
        output_dir: Path,
        codec: str = "copy",
    ):
        self._output_dir = Path(output_dir)
        self._codec = codec
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def extract(
        self,
        source_path: Path,
        trigger: Trigger,
        output_filename: Optional[str] = None,
    ) -> ClipResult:
        """Extract a clip from the source video based on trigger.

        Args:
            source_path: Path to source video file.
            trigger: Trigger defining the clip boundaries.
            output_filename: Optional custom output filename.

        Returns:
            ClipResult with extraction status and output path.
        """
        source_path = Path(source_path)
        if not source_path.exists():
            return ClipResult(
                success=False,
                output_path=None,
                trigger=trigger,
                error=f"Source file not found: {source_path}",
            )

        # Generate output filename
        if output_filename is None:
            output_filename = self._generate_filename(source_path, trigger)

        output_path = self._output_dir / output_filename

        # Build ffmpeg command
        start_sec = trigger.clip_start_sec
        duration_sec = trigger.clip_duration_sec

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-ss", str(start_sec),  # Seek to start (before -i for fast seek)
            "-i", str(source_path),
            "-t", str(duration_sec),  # Duration
        ]

        # Codec settings
        if self._codec == "copy":
            cmd.extend(["-c", "copy"])
        else:
            cmd.extend(["-c:v", self._codec, "-c:a", "aac"])

        # Avoid negative timestamps
        cmd.extend(["-avoid_negative_ts", "make_zero"])

        cmd.append(str(output_path))

        # Execute ffmpeg
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return ClipResult(
                    success=False,
                    output_path=None,
                    trigger=trigger,
                    error=f"ffmpeg error: {result.stderr}",
                )

            # Verify output and get actual duration
            actual_duration = self._get_duration(output_path)

            return ClipResult(
                success=True,
                output_path=output_path,
                trigger=trigger,
                duration_sec=actual_duration,
            )

        except subprocess.TimeoutExpired:
            return ClipResult(
                success=False,
                output_path=None,
                trigger=trigger,
                error="ffmpeg timed out",
            )
        except FileNotFoundError:
            return ClipResult(
                success=False,
                output_path=None,
                trigger=trigger,
                error="ffmpeg not found. Please install ffmpeg.",
            )
        except Exception as e:
            return ClipResult(
                success=False,
                output_path=None,
                trigger=trigger,
                error=str(e),
            )

    def _generate_filename(self, source_path: Path, trigger: Trigger) -> str:
        """Generate output filename based on trigger info."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = trigger.label or "clip"
        score_str = f"_{trigger.score:.2f}" if trigger.score > 0 else ""
        start_sec = int(trigger.clip_start_sec)

        return f"{timestamp}_{label}{score_str}_t{start_sec}.mp4"

    def _get_duration(self, video_path: Path) -> float:
        """Get video duration using ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(video_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data.get("format", {}).get("duration", 0))
        except Exception:
            pass
        return 0.0

    @property
    def output_dir(self) -> Path:
        """Output directory for clips."""
        return self._output_dir
