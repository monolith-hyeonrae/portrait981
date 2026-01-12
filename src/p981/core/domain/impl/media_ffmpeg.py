from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse
from urllib.request import urlopen
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import re

from ...common import ObservationEvent
from ...ports import BlobStore, NoopObservationPort, ObservationPort
from ...types import AssetRef, TimeRange, VideoRef
from ..media import MediaService


class FFmpegMediaService(MediaService):
    def __init__(
        self,
        blob_store: BlobStore,
        ffmpeg_path: str = "ffmpeg",
        observer: ObservationPort | None = None,
    ) -> None:
        self._blob_store = blob_store
        self._ffmpeg_path = ffmpeg_path
        self._tmp = tempfile.TemporaryDirectory(prefix="p981_media_")
        self._cache: dict[str, Path] = {}
        self._observer = observer or NoopObservationPort()

    def register_video(self, video_ref: VideoRef) -> AssetRef:
        self._ensure_local(video_ref)
        return video_ref

    def extract_keyframes(self, video_ref: VideoRef, timestamps_ms: Sequence[int]) -> AssetRef:
        source_path = self._ensure_local(video_ref)
        frames_dir = Path(self._tmp.name) / f"frames_{uuid4().hex}"
        frames_dir.mkdir(parents=True, exist_ok=True)

        for index, timestamp_ms in enumerate(timestamps_ms):
            output_path = frames_dir / f"frame_{index:03d}_{timestamp_ms}.jpg"
            self._run_ffmpeg(
                [
                    self._ffmpeg_path,
                    "-y",
                    "-ss",
                    f"{timestamp_ms / 1000:.3f}",
                    "-i",
                    str(source_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(output_path),
                ]
            )
            avg_luma = self._frame_avg_luma(output_path)
            self._observer.emit(
                ObservationEvent(
                    kind="media.frame",
                    payload={
                        "video_ref": video_ref,
                        "frame_index": index,
                        "timestamp_ms": timestamp_ms,
                        "avg_luma": avg_luma,
                        "frame_path": str(output_path),
                    },
                    timestamp_ms=timestamp_ms,
                )
            )

        buffer = io.BytesIO()
        with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as zip_file:
            for frame_path in sorted(frames_dir.glob("*.jpg")):
                zip_file.write(frame_path, arcname=frame_path.name)
        return self._blob_store.put(buffer.getvalue())

    def extract_clip(self, video_ref: VideoRef, time_range: TimeRange) -> AssetRef:
        source_path = self._ensure_local(video_ref)
        output_path = Path(self._tmp.name) / f"clip_{uuid4().hex}.mp4"
        duration_ms = max(0, time_range.end_ms - time_range.start_ms)
        self._run_ffmpeg(
            [
                self._ffmpeg_path,
                "-y",
                "-ss",
                f"{time_range.start_ms / 1000:.3f}",
                "-t",
                f"{duration_ms / 1000:.3f}",
                "-i",
                str(source_path),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
        return self._blob_store.put(output_path.read_bytes())

    def _ensure_local(self, video_ref: VideoRef) -> Path:
        if video_ref in self._cache:
            return self._cache[video_ref]

        parsed = urlparse(video_ref)
        if parsed.scheme in {"http", "https"}:
            path = self._download_http(video_ref)
        elif parsed.scheme == "s3":
            path = self._download_s3(video_ref)
        elif parsed.scheme == "file":
            path = Path(parsed.path)
        elif Path(video_ref).exists():
            path = Path(video_ref)
        elif video_ref.startswith("blob:") or video_ref.startswith("blob_"):
            blob_ref = video_ref[len("blob:") :] if video_ref.startswith("blob:") else video_ref
            data = self._blob_store.get(blob_ref)
            path = self._write_temp(data, suffix=".mp4")
        else:
            raise ValueError(f"Unsupported video_ref: {video_ref}")

        self._cache[video_ref] = path
        return path

    def _write_temp(self, data: bytes, suffix: str) -> Path:
        path = Path(self._tmp.name) / f"video_{uuid4().hex}{suffix}"
        path.write_bytes(data)
        return path

    def _download_http(self, url: str) -> Path:
        path = Path(self._tmp.name) / f"video_{uuid4().hex}.mp4"
        with urlopen(url) as response, path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        return path

    def _download_s3(self, url: str) -> Path:
        parsed = urlparse(url)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            raise ValueError(f"Invalid s3 url: {url}")
        try:
            import boto3  # type: ignore
        except ImportError as exc:
            raise RuntimeError("boto3 is required to resolve s3:// video_ref") from exc
        path = Path(self._tmp.name) / f"video_{uuid4().hex}.mp4"
        boto3.client("s3").download_file(bucket, key, str(path))
        return path

    def _frame_avg_luma(self, frame_path: Path) -> float | None:
        command = [
            self._ffmpeg_path,
            "-hide_banner",
            "-i",
            str(frame_path),
            "-vf",
            "signalstats,metadata=print:file=-",
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        output = f"{result.stdout}\n{result.stderr}"
        match = re.search(r"lavfi\\.signalstats\\.YAVG=(\\d+(?:\\.\\d+)?)", output)
        if not match:
            return None
        return float(match.group(1))

    @staticmethod
    def _run_ffmpeg(command: list[str]) -> None:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(f"ffmpeg failed: {stderr}")
