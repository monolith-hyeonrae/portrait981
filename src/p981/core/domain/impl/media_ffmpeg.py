"""FFmpeg 기반 미디어 구현체로 프레임/클립 추출을 담당한다."""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse
from urllib.request import urlopen
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from ...common import ObservationEvent
from ...ports import MediaPorts
from ...types import AssetRef, FrameSample, FrameSource, MediaHandle, TimeRange, VideoRef
from ..media import MediaService


class FFmpegMediaService(MediaService):
    """FFmpeg 기반 미디어 서비스 (로컬/다운로드 입력 지원)."""
    def __init__(
        self,
        ports: MediaPorts,
        ffmpeg_path: str = "ffmpeg",
    ) -> None:
        self._blob_store = ports.blob_store
        self._ffmpeg_path = ffmpeg_path
        self._tmp = tempfile.TemporaryDirectory(prefix="p981_media_")
        self._cache: dict[str, Path] = {}
        self._observer = ports.observer

    def register_video(self, video_ref: VideoRef) -> MediaHandle:
        self._ensure_local(video_ref)
        return video_ref

    def open_frame_source(
        self,
        media_handle: MediaHandle,
        fps: int,
        time_range: TimeRange | None = None,
        max_frames: int | None = None,
    ) -> FrameSource:
        return _FFmpegFrameSource(self, media_handle, fps, time_range, max_frames)

    def extract_keyframes(self, media_handle: MediaHandle, timestamps_ms: Sequence[int]) -> AssetRef:
        source_path = self._ensure_local(media_handle)
        frames_dir = self._make_frames_dir("frames")

        # 개별 프레임을 추출하고 관측 이벤트를 발행한다.
        frame_paths: list[Path] = []
        for index, timestamp_ms in enumerate(timestamps_ms):
            frame_paths.append(
                self._extract_frame(
                    source_path=source_path,
                    media_handle=media_handle,
                    frames_dir=frames_dir,
                    index=index,
                    timestamp_ms=timestamp_ms,
                )
            )

        return self._zip_frames(frame_paths)

    def extract_clip(self, media_handle: MediaHandle, time_range: TimeRange) -> AssetRef:
        source_path = self._ensure_local(media_handle)
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

    def _make_frames_dir(self, prefix: str) -> Path:
        """추출 프레임을 담을 임시 디렉터리를 만든다."""
        frames_dir = Path(self._tmp.name) / f"{prefix}_{uuid4().hex}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        return frames_dir

    def _extract_frame(
        self,
        *,
        source_path: Path,
        media_handle: MediaHandle,
        frames_dir: Path,
        index: int,
        timestamp_ms: int,
    ) -> Path:
        """지정된 타임스탬프의 단일 프레임을 추출한다."""
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
                    "video_ref": media_handle,
                    "frame_index": index,
                    "timestamp_ms": timestamp_ms,
                    "avg_luma": avg_luma,
                    "frame_path": str(output_path),
                },
                timestamp_ms=timestamp_ms,
            )
        )
        return output_path

    def _zip_frames(self, frame_paths: Sequence[Path]) -> AssetRef:
        """추출된 프레임을 zip으로 묶어 blob_ref를 반환한다."""
        buffer = io.BytesIO()
        with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as zip_file:
            for frame_path in sorted(frame_paths):
                zip_file.write(frame_path, arcname=frame_path.name)
        return self._blob_store.put(buffer.getvalue())

    def _ensure_local(self, video_ref: VideoRef) -> Path:
        """video_ref를 로컬 경로로 해석하고 필요 시 다운로드한다."""
        if video_ref in self._cache:
            return self._cache[video_ref]

        # video_ref를 로컬 경로로 해석하고 필요 시 다운로드/임시 저장한다.
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
        """바이트를 임시 파일로 저장하고 경로를 반환한다."""
        path = Path(self._tmp.name) / f"video_{uuid4().hex}{suffix}"
        path.write_bytes(data)
        return path

    def _download_http(self, url: str) -> Path:
        """HTTP로 원격 비디오를 다운로드한다."""
        path = Path(self._tmp.name) / f"video_{uuid4().hex}.mp4"
        with urlopen(url) as response, path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        return path

    def _download_s3(self, url: str) -> Path:
        """S3에서 원격 비디오를 다운로드한다."""
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
        """ffmpeg로 단일 프레임 평균 밝기를 계산한다."""
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
        """ffmpeg를 실행하고 실패 시 예외를 발생시킨다."""
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(f"ffmpeg failed: {stderr}")


class _FFmpegFrameSource:
    """ffmpeg로 샘플링된 프레임을 디스크에 생성하는 FrameSource."""
    def __init__(
        self,
        media: FFmpegMediaService,
        media_handle: MediaHandle,
        fps: int,
        time_range: TimeRange | None,
        max_frames: int | None,
    ) -> None:
        self._media = media
        self.media_handle = media_handle
        self.fps = fps
        self._time_range = time_range
        self._max_frames = max_frames
        self._frames: list[FrameSample] | None = None

    def iter_frames(self) -> Sequence[FrameSample]:
        if self._frames is not None:
            return list(self._frames)
        if self.fps <= 0:
            self._frames = []
            return []

        start_ms = self._time_range.start_ms if self._time_range else 0
        duration_ms = None
        if self._time_range is not None:
            duration_ms = max(0, self._time_range.end_ms - self._time_range.start_ms)

        source_path = self._media._ensure_local(self.media_handle)
        frames_dir = self._media._make_frames_dir("framesrc")

        command = [
            self._media._ffmpeg_path,
            "-y",
        ]
        if start_ms:
            command += ["-ss", f"{start_ms / 1000:.3f}"]
        if duration_ms is not None:
            command += ["-t", f"{duration_ms / 1000:.3f}"]
        command += [
            "-i",
            str(source_path),
            "-vf",
            f"fps={self.fps}",
        ]
        if self._max_frames is not None:
            command += ["-frames:v", str(self._max_frames)]
        command += [
            "-q:v",
            "2",
            str(frames_dir / "frame_%06d.jpg"),
        ]
        self._media._run_ffmpeg(command)

        frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
        step_ms = max(1, int(1000 / self.fps))
        frames = [
            FrameSample(
                frame_index=index,
                timestamp_ms=start_ms + index * step_ms,
                frame_path=str(path),
            )
            for index, path in enumerate(frame_paths)
        ]
        self._frames = frames
        return list(frames)
