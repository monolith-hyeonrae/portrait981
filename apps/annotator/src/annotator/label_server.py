"""Server-based label tool — 라벨링 결과를 데이터셋에 직접 머지.

annotator label video.mp4 --dataset data/datasets/portrait-v1
→ localhost:8766 에서 라벨링 UI
→ 라벨은 staging (서버 메모리 + .staging.json)에 저장
→ Confirm Merge 시 → 이미지+라벨을 dataset에 머지

staging 파일: {dataset}/.staging_{video_name}.json
  - 서버 재시작해도 작업 이어가기 가능
  - merge 완료 후 자동 삭제
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import subprocess
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

logger = logging.getLogger("annotator.label_server")


def _extract_frames(video_path: Path, fps: int, max_frames: int) -> list[tuple[int, np.ndarray]]:
    """Extract frames from video. Returns [(frame_index, bgr_array), ...]."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0 and len(frames) < max_frames:
            frames.append((idx, frame.copy()))
        idx += 1
    cap.release()
    logger.info("Extracted %d frames (interval=%d)", len(frames), frame_interval)
    return frames


def _frame_to_jpeg(frame_bgr: np.ndarray, max_width: int = 640) -> bytes:
    """OpenCV frame -> JPEG bytes."""
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return bytes(buf)


def _compress_video(
    src: Path, dst: Path, max_height: int = 720, fps: int = 3, bitrate: str = "340k",
) -> bool:
    """Compress video with ffmpeg (2-pass ABR).

    Defaults: 720p / 3fps / 340kbps → ~4MB per 2min video.
    """
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not found — skipping video compression")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = src.resolve()
    dst = dst.resolve()
    cwd = str(dst.parent)
    vf = f"fps={fps},scale=-2:'min({max_height},ih)':flags=lanczos"
    pass1 = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264", "-b:v", bitrate, "-pass", "1",
        "-an", "-f", "null", "/dev/null",
    ]
    pass2 = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264", "-b:v", bitrate, "-pass", "2", "-preset", "slow",
        "-an", "-movflags", "+faststart",
        str(dst),
    ]
    success = False
    try:
        logger.info("Compressing video: %s → %s", src.name, dst.name)
        logger.info("Pass 1: %s", " ".join(pass1))
        result = subprocess.run(pass1, capture_output=True, timeout=300, cwd=cwd)
        if result.returncode != 0:
            logger.warning("ffmpeg pass 1 failed (rc=%d): %s", result.returncode, result.stderr.decode()[-300:])
            return False
        logger.info("Pass 1 done, starting pass 2...")
        result = subprocess.run(pass2, capture_output=True, timeout=300, cwd=cwd)
        if result.returncode == 0:
            src_mb = src.stat().st_size / 1e6
            dst_mb = dst.stat().st_size / 1e6
            logger.info("Video compressed: %.1fMB → %.1fMB (%s)", src_mb, dst_mb, dst.name)
            success = True
        else:
            logger.warning("ffmpeg pass 2 failed (rc=%d): %s", result.returncode, result.stderr.decode()[-300:])
    except Exception as e:
        logger.warning("Video compression failed: %s", e)
    finally:
        for f in dst.parent.glob("ffmpeg2pass-*"):
            f.unlink(missing_ok=True)
    return success


class LabelHandler(SimpleHTTPRequestHandler):
    """Handles label server API requests."""

    video_name: str
    video_stem: str
    video_path: Path
    dataset_dir: Path
    video_dir: Path | None  # directory with source videos
    fps: int
    max_frames: int
    frames: list[tuple[int, np.ndarray]]  # (original_idx, bgr)
    frame_jpegs: dict[int, bytes]  # frame_index -> jpeg bytes
    staging: dict  # {labels: {idx: expr}, poses: {idx: pose}, moments: {idx: moment}, video_meta: {...}}
    staging_path: Path

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_page("label/label.html", {
                "video_name": self.video_name,
                "frame_count": len(self.frames),
                "dataset_dir": str(self.dataset_dir),
            })
        elif parsed.path.startswith("/static/"):
            self._serve_static(parsed.path[len("/static/"):])
        elif parsed.path == "/api/frames":
            # Return frame metadata (without image data)
            info = [{"index": i, "original_idx": orig_idx}
                    for i, (orig_idx, _) in enumerate(self.frames)]
            self._send_json(info)
        elif parsed.path.startswith("/api/frame/"):
            idx = int(parsed.path.split("/")[-1])
            self._serve_frame(idx)
        elif parsed.path == "/api/staging":
            self._send_json(self.staging)
        elif parsed.path == "/api/preview":
            self._send_json(self._build_preview())
        elif parsed.path == "/api/video_list":
            self._send_json(self._get_video_list())
        elif parsed.path.startswith("/api/signal/"):
            idx = int(parsed.path.split("/")[-1])
            self._send_json(self._get_signal(idx))
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {}

        if parsed.path == "/api/save_label":
            idx = str(body["index"])
            # Handle reset
            if idx == "-1" and body.get("expression") == "__reset__":
                self.staging = {"labels": {}, "poses": {}, "lightings": {}, "moments": {}, "video_meta": self.staging.get("video_meta")}
                self._save_staging()
                self._send_json({"ok": True})
                return
            self.staging["labels"][idx] = body.get("expression", "")
            if "pose" in body:
                self.staging["poses"][idx] = body["pose"]
            if "lighting" in body:
                if "lightings" not in self.staging:
                    self.staging["lightings"] = {}
                self.staging["lightings"][idx] = body["lighting"]
            if "moment" in body:
                self.staging["moments"][idx] = body["moment"]
            self._save_staging()
            self._send_json({"ok": True})

        elif parsed.path == "/api/save_meta":
            self.staging["video_meta"] = body
            self._save_staging()
            self._send_json({"ok": True})

        elif parsed.path == "/api/confirm_merge":
            result = self._do_merge()
            self._send_json(result)

        elif parsed.path == "/api/load_video":
            result = self._load_video(body.get("filename"))
            self._send_json(result)

        else:
            self.send_error(404)

    # Signal cache: idx → signal dict (reset per server start)
    _signal_cache = {}
    _signal_analyzers = None
    _signal_load_attempted = False

    def _get_signal(self, idx: int) -> dict:
        """Extract key signals for a single frame (on-demand, cached)."""
        if idx in self._signal_cache:
            return self._signal_cache[idx]

        if idx < 0 or idx >= len(self.frames):
            return {}

        # Lazy-load analyzers on first call
        if not self.__class__._signal_load_attempted:
            self.__class__._signal_load_attempted = True
            import sys
            scripts_dir = str(Path(__file__).parents[4] / "scripts")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            try:
                from extract_signals import load_analyzers, extract_signals_from_image
                self.__class__._signal_analyzers = load_analyzers()
                self.__class__._extract_fn = extract_signals_from_image
                logger.info("Signal analyzers loaded for hints (%d)", len(self._signal_analyzers))
            except Exception:
                import traceback
                logger.warning("Cannot load signal analyzers:\n%s", traceback.format_exc())

        if not self._signal_analyzers:
            return {}

        _, frame_bgr = self.frames[idx]
        try:
            extract_fn = self.__class__._extract_fn
            raw = extract_fn(frame_bgr, self._signal_analyzers, idx)
            if not raw:
                logger.info("No face detected for frame %d", idx)
                return {}
            from visualbind.signals import normalize_signal
            result = {}
            for key in ("head_yaw_dev", "head_pitch", "em_happy", "em_neutral",
                         "face_confidence", "face_exposure", "mouth_open_ratio",
                         "eye_visible_ratio", "glasses_ratio", "backlight_score"):
                if key in raw:
                    result[key] = round(normalize_signal(raw[key], key), 3)
            self._signal_cache[idx] = result
            logger.info("Signal extracted for frame %d: %d keys", idx, len(result))
            return result
        except Exception as e:
            import traceback
            logger.warning("Signal extraction failed for frame %d: %s\n%s", idx, e, traceback.format_exc())
            return {}

    def _send_json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _serve_frame(self, idx: int):
        jpeg = self.frame_jpegs.get(idx)
        if jpeg is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", len(jpeg))
        self.end_headers()
        self.wfile.write(jpeg)

    def _get_video_list(self) -> dict:
        """List available videos from video_dir and dataset/videos/."""
        videos = []
        seen = set()

        # From video_dir
        if self.video_dir and self.video_dir.exists():
            for p in sorted(self.video_dir.iterdir()):
                if p.suffix.lower() == ".mp4" and p.name not in seen:
                    seen.add(p.name)
                    videos.append({
                        "filename": p.name,
                        "stem": p.stem,
                        "path": str(p),
                        "size_mb": round(p.stat().st_size / 1e6, 1),
                        "source": "video_dir",
                        "current": p.name == self.video_name,
                    })

        return {
            "videos": videos,
            "current": self.video_name,
            "video_dir": str(self.video_dir) if self.video_dir else None,
        }

    def _load_video(self, filename: str) -> dict:
        """Switch to a different video."""
        if not self.video_dir:
            return {"ok": False, "error": "No video directory configured"}

        video_path = self.video_dir / filename
        if not video_path.exists():
            return {"ok": False, "error": f"Video not found: {filename}"}

        # Save current staging before switching
        self._save_staging()

        # Extract frames from new video
        logger.info("Loading video: %s", filename)
        try:
            frames = _extract_frames(video_path, self.fps, self.max_frames)
        except Exception as e:
            return {"ok": False, "error": str(e)}

        if not frames:
            return {"ok": False, "error": "No frames extracted"}

        # Update handler state
        video_name = video_path.name
        video_stem = video_path.stem
        self.__class__.video_name = video_name
        self.__class__.video_stem = video_stem
        self.__class__.video_path = video_path
        self.__class__.frames = frames

        # Re-encode JPEGs
        frame_jpegs = {}
        for i, (orig_idx, frame_bgr) in enumerate(frames):
            frame_jpegs[i] = _frame_to_jpeg(frame_bgr)
        self.__class__.frame_jpegs = frame_jpegs

        # Load or init staging for new video
        staging_path = self.dataset_dir / f".staging_{video_stem}.json"
        self.__class__.staging_path = staging_path

        if staging_path.exists():
            staging = json.loads(staging_path.read_text())
            logger.info("Resumed staging: %s (%d labels)", staging_path, len(staging.get("labels", {})))
        else:
            staging = {"labels": {}, "poses": {}, "moments": {}, "video_meta": None}
            # Sync from dataset
            self._sync_from_dataset(staging, video_stem)
            if staging["labels"] or staging["video_meta"]:
                staging_path.write_text(json.dumps(staging, ensure_ascii=False, indent=2))

        self.__class__.staging = staging
        self.__class__._signal_cache = {}  # clear cache on video switch
        logger.info("Switched to %s (%d frames)", video_name, len(frames))

        return {
            "ok": True,
            "video_name": video_name,
            "frame_count": len(frames),
            "existing_labels": len(staging.get("labels", {})),
        }

    def _sync_from_dataset(self, staging: dict, video_stem: str):
        """Sync labels and video meta from existing dataset."""
        labels_path = self.dataset_dir / "labels.csv"
        videos_path = self.dataset_dir / "videos.csv"

        if labels_path.exists():
            with open(labels_path, newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("workflow_id") != video_stem:
                        continue
                    fname = r["filename"]
                    prefix = f"{video_stem}_"
                    if not fname.startswith(prefix):
                        continue
                    try:
                        idx = str(int(fname[len(prefix):].split(".")[0]))
                    except ValueError:
                        continue
                    if r.get("expression"):
                        staging["labels"][idx] = r["expression"]
                    if r.get("pose"):
                        staging["poses"][idx] = r["pose"]
                    if r.get("moment"):
                        staging["moments"][idx] = r["moment"]

        if videos_path.exists():
            with open(videos_path, newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("workflow_id") == video_stem:
                        staging["video_meta"] = {k: v for k, v in r.items() if v}
                        break

    def _save_staging(self):
        self.staging_path.write_text(json.dumps(self.staging, ensure_ascii=False, indent=2))

    def _build_preview(self) -> dict:
        """Build merge preview: what will be added to dataset."""
        labels = self.staging.get("labels", {})
        poses = self.staging.get("poses", {})
        moments = self.staging.get("moments", {})
        meta = self.staging.get("video_meta")

        items = []
        for idx_str, expr in labels.items():
            if not expr or expr == "__shoot__":
                continue
            items.append({
                "index": int(idx_str),
                "filename": f"{self.video_stem}_{int(idx_str):04d}.jpg",
                "expression": expr,
                "pose": poses.get(idx_str, ""),
                "moment": moments.get(idx_str, ""),
            })

        # Bucket counts
        buckets = {}
        for item in items:
            key = f"{item['expression']}|{item['pose'] or '(none)'}"
            buckets[key] = buckets.get(key, 0) + 1

        return {
            "total": len(items),
            "items": items,
            "buckets": buckets,
            "video_meta": meta,
            "dataset_dir": str(self.dataset_dir),
        }

    def _do_merge(self) -> dict:
        """Merge staging into dataset."""
        labels_data = self.staging.get("labels", {})
        poses_data = self.staging.get("poses", {})
        moments_data = self.staging.get("moments", {})
        meta = self.staging.get("video_meta")

        images_dir = self.dataset_dir / "images"
        labels_path = self.dataset_dir / "labels.csv"
        videos_path = self.dataset_dir / "videos.csv"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Collect items to merge (skip __shoot__ and empty)
        merge_items = []
        for idx_str, expr in labels_data.items():
            if not expr or expr == "__shoot__":
                continue
            idx = int(idx_str)
            fname = f"{self.video_stem}_{idx:04d}.jpg"
            merge_items.append({
                "index": idx,
                "filename": fname,
                "expression": expr,
                "pose": poses_data.get(idx_str, ""),
                "moment": moments_data.get(idx_str, ""),
            })

        if not merge_items:
            return {"ok": False, "error": "No labeled frames to merge"}

        # Save images
        for item in merge_items:
            jpeg = self.frame_jpegs.get(item["index"])
            if jpeg:
                # Save full resolution
                orig_idx, frame_bgr = self.frames[item["index"]]
                _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (images_dir / item["filename"]).write_bytes(bytes(buf))

        # Read existing labels
        fieldnames = ["filename", "workflow_id", "expression", "pose", "moment", "source"]
        existing_rows = []
        existing_fnames = set()
        if labels_path.exists():
            with open(labels_path, newline="") as f:
                existing_rows = list(csv.DictReader(f))
                existing_fnames = {r["filename"] for r in existing_rows}

        # Update existing labels or append new ones
        merge_map = {}
        for item in merge_items:
            merge_map[item["filename"]] = {
                "filename": item["filename"],
                "workflow_id": self.video_stem,
                "expression": item["expression"],
                "pose": item["pose"],
                "moment": item["moment"],
                "source": "operational",
            }

        updated = 0
        for r in existing_rows:
            if r["filename"] in merge_map:
                r.update(merge_map.pop(r["filename"]))
                updated += 1

        new_rows = list(merge_map.values())
        all_rows = existing_rows + new_rows
        if updated:
            logger.info("Updated %d existing labels, added %d new", updated, len(new_rows))
        with open(labels_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})

        # Compress and save video
        videos_dir = self.dataset_dir / "videos"
        compressed_name = f"{self.video_stem}.mp4"
        compressed_path = videos_dir / compressed_name
        video_saved = False
        logger.info("About to compress: src=%s dst=%s", self.video_path, compressed_path)
        try:
            video_saved = _compress_video(self.video_path, compressed_path)
        except Exception as e:
            logger.warning("Compress call failed: %s", e)

        # Build expression summary
        expr_counts = {}
        for item in merge_items:
            e = item["expression"]
            expr_counts[e] = expr_counts.get(e, 0) + 1
        summary_str = " ".join(f"{k}={v}" for k, v in sorted(expr_counts.items()))

        # Save video metadata + summary
        video_fieldnames = ["workflow_id", "scene", "main_gender", "main_ethnicity",
                           "passenger_gender", "passenger_ethnicity", "member_id",
                           "source_video", "total_frames", "labeled_count", "summary", "notes"]
        wf_id = meta.get("workflow_id", self.video_stem) if meta else self.video_stem
        video_row = dict(meta) if meta else {}
        video_row.update({
            "workflow_id": wf_id,
            "source_video": compressed_name if video_saved else self.video_name,
            "total_frames": str(len(self.frames)),
            "labeled_count": str(len(merge_items)),
            "summary": summary_str,
        })

        videos = {}
        if videos_path.exists():
            with open(videos_path, newline="") as f:
                videos = {r["workflow_id"]: r for r in csv.DictReader(f)}
        videos[wf_id] = video_row
        with open(videos_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=video_fieldnames)
            writer.writeheader()
            for v in videos.values():
                writer.writerow({k: v.get(k, "") for k in video_fieldnames})

        # Re-sync staging from dataset (merged labels are now in CSV)
        if self.staging_path.exists():
            self.staging_path.unlink()
        new_staging = {
            "labels": {}, "poses": {}, "moments": {},
            "video_meta": self.staging.get("video_meta"),
        }
        self._sync_from_dataset(new_staging, self.video_stem)
        self.__class__.staging = new_staging
        logger.info("Staging re-synced: %d labels from dataset", len(new_staging["labels"]))

        logger.info("Merged %d images + %d new labels into %s",
                     len(merge_items), len(new_rows), self.dataset_dir)

        return {
            "ok": True,
            "merged_images": len(merge_items),
            "new_labels": len(new_rows),
            "updated_labels": updated,
            "video_saved": video_saved,
            "video_path": str(compressed_path) if video_saved else None,
        }

    def _serve_page(self, rel_path: str, config: dict):
        html_path = _STATIC_DIR / rel_path
        if not html_path.exists():
            self.send_error(404)
            return
        html = html_path.read_text("utf-8")
        html = html.replace("__CONFIG_PLACEHOLDER__", json.dumps(config, ensure_ascii=False))
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, rel_path: str):
        file_path = (_STATIC_DIR / rel_path).resolve()
        if not str(file_path).startswith(str(_STATIC_DIR.resolve())) or not file_path.exists():
            self.send_error(404)
            return
        ct = {".html": "text/html", ".css": "text/css",
              ".js": "application/javascript"}.get(file_path.suffix, "application/octet-stream")
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{ct}; charset=utf-8")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        # Log errors, suppress normal requests
        if args and str(args[0]).startswith(("4", "5")):
            logger.warning(fmt, *args)



_STATIC_DIR = Path(__file__).parent / "static"


def start_label_server(
    video_path: str | Path,
    dataset_dir: str | Path,
    fps: int = 2,
    max_frames: int = 500,
    port: int = 8766,
    video_dir: str | Path | None = None,
):
    """Start label server for a video targeting a dataset."""
    video_path = Path(video_path)
    dataset_dir = Path(dataset_dir)
    video_name = video_path.name
    video_stem = video_path.stem

    # Extract frames
    logger.info("Extracting frames from %s (fps=%d, max=%d)", video_path, fps, max_frames)
    frames = _extract_frames(video_path, fps, max_frames)
    if not frames:
        raise RuntimeError("No frames extracted from video")

    # Pre-encode JPEG for serving
    frame_jpegs = {}
    for i, (orig_idx, frame_bgr) in enumerate(frames):
        frame_jpegs[i] = _frame_to_jpeg(frame_bgr)
    logger.info("Encoded %d frame thumbnails", len(frame_jpegs))

    # Load or init staging
    staging_path = dataset_dir / f".staging_{video_stem}.json"
    if staging_path.exists():
        staging = json.loads(staging_path.read_text())
        logger.info("Resumed staging: %s (%d labels)", staging_path, len(staging.get("labels", {})))
    else:
        staging = {"labels": {}, "poses": {}, "moments": {}, "video_meta": None}

        # Sync from existing dataset
        labels_path = dataset_dir / "labels.csv"
        videos_path = dataset_dir / "videos.csv"

        # Import existing frame labels for this workflow_id
        if labels_path.exists():
            with open(labels_path, newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("workflow_id") != video_stem:
                        continue
                    # Parse frame index from filename: {stem}_{idx:04d}.jpg
                    fname = r["filename"]
                    prefix = f"{video_stem}_"
                    if not fname.startswith(prefix):
                        continue
                    try:
                        idx_str = fname[len(prefix):].split(".")[0]
                        idx = str(int(idx_str))  # normalize "0012" → "12"
                    except ValueError:
                        continue
                    if r.get("expression"):
                        staging["labels"][idx] = r["expression"]
                    if r.get("pose"):
                        staging["poses"][idx] = r["pose"]
                    if r.get("moment"):
                        staging["moments"][idx] = r["moment"]

            synced = len(staging["labels"])
            if synced:
                logger.info("Synced %d existing labels from dataset for %s", synced, video_stem)

        # Import existing video metadata
        if videos_path.exists():
            with open(videos_path, newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("workflow_id") == video_stem:
                        staging["video_meta"] = {k: v for k, v in r.items() if v}
                        logger.info("Synced video metadata for %s", video_stem)
                        break

        # Save synced staging
        if staging["labels"] or staging["video_meta"]:
            staging_path.write_text(json.dumps(staging, ensure_ascii=False, indent=2))

    if video_dir:
        video_dir = Path(video_dir)

    # Configure handler
    LabelHandler.video_name = video_name
    LabelHandler.video_stem = video_stem
    LabelHandler.video_path = video_path
    LabelHandler.dataset_dir = dataset_dir
    LabelHandler.video_dir = video_dir
    LabelHandler.fps = fps
    LabelHandler.max_frames = max_frames
    LabelHandler.frames = frames
    LabelHandler.frame_jpegs = frame_jpegs
    LabelHandler.staging = staging
    LabelHandler.staging_path = staging_path

    server = HTTPServer(("localhost", port), LabelHandler)
    logger.info("Label server: http://localhost:%d", port)
    logger.info("Video: %s (%d frames)", video_name, len(frames))
    logger.info("Dataset: %s", dataset_dir)
    logger.info("Staging: %s", staging_path)
    logger.info("Press Ctrl+C to stop")

    try:
        import webbrowser
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped")
        server.server_close()
