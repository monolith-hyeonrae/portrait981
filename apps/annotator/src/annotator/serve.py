"""Local review server — 라벨 수정/삭제가 파일에 즉시 반영.

annotator review data/datasets/portrait-v1 --serve
→ localhost:8765 에서 리뷰 UI
→ 라벨 수정 → labels.csv 즉시 저장
→ 이미지 삭제 → images/ + labels.csv에서 즉시 제거

images/ 하위 디렉토리 지원:
  labels.csv filename은 파일명만 저장 (경로 없음).
  서버가 재귀 스캔하여 이름→경로 매핑.
  동일 파일명이 여러 폴더에 있으면 경고.
"""

from __future__ import annotations

import csv
import json
import logging
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

logger = logging.getLogger("annotator.serve")

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".avif", ".webp"}
_STATIC_DIR = Path(__file__).parent / "static"


def _scan_images(images_dir: Path) -> tuple[dict[str, Path], dict[str, list[str]]]:
    """Scan images/ recursively. Returns (name→path, duplicates: name→[rel_paths])."""
    name_to_path: dict[str, Path] = {}
    all_paths: dict[str, list[str]] = {}

    if not images_dir.exists():
        return name_to_path, {}

    for p in sorted(images_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in _IMG_EXTS:
            continue
        name = p.name
        rel = str(p.relative_to(images_dir))
        all_paths.setdefault(name, []).append(rel)
        if name not in name_to_path:
            name_to_path[name] = p

    duplicates = {k: v for k, v in all_paths.items() if len(v) > 1}
    return name_to_path, duplicates


class ReviewHandler(SimpleHTTPRequestHandler):
    """Handles API requests + serves static review HTML."""

    dataset_dir: Path
    labels_path: Path
    videos_path: Path
    images_dir: Path
    image_index: dict[str, Path]
    duplicates: dict[str, list[str]]

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._serve_page("review/review.html", {"dataset_dir": str(self.dataset_dir)})
        elif parsed.path.startswith("/static/"):
            self._serve_static(parsed.path[len("/static/"):])
        elif parsed.path == "/api/labels":
            self._send_json(self._read_labels())
        elif parsed.path == "/api/videos":
            self._send_json(self._read_videos())
        elif parsed.path == "/api/warnings":
            self._send_json(self._get_warnings())
        elif parsed.path == "/api/folders":
            self._send_json(self._get_folders())
        elif parsed.path.startswith("/api/image/"):
            fname = unquote(parsed.path[len("/api/image/"):])
            self._serve_image(fname)
        elif parsed.path.startswith("/api/video/"):
            fname = unquote(parsed.path[len("/api/video/"):])
            self._serve_video(fname)
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        body = self._read_body()

        if parsed.path == "/api/update_label":
            data = json.loads(body)
            self._update_label(data)
            self._send_json({"ok": True})

        elif parsed.path == "/api/update_video":
            data = json.loads(body)
            self._update_video(data)
            self._send_json({"ok": True})

        elif parsed.path == "/api/delete":
            data = json.loads(body)
            self._delete_image(data["filename"])
            self._send_json({"ok": True})

        elif parsed.path == "/api/delete_batch":
            data = json.loads(body)
            count = 0
            for fname in data.get("filenames", []):
                self._delete_image(fname)
                count += 1
            self._send_json({"ok": True, "deleted": count})

        else:
            self.send_error(404)

    # --- Utilities ---

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def _send_json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

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

    # --- Image/Video serving ---

    def _serve_image(self, fname: str):
        img_path = self.image_index.get(fname)
        if not img_path or not img_path.exists():
            self._refresh_index()
            img_path = self.image_index.get(fname)
        if not img_path or not img_path.exists():
            self.send_error(404)
            return
        data = img_path.read_bytes()
        suffix = img_path.suffix.lower()
        ct = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
              ".avif": "image/avif", ".webp": "image/webp"}.get(suffix, "image/jpeg")
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _serve_video(self, fname: str):
        videos_dir = self.dataset_dir / "videos"
        video_path = videos_dir / fname
        if not video_path.exists() or not str(video_path.resolve()).startswith(str(videos_dir.resolve())):
            self.send_error(404)
            return
        data = video_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    # --- CSV operations ---

    def _read_labels(self) -> list[dict]:
        if not self.labels_path.exists():
            return []
        with open(self.labels_path, newline="") as f:
            return list(csv.DictReader(f))

    def _read_videos(self) -> dict[str, dict]:
        if not self.videos_path.exists():
            return {}
        with open(self.videos_path, newline="") as f:
            return {r["workflow_id"]: r for r in csv.DictReader(f)}

    def _write_labels(self, rows: list[dict]):
        fieldnames = ["filename", "workflow_id", "expression", "pose", "moment", "source"]
        with open(self.labels_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})

    def _write_videos(self, videos: dict[str, dict]):
        fieldnames = ["workflow_id", "scene", "main_gender", "main_ethnicity",
                       "passenger_gender", "passenger_ethnicity", "member_id",
                       "source_video", "total_frames", "labeled_count", "summary", "notes"]
        with open(self.videos_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for v in videos.values():
                writer.writerow({k: v.get(k, "") for k in fieldnames})

    def _update_label(self, data: dict):
        rows = self._read_labels()
        fname = data["filename"]
        found = False
        for r in rows:
            if r["filename"] == fname:
                for k, v in data.items():
                    if k != "filename":
                        r[k] = v
                found = True
                break
        if not found:
            rows.append(data)
        self._write_labels(rows)
        logger.info("Label updated: %s → %s", fname, {k: v for k, v in data.items() if k != "filename"})

    def _update_video(self, data: dict):
        videos = self._read_videos()
        vid = data["workflow_id"]
        if vid in videos:
            videos[vid].update(data)
        else:
            videos[vid] = data
        self._write_videos(videos)
        logger.info("Video updated: %s", vid)

    def _refresh_index(self):
        """Rescan images directory and update index."""
        new_index, new_dupes = _scan_images(self.images_dir)
        self.__class__.image_index = new_index
        self.__class__.duplicates = new_dupes

    def _delete_image(self, fname: str):
        rows = self._read_labels()
        rows = [r for r in rows if r["filename"] != fname]
        self._write_labels(rows)
        img_path = self.image_index.pop(fname, None)
        if img_path and img_path.exists():
            img_path.unlink()
            logger.info("Deleted: %s", fname)
        else:
            logger.warning("File not found for deletion: %s", fname)

    def _get_warnings(self) -> list[str]:
        warnings = []
        for name, paths in self.duplicates.items():
            warnings.append(f"Duplicate filename '{name}' in: {', '.join(paths)}")
        return warnings

    def _get_folders(self) -> list[str]:
        """Return list of subfolder names under images/."""
        folders = set()
        for p in self.image_index.values():
            rel = p.relative_to(self.images_dir)
            if len(rel.parts) > 1:
                folders.add(str(rel.parent))
        return sorted(folders)

    def log_message(self, format, *args):
        pass


def start_server(dataset_dir: str | Path, port: int = 8765):
    """Start local review server."""
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    labels_path = dataset_dir / "labels.csv"
    videos_path = dataset_dir / "videos.csv"

    # Build image index (name → path), detect duplicates
    image_index, duplicates = _scan_images(images_dir)
    if duplicates:
        for name, paths in duplicates.items():
            logger.warning("Duplicate filename '%s' found in: %s", name, ", ".join(paths))
        logger.warning(
            "%d filename conflict(s) detected. "
            "Only the first occurrence is used. Rename to avoid label mismatch.",
            len(duplicates),
        )
    logger.info("Indexed %d images", len(image_index))

    # Register new images not yet in labels.csv
    if labels_path.exists():
        with open(labels_path, newline="") as f:
            existing = {r["filename"] for r in csv.DictReader(f)}
        new_files = [name for name in sorted(image_index) if name not in existing]
        if new_files:
            fieldnames = ["filename", "workflow_id", "expression", "pose", "moment", "source"]
            with open(labels_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                for fname in new_files:
                    writer.writerow({"filename": fname, "workflow_id": "", "expression": "",
                                     "pose": "", "moment": "", "source": "reference"})
            logger.info("Added %d new images to labels.csv", len(new_files))

    # Configure handler
    ReviewHandler.dataset_dir = dataset_dir
    ReviewHandler.labels_path = labels_path
    ReviewHandler.videos_path = videos_path
    ReviewHandler.images_dir = images_dir
    ReviewHandler.image_index = image_index
    ReviewHandler.duplicates = duplicates

    server = HTTPServer(("localhost", port), ReviewHandler)
    logger.info("Review server: http://localhost:%d", port)
    logger.info("Dataset: %s", dataset_dir)
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
