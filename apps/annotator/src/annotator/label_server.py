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
    try:
        result = subprocess.run(pass1, capture_output=True, timeout=300, cwd=str(dst.parent))
        if result.returncode != 0:
            logger.warning("ffmpeg pass 1 failed: %s", result.stderr.decode()[-200:])
            return False
        result = subprocess.run(pass2, capture_output=True, timeout=300, cwd=str(dst.parent))
        if result.returncode == 0:
            src_mb = src.stat().st_size / 1e6
            dst_mb = dst.stat().st_size / 1e6
            logger.info("Video compressed: %.1fMB → %.1fMB (%s)", src_mb, dst_mb, dst.name)
            return True
        logger.warning("ffmpeg pass 2 failed: %s", result.stderr.decode()[-200:])
    except Exception as e:
        logger.warning("Video compression failed: %s", e)
    finally:
        # Clean up 2-pass log files
        for f in dst.parent.glob("ffmpeg2pass-*"):
            f.unlink(missing_ok=True)
    return False


class LabelHandler(SimpleHTTPRequestHandler):
    """Handles label server API requests."""

    video_name: str
    video_stem: str
    video_path: Path
    dataset_dir: Path
    frames: list[tuple[int, np.ndarray]]  # (original_idx, bgr)
    frame_jpegs: dict[int, bytes]  # frame_index -> jpeg bytes
    staging: dict  # {labels: {idx: expr}, poses: {idx: pose}, chems: {idx: chem}, video_meta: {...}}
    staging_path: Path

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_html()
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
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))

        if parsed.path == "/api/save_label":
            idx = str(body["index"])
            # Handle reset
            if idx == "-1" and body.get("expression") == "__reset__":
                self.staging = {"labels": {}, "poses": {}, "chems": {}, "video_meta": self.staging.get("video_meta")}
                self._save_staging()
                self._send_json({"ok": True})
                return
            self.staging["labels"][idx] = body.get("expression", "")
            if "pose" in body:
                self.staging["poses"][idx] = body["pose"]
            if "chemistry" in body:
                self.staging["chems"][idx] = body["chemistry"]
            self._save_staging()
            self._send_json({"ok": True})

        elif parsed.path == "/api/save_meta":
            self.staging["video_meta"] = body
            self._save_staging()
            self._send_json({"ok": True})

        elif parsed.path == "/api/confirm_merge":
            result = self._do_merge()
            self._send_json(result)

        else:
            self.send_error(404)

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

    def _save_staging(self):
        self.staging_path.write_text(json.dumps(self.staging, ensure_ascii=False, indent=2))

    def _build_preview(self) -> dict:
        """Build merge preview: what will be added to dataset."""
        labels = self.staging.get("labels", {})
        poses = self.staging.get("poses", {})
        chems = self.staging.get("chems", {})
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
                "chemistry": chems.get(idx_str, ""),
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
        chems_data = self.staging.get("chems", {})
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
                "chemistry": chems_data.get(idx_str, ""),
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
        fieldnames = ["filename", "workflow_id", "expression", "pose", "chemistry", "source"]
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
                "chemistry": item["chemistry"],
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
        video_saved = _compress_video(self.video_path, compressed_path)

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

        # Remove staging file
        if self.staging_path.exists():
            self.staging_path.unlink()
            logger.info("Staging file removed: %s", self.staging_path)

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

    def _serve_html(self):
        html = _build_label_html(self.video_name, len(self.frames), str(self.dataset_dir))
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def _build_label_html(video_name: str, frame_count: int, dataset_dir: str = "") -> str:
    """Build the label UI HTML.

    Key difference from label.py:
    - Images loaded from server (/api/frame/{idx}) instead of inline base64
    - Labels saved to server (/api/save_label) instead of localStorage
    - "Confirm Merge" instead of "Export ZIP"
    """
    # We reuse the same styling and keyboard flow from label.py
    # but swap localStorage for server API calls
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Label Tool — {video_name}</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, sans-serif; margin: 0; background: #f5f5f5; color: #333;
    display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}
.toolbar {{ background: #fff; padding: 10px 20px; border-bottom: 2px solid #e94560;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap; flex-shrink: 0; }}
.toolbar h1 {{ margin: 0; font-size: 18px; color: #e94560; }}
.progress {{ font-size: 14px; color: #888; }}
.progress b {{ color: #4CAF50; font-size: 18px; }}
.toolbar button {{ padding: 6px 14px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; }}
.btn-merge {{ background: #4CAF50; color: #fff; }}
.btn-merge:disabled {{ opacity: 0.4; cursor: default; }}
.btn-reset {{ background: #999; color: #fff; }}
.filter-group {{ display: flex; gap: 6px; }}
.filter-btn {{ background: #e8e8e8; color: #555; padding: 4px 10px; border: 1px solid #ccc;
    border-radius: 3px; cursor: pointer; font-size: 12px; }}
.filter-btn.active {{ background: #e94560; color: #fff; border-color: #e94560; }}
.main {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
.focus-panel {{ flex: 1; display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 20px; min-width: 0; }}
.focus-img {{ max-width: 100%; max-height: 55vh; border-radius: 8px; object-fit: contain;
    box-shadow: 0 2px 12px rgba(0,0,0,0.1); }}
.focus-meta {{ margin: 12px 0 6px; font-size: 14px; color: #888; }}
.focus-label {{ font-size: 16px; color: #4CAF50; margin: 8px 0; font-weight: bold; }}
.buttons {{ display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; margin-top: 10px; }}
.cat-btn {{ padding: 8px 18px; border: 2px solid #ddd; border-radius: 6px; background: #fff;
    color: #666; cursor: pointer; font-size: 14px; transition: all .15s; }}
.cat-btn:hover {{ background: #f0f0f0; color: #333; transform: scale(1.05); }}
.cat-btn.selected {{ color: #fff; font-weight: bold; }}
.nav {{ display: flex; gap: 12px; margin-top: 14px; align-items: center; }}
.nav button {{ padding: 10px 24px; border: none; border-radius: 6px; background: #e8e8e8;
    color: #444; cursor: pointer; font-size: 16px; }}
.nav button:hover {{ background: #ddd; }}
.nav button:disabled {{ opacity: 0.3; cursor: default; }}
.nav .pos {{ font-size: 14px; color: #888; min-width: 80px; text-align: center; }}
.strip {{ display: flex; overflow-x: auto; background: #fff; border-top: 1px solid #e0e0e0;
    flex-shrink: 0; padding: 4px; gap: 3px; }}
.thumb {{ flex-shrink: 0; cursor: pointer; opacity: 0.5; transition: opacity .15s;
    border-bottom: 3px solid transparent; }}
.thumb:hover {{ opacity: 0.8; }}
.thumb.active {{ opacity: 1; border-bottom-color: #e94560; }}
.thumb img {{ height: 56px; border-radius: 3px; display: block; }}
.shortcut-hint {{ font-size: 11px; color: #aaa; text-align: center; margin-top: 8px; }}
.bucket-bar {{ padding: 6px 20px; background: #fff; border-bottom: 1px solid #e0e0e0; flex-shrink: 0; }}
.timeline-bar {{ display: flex; height: 16px; background: #e8e8e8; border-bottom: 1px solid #ddd; flex-shrink: 0; cursor: pointer; }}
.timeline-bar .tl-frame {{ flex: 1; min-width: 1px; transition: opacity 0.1s; }}
.bucket-grid {{ display: grid; gap: 2px; }}
.bucket-cell {{ display: flex; align-items: center; justify-content: center;
    height: 28px; border-radius: 4px; font-size: 12px; font-weight: 600; position: relative; overflow: hidden;
    transition: transform 0.1s; }}
.bucket-cell:hover {{ transform: scale(1.06); z-index: 1; }}
.bucket-cell .bc-fill {{ position: absolute; inset: 0; border-radius: 4px; }}
.bucket-cell .bc-num {{ position: relative; z-index: 1; }}
.modal-overlay {{ position: fixed; bottom: 0; left: 0; width: 100%;
    background: rgba(0,0,0,0.5); z-index: 9999; display: flex; justify-content: center; padding: 12px 0; }}
.modal-box {{ background: #fff; border: 2px solid #e94560; border-radius: 12px 12px 0 0;
    padding: 20px 32px; min-width: 360px; max-width: 560px; color: #333; }}
.modal-box h2 {{ margin: 0 0 20px; color: #e94560; font-size: 20px; }}
.modal-row {{ margin: 12px 0; }}
.modal-row label {{ display: block; font-size: 13px; color: #888; margin-bottom: 6px; }}
.modal-opts {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.modal-opt {{ padding: 6px 16px; border: 2px solid #ddd; border-radius: 6px; background: #f5f5f5;
    color: #666; cursor: pointer; font-size: 13px; transition: all .15s; }}
.modal-opt:hover {{ background: #e8e8e8; color: #333; }}
.modal-opt.selected {{ background: #e94560; color: #fff; border-color: #e94560; }}
.modal-start {{ margin-top: 24px; padding: 10px 32px; border: none; border-radius: 6px;
    background: #4CAF50; color: #fff; font-size: 15px; cursor: pointer; width: 100%; }}
.modal-start:hover {{ background: #45a049; }}
.modal-start:disabled {{ opacity: 0.4; cursor: default; }}
.video-meta-indicator {{ font-size: 12px; color: #888; background: #f0f0f0; padding: 3px 10px;
    border-radius: 3px; border: 1px solid #ddd; }}
.btn-meta {{ background: #e8e8e8; color: #666; font-size: 12px; padding: 4px 10px; }}
/* Confirm overlay */
.confirm-overlay {{ position: fixed; inset: 0; background: rgba(0,0,0,0.5); z-index: 10000;
    display: flex; align-items: center; justify-content: center; }}
.confirm-box {{ background: #fff; border: 2px solid #4CAF50; border-radius: 12px;
    padding: 28px 36px; max-width: 600px; width: 90%; max-height: 80vh; overflow-y: auto; color: #333; }}
.confirm-box h2 {{ margin: 0 0 16px; color: #4CAF50; }}
.confirm-summary {{ font-size: 13px; color: #555; line-height: 1.6; }}
.confirm-btns {{ display: flex; gap: 12px; margin-top: 20px; }}
.confirm-btns button {{ flex: 1; padding: 12px; border: none; border-radius: 6px; font-size: 15px; cursor: pointer; }}
</style>
</head><body>

<div class="toolbar">
    <h1>Label Tool</h1>
    <div class="progress">
        <b id="count">0</b> / <span id="total">{frame_count}</span> labeled
        &nbsp;| {video_name}
    </div>
    <span style="font-size:11px;color:#999;background:#f0f0f0;padding:2px 8px;border-radius:3px;border:1px solid #ddd">{dataset_dir}</span>
    <div class="filter-group">
        <button class="filter-btn active" data-filter="all">All</button>
        <button class="filter-btn" data-filter="unlabeled">Unlabeled</button>
        <button class="filter-btn" data-filter="labeled">Labeled</button>
    </div>
    <span class="video-meta-indicator" id="metaIndicator"></span>
    <button class="btn-meta" onclick="showMetaModal()">Edit Meta</button>
    <button class="btn-merge" id="mergeBtn" onclick="showConfirm()">Confirm Merge</button>
    <button class="btn-reset" onclick="resetLabels()">Reset All</button>
</div>

<div class="bucket-bar" id="bucketBar"></div>
<div class="timeline-bar" id="timelineBar"></div>
<div class="main">
    <div class="focus-panel" id="focus"></div>
    <div class="strip" id="strip"></div>
</div>

<div class="modal-overlay" id="metaModal" style="display:none">
    <div class="modal-box">
        <h2>Video Setup: {video_name}</h2>
        <div class="modal-row"><label>Scene</label>
            <div class="modal-opts" id="metaScene">
                <div class="modal-opt" data-value="solo" onclick="selectMetaOpt('scene','solo')"><span style="opacity:0.5;font-size:11px">1 </span>solo</div>
                <div class="modal-opt" data-value="duo" onclick="selectMetaOpt('scene','duo')"><span style="opacity:0.5;font-size:11px">2 </span>duo</div>
            </div>
        </div>
        <div class="modal-row"><label>Main Gender</label>
            <div class="modal-opts" id="metaMain_gender">
                <div class="modal-opt" data-value="male" onclick="selectMetaOpt('main_gender','male')"><span style="opacity:0.5;font-size:11px">1 </span>male</div>
                <div class="modal-opt" data-value="female" onclick="selectMetaOpt('main_gender','female')"><span style="opacity:0.5;font-size:11px">2 </span>female</div>
            </div>
        </div>
        <div class="modal-row"><label>Main Ethnicity</label>
            <div class="modal-opts" id="metaMain_ethnicity">
                <div class="modal-opt" data-value="asian" onclick="selectMetaOpt('main_ethnicity','asian')"><span style="opacity:0.5;font-size:11px">1 </span>asian</div>
                <div class="modal-opt" data-value="western" onclick="selectMetaOpt('main_ethnicity','western')"><span style="opacity:0.5;font-size:11px">2 </span>western</div>
                <div class="modal-opt" data-value="other" onclick="selectMetaOpt('main_ethnicity','other')"><span style="opacity:0.5;font-size:11px">3 </span>other</div>
            </div>
        </div>
        <div class="modal-row" id="passengerGenderRow" style="display:none"><label>Passenger Gender</label>
            <div class="modal-opts" id="metaPassenger_gender">
                <div class="modal-opt" data-value="male" onclick="selectMetaOpt('passenger_gender','male')"><span style="opacity:0.5;font-size:11px">1 </span>male</div>
                <div class="modal-opt" data-value="female" onclick="selectMetaOpt('passenger_gender','female')"><span style="opacity:0.5;font-size:11px">2 </span>female</div>
            </div>
        </div>
        <div class="modal-row" id="passengerEthnicityRow" style="display:none"><label>Passenger Ethnicity</label>
            <div class="modal-opts" id="metaPassenger_ethnicity">
                <div class="modal-opt" data-value="asian" onclick="selectMetaOpt('passenger_ethnicity','asian')"><span style="opacity:0.5;font-size:11px">1 </span>asian</div>
                <div class="modal-opt" data-value="western" onclick="selectMetaOpt('passenger_ethnicity','western')"><span style="opacity:0.5;font-size:11px">2 </span>western</div>
                <div class="modal-opt" data-value="other" onclick="selectMetaOpt('passenger_ethnicity','other')"><span style="opacity:0.5;font-size:11px">3 </span>other</div>
            </div>
        </div>
        <div class="modal-row"><label>Member ID (optional)</label>
            <input type="text" id="metaMemberId" placeholder="예: park_042"
                style="background:#fff;border:1px solid #ccc;color:#333;padding:6px 12px;border-radius:4px;width:100%;font-size:14px"
                oninput="metaDraft.member_id=this.value">
        </div>
        <button class="modal-start" id="metaStartBtn" onclick="saveMetaAndStart()" disabled>Start Labeling</button>
    </div>
</div>

<div class="confirm-overlay" id="confirmOverlay" style="display:none">
    <div class="confirm-box">
        <h2>Merge Preview</h2>
        <div class="confirm-summary" id="confirmSummary">Loading...</div>
        <div class="confirm-btns">
            <button style="background:#e0e0e0;color:#555" onclick="hideConfirm()">Back to Edit</button>
            <button style="background:#4CAF50;color:#fff" id="confirmBtn" onclick="doMerge()">Confirm Merge</button>
        </div>
    </div>
</div>

<script>
const FRAME_COUNT = {frame_count};
const VIDEO_NAME = "{video_name}";
const VIDEO_STEM = VIDEO_NAME.replace(/\\.[^.]+$/, '');
const COLORS = {{
    cheese:'#4CAF50', goofy:'#E91E63', chill:'#2196F3', edge:'#FF5722', hype:'#9C27B0',
    cut:'#d32f2f', occluded:'#795548', front:'#00BCD4', angle:'#FF9800', side:'#795548',
    sync:'#FFD700', interact:'#00E676',
}};
const EXPRESSIONS = ['cheese','goofy','chill','edge','hype','occluded'];
const POSES_LIST = ['front','angle','side'];
const DESC = {{
    '__shoot__': 'SHOOT', 'cut': 'CUT',
    'cheese': '\uc5bc\uad74\uc774 \uc8fc\uc778\uacf5 \u2014 \ud504\ub85c\ud544 \uc0ac\uc9c4', 'goofy': '\uc7a5\ub09c\uc2a4\ub7ec\uc6b4 \ud45c\uc815 \u2014 \ud600 \ub0b4\ubc00\uae30, \uc719\ud06c',
    'chill': '\ucfe8\ud558\uace0 \uc5ec\uc720\ub85c\uc6b4', 'edge': '\ub0a0\uce74\ub86d\uace0 \uac15\ub82c\ud55c', 'hype': '\uc21c\uac04\uc774 \uc8fc\uc778\uacf5 \u2014 \uc5d0\ub108\uc9c0 \ud3ed\ubc1c',
    'occluded': '\uc5bc\uad74 \uac00\ub824\uc9d0', 'front': '\uc815\uba74', 'angle': '3/4', 'side': '\uce21\uba74',
    'sync': '\ub3d9\uc2dc \ubc18\uc751', 'interact': '\uad50\uac10',
}};

let labels = {{}};
let poses = {{}};
let chemistries = {{}};
let videoMeta = null;
let metaDraft = {{}};
let currentPos = 0;
let currentFilter = 'all';
let filteredList = [];
let manualStep = null;
let modalField = 0;
let saving = false;

function getColor(c) {{ return COLORS[c] || '#666'; }}

// --- Server communication ---
async function loadStaging() {{
    const data = await (await fetch('/api/staging')).json();
    labels = data.labels || {{}};
    poses = data.poses || {{}};
    chemistries = data.chems || {{}};
    videoMeta = data.video_meta || null;
    buildFilteredList();
    renderAll();
    updateMetaIndicator();
    if (!videoMeta) showMetaModal();
}}

async function saveToServer(index) {{
    if (saving) return;
    saving = true;
    try {{
        await fetch('/api/save_label', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{
                index: index,
                expression: labels[index] || '',
                pose: poses[index] || '',
                chemistry: chemistries[index] || '',
            }}),
        }});
    }} finally {{ saving = false; }}
}}

async function saveMetaToServer() {{
    await fetch('/api/save_meta', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(videoMeta),
    }});
}}

// --- Frame list ---
function buildFilteredList() {{
    filteredList = [];
    for (let i = 0; i < FRAME_COUNT; i++) {{
        const lbl = labels[i];
        if (currentFilter === 'unlabeled' && lbl !== undefined) continue;
        if (currentFilter === 'labeled' && lbl === undefined) continue;
        filteredList.push(i);
    }}
    filteredList.sort((a, b) => a - b);
}}

// --- Rendering ---
function renderAll() {{
    renderFocus();
    renderStrip();
    renderTimeline();
    updateCount();
}}

function renderStrip() {{
    const sb = document.getElementById('strip');
    sb.innerHTML = '';
    filteredList.forEach((idx, pos) => {{
        const div = document.createElement('div');
        const lbl = labels[idx];
        const isLabeled = lbl !== undefined;
        const borderColor = isLabeled ? (getColor(lbl) || '#4CAF50') : 'transparent';
        div.className = 'thumb' + (pos === currentPos ? ' active' : '');
        div.style.borderBottomColor = borderColor;
        div.style.opacity = isLabeled ? '0.7' : (pos === currentPos ? '1' : '0.5');
        div.innerHTML = `<img src="/api/frame/${{idx}}" loading="lazy">`;
        div.onclick = () => {{ currentPos = pos; renderAll(); }};
        sb.appendChild(div);
        if (pos === currentPos) {{
            requestAnimationFrame(() => {{
                const left = div.offsetLeft - sb.clientWidth / 2 + div.offsetWidth / 2;
                sb.scrollLeft = left;
            }});
        }}
    }});
}}

function renderFocus() {{
    const panel = document.getElementById('focus');
    if (!filteredList.length) {{ panel.innerHTML = '<p style="color:#888">No frames</p>'; return; }}
    if (currentPos >= filteredList.length) currentPos = filteredList.length - 1;
    if (currentPos < 0) currentPos = 0;

    const idx = filteredList[currentPos];
    const label = labels[idx];
    const pose = poses[idx];
    const chem = chemistries[idx];
    const scene = videoMeta ? videoMeta.scene : null;
    const isDuo = scene === 'duo';
    const isAccepted = label && label !== 'cut' && label !== '__shoot__';
    const displayLabel = label === '__shoot__' ? 'SHOOT' : label;
    const parts = [displayLabel, pose, chem].filter(x => x && x !== '__shoot__');
    const labelColor = label === 'cut' ? '#d32f2f' : label === '__shoot__' ? '#FF9800' : label === 'occluded' ? '#795548' : '#4CAF50';
    const labelHtml = parts.length > 0
        ? `<div class="focus-label" style="color:${{labelColor}}">${{parts.join(' + ')}}</div>`
        : `<div class="focus-label" style="color:#888">Unlabeled</div>`;

    const K = '<span style="font-size:10px;opacity:0.5;margin-left:4px">';
    const FOCUS = 'border:3px solid #e94560;';
    const descHtml = (val) => val && DESC[val] ? `<div style="font-size:11px;color:#888;margin-top:2px;text-align:center">${{DESC[val]}}</div>` : '';

    const autoStep = getStep(idx);
    const step = (manualStep !== null && manualStep >= -1) ? manualStep : autoStep;

    let btnsHtml = '';
    // Step 0: SHOOT/CUT
    const isShot = label && label !== 'cut';
    const isCut = label === 'cut';
    btnsHtml += `<div class="buttons" style="${{step === 0 ? FOCUS + 'padding:4px;border-radius:8px;' : ''}}">`;
    btnsHtml += `<button class="cat-btn${{isShot ? ' selected' : ''}}" style="${{isShot ? 'background:#4CAF50;color:#fff;' : ''}}" onclick="setLabel(${{idx}},'__shoot__')">SHOOT ${{K}}Q</span></button>`;
    btnsHtml += `<button class="cat-btn${{isCut ? ' selected' : ''}}" style="${{isCut ? 'background:#d32f2f;color:#fff;' : 'background:#f5f5f5;color:#999;'}}" onclick="setLabel(${{idx}},'cut')">CUT ${{K}}W</span></button>`;
    btnsHtml += '</div>';

    // Step 1: CHEMISTRY (duo)
    if (isDuo) {{
        btnsHtml += `<div class="buttons" style="margin-top:6px;${{step === 1 ? FOCUS + 'padding:4px;border-radius:8px;' : ''}}">`;
        for (const [c, key] of [['sync','Q'],['interact','W']]) {{
            const sel = chem === c;
            btnsHtml += `<button class="cat-btn${{sel ? ' selected' : ''}}" style="${{sel ? 'background:'+getColor(c)+';color:#fff;' : ''}}" onclick="setChemistry(${{idx}},'${{c}}')">${{c}} ${{K}}${{key}}</span></button>`;
        }}
        btnsHtml += '</div>';
        btnsHtml += descHtml(chem);
    }}

    // Step 2: EXPRESSION
    btnsHtml += `<div class="buttons" style="margin-top:6px;${{step === 2 ? FOCUS + 'padding:4px;border-radius:8px;' : ''}}">`;
    for (const [cat, key] of [['cheese','Q'],['goofy','W'],['chill','E'],['edge','R'],['hype','T'],['occluded','Y']]) {{
        const sel = label === cat;
        btnsHtml += `<button class="cat-btn${{sel ? ' selected' : ''}}" style="${{sel ? 'background:'+getColor(cat)+';color:#fff;' : ''}}" onclick="setLabel(${{idx}},'${{cat}}')">${{cat}} ${{K}}${{key}}</span></button>`;
    }}
    btnsHtml += '</div>';
    btnsHtml += descHtml(isAccepted ? label : null);

    // Step 3: POSE
    btnsHtml += `<div class="buttons" style="margin-top:6px;${{step === 3 ? FOCUS + 'padding:4px;border-radius:8px;' : ''}}">`;
    for (const [p, key] of [['front','Q'],['angle','W'],['side','E']]) {{
        const sel = pose === p;
        btnsHtml += `<button class="cat-btn${{sel ? ' selected' : ''}}" style="${{sel ? 'background:'+getColor(p)+';color:#fff;' : ''}}" onclick="setPose(${{idx}},'${{p}}')">${{p}} ${{K}}${{key}}</span></button>`;
    }}
    btnsHtml += '</div>';
    btnsHtml += descHtml(pose);
    if (step === 4) btnsHtml += '<div style="color:#4CAF50;font-size:13px;margin-top:8px;text-align:center">Complete</div>';

    const stepHint = isDuo ? 'shoot\u2192chemistry\u2192expression\u2192pose' : 'shoot\u2192expression\u2192pose';
    panel.innerHTML = `
        <img class="focus-img" src="/api/frame/${{idx}}">
        <div class="focus-meta">Frame #${{idx}} &nbsp; ${{currentPos + 1}} / ${{filteredList.length}}</div>
        ${{labelHtml}}
        ${{btnsHtml}}
        <div class="nav">
            <button onclick="go(-1)" ${{currentPos <= 0 ? 'disabled' : ''}}>Prev ${{K}}H</span></button>
            <div class="pos">${{currentPos + 1}} / ${{filteredList.length}}</div>
            <button onclick="go(1)" ${{currentPos >= filteredList.length - 1 ? 'disabled' : ''}}>Next ${{K}}L</span></button>
        </div>
        <div class="shortcut-hint">H prev L next | J step\u2193 K step\u2191 | Q W E R T Y = select | ${{stepHint}}</div>
    `;
}}

function go(delta) {{
    currentPos = Math.max(0, Math.min(filteredList.length - 1, currentPos + delta));
    manualStep = null;
    renderAll();
}}

function getStep(idx) {{
    const lbl = labels[idx];
    const c = chemistries[idx];
    const p = poses[idx];
    const isDuo = videoMeta && videoMeta.scene === 'duo';
    const isAccepted = lbl && lbl !== 'cut' && lbl !== '__shoot__';
    if (!lbl) return 0;
    if (lbl === 'cut') return -1;
    if (isDuo && !c) return 1;
    if (lbl === '__shoot__' || !isAccepted) return 2;
    if (!p) return 3;
    return 4;
}}

function setLabel(idx, value) {{
    if (labels[idx] === value) delete labels[idx];
    else labels[idx] = value;
    saveToServer(idx);
    if (value === 'cut') setTimeout(() => go(1), 300);
    renderAll();
}}

function setPose(idx, value) {{
    if (poses[idx] === value) delete poses[idx];
    else poses[idx] = value;
    saveToServer(idx);
    checkAutoAdvance(idx);
    renderAll();
}}

function setChemistry(idx, value) {{
    if (chemistries[idx] === value) delete chemistries[idx];
    else chemistries[idx] = value;
    saveToServer(idx);
    checkAutoAdvance(idx);
    renderAll();
}}

function checkAutoAdvance(idx) {{
    const lbl = labels[idx];
    const p = poses[idx];
    const c = chemistries[idx];
    const isDuo = videoMeta && videoMeta.scene === 'duo';
    const isComplete = lbl && lbl !== '__shoot__' && lbl !== 'cut' && p && (!isDuo || c);
    if (isComplete) setTimeout(() => go(1), 400);
}}

function updateCount() {{
    const total = Object.keys(labels).length;
    document.getElementById('count').textContent = total;

    const EXPRS = [...EXPRESSIONS, 'cut'];
    const ALL_POSES = [...POSES_LIST, ''];
    const poseLabel = p => p || '(none)';

    const counts = {{}};
    EXPRS.forEach(e => {{ counts[e] = {{}}; ALL_POSES.forEach(p => counts[e][p] = 0); }});
    let maxCount = 1;

    for (const [idx, lbl] of Object.entries(labels)) {{
        if (!lbl || lbl === '__shoot__') continue;
        const p = poses[idx] || '';
        if (counts[lbl]) counts[lbl][p] = (counts[lbl][p]||0) + 1;
    }}
    EXPRS.forEach(e => ALL_POSES.forEach(p => {{ if (counts[e][p] > maxCount) maxCount = counts[e][p]; }}));

    const unlabeled = FRAME_COUNT - Object.keys(labels).length;
    const bar = document.getElementById('bucketBar');

    let html = `<div class="bucket-grid" style="grid-template-columns:50px repeat(${{EXPRS.length}}, 1fr) 36px;max-width:${{60 + EXPRS.length * 48 + 40}}px">`;
    html += '<div></div>';
    EXPRS.forEach(e => html += `<div style="text-align:center;font-size:9px;font-weight:600;color:${{getColor(e)}}">${{e}}</div>`);
    html += '<div></div>';
    for (const p of ALL_POSES) {{
        html += `<div style="display:flex;align-items:center;justify-content:flex-end;padding-right:4px;font-size:10px;font-weight:600;color:${{getColor(p)||'#666'}}">${{poseLabel(p)}}</div>`;
        EXPRS.forEach(e => {{
            const v = counts[e][p] || 0;
            const t = v > 0 ? Math.min(v/maxCount,1) : 0;
            const opacity = v > 0 ? 0.15 + 0.85*t : 0;
            const textColor = opacity > 0.45 ? '#fff' : v > 0 ? '#444' : '#d0d0d0';
            html += `<div class="bucket-cell"><div class="bc-fill" style="background:${{getColor(e)}};opacity:${{opacity}}"></div><span class="bc-num" style="color:${{textColor}}">${{v||'-'}}</span></div>`;
        }});
        const rowTotal = ALL_POSES.length > 0 ? EXPRS.reduce((s,e) => s+(counts[e][p]||0), 0) : 0;
        html += `<div style="display:flex;align-items:center;justify-content:center;font-size:10px;color:#555">${{rowTotal}}</div>`;
    }}
    html += '</div>';
    if (unlabeled > 0) html += `<span style="font-size:11px;color:#e94560;margin-left:12px">${{unlabeled}} unlabeled</span>`;

    bar.innerHTML = html;
}}

const TL_COLORS = {{ '':'#ddd', '__shoot__':'#bbb', cut:'#d32f2f', occluded:'#795548',
    cheese:'#4CAF50', goofy:'#E91E63', chill:'#2196F3', edge:'#FF5722', hype:'#9C27B0' }};

function renderTimeline() {{
    const bar = document.getElementById('timelineBar');
    bar.innerHTML = '';
    const currentIdx = filteredList[currentPos];
    for (let i = 0; i < FRAME_COUNT; i++) {{
        const el = document.createElement('div');
        el.className = 'tl-frame';
        el.style.background = TL_COLORS[labels[i] || ''] || '#ddd';
        if (i === currentIdx) {{ el.style.outline = '2px solid #e94560'; el.style.zIndex = '1'; }}
        el.onclick = () => {{
            const pos = filteredList.indexOf(i);
            if (pos >= 0) {{ currentPos = pos; manualStep = null; renderAll(); }}
        }};
        bar.appendChild(el);
    }}
}}

// --- Meta modal ---
const MODAL_VALUES = {{
    scene: ['solo', 'duo'], main_gender: ['male', 'female'],
    main_ethnicity: ['asian', 'western', 'other'],
    passenger_gender: ['male', 'female'], passenger_ethnicity: ['asian', 'western', 'other'],
}};
const MODAL_FIELDS_SOLO = ['scene', 'main_gender', 'main_ethnicity'];
const MODAL_FIELDS_DUO = ['scene', 'main_gender', 'main_ethnicity', 'passenger_gender', 'passenger_ethnicity'];
function getModalFields() {{ return metaDraft.scene === 'duo' ? MODAL_FIELDS_DUO : MODAL_FIELDS_SOLO; }}

function getMetaContainerId(f) {{ return 'meta' + f.charAt(0).toUpperCase() + f.slice(1); }}

function updateMetaIndicator() {{
    const el = document.getElementById('metaIndicator');
    if (videoMeta) {{
        if (videoMeta.scene === 'duo') {{
            el.textContent = 'duo | M:' + videoMeta.main_gender + '/' + videoMeta.main_ethnicity
                + ' P:' + videoMeta.passenger_gender + '/' + videoMeta.passenger_ethnicity;
        }} else {{
            el.textContent = 'solo | ' + videoMeta.main_gender + ' | ' + videoMeta.main_ethnicity;
        }}
        el.style.color = '#aaa';
    }} else {{
        el.textContent = '(no video meta)';
        el.style.color = '#e94560';
    }}
}}

function selectMetaOpt(field, value) {{
    metaDraft[field] = value;
    if (field === 'scene') {{
        document.getElementById('passengerGenderRow').style.display = value === 'duo' ? '' : 'none';
        document.getElementById('passengerEthnicityRow').style.display = value === 'duo' ? '' : 'none';
        if (value !== 'duo') {{ metaDraft.passenger_gender = null; metaDraft.passenger_ethnicity = null; }}
    }}
    renderMetaSelections();
}}

function renderMetaSelections() {{
    for (const field of MODAL_FIELDS_DUO) {{
        const c = document.getElementById(getMetaContainerId(field));
        if (!c) continue;
        c.querySelectorAll('.modal-opt').forEach(el => el.classList.toggle('selected', el.dataset.value === metaDraft[field]));
        const fields = getModalFields();
        const idx = fields.indexOf(field);
        c.parentElement.style.background = (idx >= 0 && modalField === idx) ? 'rgba(233,69,96,0.1)' : '';
        c.parentElement.style.borderRadius = '6px';
    }}
    const isDuo = metaDraft.scene === 'duo';
    const baseOk = metaDraft.scene && metaDraft.main_gender && metaDraft.main_ethnicity;
    const passengerOk = !isDuo || (metaDraft.passenger_gender && metaDraft.passenger_ethnicity);
    document.getElementById('metaStartBtn').disabled = !(baseOk && passengerOk);
}}

function showMetaModal() {{
    metaDraft = videoMeta
        ? {{ ...videoMeta, member_id: videoMeta.member_id || '' }}
        : {{ scene: null, main_gender: null, main_ethnicity: null, passenger_gender: null, passenger_ethnicity: null, member_id: '' }};
    document.getElementById('metaMemberId').value = metaDraft.member_id || '';
    modalField = 0;
    const isDuo = metaDraft.scene === 'duo';
    document.getElementById('passengerGenderRow').style.display = isDuo ? '' : 'none';
    document.getElementById('passengerEthnicityRow').style.display = isDuo ? '' : 'none';
    document.getElementById('metaModal').style.display = 'flex';
    renderMetaSelections();
}}

function hideMetaModal() {{ document.getElementById('metaModal').style.display = 'none'; }}

function saveMetaAndStart() {{
    const isDuo = metaDraft.scene === 'duo';
    if (!metaDraft.scene || !metaDraft.main_gender || !metaDraft.main_ethnicity) return;
    if (isDuo && (!metaDraft.passenger_gender || !metaDraft.passenger_ethnicity)) return;
    videoMeta = {{ workflow_id: VIDEO_STEM, scene: metaDraft.scene, main_gender: metaDraft.main_gender, main_ethnicity: metaDraft.main_ethnicity }};
    if (isDuo) {{ videoMeta.passenger_gender = metaDraft.passenger_gender; videoMeta.passenger_ethnicity = metaDraft.passenger_ethnicity; }}
    if (metaDraft.member_id) videoMeta.member_id = metaDraft.member_id;
    saveMetaToServer();
    hideMetaModal();
    updateMetaIndicator();
    renderAll();
}}

// --- Confirm merge ---
async function showConfirm() {{
    document.getElementById('confirmOverlay').style.display = 'flex';
    const preview = await (await fetch('/api/preview')).json();
    const s = document.getElementById('confirmSummary');
    if (preview.total === 0) {{
        s.innerHTML = '<p style="color:#e94560">No completed labels to merge.</p>';
        document.getElementById('confirmBtn').disabled = true;
        return;
    }}
    document.getElementById('confirmBtn').disabled = false;

    let html = `<p>Merge <b>${{preview.total}}</b> labeled frames into <code>${{preview.dataset_dir}}</code></p>`;
    html += '<table style="border-collapse:collapse;font-size:12px;margin:12px 0">';
    html += '<tr><th style="padding:4px 8px;text-align:left;color:#888">Bucket</th><th style="padding:4px 8px;color:#888">Count</th></tr>';
    const sorted = Object.entries(preview.buckets).sort((a,b) => b[1]-a[1]);
    for (const [key, count] of sorted) {{
        const [expr, pose] = key.split('|');
        const color = getColor(expr);
        html += `<tr><td style="padding:4px 8px"><span style="color:${{color}}">${{expr}}</span> \u00d7 ${{pose}}</td><td style="padding:4px 8px;text-align:center">${{count}}</td></tr>`;
    }}
    html += '</table>';
    if (preview.video_meta) {{
        const m = preview.video_meta;
        html += `<p style="font-size:12px;color:#888">Video: ${{m.workflow_id}} (${{m.scene}}, ${{m.main_gender}}/${{m.main_ethnicity}})</p>`;
    }}
    s.innerHTML = html;
}}

function hideConfirm() {{ document.getElementById('confirmOverlay').style.display = 'none'; }}

async function doMerge() {{
    const btn = document.getElementById('confirmBtn');
    btn.textContent = 'Merging...';
    btn.disabled = true;
    try {{
        const res = await (await fetch('/api/confirm_merge', {{ method: 'POST', headers: {{'Content-Type':'application/json'}}, body: '{{}}' }})).json();
        if (res.ok) {{
            const s = document.getElementById('confirmSummary');
            let detail = `${{res.merged_images}} images saved, ${{res.new_labels}} new labels added`;
            if (res.updated_labels > 0) detail += `, ${{res.updated_labels}} labels updated`;
            if (res.video_saved) detail += `<br>Video saved: <code>${{res.video_path}}</code>`;
            else detail += '<br><span style="color:#FF9800">Video not saved (ffmpeg not found)</span>';
            s.innerHTML = `<div style="color:#4CAF50;font-size:18px;margin:20px 0">Merge complete!</div><p>${{detail}}</p><p style="color:#888;font-size:12px">You can close this tab or continue labeling.</p>`;
            document.querySelector('.confirm-btns').innerHTML = '<button style="background:#e0e0e0;color:#555;flex:1;padding:12px;border:none;border-radius:6px;font-size:15px;cursor:pointer" onclick="hideConfirm()">Close</button>';
        }} else {{
            alert('Merge failed: ' + (res.error || 'unknown error'));
        }}
    }} catch(e) {{
        alert('Merge failed: ' + e.message);
    }}
    btn.textContent = 'Confirm Merge';
    btn.disabled = false;
}}

function resetLabels() {{
    if (!confirm('Reset all labels?')) return;
    labels = {{}};
    poses = {{}};
    chemistries = {{}};
    fetch('/api/save_label', {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{index:-1,expression:'__reset__'}}) }});
    buildFilteredList();
    currentPos = 0;
    renderAll();
}}

// --- Keyboard ---
const STEP_OPTIONS_SOLO = {{
    '-1': [['__shoot__','setLabel']],
    '0': [['__shoot__','setLabel'], ['cut','setLabel']],
    '2': [['cheese','setLabel'], ['goofy','setLabel'], ['chill','setLabel'], ['edge','setLabel'], ['hype','setLabel'], ['occluded','setLabel']],
    '3': [['front','setPose'], ['angle','setPose'], ['side','setPose']],
}};
const STEP_OPTIONS_DUO = {{
    '-1': [['__shoot__','setLabel']],
    '0': [['__shoot__','setLabel'], ['cut','setLabel']],
    '1': [['sync','setChemistry'], ['interact','setChemistry']],
    '2': [['cheese','setLabel'], ['goofy','setLabel'], ['chill','setLabel'], ['edge','setLabel'], ['hype','setLabel'], ['occluded','setLabel']],
    '3': [['front','setPose'], ['angle','setPose'], ['side','setPose']],
}};

function isModalOpen() {{
    return document.getElementById('metaModal').style.display !== 'none'
        || document.getElementById('confirmOverlay').style.display !== 'none';
}}

document.addEventListener('keydown', e => {{
    if (isModalOpen()) {{
        if (document.getElementById('confirmOverlay').style.display !== 'none') {{
            if (e.key === 'Escape') hideConfirm();
            return;
        }}
        const n = parseInt(e.key);
        const fields = getModalFields();
        if (e.key === 'Enter') {{ const isDuo = metaDraft.scene === 'duo'; const baseOk = metaDraft.scene && metaDraft.main_gender && metaDraft.main_ethnicity; const passengerOk = !isDuo || (metaDraft.passenger_gender && metaDraft.passenger_ethnicity); if (baseOk && passengerOk) saveMetaAndStart(); return; }}
        if (e.key === 'Escape') {{ if (videoMeta) hideMetaModal(); return; }}
        if (e.key === 'Tab') {{ e.preventDefault(); modalField = (modalField + 1) % fields.length; renderMetaSelections(); return; }}
        if (n >= 1) {{
            const field = fields[modalField];
            const values = MODAL_VALUES[field];
            if (n <= values.length) {{
                selectMetaOpt(field, values[n - 1]);
                const updatedFields = getModalFields();
                for (let i = 0; i < updatedFields.length; i++) {{
                    const next = (modalField + 1 + i) % updatedFields.length;
                    if (!metaDraft[updatedFields[next]]) {{ modalField = next; renderMetaSelections(); return; }}
                }}
                renderMetaSelections();
            }}
        }}
        return;
    }}

    const idx = filteredList[currentPos];
    if (idx === undefined) return;
    const autoStep = getStep(idx);
    const step = (manualStep !== null) ? manualStep : autoStep;
    const isDuo = videoMeta && videoMeta.scene === 'duo';
    const STEP_OPTIONS = isDuo ? STEP_OPTIONS_DUO : STEP_OPTIONS_SOLO;

    if (e.key === 'h') {{ go(-1); return; }}
    if (e.key === 'l') {{ go(1); return; }}
    if (e.key === 'j') {{ if (manualStep === null) manualStep = step; manualStep = Math.min(3, manualStep + 1); renderAll(); return; }}
    if (e.key === 'k') {{ if (manualStep === null) manualStep = step; manualStep = Math.max(-1, manualStep - 1); renderAll(); return; }}

    const QWER = {{'q':1,'w':2,'e':3,'r':4,'t':5,'y':6}};
    const n = QWER[e.key];
    const activeStep = String(step);
    if (n && STEP_OPTIONS[activeStep]) {{
        const opts = STEP_OPTIONS[activeStep];
        if (n <= opts.length) {{
            const [value, fn] = opts[n - 1];
            if (fn === 'setLabel') setLabel(idx, value);
            else if (fn === 'setChemistry') setChemistry(idx, value);
            else if (fn === 'setPose') setPose(idx, value);
            manualStep = null;
        }}
    }}
}});

// Filter buttons
document.querySelectorAll('.filter-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentFilter = btn.dataset.filter;
        buildFilteredList();
        currentPos = 0;
        renderAll();
    }});
}});

// Init
loadStaging();
</script>
</body></html>"""


def start_label_server(
    video_path: str | Path,
    dataset_dir: str | Path,
    fps: int = 2,
    max_frames: int = 500,
    port: int = 8766,
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
        staging = {"labels": {}, "poses": {}, "chems": {}, "video_meta": None}

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
                    if r.get("chemistry"):
                        staging["chems"][idx] = r["chemistry"]

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

    # Configure handler
    LabelHandler.video_name = video_name
    LabelHandler.video_stem = video_stem
    LabelHandler.video_path = video_path
    LabelHandler.dataset_dir = dataset_dir
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
