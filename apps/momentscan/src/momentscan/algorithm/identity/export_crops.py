"""Identity crop image extraction from video.

선택된 identity 프레임의 face/body crop을 비디오에서 추출하여 저장.
reportrait 파이프라인에서 diffusion 입력으로 사용.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

from momentscan.algorithm.identity.types import (
    IdentityFrame,
    IdentityRecord,
    IdentityResult,
)

logger = logging.getLogger(__name__)

JPEG_QUALITY = 95


def export_identity_crops(
    video_path: Path,
    result: IdentityResult,
    records: List[IdentityRecord],
    output_dir: Path,
) -> None:
    """선택된 identity 프레임의 face/body crop을 비디오에서 추출한다.

    Args:
        video_path: 원본 비디오 경로.
        result: IdentityBuilder.build() 결과 (선택된 프레임).
        records: 전체 IdentityRecord (timestamp lookup용).
        output_dir: 출력 루트 디렉토리. identity/person_{id}/crops/ 하위에 저장.
    """
    if not result.persons:
        return

    # frame_idx → timestamp_ms
    fidx_to_ts: Dict[int, float] = {r.frame_idx: r.timestamp_ms for r in records}

    for pid, person in result.persons.items():
        crops_dir = output_dir / "identity" / f"person_{pid}" / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        all_frames = (
            person.anchor_frames
            + person.coverage_frames
            + person.challenge_frames
        )
        if not all_frames:
            continue

        # (frame, timestamp_ms) pairs — timestamp 순 정렬 (seek 최적화)
        targets: List[Tuple[IdentityFrame, float]] = []
        for f in all_frames:
            ts = fidx_to_ts.get(f.frame_idx, f.timestamp_ms)
            targets.append((f, ts))
        targets.sort(key=lambda x: x[1])

        saved = _extract_crops(video_path, targets, crops_dir)
        logger.info(
            "Exported %d crops for person %d to %s",
            saved, pid, crops_dir,
        )


def _extract_crops(
    video_path: Path,
    targets: List[Tuple[IdentityFrame, float]],
    crops_dir: Path,
) -> int:
    """비디오에서 face/body crop을 추출하여 저장한다.

    Returns:
        저장된 이미지 수.
    """
    from visualbase.sources.file import FileSource

    source = FileSource(str(video_path))
    source.open()

    saved = 0
    seen_idx = 0  # set_type별 순번

    try:
        for identity_frame, ts_ms in targets:
            t_ns = int(ts_ms * 1_000_000)
            if not source.seek(t_ns):
                continue

            frame = source.read()
            if frame is None:
                continue

            img = frame.data
            f = identity_frame
            prefix = f"{f.set_type}_{f.frame_idx}"

            # Head crop
            if f.head_crop_box is not None:
                head_img = _crop_image(img, f.head_crop_box)
                if head_img is not None:
                    path = crops_dir / f"{prefix}_head.jpg"
                    cv2.imwrite(
                        str(path), head_img,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                    )
                    saved += 1
    finally:
        source.close()

    return saved


def _crop_image(
    img,
    box: tuple[int, int, int, int],
) -> "cv2.Mat | None":
    """이미지에서 (x1, y1, x2, y2) 픽셀 좌표로 크롭한다.

    좌표가 이미지 범위를 벗어나면 클램핑한다.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box

    # Clamp to image bounds
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]
