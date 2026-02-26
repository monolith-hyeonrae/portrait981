"""Identity result export.

identity metadata를 JSON으로 출력.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from momentscan.algorithm.identity.types import IdentityResult

logger = logging.getLogger(__name__)


def export_identity_metadata(result: IdentityResult, output_dir: Path) -> None:
    """Identity 결과를 meta.json으로 출력한다.

    Args:
        result: IdentityBuilder 분석 결과.
        output_dir: 출력 루트 디렉토리. identity/person_{id}/ 하위에 저장.
    """
    if not result.persons:
        return

    for pid, person in result.persons.items():
        person_dir = output_dir / "identity" / f"person_{pid}"
        person_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "person_id": person.person_id,
            "prototype_frame_idx": person.prototype_frame_idx,
            "anchor_count": len(person.anchor_frames),
            "coverage_count": len(person.coverage_frames),
            "challenge_count": len(person.challenge_frames),
            "yaw_coverage": person.yaw_coverage,
            "pitch_coverage": person.pitch_coverage,
            "expression_coverage": person.expression_coverage,
            "anchors": [_frame_to_dict(f) for f in person.anchor_frames],
            "coverage": [_frame_to_dict(f) for f in person.coverage_frames],
            "challenges": [_frame_to_dict(f) for f in person.challenge_frames],
        }

        meta_path = person_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(
            "Exported identity metadata for person %d: %s "
            "(%d anchors, %d coverage, %d challenge)",
            pid, meta_path,
            len(person.anchor_frames),
            len(person.coverage_frames),
            len(person.challenge_frames),
        )


def _frame_to_dict(frame) -> dict:
    """IdentityFrame → JSON-serializable dict."""
    d = {
        "frame_idx": frame.frame_idx,
        "timestamp_ms": frame.timestamp_ms,
        "set_type": frame.set_type,
        "bucket": frame.bucket.key,
        "quality_score": round(frame.quality_score, 4),
        "stable_score": round(frame.stable_score, 4),
        "novelty_score": round(frame.novelty_score, 4),
        "face_crop_box": frame.face_crop_box,
        "image_size": frame.image_size,
    }
    if frame.pivot_name is not None:
        d["pivot_name"] = frame.pivot_name
        d["pivot_distance"] = round(frame.pivot_distance, 4)
    return d
