"""IdentityResult → MemoryBank bridge.

IdentityBuilder의 per-video 선택 결과를 momentbank MemoryBank에 등록.
MemoryBank는 다중 비디오에 걸친 장기 메모리 역할.

Flow:
    IdentityBuilder.build() → IdentityResult
    export_identity_crops() → face/body crop images
    register_to_bank() → MemoryBank.update() × N → save_bank()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from momentscan.algorithm.identity.types import (
    IdentityFrame,
    IdentityRecord,
    IdentityResult,
)

logger = logging.getLogger(__name__)


@dataclass
class BankRegistrationResult:
    """register_to_bank() 결과 요약."""

    persons_registered: int = 0
    frames_registered: int = 0
    nodes_created: Dict[int, int] = field(default_factory=dict)  # pid → node count
    bank_paths: Dict[int, str] = field(default_factory=dict)  # pid → bank.json path


def register_to_bank(
    result: IdentityResult,
    records: List[IdentityRecord],
    output_dir: Path,
    *,
    bank_dir: Optional[Path] = None,
) -> BankRegistrationResult:
    """IdentityBuilder 결과를 MemoryBank에 등록한다.

    각 person의 선택된 프레임 (anchor/coverage/challenge)을
    MemoryBank.update()로 등록. 기존 bank.json이 있으면 로드하여 누적.

    Args:
        result: IdentityBuilder.build() 결과.
        records: 전체 IdentityRecord (임베딩 lookup용).
        output_dir: crop 이미지가 저장된 출력 디렉토리.
        bank_dir: MemoryBank 저장 디렉토리. None이면 output_dir 사용.

    Returns:
        BankRegistrationResult 요약.
    """
    from momentbank import MemoryBank, save_bank, load_bank

    if not result.persons:
        return BankRegistrationResult()

    bank_root = bank_dir or output_dir

    # frame_idx → IdentityRecord lookup
    record_map: Dict[int, IdentityRecord] = {r.frame_idx: r for r in records}

    reg_result = BankRegistrationResult()

    for pid, person in result.persons.items():
        bank_path = bank_root / "identity" / f"person_{pid}" / "memory_bank.json"

        # Load existing bank or create new
        bank: MemoryBank
        if bank_path.exists():
            bank = load_bank(bank_path)
            logger.info("Loaded existing bank for person %d (%d nodes)", pid, len(bank.nodes))
        else:
            bank = MemoryBank(person_id=pid)

        # Collect all selected frames, anchor first (higher priority)
        all_frames: List[IdentityFrame] = (
            person.anchor_frames
            + person.coverage_frames
            + person.challenge_frames
        )

        registered = 0
        for frame in all_frames:
            record = record_map.get(frame.frame_idx)
            if record is None or record.e_id is None:
                continue

            # Bucket metadata for NodeMeta histogram
            meta = {
                "yaw": frame.bucket.yaw_bin,
                "pitch": frame.bucket.pitch_bin,
                "expression": frame.bucket.expression_bin,
            }

            # Face crop image path (from export_crops)
            crops_dir = output_dir / "identity" / f"person_{pid}" / "crops"
            image_path = str(
                crops_dir / f"{frame.set_type}_{frame.frame_idx}_face.jpg"
            )

            bank.update(
                e_id=record.e_id.copy(),
                quality=frame.quality_score,
                meta=meta,
                image_path=image_path,
            )
            registered += 1

        # Save bank
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        save_bank(bank, bank_path)

        reg_result.persons_registered += 1
        reg_result.frames_registered += registered
        reg_result.nodes_created[pid] = len(bank.nodes)
        reg_result.bank_paths[pid] = str(bank_path)

        logger.info(
            "Registered %d frames for person %d → %d nodes (%s)",
            registered, pid, len(bank.nodes), bank_path,
        )

    return reg_result
