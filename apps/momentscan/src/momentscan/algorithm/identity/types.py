"""Identity builder data types.

identity_builder.md 설계를 따르는 데이터 구조 정의.
Phase 3: 인물별 다양한 참조 이미지 수집 + 맥락 기반 top-k 추출.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class IdentityRecord:
    """on_frame()에서 축적되는 프레임별 임베딩 + 메타데이터."""

    frame_idx: int
    timestamp_ms: float

    # ArcFace 512D (face.detect → FaceObservation.embedding)
    e_id: Optional[np.ndarray] = None

    # DINOv2 384D (face.embed → e_face, body.embed → e_body)
    e_face: Optional[np.ndarray] = None
    e_body: Optional[np.ndarray] = None

    # 포즈/표정/품질 (버킷 분류 + 게이트용)
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0
    smile_intensity: float = 0.0
    mouth_open_ratio: float = 0.0
    eye_open_ratio: float = 0.0
    face_confidence: float = 0.0
    face_area_ratio: float = 0.0
    blur_score: float = 0.0
    brightness: float = 0.0

    # 크롭 좌표 (이미지 추출용)
    face_bbox: Optional[tuple[float, float, float, float]] = None  # 정규화
    face_crop_box: Optional[tuple[int, int, int, int]] = None  # 픽셀
    body_crop_box: Optional[tuple[int, int, int, int]] = None  # 픽셀
    image_size: Optional[tuple[int, int]] = None  # (w, h)

    person_id: int = 0  # 0=main (MVP에서는 main만)


@dataclass
class BucketLabel:
    """프레임의 yaw/pitch/expression 복합 버킷."""

    yaw_bin: str
    pitch_bin: str
    expression_bin: str

    @property
    def key(self) -> str:
        return f"{self.yaw_bin}|{self.pitch_bin}|{self.expression_bin}"


@dataclass
class IdentityFrame:
    """선택된 프레임 메타데이터."""

    frame_idx: int
    timestamp_ms: float
    set_type: str  # "anchor" | "coverage" | "challenge"
    bucket: BucketLabel
    quality_score: float = 0.0
    stable_score: float = 0.0
    novelty_score: float = 0.0
    face_crop_box: Optional[tuple[int, int, int, int]] = None
    body_crop_box: Optional[tuple[int, int, int, int]] = None
    image_size: Optional[tuple[int, int]] = None


@dataclass
class IdentityConfig:
    """Identity builder 설정."""

    tau_id: float = 0.35  # ID 안정성 임계값
    anchor_count: int = 5  # 앵커 프레임 수
    anchor_max_yaw: float = 15.0  # 앵커 최대 yaw
    anchor_min_interval_ms: float = 1500.0  # 앵커 간 최소 시간 간격
    anchor_max_similarity: float = 0.90  # 앵커 DINOv2 코사인 유사도 상한

    # Strict gate
    gate_face_confidence: float = 0.7
    gate_blur_min: float = 50.0

    # Loose gate
    loose_face_confidence: float = 0.5
    loose_blur_min: float = 25.0

    # Coverage/Challenge
    coverage_max_per_bucket: int = 2
    coverage_min_interval_ms: float = 2000.0  # 같은 버킷 내 최소 시간 간격
    coverage_max_similarity: float = 0.85  # DINOv2 코사인 유사도 상한 (전체 coverage)
    challenge_count: int = 8
    challenge_min_stable: float = 0.4

    # Medoid computation
    medoid_max_candidates: int = 200


@dataclass
class PersonIdentity:
    """한 인물의 identity 정보."""

    person_id: int
    prototype_frame_idx: int
    anchor_frames: List[IdentityFrame] = field(default_factory=list)
    coverage_frames: List[IdentityFrame] = field(default_factory=list)
    challenge_frames: List[IdentityFrame] = field(default_factory=list)
    yaw_coverage: Dict[str, int] = field(default_factory=dict)
    pitch_coverage: Dict[str, int] = field(default_factory=dict)
    expression_coverage: Dict[str, int] = field(default_factory=dict)

    def query(
        self,
        *,
        yaw_bin: Optional[str] = None,
        pitch_bin: Optional[str] = None,
        expression_bin: Optional[str] = None,
        top_k: int = 3,
    ) -> List[IdentityFrame]:
        """맥락 기반 top-k 선택 (하이라이트 대용).

        지정된 bucket 조건에 맞는 프레임을 quality_score 순으로 반환.
        """
        all_frames = self.anchor_frames + self.coverage_frames + self.challenge_frames

        filtered = []
        for f in all_frames:
            if yaw_bin is not None and f.bucket.yaw_bin != yaw_bin:
                continue
            if pitch_bin is not None and f.bucket.pitch_bin != pitch_bin:
                continue
            if expression_bin is not None and f.bucket.expression_bin != expression_bin:
                continue
            filtered.append(f)

        filtered.sort(key=lambda f: f.quality_score, reverse=True)
        return filtered[:top_k]


@dataclass
class IdentityResult:
    """Identity builder 전체 결과."""

    persons: Dict[int, PersonIdentity] = field(default_factory=dict)
    frame_count: int = 0
    config: Optional[IdentityConfig] = None
