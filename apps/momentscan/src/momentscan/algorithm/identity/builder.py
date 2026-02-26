"""Identity builder engine.

IdentityRecord 리스트에서 인물별 다양한 참조 프레임을 선택:
1. Quality gate → 사용 가능한 프레임 필터
2. Medoid prototype 계산 (ArcFace 코사인 유사도)
3. ID 안정성 검사 (cos(e_id, prototype) > tau_id)
4. Bucket 분류 (yaw × pitch × expression)
5. Anchor 선택 (정면 + 고품질 top-k)
6. Coverage 선택 (버킷별 최고 품질)
7. Challenge 선택 (극단 조건 + 안정 + 신규성)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from momentscan.algorithm.identity.buckets import classify_frame
from momentscan.algorithm.identity.pivots import (
    PivotAssignment,
    assign_pivot,
    assign_pivot_fallback,
    pivot_to_bucket,
)
from momentscan.algorithm.identity.types import (
    BucketLabel,
    IdentityConfig,
    IdentityFrame,
    IdentityRecord,
    IdentityResult,
    PersonIdentity,
)

logger = logging.getLogger(__name__)


class IdentityBuilder:
    """인물별 다양한 참조 프레임을 선택하는 엔진."""

    def __init__(self, config: Optional[IdentityConfig] = None):
        self.config = config or IdentityConfig()

    def build(self, records: List[IdentityRecord]) -> IdentityResult:
        """IdentityRecord 리스트에서 인물별 identity를 구축한다."""
        if not records:
            return IdentityResult(frame_count=0, config=self.config)

        # person_id별 그룹핑
        groups: Dict[int, List[IdentityRecord]] = defaultdict(list)
        for r in records:
            groups[r.person_id].append(r)

        persons: Dict[int, PersonIdentity] = {}
        for pid, pid_records in groups.items():
            person = self._build_person(pid, pid_records)
            if person is not None:
                persons[pid] = person

        return IdentityResult(
            persons=persons,
            frame_count=len(records),
            config=self.config,
        )

    def _build_person(
        self, person_id: int, records: List[IdentityRecord]
    ) -> Optional[PersonIdentity]:
        """한 인물의 identity를 구축한다."""
        cfg = self.config

        # (a) Strict gate → medoid 후보
        strict_records = [r for r in records if self._pass_strict_gate(r)]
        if len(strict_records) < 3:
            logger.info(
                "Person %d: too few strict-gate frames (%d) — skipping",
                person_id, len(strict_records),
            )
            return None

        # (b) Medoid: 정면 고품질 임베딩의 중심
        medoid_candidates = [
            r for r in strict_records
            if abs(r.head_yaw) <= cfg.anchor_max_yaw
        ]
        if not medoid_candidates:
            medoid_candidates = strict_records

        prototype, proto_idx = self._compute_medoid(medoid_candidates)

        # (c) stable_score for all records with e_id
        scored: List[Tuple[IdentityRecord, float, BucketLabel, float, Optional[PivotAssignment]]] = []
        for r in records:
            if r.e_id is None:
                continue
            stable = float(np.dot(r.e_id, prototype))
            if stable < cfg.tau_id:
                continue

            # Pivot assignment (AU-based or fallback)
            if r.au_intensities:
                assignment = assign_pivot(r.head_yaw, r.head_pitch, r.au_intensities)
            else:
                assignment = assign_pivot_fallback(
                    r.head_yaw, r.head_pitch,
                    r.smile_intensity, r.mouth_open_ratio,
                )

            if assignment is not None:
                bucket = pivot_to_bucket(assignment)
            else:
                # Fallback to fixed grid for out-of-range poses
                bucket = classify_frame(
                    r.head_yaw, r.head_pitch,
                    r.smile_intensity, r.mouth_open_ratio, r.eye_open_ratio,
                )

            quality = self._compute_quality(r)
            scored.append((r, stable, bucket, quality, assignment))

        if not scored:
            return None

        # (f) Select anchors
        anchor_frames = self._select_anchors(scored)

        # (g) Select coverage
        coverage_frames = self._select_coverage(scored, anchor_frames)

        # (h) Select challenges
        selected_so_far = anchor_frames + coverage_frames
        challenge_frames = self._select_challenges(scored, selected_so_far)

        # Coverage stats
        all_selected = anchor_frames + coverage_frames + challenge_frames
        yaw_cov: Dict[str, int] = defaultdict(int)
        pitch_cov: Dict[str, int] = defaultdict(int)
        expr_cov: Dict[str, int] = defaultdict(int)
        pivot_cov: Dict[str, int] = defaultdict(int)
        for f in all_selected:
            yaw_cov[f.bucket.yaw_bin] += 1
            pitch_cov[f.bucket.pitch_bin] += 1
            expr_cov[f.bucket.expression_bin] += 1
            if f.pivot_name:
                pivot_cov[f.pivot_name] += 1

        logger.info(
            "Person %d: %d anchors, %d coverage, %d challenge (from %d scored / %d total)",
            person_id, len(anchor_frames), len(coverage_frames),
            len(challenge_frames), len(scored), len(records),
        )

        return PersonIdentity(
            person_id=person_id,
            prototype_frame_idx=proto_idx,
            anchor_frames=anchor_frames,
            coverage_frames=coverage_frames,
            challenge_frames=challenge_frames,
            yaw_coverage=dict(yaw_cov),
            pitch_coverage=dict(pitch_cov),
            expression_coverage=dict(expr_cov),
            pivot_coverage=dict(pivot_cov),
        )

    def _compute_medoid(
        self, records: List[IdentityRecord]
    ) -> Tuple[np.ndarray, int]:
        """ArcFace 임베딩의 medoid (평균 유사도 최대점)를 계산한다.

        O(n²) 코사인 유사도. medoid_max_candidates로 서브샘플링.
        """
        cfg = self.config

        # Subsample if too many
        if len(records) > cfg.medoid_max_candidates:
            indices = np.random.default_rng(42).choice(
                len(records), cfg.medoid_max_candidates, replace=False
            )
            candidates = [records[i] for i in indices]
        else:
            candidates = records

        embeddings = np.array([r.e_id for r in candidates])  # (N, D)
        # Cosine similarity matrix (embeddings are L2-normalized)
        sim_matrix = embeddings @ embeddings.T  # (N, N)
        avg_sim = sim_matrix.mean(axis=1)
        medoid_idx = int(np.argmax(avg_sim))

        return embeddings[medoid_idx].copy(), candidates[medoid_idx].frame_idx

    def _compute_quality(self, r: IdentityRecord) -> float:
        """프레임 품질 점수 계산.

        blur_norm * 0.3 + face_size_norm * 0.3 + frontalness * 0.2 + confidence * 0.2
        """
        # blur: 0~500 range → 0~1
        blur_norm = min(r.blur_score / 500.0, 1.0) if r.blur_score > 0 else 0.5

        # face size: area_ratio 0~0.3 → 0~1
        face_size_norm = min(r.face_area_ratio / 0.3, 1.0)

        # frontalness: 1 - |yaw|/45
        frontalness = max(0.0, 1.0 - abs(r.head_yaw) / 45.0)

        # confidence: already 0~1
        confidence = r.face_confidence

        return (
            0.3 * blur_norm
            + 0.3 * face_size_norm
            + 0.2 * frontalness
            + 0.2 * confidence
        )

    def _compute_novelty(
        self, r: IdentityRecord, selected_embeddings: List[np.ndarray]
    ) -> float:
        """Novelty score — always 1.0 (no visual embedding available).

        Previously used DINOv2 e_head for diversity computation.
        """
        return 1.0

    def _pass_strict_gate(self, r: IdentityRecord) -> bool:
        cfg = self.config
        if r.e_id is None:
            return False
        if r.face_confidence < cfg.gate_face_confidence:
            return False
        if r.blur_score > 0 and r.blur_score < cfg.gate_blur_min:
            return False
        return True

    def _pass_loose_gate(self, r: IdentityRecord) -> bool:
        cfg = self.config
        if r.e_id is None:
            return False
        if r.face_confidence < cfg.loose_face_confidence:
            return False
        if r.blur_score > 0 and r.blur_score < cfg.loose_blur_min:
            return False
        return True

    def _select_anchors(
        self,
        scored: List[Tuple[IdentityRecord, float, BucketLabel, float, Optional[PivotAssignment]]],
    ) -> List[IdentityFrame]:
        """정면 + strict gate + quality top-k로 앵커 선택 (중복 제거).

        Coverage와 동일한 시간 간격 + DINOv2 유사도 기반 greedy dedup.
        """
        cfg = self.config

        # Filter: frontal + strict gate
        candidates = [
            (r, stable, bucket, quality, assignment)
            for r, stable, bucket, quality, assignment in scored
            if abs(r.head_yaw) <= cfg.anchor_max_yaw
            and self._pass_strict_gate(r)
        ]

        # Sort by quality descending
        candidates.sort(key=lambda x: x[3], reverse=True)

        anchors = []
        selected_timestamps: List[float] = []

        for r, stable, bucket, quality, assignment in candidates:
            if len(anchors) >= cfg.anchor_count:
                break

            if self._too_close_temporally(
                r.timestamp_ms, selected_timestamps, cfg.anchor_min_interval_ms
            ):
                continue

            anchors.append(IdentityFrame(
                frame_idx=r.frame_idx,
                timestamp_ms=r.timestamp_ms,
                set_type="anchor",
                bucket=bucket,
                quality_score=quality,
                stable_score=stable,
                novelty_score=0.0,
                pivot_name=assignment.pivot_name if assignment else None,
                pivot_distance=assignment.pose_distance if assignment else 0.0,
                face_crop_box=r.face_crop_box,
                image_size=r.image_size,
            ))
            selected_timestamps.append(r.timestamp_ms)

        return anchors

    def _select_coverage(
        self,
        scored: List[Tuple[IdentityRecord, float, BucketLabel, float, Optional[PivotAssignment]]],
        already_selected: List[IdentityFrame],
    ) -> List[IdentityFrame]:
        """버킷별 best quality로 coverage 선택 (중복 제거).

        두 가지 중복 방지:
        1. 시간 간격: 같은 버킷 내 이미 선택된 프레임과 min_interval_ms 이상 떨어져야 함
        2. 시각 유사도: DINOv2 e_face가 있으면 전체 coverage 중 max_similarity 미만이어야 함
        """
        cfg = self.config
        selected_indices = {f.frame_idx for f in already_selected}

        # Group by bucket key
        buckets: Dict[str, List[Tuple[IdentityRecord, float, BucketLabel, float, Optional[PivotAssignment]]]] = defaultdict(list)
        for item in scored:
            r = item[0]
            if r.frame_idx not in selected_indices:
                buckets[item[2].key].append(item)

        # Track per-bucket timestamps
        bucket_timestamps: Dict[str, List[float]] = defaultdict(list)

        coverage = []
        for key, candidates in buckets.items():
            candidates.sort(key=lambda x: x[3], reverse=True)
            count = 0
            for r, stable, bucket, quality, assignment in candidates:
                if count >= cfg.coverage_max_per_bucket:
                    break

                # Temporal gap within same bucket
                if self._too_close_temporally(
                    r.timestamp_ms, bucket_timestamps[key], cfg.coverage_min_interval_ms
                ):
                    continue

                frame = IdentityFrame(
                    frame_idx=r.frame_idx,
                    timestamp_ms=r.timestamp_ms,
                    set_type="coverage",
                    bucket=bucket,
                    quality_score=quality,
                    stable_score=stable,
                    novelty_score=0.0,
                    pivot_name=assignment.pivot_name if assignment else None,
                    pivot_distance=assignment.pose_distance if assignment else 0.0,
                    face_crop_box=r.face_crop_box,
                    image_size=r.image_size,
                )
                coverage.append(frame)
                count += 1

                bucket_timestamps[key].append(r.timestamp_ms)

        return coverage

    @staticmethod
    def _too_close_temporally(
        timestamp_ms: float,
        existing_timestamps: List[float],
        min_interval_ms: float,
    ) -> bool:
        """이미 선택된 타임스탬프와 너무 가까운지 검사."""
        for ts in existing_timestamps:
            if abs(timestamp_ms - ts) < min_interval_ms:
                return True
        return False

    @staticmethod
    def _too_similar_visually(
        record: IdentityRecord,
        selected_embeddings: List[np.ndarray],
        max_similarity: float,
    ) -> bool:
        """Visual dedup — always False (no visual embedding available)."""
        return False

    def _select_challenges(
        self,
        scored: List[Tuple[IdentityRecord, float, BucketLabel, float, Optional[PivotAssignment]]],
        already_selected: List[IdentityFrame],
    ) -> List[IdentityFrame]:
        """극단 조건 + 안정 + 신규성으로 challenge 선택."""
        cfg = self.config
        selected_indices = {f.frame_idx for f in already_selected}

        # Candidates: not yet selected, loose gate, stable
        candidates = []
        for r, stable, bucket, quality, assignment in scored:
            if r.frame_idx in selected_indices:
                continue
            if not self._pass_loose_gate(r):
                continue
            if stable < cfg.challenge_min_stable:
                continue
            novelty = self._compute_novelty(r, [])
            # Combined score: novelty dominant, with quality and stability bonus
            combined = 0.5 * novelty + 0.3 * quality + 0.2 * stable
            candidates.append((r, stable, bucket, quality, assignment, novelty, combined))

        # Sort by combined score descending
        candidates.sort(key=lambda x: x[6], reverse=True)

        challenges = []
        for r, stable, bucket, quality, assignment, novelty, combined in candidates[:cfg.challenge_count]:
            challenges.append(IdentityFrame(
                frame_idx=r.frame_idx,
                timestamp_ms=r.timestamp_ms,
                set_type="challenge",
                bucket=bucket,
                quality_score=quality,
                stable_score=stable,
                novelty_score=novelty,
                pivot_name=assignment.pivot_name if assignment else None,
                pivot_distance=assignment.pose_distance if assignment else 0.0,
                face_crop_box=r.face_crop_box,
                image_size=r.image_size,
            ))

        return challenges
