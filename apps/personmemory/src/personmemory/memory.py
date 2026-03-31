"""PersonMemory — per-member person-conditioned distribution store.

한 사람의 기억. Entity store + Online stats + ANN index.

Usage:
    from personmemory import PersonMemory

    memory = PersonMemory("test_3")
    memory.ingest(workflow_id="ride_1", frames=shoot_results)
    refs = memory.get_reference(expression="cheese", pose="front")
    profile = memory.profile()

    # 전체 관리
    PersonMemory.list_all()
    PersonMemory.rename("test_3", "member_042")
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("personmemory")

def _get_member_dir(member_id: str) -> Path:
    """~/.portrait981/personmemory/{shard}/{member_id}/"""
    from personmemory.paths import get_bank_dir
    return get_bank_dir(member_id)


def _get_base_dir() -> Path:
    """~/.portrait981/personmemory/"""
    from personmemory.paths import get_bank_base_dir
    return get_bank_base_dir()


@dataclass
class MemoryNode:
    """Expression × Pose 버킷의 기억 노드."""
    expression: str
    pose: str
    signal_mean: np.ndarray          # 43D
    signal_var: np.ndarray           # 43D (variance, not std)
    n_observed: int = 0
    best_frame_path: str = ""
    best_confidence: float = 0.0
    workflows: list[str] = field(default_factory=list)
    last_seen: str = ""

    def update_welford(self, signal: np.ndarray) -> None:
        """Welford online update for mean and variance."""
        self.n_observed += 1
        delta = signal - self.signal_mean
        self.signal_mean += delta / self.n_observed
        delta2 = signal - self.signal_mean
        self.signal_var += delta * delta2

    @property
    def signal_std(self) -> np.ndarray:
        if self.n_observed < 2:
            return np.zeros_like(self.signal_var)
        return np.sqrt(self.signal_var / (self.n_observed - 1))


@dataclass
class Profile:
    """Member 전체 프로필 — 외부 공개용."""
    member_id: str
    n_visits: int
    n_total_frames: int
    n_nodes: int
    expression_dist: dict[str, float]
    pose_dist: dict[str, float]
    signal_mean: np.ndarray
    signal_std: np.ndarray
    coverage: dict[str, bool]
    gaps: list[str]


def _portrait_quality_score(frame) -> float:
    """Portrait quality score — lighting + expression + sharpness.

    dramatic lighting × expression confidence × sharpness.
    personmemory의 best frame 선정에 사용.
    """
    signals = getattr(frame, "signals", {})
    conf = getattr(frame, "expression_conf", 0.0)

    lighting_ratio = signals.get("lighting_ratio", 1.0)
    brightness_std = signals.get("face_brightness_std", 0.0)
    blur = signals.get("face_blur", 0.0)

    # Lighting quality: dramatic > natural > flat
    lighting_q = lighting_ratio * brightness_std / 100.0  # normalize ~0-1

    # Sharpness: blur > 50 is good
    sharpness = min(blur / 50.0, 1.0)

    # Combined: lighting × confidence × sharpness
    return lighting_q * conf * sharpness


class PersonMemory:
    """한 사람의 기억 — conditional distribution store.

    identity-anchored multi-vector store:
      - Stateful accumulation (Welford μ/Σ)
      - Heterogeneous retrieval (images, distribution, embedding)
      - Marginal value scoring (new frame value vs existing memory)
    """

    def __init__(self, member_id: str, store_dir: Optional[Path] = None):
        self.member_id = member_id
        self._store_dir = Path(store_dir) if store_dir else _get_member_dir(member_id)

        self.nodes: list[MemoryNode] = []
        self.n_visits: int = 0
        self.n_total_frames: int = 0

        # Global distribution (43D)
        self._global_n: int = 0
        self._global_mean: Optional[np.ndarray] = None
        self._global_m2: Optional[np.ndarray] = None

        # Face embeddings
        self._embeddings: list[np.ndarray] = []
        self._dominant_embedding: Optional[np.ndarray] = None

        # Auto-load if exists
        self._try_load()

    # ── Ingest ──

    def ingest(
        self,
        workflow_id: str,
        frames: list,
        timestamp: str = "",
        auto_save: bool = True,
    ) -> dict:
        """Workflow 결과를 기억에 축적.

        Args:
            workflow_id: 탑승 식별자
            frames: list[FrameResult] — SHOOT 프레임들
            timestamp: ISO timestamp
            auto_save: 자동 저장 여부

        Returns:
            dict with ingest stats
        """
        from visualbind.signals import SIGNAL_FIELDS
        ndim = len(SIGNAL_FIELDS)
        fields = list(SIGNAL_FIELDS)

        if self._global_mean is None:
            self._global_mean = np.zeros(ndim)
            self._global_m2 = np.zeros(ndim)

        new_nodes = 0
        updated_nodes = 0

        for frame in frames:
            if not getattr(frame, "is_shoot", False):
                continue

            expr = frame.expression
            pose = frame.pose
            if not expr or not pose:
                continue

            signal = np.array([frame.signals.get(f, 0.0) for f in fields])

            # Global Welford update
            self._global_n += 1
            delta = signal - self._global_mean
            self._global_mean += delta / self._global_n
            delta2 = signal - self._global_mean
            self._global_m2 += delta * delta2

            # Node update
            node = self._find_node(expr, pose)
            if node is None:
                node = MemoryNode(
                    expression=expr, pose=pose,
                    signal_mean=signal.copy(),
                    signal_var=np.zeros(ndim),
                    n_observed=0,
                )
                self.nodes.append(node)
                new_nodes += 1

            node.update_welford(signal)
            if not node.workflows or node.workflows[-1] != workflow_id:
                node.workflows.append(workflow_id)
            node.last_seen = timestamp

            # Best frame update — portrait quality score
            portrait_score = _portrait_quality_score(frame)
            if portrait_score > node.best_confidence:
                node.best_confidence = portrait_score
                img = getattr(frame, "image", None)
                if img is not None:
                    node.best_frame_path = self._save_frame(
                        img, expr, pose, frame.frame_idx, workflow_id
                    )
                updated_nodes += 1

            # Face embedding
            emb = getattr(frame, "face_embedding", None)
            if emb is not None:
                self._embeddings.append(emb)

            self.n_total_frames += 1

        self.n_visits += 1

        # Dominant embedding
        if self._embeddings:
            self._dominant_embedding = np.mean(self._embeddings[-20:], axis=0)
            self._dominant_embedding /= np.linalg.norm(self._dominant_embedding) + 1e-8

        logger.info("Ingest %s workflow=%s: %d new nodes, %d updated",
                    self.member_id, workflow_id, new_nodes, updated_nodes)

        if auto_save:
            self.save()

        return {"new_nodes": new_nodes, "updated_nodes": updated_nodes,
                "n_frames": self.n_total_frames, "total_nodes": len(self.nodes)}

    # ── Retrieval ──

    def get_reference(
        self,
        expression: Optional[str] = None,
        pose: Optional[str] = None,
        top_k: int = 3,
    ) -> list[str]:
        """맥락에 적합한 참조 이미지 경로 반환."""
        candidates = []
        for node in self.nodes:
            if expression and node.expression != expression:
                continue
            if pose and node.pose != pose:
                continue
            if node.best_frame_path:
                candidates.append((node.best_confidence, node.best_frame_path))

        candidates.sort(key=lambda x: -x[0])
        return [path for _, path in candidates[:top_k]]

    def get_profile_reference(self, top_k: int = 1) -> list[str]:
        """이 사람을 가장 잘 대표하는 참조 이미지."""
        if not self.nodes:
            return []
        ranked = sorted(self.nodes, key=lambda n: n.n_observed * n.best_confidence, reverse=True)
        return [n.best_frame_path for n in ranked[:top_k] if n.best_frame_path]

    # ── Profile ──

    def profile(self) -> Profile:
        """전체 프로필."""
        total = sum(n.n_observed for n in self.nodes)
        expr_counts: dict[str, int] = {}
        pose_counts: dict[str, int] = {}
        for n in self.nodes:
            expr_counts[n.expression] = expr_counts.get(n.expression, 0) + n.n_observed
            pose_counts[n.pose] = pose_counts.get(n.pose, 0) + n.n_observed

        expr_dist = {k: v / total for k, v in expr_counts.items()} if total > 0 else {}
        pose_dist = {k: v / total for k, v in pose_counts.items()} if total > 0 else {}

        all_exprs = ["cheese", "chill", "edge", "goofy", "hype"]
        all_poses = ["front", "angle", "side"]
        existing = {f"{n.expression}|{n.pose}" for n in self.nodes}
        coverage = {}
        gaps = []
        for e in all_exprs:
            for p in all_poses:
                key = f"{e}|{p}"
                has = key in existing
                coverage[key] = has
                if not has:
                    gaps.append(key)

        global_mean = self._global_mean if self._global_mean is not None else np.zeros(1)
        global_std = np.zeros_like(global_mean)
        if self._global_n > 1 and self._global_m2 is not None:
            global_std = np.sqrt(self._global_m2 / (self._global_n - 1))

        return Profile(
            member_id=self.member_id, n_visits=self.n_visits,
            n_total_frames=self.n_total_frames, n_nodes=len(self.nodes),
            expression_dist=expr_dist, pose_dist=pose_dist,
            signal_mean=global_mean, signal_std=global_std,
            coverage=coverage, gaps=gaps,
        )

    # ── Marginal Value ──

    def marginal_value(self, signal: np.ndarray) -> float:
        """새 프레임의 marginal value — 기존 분포 대비 새로운 정도."""
        if self._global_n < 2 or self._global_m2 is None:
            return 1.0
        var = self._global_m2 / (self._global_n - 1)
        var = np.maximum(var, 1e-8)
        z = (signal - self._global_mean) / np.sqrt(var)
        return float(np.sqrt(np.mean(z ** 2)))

    # ── Persistence ──

    def save(self) -> Path:
        """디스크에 저장."""
        self._store_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "member_id": self.member_id,
            "n_visits": self.n_visits,
            "n_total_frames": self.n_total_frames,
            "global_n": self._global_n,
            "nodes": [
                {
                    "expression": n.expression, "pose": n.pose,
                    "signal_mean": n.signal_mean.tolist(),
                    "signal_var": n.signal_var.tolist(),
                    "n_observed": n.n_observed,
                    "best_frame_path": n.best_frame_path,
                    "best_confidence": n.best_confidence,
                    "workflows": n.workflows, "last_seen": n.last_seen,
                }
                for n in self.nodes
            ],
        }
        if self._global_mean is not None:
            data["global_mean"] = self._global_mean.tolist()
            data["global_m2"] = self._global_m2.tolist()
        if self._dominant_embedding is not None:
            data["dominant_embedding"] = self._dominant_embedding.tolist()

        path = self._store_dir / "memory.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved: %s (%d nodes)", self.member_id, len(self.nodes))
        return path

    def _try_load(self) -> None:
        """디스크에서 로딩 (있으면)."""
        path = self._store_dir / "memory.json"
        if not path.exists():
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        self.n_visits = data.get("n_visits", 0)
        self.n_total_frames = data.get("n_total_frames", 0)
        self._global_n = data.get("global_n", 0)

        if "global_mean" in data:
            self._global_mean = np.array(data["global_mean"])
            self._global_m2 = np.array(data["global_m2"])
        if "dominant_embedding" in data:
            self._dominant_embedding = np.array(data["dominant_embedding"])

        for nd in data.get("nodes", []):
            self.nodes.append(MemoryNode(
                expression=nd["expression"], pose=nd["pose"],
                signal_mean=np.array(nd["signal_mean"]),
                signal_var=np.array(nd["signal_var"]),
                n_observed=nd["n_observed"],
                best_frame_path=nd.get("best_frame_path", ""),
                best_confidence=nd.get("best_confidence", 0.0),
                workflows=nd.get("workflows", []),
                last_seen=nd.get("last_seen", ""),
            ))

        logger.info("Loaded: %s (%d nodes, %d visits)", self.member_id, len(self.nodes), self.n_visits)

    # ── Class methods (전체 관리) ──

    @classmethod
    def list_all(cls) -> list[str]:
        """등록된 전체 member 목록."""
        from personmemory.paths import list_member_ids
        return list_member_ids()

    @classmethod
    def rename(cls, old_id: str, new_id: str) -> None:
        """Member ID 변경."""
        old_dir = _get_member_dir(old_id)
        new_dir = _get_member_dir(new_id)
        if not old_dir.exists():
            raise FileNotFoundError(f"Member not found: {old_id}")
        if new_dir.exists():
            raise FileExistsError(f"Member already exists: {new_id}")

        new_dir.parent.mkdir(parents=True, exist_ok=True)
        old_dir.rename(new_dir)
        mem = cls(new_id)
        mem.member_id = new_id
        mem.save()
        logger.info("Renamed: %s → %s", old_id, new_id)

    @classmethod
    def delete(cls, member_id: str) -> None:
        """Member 기억 삭제."""
        member_dir = _get_member_dir(member_id)
        if member_dir.exists():
            shutil.rmtree(member_dir)
        logger.info("Deleted: %s", member_id)

    # ── Internal ──

    def _find_node(self, expression: str, pose: str) -> Optional[MemoryNode]:
        for n in self.nodes:
            if n.expression == expression and n.pose == pose:
                return n
        return None

    def _save_frame(self, image: np.ndarray, expr: str, pose: str,
                    frame_idx: int, workflow_id: str) -> str:
        import cv2
        images_dir = self._store_dir / "frames"
        images_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{expr}_{pose}_{workflow_id}_{frame_idx:04d}.jpg"
        path = images_dir / filename
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return str(path)
