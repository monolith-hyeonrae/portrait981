"""Momentscan App — visualpath App + visualbind 판단.

visualbase(I/O) → visualpath(App/FlowGraph) → vpx+plugins(측정) → visualbind(결합+판단)

Usage:
    from momentscan.app import Momentscan

    app = Momentscan(
        quality_model="models/quality_v1.pkl",
        expression_model="models/bind_v12.pkl",
        pose_model="models/pose_v10.pkl",
    )
    results = app.run("video.mp4", fps=2)
    summary = app.summary()
    selected = app.select_frames(results)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import visualpath as vp

from visualbind import VisualBind, HeuristicStrategy, TreeStrategy, bind_observations
from visualbind.judge import JudgmentResult
from visualbind.signals import SIGNAL_FIELDS

logger = logging.getLogger("momentscan.app")

# momentscan이 사용하는 analyzer 목록
MOMENTSCAN_MODULES = [
    "face.detect", "face.au", "face.expression", "head.pose",
    "face.parse", "face.quality", "face.lighting", "frame.quality",
]


@dataclass
class FrameResult:
    """프레임별 분석 결과."""
    frame_idx: int = 0
    timestamp_ms: float = 0.0
    image: Optional[np.ndarray] = field(default=None, repr=False)
    signals: dict = field(default_factory=dict)
    judgment: JudgmentResult = field(default_factory=JudgmentResult)
    face_detected: bool = False
    face_count: int = 0
    face_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    z_score: float = 0.0  # 비디오 내 상대적 특별함 (expression signal 기준)

    @property
    def gate_passed(self): return self.judgment.gate_passed
    @property
    def expression(self): return self.judgment.expression
    @property
    def expression_conf(self): return self.judgment.expression_conf
    @property
    def pose(self): return self.judgment.pose
    @property
    def pose_conf(self): return self.judgment.pose_conf
    @property
    def is_shoot(self): return self.face_detected and self.judgment.is_shoot


@dataclass
class SignalSummary:
    """Signal 분포 요약 — personmemory 전달용."""
    mean: np.ndarray
    cov: np.ndarray
    n_frames: int = 0
    n_shoot: int = 0
    expression_dist: dict = field(default_factory=dict)
    pose_dist: dict = field(default_factory=dict)
    face_embeddings: list = field(default_factory=list)


class Momentscan(vp.App):
    """간결한 momentscan — vp.App 상속 + visualbind 판단."""

    modules = MOMENTSCAN_MODULES
    fps = 2
    backend = "simple"

    def __init__(self, quality_model=None, expression_model=None, pose_model=None, **kwargs):
        self._quality_model = quality_model
        self._expression_model = expression_model
        self._pose_model = pose_model
        self.judge = None
        self._results: list[FrameResult] = []

    def setup(self):
        self.judge = VisualBind(
            gate=HeuristicStrategy(),
            quality=TreeStrategy.load(self._quality_model) if self._quality_model else None,
            expression=TreeStrategy.load(self._expression_model) if self._expression_model else None,
            pose=TreeStrategy.load(self._pose_model) if self._pose_model else None,
        )
        self._results = []
        logger.info("Momentscan ready")

    def on_frame(self, frame, terminal_results):
        observations = []
        for flow_data in terminal_results:
            observations.extend(getattr(flow_data, "observations", []))

        signals = bind_observations(observations)
        face_detected = signals.get("face_confidence", 0.0) > 0
        judgment = self.judge(signals) if face_detected else JudgmentResult()

        face_embedding = _extract_face_embedding(observations) if face_detected else None

        self._results.append(FrameResult(
            frame_idx=getattr(frame, "frame_id", 0),
            timestamp_ms=getattr(frame, "t_src_ns", 0) / 1_000_000,
            image=getattr(frame, "data", None),
            signals=signals,
            judgment=judgment,
            face_detected=face_detected,
            face_count=sum(1 for obs in observations
                          if getattr(obs, "source", "") == "face.detect"
                          and obs.signals.get("face_count", 0) > 0),
            face_embedding=face_embedding,
        ))
        return True

    # Expression signal fields (AU + Emotion) — z_score 계산에 사용
    _EXPR_FIELDS = [f for f in SIGNAL_FIELDS if f.startswith("au") or f.startswith("em_")]

    def after_run(self, result):
        """vp.App hook: temporal smoothing + Fast-Slow z_score.

        1. Temporal smoothing: 전체 65D signal에 이동평균 적용
        2. Fast-Slow z_score: expression signal (20D)만으로 계산
        """
        fields = list(SIGNAL_FIELDS)
        face_indices = [i for i, r in enumerate(self._results) if r.face_detected]

        # 1. Temporal smoothing (moving average, window=3)
        if len(face_indices) > 2:
            window = 3
            half = window // 2
            vectors = np.array([
                [self._results[i].signals.get(f, 0.0) for f in fields]
                for i in face_indices
            ])
            smoothed = np.copy(vectors)
            for j in range(len(face_indices)):
                start = max(0, j - half)
                end = min(len(face_indices), j + half + 1)
                smoothed[j] = vectors[start:end].mean(axis=0)

            for j, idx in enumerate(face_indices):
                for k, f in enumerate(fields):
                    self._results[idx].signals[f] = float(smoothed[j, k])

        # 2. Fast-Slow z_score (expression signal only, 20D)
        expr_fields = self._EXPR_FIELDS
        face_results = [self._results[i] for i in face_indices]

        if len(face_results) > 2 and expr_fields:
            expr_vectors = np.array([
                [r.signals.get(f, 0.0) for f in expr_fields]
                for r in face_results
            ])
            mu = expr_vectors.mean(axis=0)
            var = expr_vectors.var(axis=0)
            var = np.maximum(var, 1e-8)

            for r in face_results:
                sig = np.array([r.signals.get(f, 0.0) for f in expr_fields])
                z = (sig - mu) / np.sqrt(var)
                r.z_score = float(np.sqrt(np.mean(z ** 2)))

        logger.info("Analyzed %d frames, %d SHOOT",
                    len(self._results),
                    sum(1 for r in self._results if r.is_shoot))
        return self._results

    def teardown(self):
        pass

    def summary(self, results: list[FrameResult] | None = None) -> SignalSummary:
        """SHOOT 프레임의 signal 분포 요약."""
        results = results or self._results
        shoot_results = [r for r in results if r.is_shoot]

        if not shoot_results:
            ndim = len(SIGNAL_FIELDS)
            return SignalSummary(mean=np.zeros(ndim), cov=np.zeros((ndim, ndim)))

        fields = list(SIGNAL_FIELDS)
        vectors = np.array([[r.signals.get(f, 0.0) for f in fields] for r in shoot_results])
        mean = vectors.mean(axis=0)
        cov = np.cov(vectors, rowvar=False) if len(vectors) > 1 else np.zeros((len(fields), len(fields)))

        expr_counts: dict[str, int] = {}
        for r in shoot_results:
            expr_counts[r.expression] = expr_counts.get(r.expression, 0) + 1
        total_expr = sum(expr_counts.values())
        expr_dist = {k: v / total_expr for k, v in expr_counts.items()} if total_expr > 0 else {}

        pose_counts: dict[str, int] = {}
        for r in shoot_results:
            if r.pose:
                pose_counts[r.pose] = pose_counts.get(r.pose, 0) + 1
        total_pose = sum(pose_counts.values())
        pose_dist = {k: v / total_pose for k, v in pose_counts.items()} if total_pose > 0 else {}

        embeddings = [r.face_embedding for r in shoot_results if r.face_embedding is not None]

        return SignalSummary(
            mean=mean, cov=cov, n_frames=len(shoot_results), n_shoot=len(shoot_results),
            expression_dist=expr_dist, pose_dist=pose_dist, face_embeddings=embeddings,
        )

    @staticmethod
    def _qez_score(r: FrameResult) -> float:
        """q×e×z 결합 스코어: 품질 × 표정확신 × 상대적특별함."""
        q = r.judgment.quality_conf if r.judgment.quality == "shoot" else r.judgment.quality_conf * 0.5
        e = r.expression_conf
        z = max(r.z_score, 0.1)  # z_score 0인 경우 (프레임 부족 등) 최소값
        return q * e * z

    def select_frames(self, results, top_k=10):
        """다양성 기반 프레임 선택 (expression × pose 버킷별 best q×e×z)."""
        buckets = {}
        for r in results:
            if not r.is_shoot:
                continue
            key = f"{r.expression}|{r.pose}"
            score = self._qez_score(r)
            if key not in buckets or score > self._qez_score(buckets[key]):
                buckets[key] = r
        return sorted(buckets.values(), key=lambda r: -self._qez_score(r))[:top_k]

    def run_single_image(self, image: np.ndarray) -> FrameResult:
        """단일 이미지 → FlowGraph 경로로 65D signal 추출.

        비디오 파이프라인과 동일한 analyzer + bind_observations 경로를 사용.
        SignalExtractor를 대체하는 정상 경로.

        Args:
            image: BGR 이미지 (np.ndarray).

        Returns:
            FrameResult with 65D signals + judgment.
        """
        from visualbase.core.frame import Frame
        from visualpath.runner import get_backend, resolve_modules as _resolve_modules

        self.setup()
        try:
            resolved = self.configure_modules(list(self.modules))
            graph = self.configure_graph(resolved)

            h, w = image.shape[:2]
            frame = Frame.from_array(image, frame_id=0, t_src_ns=0)

            engine = get_backend(self.backend)
            engine.execute(iter([frame]), graph, on_frame=self.on_frame)

            if self._results:
                return self._results[0]
            return FrameResult(frame_idx=0, timestamp_ms=0.0, image=image,
                               signals={}, judgment=JudgmentResult())
        finally:
            self.teardown()


def _extract_face_embedding(observations: list) -> np.ndarray | None:
    for obs in observations:
        if getattr(obs, "source", "") != "face.detect":
            continue
        data = getattr(obs, "data", None)
        if data is None:
            continue
        faces = getattr(data, "faces", [])
        if not faces:
            continue
        face = max(faces, key=lambda f: getattr(f, "area_ratio", 0.0))
        embedding = getattr(face, "embedding", None)
        if embedding is not None:
            return np.array(embedding, dtype=np.float32)
    return None
