#!/usr/bin/env python3
"""VisualBind Demo: Pose + Expression → unified embedding.

Scenario A: Two independent observers (pose model, expression model)
produce signals about the same person. VisualBind learns a unified
embedding where similar states cluster together.

Synthetic data simulates 4 person states:
  1. happy_frontal  — smiling, facing camera
  2. happy_turned   — smiling, head turned
  3. neutral_frontal — neutral, facing camera
  4. surprise_open   — surprised, mouth open

Each state has characteristic pose + expression + AU signals.
The demo runs the full 4-stage pipeline:
  Collect → Agree → Pair → Encode

Usage:
    python -m demos.demo_pose_expression
    # or
    cd libs/visualbind && uv run python demos/demo_pose_expression.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from visualbind import (
    AgreementEngine,
    CrossCheck,
    HintCollector,
    PairMiner,
    SourceSpec,
    TripletEncoder,
)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

@dataclass
class PersonState:
    """Ground-truth person state for synthetic data."""
    name: str
    label: int
    # Signal distributions: {source: {signal: (mean, std)}}
    distributions: Dict[str, Dict[str, Tuple[float, float]]]


STATES = [
    PersonState(
        name="happy_frontal",
        label=0,
        distributions={
            "face.expression": {
                "em_happy": (0.85, 0.08),
                "em_neutral": (0.10, 0.05),
                "em_surprise": (0.03, 0.02),
                "em_angry": (0.02, 0.01),
            },
            "face.au": {
                "AU6": (3.5, 0.5),   # cheek raiser (Duchenne)
                "AU12": (4.0, 0.4),  # lip corner puller
                "AU25": (1.0, 0.5),  # lips part
                "AU26": (0.5, 0.3),  # jaw drop
            },
            "head.pose": {
                "yaw": (0.0, 5.0),     # frontal
                "pitch": (0.0, 3.0),
                "roll": (0.0, 2.0),
            },
        },
    ),
    PersonState(
        name="happy_turned",
        label=1,
        distributions={
            "face.expression": {
                "em_happy": (0.75, 0.10),
                "em_neutral": (0.15, 0.08),
                "em_surprise": (0.05, 0.03),
                "em_angry": (0.02, 0.01),
            },
            "face.au": {
                "AU6": (3.0, 0.6),
                "AU12": (3.5, 0.5),
                "AU25": (0.8, 0.4),
                "AU26": (0.3, 0.2),
            },
            "head.pose": {
                "yaw": (35.0, 8.0),   # turned
                "pitch": (5.0, 4.0),
                "roll": (3.0, 3.0),
            },
        },
    ),
    PersonState(
        name="neutral_frontal",
        label=2,
        distributions={
            "face.expression": {
                "em_happy": (0.05, 0.03),
                "em_neutral": (0.85, 0.08),
                "em_surprise": (0.05, 0.03),
                "em_angry": (0.05, 0.03),
            },
            "face.au": {
                "AU6": (0.3, 0.2),
                "AU12": (0.2, 0.2),
                "AU25": (0.5, 0.3),
                "AU26": (0.3, 0.2),
            },
            "head.pose": {
                "yaw": (0.0, 4.0),
                "pitch": (0.0, 3.0),
                "roll": (0.0, 2.0),
            },
        },
    ),
    PersonState(
        name="surprise_open",
        label=3,
        distributions={
            "face.expression": {
                "em_happy": (0.05, 0.03),
                "em_neutral": (0.05, 0.03),
                "em_surprise": (0.80, 0.10),
                "em_angry": (0.05, 0.03),
            },
            "face.au": {
                "AU6": (0.5, 0.3),
                "AU12": (0.3, 0.2),
                "AU25": (3.5, 0.5),  # lips part
                "AU26": (4.0, 0.4),  # jaw drop
            },
            "head.pose": {
                "yaw": (0.0, 8.0),
                "pitch": (-5.0, 5.0),
                "roll": (0.0, 3.0),
            },
        },
    ),
]


def generate_samples(
    n_per_state: int = 100,
    seed: int = 42,
) -> Tuple[List[Dict[str, Dict[str, float]]], List[int], List[str]]:
    """Generate synthetic multi-observer signal samples.

    Returns:
        (signals_list, labels, state_names)
        signals_list: [{source: {signal: value}}, ...]
        labels: ground-truth state index
        state_names: state name per sample
    """
    rng = np.random.default_rng(seed)
    signals_list = []
    labels = []
    names = []

    for state in STATES:
        for _ in range(n_per_state):
            signals: Dict[str, Dict[str, float]] = {}
            for source, sig_dists in state.distributions.items():
                signals[source] = {}
                for sig_name, (mean, std) in sig_dists.items():
                    val = rng.normal(mean, std)
                    signals[source][sig_name] = float(val)
            signals_list.append(signals)
            labels.append(state.label)
            names.append(state.name)

    return signals_list, labels, names


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_demo(n_per_state: int = 200, seed: int = 42, verbose: bool = True):
    """Run the full 4-stage visualbind pipeline on synthetic data."""

    if verbose:
        print("=" * 60)
        print("VisualBind Demo: Pose + Expression → Unified Embedding")
        print("=" * 60)

    # --- Generate data ---
    signals_list, labels, state_names = generate_samples(n_per_state, seed)
    n_total = len(signals_list)
    if verbose:
        print(f"\n[Data] {n_total} samples, {len(STATES)} states "
              f"({n_per_state} per state)")
        for s in STATES:
            print(f"  - {s.name} (label={s.label})")

    # --- Stage 1: Collect + Normalize ---
    collector = HintCollector({
        "face.expression": SourceSpec(
            signals=("em_happy", "em_neutral", "em_surprise", "em_angry"),
            normalize="softmax",
        ),
        "face.au": SourceSpec(
            signals=("AU6", "AU12", "AU25", "AU26"),
            normalize="minmax",
            range=(0.0, 5.0),
        ),
        "head.pose": SourceSpec(
            signals=("yaw", "pitch", "roll"),
            normalize="minmax",
            range=(-90.0, 90.0),
        ),
    })

    hint_frames = [
        collector.collect_from_signals(sig, frame_id=i)
        for i, sig in enumerate(signals_list)
    ]

    input_dim = len(hint_frames[0].flat_vector())
    if verbose:
        print(f"\n[Stage 1: Collect] {input_dim}D hint vectors")
        print(f"  Sources: {collector.sources}")
        print(f"  Normalizations: "
              f"{', '.join(s.normalize for s in collector.specs.values())}")

    # --- Stage 2: Agreement ---
    engine = AgreementEngine([
        CrossCheck("face.expression", "em_happy", "face.au", "AU12",
                   relation="positive", weight=2.0,
                   description="smile ↔ lip corner puller"),
        CrossCheck("face.expression", "em_happy", "face.au", "AU6",
                   relation="positive", weight=1.5,
                   description="smile ↔ cheek raiser (Duchenne)"),
        CrossCheck("face.expression", "em_surprise", "face.au", "AU26",
                   relation="positive", weight=1.0,
                   description="surprise ↔ jaw drop"),
        CrossCheck("face.expression", "em_surprise", "face.au", "AU25",
                   relation="positive", weight=1.0,
                   description="surprise ↔ lips part"),
    ])

    agreements = [engine.compute(f) for f in hint_frames]
    scores = np.array([a.score for a in agreements])

    if verbose:
        print(f"\n[Stage 2: Agreement] {len(engine.checks)} cross-checks")
        for state in STATES:
            mask = np.array(labels) == state.label
            state_scores = scores[mask]
            print(f"  {state.name:20s}  "
                  f"agreement={state_scores.mean():.3f} ± {state_scores.std():.3f}")

    # --- Stage 3: Pair Mining ---
    # Use median as adaptive threshold
    median_score = float(np.median(scores))
    pos_thresh = max(median_score * 1.2, 0.1)
    neg_thresh = max(median_score * 0.5, 0.01)

    miner = PairMiner(
        positive_threshold=pos_thresh,
        negative_threshold=neg_thresh,
        max_pairs_per_anchor=2,
        seed=seed,
    )
    mining_result = miner.mine(hint_frames, agreements)

    if verbose:
        print(f"\n[Stage 3: Pair Mining] "
              f"pos_thresh={pos_thresh:.3f}, neg_thresh={neg_thresh:.3f}")
        print(f"  High agreement: {mining_result.n_high}")
        print(f"  Low agreement:  {mining_result.n_low}")
        print(f"  Ambiguous:      {mining_result.n_ambiguous}")
        print(f"  Triplets mined: {len(mining_result.pairs)}")

    if not mining_result.pairs:
        print("\n[!] No pairs mined. Adjust thresholds or data.")
        return None

    anchors, positives, negatives = mining_result.as_arrays()

    # --- Stage 4: Encode ---
    embed_dim = 4
    encoder = TripletEncoder(
        input_dim=input_dim,
        embed_dim=embed_dim,
        margin=0.3,
        lr=0.005,
        seed=seed,
    )

    if verbose:
        print(f"\n[Stage 4: Encode] {input_dim}D → {embed_dim}D")
        print(f"  Training on {len(mining_result.pairs)} triplets...")

    history = encoder.fit(
        anchors, positives, negatives,
        epochs=300,
        batch_size=64,
        verbose=verbose,
    )

    if verbose:
        print(f"\n  Final loss:    {history.final_loss:.4f}")
        print(f"  Converged:     {history.converged}")
        print(f"  Separation:    {history.separation:.4f} "
              f"(neg_d - pos_d, higher = better)")

    # --- Evaluation ---
    all_vectors = np.array([f.flat_vector() for f in hint_frames], dtype=np.float32)
    embeddings = encoder.encode(all_vectors)
    labels_arr = np.array(labels)

    if verbose:
        print(f"\n[Evaluation]")

    # Intra-class vs inter-class cosine similarity
    intra_sims = []
    inter_sims = []
    for i in range(n_total):
        for j in range(i + 1, min(i + 20, n_total)):  # sample pairs
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if labels_arr[i] == labels_arr[j]:
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)

    intra_arr = np.array(intra_sims) if intra_sims else np.array([0.0])
    inter_arr = np.array(inter_sims) if inter_sims else np.array([0.0])

    if verbose:
        print(f"  Intra-class similarity: {intra_arr.mean():.3f} ± {intra_arr.std():.3f}")
        print(f"  Inter-class similarity: {inter_arr.mean():.3f} ± {inter_arr.std():.3f}")
        gap = intra_arr.mean() - inter_arr.mean()
        print(f"  Gap (intra - inter):    {gap:.3f} "
              f"({'GOOD' if gap > 0.05 else 'WEAK'} separation)")

    # Per-state centroids
    if verbose:
        print(f"\n  State centroids ({embed_dim}D):")
        for state in STATES:
            mask = labels_arr == state.label
            centroid = embeddings[mask].mean(axis=0)
            print(f"    {state.name:20s}  [{', '.join(f'{v:.3f}' for v in centroid)}]")

    # Simple kNN accuracy (k=5)
    k = 5
    correct = 0
    for i in range(n_total):
        dists = np.linalg.norm(embeddings - embeddings[i], axis=1)
        dists[i] = float("inf")  # exclude self
        nn_indices = np.argsort(dists)[:k]
        nn_labels = labels_arr[nn_indices]
        pred = int(np.bincount(nn_labels).argmax())
        if pred == labels_arr[i]:
            correct += 1

    accuracy = correct / n_total
    if verbose:
        print(f"\n  kNN accuracy (k={k}): {accuracy:.1%}")
        print(f"\n{'=' * 60}")
        if accuracy > 0.7:
            print("SUCCESS: Unified embedding separates person states")
        else:
            print("PARTIAL: Some separation, needs tuning")
        print(f"{'=' * 60}")

    return {
        "accuracy": accuracy,
        "intra_sim": float(intra_arr.mean()),
        "inter_sim": float(inter_arr.mean()),
        "gap": float(intra_arr.mean() - inter_arr.mean()),
        "history": history,
    }


if __name__ == "__main__":
    run_demo()
