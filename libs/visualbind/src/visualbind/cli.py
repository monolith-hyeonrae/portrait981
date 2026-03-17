"""CLI entry points for visualbind (analyze, train, eval)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .signals import SIGNAL_FIELDS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_parquet(path: str) -> tuple[np.ndarray, tuple[str, ...]]:
    """Load signal vectors from parquet. Returns (N,D) array + field names."""
    table = pq.read_table(path)
    all_cols = table.column_names
    fields = [f for f in SIGNAL_FIELDS if f in all_cols]
    if not fields:
        fields = [
            c for c in all_cols
            if table.schema.field(c).type
            in ("float", "double", "float32", "float64", "int32", "int64")
        ]
    if not fields:
        raise SystemExit(f"No signal columns found in {path}")
    df = table.select(fields).to_pandas()
    return df.values.astype(np.float64), tuple(fields)


def _load_labels_json(path: str) -> np.ndarray:
    """Load labels from JSON (flat list or {\"labels\": [...]})."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return np.array(data, dtype=str)
    if isinstance(data, dict) and "labels" in data:
        return np.array(data["labels"], dtype=str)
    raise SystemExit(f"Cannot parse labels from {path}")


def _labels_from_catalog(vectors: np.ndarray, catalog_path: Path) -> np.ndarray:
    """Assign pseudo-labels by nearest catalog profile."""
    from .profile import load_profiles
    from .strategies.catalog import CatalogStrategy

    strategy = CatalogStrategy(profiles=load_profiles(catalog_path))
    labels = []
    for vec in vectors:
        scores = strategy.predict(vec)
        labels.append(max(scores, key=scores.get) if scores else "")  # type: ignore[arg-type]
    return np.array(labels, dtype=str)


def _resolve_labels(args: argparse.Namespace, vectors: np.ndarray) -> np.ndarray:
    """Resolve labels from --labels JSON or --catalog pseudo-labels."""
    if args.labels:
        labels = _load_labels_json(args.labels)
        if len(labels) != len(vectors):
            raise SystemExit(f"Label count ({len(labels)}) != sample count ({len(vectors)})")
        return labels
    if args.catalog:
        return _labels_from_catalog(vectors, Path(args.catalog))
    raise SystemExit("Either --catalog or --labels is required.")


def _group_by_label(vectors: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Group vectors by label into category -> (N, D) dict."""
    groups: dict[str, list[np.ndarray]] = {}
    for vec, lbl in zip(vectors, labels):
        if lbl:
            groups.setdefault(lbl, []).append(vec)
    return {k: np.array(v) for k, v in groups.items()}


def _load_tree_model(pkl_path: Path):
    """Load a pickled TreeStrategy."""
    import pickle
    from .strategies.tree import TreeStrategy

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)  # noqa: S301
    strategy = TreeStrategy()
    strategy._model = data["model"]
    strategy._classes = data["classes"]
    return strategy


def _write_html(text: str, path: Path, title: str = "VisualBind Report") -> None:
    """Wrap plain text in minimal HTML."""
    html = (
        f"<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\"><title>{title}</title>\n"
        f"<style>body{{font-family:monospace;white-space:pre-wrap;margin:2em;}}</style></head>\n"
        f"<body><h1>{title}</h1>\n<pre>{text}</pre>\n</body></html>"
    )
    path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_analyze(args: argparse.Namespace) -> None:
    """Day 0 analysis: N_eff, correlations, distribution diagnostics."""
    from .analyzer import generate_report

    vectors, fields = _load_parquet(args.data)
    report = generate_report(vectors, fields, corr_threshold=args.corr_threshold)
    print(report)

    if args.output:
        out = Path(args.output)
        if out.suffix in (".html", ".htm"):
            _write_html(report, out, title="VisualBind Day 0 Analysis")
        else:
            out.write_text(report, encoding="utf-8")
        print(f"\nReport saved to {out}")


def cmd_train(args: argparse.Namespace) -> None:
    """Train a binding strategy and save the model."""
    vectors, fields = _load_parquet(args.data)
    labels = _resolve_labels(args, vectors)

    cat_vecs = _group_by_label(vectors, labels)
    n_samples = sum(len(v) for v in cat_vecs.values())
    print(f"Training data: {n_samples} samples, {len(cat_vecs)} categories")
    for name, vecs in sorted(cat_vecs.items()):
        print(f"  {name}: {len(vecs)} samples")

    # Build strategy
    strategy_name = args.strategy
    if strategy_name in ("xgboost", "logistic"):
        from .strategies.tree import TreeStrategy
        strategy = TreeStrategy(use_xgboost=(strategy_name == "xgboost"))
    elif strategy_name == "catalog":
        from .strategies.catalog import CatalogStrategy
        strategy = CatalogStrategy()
    else:
        raise SystemExit(f"Unknown strategy: {strategy_name}")

    strategy.fit(cat_vecs)

    # Save
    out = Path(args.output)
    meta = {
        "strategy": strategy_name,
        "signal_fields": list(fields),
        "n_samples": n_samples,
        "categories": sorted(cat_vecs.keys()),
    }

    if strategy_name == "catalog":
        meta["profiles"] = [
            {
                "name": p.name,
                "mean_signals": p.mean_signals.tolist(),
                "importance_weights": p.importance_weights.tolist(),
                "n_refs": p.n_refs,
            }
            for p in strategy.profiles
        ]
        out.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Model saved to {out}")
    else:
        import pickle
        meta_path = out.with_suffix(".json")
        pkl_path = out.with_suffix(".pkl")
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        with open(pkl_path, "wb") as f:
            pickle.dump({"model": strategy._model, "classes": strategy.classes}, f)
        print(f"Model saved to {pkl_path}")
        print(f"Metadata saved to {meta_path}")


def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate and compare strategies."""
    from .evaluator import compare_strategies

    vectors, fields = _load_parquet(args.data)
    labels = _resolve_labels(args, vectors)

    strats: dict[str, object] = {}

    # Catalog baseline
    if args.catalog:
        from .profile import load_profiles
        from .strategies.catalog import CatalogStrategy
        strats["catalog"] = CatalogStrategy(profiles=load_profiles(Path(args.catalog)))

    # Trained model
    if args.model:
        mp = Path(args.model)
        if mp.suffix == ".json":
            with open(mp) as f:
                mdata = json.load(f)
            if mdata.get("strategy") == "catalog":
                from .profile import CategoryProfile
                from .strategies.catalog import CatalogStrategy
                profiles = [
                    CategoryProfile(
                        name=p["name"],
                        mean_signals=np.array(p["mean_signals"]),
                        importance_weights=np.array(p["importance_weights"]),
                        n_refs=p.get("n_refs", 0),
                    )
                    for p in mdata["profiles"]
                ]
                strats["trained_catalog"] = CatalogStrategy(profiles=profiles)
            else:
                pkl = mp.with_suffix(".pkl")
                if not pkl.exists():
                    raise SystemExit(f"Pickle model not found: {pkl}")
                strats["trained_model"] = _load_tree_model(pkl)
        elif mp.suffix == ".pkl":
            strats["trained_model"] = _load_tree_model(mp)
        else:
            raise SystemExit(f"Unsupported model format: {mp.suffix}")

    if not strats:
        raise SystemExit("No strategies to evaluate. Provide --catalog and/or --model.")

    results = compare_strategies(strats, vectors, labels)  # type: ignore[arg-type]

    # Print
    print(f"{'=' * 60}\nVisualBind Evaluation Report\n{'=' * 60}")
    print(f"\nTest set: {len(vectors)} samples, {len(set(labels))} categories")
    for name, m in results.items():
        auc_str = f", AUC={m['auc']:.3f}" if "auc" in m else ""
        print(f"\n--- {name} (acc={m['accuracy']:.3f}{auc_str}) ---")
        for cls, acc in sorted(m["per_class_accuracy"].items()):
            print(f"    {cls}: {acc:.3f}")

    if args.output:
        out = Path(args.output)
        report = {"n_samples": len(vectors), "n_categories": len(set(labels)), "results": {}}
        for name, m in results.items():
            entry = dict(m)
            entry.pop("confusion_matrix", None)
            report["results"][name] = entry
        text = json.dumps(report, indent=2, ensure_ascii=False)
        if out.suffix in (".html", ".htm"):
            _write_html(text, out, title="VisualBind Evaluation Report")
        else:
            out.write_text(text, encoding="utf-8")
        print(f"\nReport saved to {out}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visualbind",
        description="VisualBind — multi-observer signal binding CLI",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # analyze
    p = sub.add_parser("analyze", help="Day 0 analysis (N_eff, correlations, distributions)")
    p.add_argument("--data", required=True, help="Signal parquet file")
    p.add_argument("--output", help="Output report file (text or .html)")
    p.add_argument("--corr-threshold", type=float, default=0.7, help="Correlation flag threshold")

    # train
    p = sub.add_parser("train", help="Train a binding strategy")
    p.add_argument("--data", required=True, help="Signal parquet file")
    p.add_argument("--catalog", help="Catalog directory for pseudo-labels")
    p.add_argument("--labels", help="JSON file with per-row human labels")
    p.add_argument("--output", default="model.json", help="Output model file (default: model.json)")
    p.add_argument("--strategy", default="xgboost", choices=["xgboost", "logistic", "catalog"],
                    help="Strategy to train (default: xgboost)")

    # eval
    p = sub.add_parser("eval", help="Evaluate and compare strategies")
    p.add_argument("--data", required=True, help="Signal parquet file")
    p.add_argument("--catalog", help="Catalog directory for baseline + labels")
    p.add_argument("--labels", help="JSON file with per-row labels")
    p.add_argument("--model", help="Trained model file (.json or .pkl)")
    p.add_argument("--output", help="Output report file (JSON or .html)")

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    {"analyze": cmd_analyze, "train": cmd_train, "eval": cmd_eval}[args.command](args)
