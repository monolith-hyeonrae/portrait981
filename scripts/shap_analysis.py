"""SHAP 분석: XGBoost feature 기여도 시각화.

전체 데이터셋의 feature importance + 개별 프레임 분석.

Usage:
    uv run python scripts/shap_analysis.py
    uv run python scripts/shap_analysis.py --model models/bind_v4.pkl --output shap_report.html
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("shap_analysis")


def fig_to_b64(fig):
    """Matplotlib figure to base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    buf.close()
    return b64


def main():
    parser = argparse.ArgumentParser(description="SHAP analysis for XGBoost models")
    parser.add_argument("--model", default="models/bind_v4.pkl")
    parser.add_argument("--dataset", default="data/datasets/portrait-v1")
    parser.add_argument("--output", "-o", default="shap_report.html")
    args = parser.parse_args()

    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load
    clf = joblib.load(args.model)
    meta_path = Path(args.model).with_suffix(".json")
    with open(meta_path) as f:
        meta = json.load(f)
    classes = meta["classes"]

    signals = pd.read_parquet(Path(args.dataset) / "signals.parquet")
    labels = pd.read_csv(Path(args.dataset) / "labels.csv")
    df = signals.merge(labels[["filename", "expression"]], on="filename", how="inner")
    df = df[df["expression"].notna() & (df["expression"] != "")]
    signal_cols = meta["feature_names"]
    X = df[signal_cols].values
    y_labels = df["expression"].values

    logger.info("Computing SHAP values for %d samples, %d features, %d classes...", len(X), len(signal_cols), len(classes))

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)  # shape: (n_samples, n_features, n_classes)
    shap_arr = np.array(shap_values)  # (n_samples, n_features, n_classes)

    logger.info("SHAP computed: %s", shap_arr.shape)

    COLORS = {
        'cheese': '#4CAF50', 'chill': '#2196F3', 'cut': '#d32f2f',
        'edge': '#FF5722', 'goofy': '#E91E63', 'hype': '#9C27B0',
    }

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SHAP Analysis</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #f5f5f5; color: #333; margin: 20px; max-width: 1200px; }}
h1 {{ color: #e94560; }}
h2 {{ margin-top: 32px; color: #444; }}
h3 {{ color: #666; }}
.summary {{ background: #fff; padding: 16px; border-radius: 8px; border: 1px solid #e0e0e0; margin: 12px 0; }}
img {{ max-width: 100%; border-radius: 6px; margin: 8px 0; }}
.class-section {{ background: #fff; padding: 16px; border-radius: 8px; border: 1px solid #e0e0e0; margin: 12px 0; }}
table {{ border-collapse: collapse; font-size: 12px; }}
th, td {{ padding: 4px 12px; border: 1px solid #e0e0e0; text-align: right; }}
th {{ background: #f5f5f5; }}
.pos {{ color: #4CAF50; }}
.neg {{ color: #d32f2f; }}
</style>
</head><body>
<h1>SHAP Feature Analysis</h1>
<div class="summary">
    Model: {meta.get('version','?')} | {len(X)} samples | {len(signal_cols)}D features | {len(classes)} classes
</div>
"""

    # --- 1. Global: mean |SHAP| per class ---
    html += "<h2>1. Feature Importance by Class</h2>"
    html += "<p>각 class 예측에 가장 크게 기여하는 feature (mean |SHAP|)</p>"

    for ci, cls in enumerate(classes):
        color = COLORS.get(cls, '#666')
        shap_cls = shap_arr[:, :, ci]  # (n_samples, n_features)
        mean_abs = np.abs(shap_cls).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:10]

        html += f'<div class="class-section"><h3 style="color:{color}">{cls}</h3>'
        html += '<table><tr><th>Feature</th><th>mean |SHAP|</th><th>Direction</th></tr>'
        for idx in top_idx:
            feat = signal_cols[idx]
            val = mean_abs[idx]
            # Direction: positive SHAP = pushes toward this class
            mean_signed = shap_cls[:, idx].mean()
            direction = "→ " + cls if mean_signed > 0 else "← away"
            dir_cls = "pos" if mean_signed > 0 else "neg"
            html += f'<tr><td style="text-align:left">{feat}</td><td>{val:.4f}</td><td class="{dir_cls}">{direction}</td></tr>'
        html += '</table></div>'

    # --- 2. Summary plot (all classes) ---
    html += "<h2>2. Global Summary Plot</h2>"
    html += "<p>전체 데이터에서 각 feature의 SHAP 분포</p>"

    # Mean absolute across classes for summary
    shap_mean_across_classes = np.abs(shap_arr).mean(axis=2)  # (n_samples, n_features)

    fig, ax = plt.subplots(figsize=(10, 8))
    mean_importance = np.abs(shap_mean_across_classes).mean(axis=0)
    top20 = np.argsort(mean_importance)[::-1][:20]
    top_names = [signal_cols[i] for i in top20]
    top_vals = mean_importance[top20]
    colors = ['#e94560' if i < 5 else '#666' for i in range(len(top20))]
    ax.barh(range(len(top20)), top_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("Mean |SHAP| (across all classes)")
    ax.set_title("Top 20 Features by Global SHAP Importance")
    plt.tight_layout()
    html += f'<img src="data:image/png;base64,{fig_to_b64(fig)}">'
    plt.close(fig)

    # --- 3. Per-class SHAP direction heatmap ---
    html += "<h2>3. Feature × Class Heatmap</h2>"
    html += "<p>각 feature가 각 class를 향해(+) 또는 반대로(-) 기여하는 정도</p>"

    top30 = np.argsort(mean_importance)[::-1][:30]
    heatmap_data = np.zeros((len(top30), len(classes)))
    for fi, feat_idx in enumerate(top30):
        for ci in range(len(classes)):
            heatmap_data[fi, ci] = shap_arr[:, feat_idx, ci].mean()

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=11, rotation=45)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels([signal_cols[i] for i in top30], fontsize=9)
    ax.set_title("Mean SHAP value (red=pushes toward class, blue=pushes away)")
    plt.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    html += f'<img src="data:image/png;base64,{fig_to_b64(fig)}">'
    plt.close(fig)

    # --- 4. Class-specific insights ---
    html += "<h2>4. Key Insights</h2>"
    html += '<div class="summary">'
    for ci, cls in enumerate(classes):
        shap_cls = shap_arr[:, :, ci]
        mean_signed = shap_cls.mean(axis=0)
        top_pos = np.argsort(mean_signed)[::-1][:3]
        top_neg = np.argsort(mean_signed)[:3]
        color = COLORS.get(cls, '#666')
        html += f'<p><b style="color:{color}">{cls}</b>: '
        html += 'pushed by <b>' + ', '.join(signal_cols[i] for i in top_pos) + '</b>'
        html += ' | blocked by <b>' + ', '.join(signal_cols[i] for i in top_neg) + '</b></p>'
    html += '</div>'

    html += "</body></html>"
    Path(args.output).write_text(html, encoding="utf-8")
    logger.info("Report saved: %s", args.output)


if __name__ == "__main__":
    main()
