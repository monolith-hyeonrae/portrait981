"""catalog-build command for momentscan CLI.

참조 이미지를 파이프라인으로 분석하여 시그널 프로파일을 생성한다.

Usage:
    momentscan catalog-build catalogs/portrait-v1
    momentscan catalog-build catalogs/portrait-v1 --no-cache
"""

import sys
from pathlib import Path

from momentscan.cli.utils import BOLD, DIM, ITALIC, RESET


def run_catalog_build(args):
    """Run catalog profile building.

    Analyzes reference images → generates signal profiles → saves _profile.json.
    """
    from momentscan.algorithm.batch.catalog_build import (
        build_catalog_profiles,
        compute_separation_metrics,
        print_separation_report,
    )

    catalog_path = Path(args.path)
    if not catalog_path.is_dir():
        print(f"Error: catalog directory not found: {catalog_path}")
        sys.exit(1)

    categories_dir = catalog_path / "categories"
    if not categories_dir.is_dir():
        print(f"Error: categories/ directory not found in {catalog_path}")
        sys.exit(1)

    cache_enabled = not getattr(args, "no_cache", False)

    print()
    print(f"{BOLD}{'Catalog':<10}{RESET}{catalog_path}")
    print(f"          {DIM}cache: {'enabled' if cache_enabled else 'disabled'}{RESET}")
    print()

    print(f"{DIM}Building signal profiles...{RESET}")
    try:
        profiles = build_catalog_profiles(
            catalog_path,
            cache_enabled=cache_enabled,
        )
    except Exception as e:
        print(f"\nError during catalog build: {e}")
        sys.exit(1)

    print()

    # Summary
    ndim = len(profiles[0].mean_signals) if profiles else 0
    print(f"{BOLD}{'Profiles':<10}{RESET}{DIM}{len(profiles)} categories ({ndim}D signal profiles){RESET}")
    for p in profiles:
        print(f"  {p.name:<20}{DIM}{p.n_refs} refs{RESET}")
    print()

    # Separation metrics + report
    metrics = compute_separation_metrics(profiles)
    report = print_separation_report(profiles, metrics)
    print(report)
    print()

    # Warnings in red
    if metrics.warnings:
        RED = "\033[31m"
        for w in metrics.warnings:
            print(f"{RED}\u26a0 {w}{RESET}")
        print()

    # Output locations
    print(f"{BOLD}{'Output':<10}{RESET}")
    for p in profiles:
        profile_path = catalog_path / "categories" / p.name / "_profile.json"
        print(f"  {DIM}{profile_path}{RESET}")
    print()

    # HTML report
    report_path = getattr(args, "report", None)
    if report_path:
        report_path = Path(report_path)
        html = _generate_catalog_report_html(profiles, metrics, catalog_path)
        report_path.write_text(html, encoding="utf-8")
        print(f"{BOLD}{'Report':<10}{RESET}{report_path}")
        print()


def _generate_catalog_report_html(profiles, metrics, catalog_path):
    """Generate standalone HTML report with PCA scatter + separation metrics."""
    import numpy as np
    import json as _json
    from momentscan.algorithm.batch.catalog_scoring import SIGNAL_FIELDS

    ndim = len(SIGNAL_FIELDS)

    # PCA 2D projection of category mean signals
    pca_data = None
    if len(profiles) >= 2:
        vecs = np.array([p.mean_signals for p in profiles])
        mean = vecs.mean(axis=0)
        centered = vecs - mean
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            n_comp = min(2, len(S))
            pc = Vt[:n_comp]
            projected = centered @ pc.T
            total_var = (S ** 2).sum()
            explained = [(S[i] ** 2) / total_var for i in range(n_comp)]

            pca_data = {
                "names": [p.name for p in profiles],
                "x": projected[:, 0].tolist() if n_comp >= 1 else [0.0] * len(profiles),
                "y": projected[:, 1].tolist() if n_comp >= 2 else [0.0] * len(profiles),
                "n_refs": [p.n_refs for p in profiles],
                "explained": explained,
            }
        except Exception:
            pass

    # Pairwise distance data for bar chart
    dist_data = {
        "labels": [],
        "values": [],
    }
    for key, d in metrics.pairwise_distances.items():
        a, b = key.split("|")
        dist_data["labels"].append(f"{a} <-> {b}")
        dist_data["values"].append(round(d, 4))

    # Signal importance heatmap data
    heatmap_data = {
        "categories": [p.name for p in profiles],
        "signals": list(SIGNAL_FIELDS),
        "weights": [p.importance_weights.tolist() for p in profiles],
        "means": [p.mean_signals.tolist() for p in profiles],
    }

    data_json = _json.dumps({
        "pca": pca_data,
        "distances": dist_data,
        "heatmap": heatmap_data,
        "metrics": {
            "n_categories": metrics.n_categories,
            "min_distance": round(metrics.min_distance, 4) if metrics.min_distance < float("inf") else 0,
            "min_pair": list(metrics.min_pair),
            "mean_distance": round(metrics.mean_distance, 4),
            "silhouette": round(metrics.silhouette_approx, 4),
            "warnings": metrics.warnings,
        },
    })

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Catalog Separation Report — {catalog_path.name}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2em; background: #fafafa; }}
h1 {{ color: #333; }} h2 {{ color: #555; margin-top: 2em; }}
.metric {{ display: inline-block; padding: 1em 2em; margin: 0.5em; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
.metric .value {{ font-size: 2em; font-weight: bold; color: #1565c0; }}
.metric .label {{ font-size: 0.9em; color: #888; }}
.warning {{ color: #c62828; margin: 0.3em 0; }}
.chart {{ background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 1em; margin: 1em 0; }}
</style>
</head><body>
<h1>Catalog Separation Report</h1>
<p>{catalog_path} — {len(profiles)} categories, {ndim}D signal space</p>

<div id="metrics"></div>
<div id="warnings"></div>

<h2>Category Profiles (PCA 2D)</h2>
<div class="chart"><div id="pca"></div></div>

<h2>Pairwise Distances</h2>
<div class="chart"><div id="distances"></div></div>

<h2>Signal Importance Weights</h2>
<div class="chart"><div id="heatmap"></div></div>

<h2>Mean Signal Values</h2>
<div class="chart"><div id="means"></div></div>

<script>
var DATA = {data_json};

// Metrics
(function() {{
  var m = DATA.metrics;
  var html = '';
  html += '<div class="metric"><div class="value">' + m.n_categories + '</div><div class="label">Categories</div></div>';
  html += '<div class="metric"><div class="value">' + m.min_distance.toFixed(3) + '</div><div class="label">Min Distance</div></div>';
  html += '<div class="metric"><div class="value">' + m.mean_distance.toFixed(3) + '</div><div class="label">Mean Distance</div></div>';
  html += '<div class="metric"><div class="value">' + m.silhouette.toFixed(2) + '</div><div class="label">Silhouette</div></div>';
  document.getElementById('metrics').innerHTML = html;

  if (m.warnings.length > 0) {{
    var whtml = '<h3>Warnings</h3>';
    m.warnings.forEach(function(w) {{ whtml += '<p class="warning">\\u26a0 ' + w + '</p>'; }});
    document.getElementById('warnings').innerHTML = whtml;
  }}
}})();

// PCA scatter
if (DATA.pca) {{
  var pca = DATA.pca;
  var COLORS = ['#2e7d32','#1565c0','#e65100','#6a1b9a','#c62828','#00838f','#4e342e','#37474f'];
  var traces = [];
  pca.names.forEach(function(name, i) {{
    traces.push({{
      x: [pca.x[i]], y: [pca.y[i]],
      text: [name + ' (' + pca.n_refs[i] + ' refs)'],
      type: 'scatter', mode: 'markers+text',
      textposition: 'top center',
      marker: {{size: 20 + pca.n_refs[i] * 2, color: COLORS[i % COLORS.length], opacity: 0.8}},
      name: name, showlegend: true
    }});
  }});
  var title = 'PCA 2D';
  if (pca.explained && pca.explained.length >= 2) {{
    title += ' (' + (pca.explained[0]*100).toFixed(0) + '% + ' + (pca.explained[1]*100).toFixed(0) + '% variance)';
  }}
  Plotly.newPlot('pca', traces, {{title: title, xaxis: {{title: 'PC1'}}, yaxis: {{title: 'PC2'}}}});
}}

// Distance bar chart
(function() {{
  var d = DATA.distances;
  if (d.labels.length === 0) return;
  var colors = d.values.map(function(v) {{ return v < 0.15 ? '#c62828' : v < 0.40 ? '#f57f17' : '#2e7d32'; }});
  Plotly.newPlot('distances', [{{
    x: d.labels, y: d.values, type: 'bar',
    marker: {{color: colors}},
    text: d.values.map(function(v) {{ return v.toFixed(3); }}),
    textposition: 'outside'
  }}], {{title: 'Pairwise Weighted Distance', yaxis: {{title: 'Distance'}}}});
}})();

// Importance weights heatmap
(function() {{
  var h = DATA.heatmap;
  if (h.categories.length === 0) return;
  Plotly.newPlot('heatmap', [{{
    z: h.weights, x: h.signals, y: h.categories,
    type: 'heatmap', colorscale: 'Viridis'
  }}], {{title: 'Importance Weights', margin: {{l: 120, b: 120}}}});
}})();

// Mean signals heatmap
(function() {{
  var h = DATA.heatmap;
  if (h.categories.length === 0) return;
  Plotly.newPlot('means', [{{
    z: h.means, x: h.signals, y: h.categories,
    type: 'heatmap', colorscale: 'RdYlGn'
  }}], {{title: 'Mean Signal Values (normalized)', margin: {{l: 120, b: 120}}}});
}})();
</script>
</body></html>"""
