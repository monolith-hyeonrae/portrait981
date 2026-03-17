"""CatalogStrategy vs TreeStrategy vs LogisticRegression 비교.

참조 이미지 프로파일(centroid)까지의 거리 기반 판단 vs
학습 기반 판단의 cross-validated 비교.

pseudo-label 순환 참조를 피하기 위해:
- 각 카테고리에 대해 "해당 카테고리 참조 이미지와 유사한 프레임" vs "다른 카테고리 참조 이미지와 유사한 프레임"으로 양성/음성 구분
- 즉, 카테고리 간 구분 능력을 테스트

Usage:
    uv run python scripts/compare_strategies.py /tmp/day0_multi.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare")


def main():
    parser = argparse.ArgumentParser(description="Strategy Comparison")
    parser.add_argument("parquet", help="21D signal parquet")
    parser.add_argument("--catalog", default="data/catalogs/portrait-v1")
    args = parser.parse_args()

    import pyarrow.parquet as pq

    # 1. Load daily data
    table = pq.read_table(args.parquet)
    df = table.to_pandas()
    X_daily = df.values
    field_names = tuple(df.columns)
    logger.info("Daily data: %d samples × %d dims", *X_daily.shape)

    # 2. Load catalog profiles
    from visualbind.profile import load_profiles
    profiles = load_profiles(Path(args.catalog))
    cat_names = [p.name for p in profiles]
    logger.info("Categories: %s", cat_names)

    # 3. CatalogStrategy로 전체 프레임 스코어링
    from visualbind.strategies.catalog import CatalogStrategy
    cat_strat = CatalogStrategy(profiles=profiles)

    # 각 프레임의 best category 할당
    assignments = []
    score_matrix = np.zeros((len(X_daily), len(cat_names)))

    for i, vec in enumerate(X_daily):
        scores = cat_strat.predict(vec)
        for j, name in enumerate(cat_names):
            score_matrix[i, j] = scores.get(name, 0.0)
        best_name = max(scores, key=scores.get)
        assignments.append(best_name)

    # 4. 카테고리 쌍별 구분 테스트 (1-vs-1)
    print("\n" + "=" * 70)
    print("Binary Classification: 카테고리 쌍별 구분 능력 비교")
    print("=" * 70)

    results = []

    for i, cat_a in enumerate(cat_names):
        for j, cat_b in enumerate(cat_names):
            if j <= i:
                continue

            # cat_a에 할당된 프레임 vs cat_b에 할당된 프레임
            mask_a = np.array([a == cat_a for a in assignments])
            mask_b = np.array([a == cat_b for a in assignments])

            n_a, n_b = mask_a.sum(), mask_b.sum()
            if n_a < 10 or n_b < 10:
                continue

            X_pair = np.vstack([X_daily[mask_a], X_daily[mask_b]])
            y_pair = np.array([1] * n_a + [0] * n_b)

            # CatalogStrategy: similarity score 차이로 판단
            scores_a_for_a = score_matrix[mask_a, i]  # cat_a 프레임의 cat_a 점수
            scores_b_for_a = score_matrix[mask_b, i]  # cat_b 프레임의 cat_a 점수
            cat_preds = np.concatenate([scores_a_for_a, scores_b_for_a])
            cat_auc = roc_auc_score(y_pair, cat_preds)

            # LogisticRegression: 5-fold CV
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            lr_scores = cross_val_score(
                LogisticRegression(max_iter=1000), X_pair, y_pair,
                cv=cv, scoring="roc_auc",
            )

            # XGBoost or fallback
            try:
                from xgboost import XGBClassifier
                xgb_scores = cross_val_score(
                    XGBClassifier(n_estimators=100, max_depth=4, verbosity=0,
                                  use_label_encoder=False, eval_metric="logloss"),
                    X_pair, y_pair, cv=cv, scoring="roc_auc",
                )
                xgb_auc = f"{xgb_scores.mean():.3f}±{xgb_scores.std():.3f}"
            except ImportError:
                xgb_auc = "N/A (xgboost not installed)"
                xgb_scores = lr_scores  # fallback

            results.append({
                "pair": f"{cat_a} vs {cat_b}",
                "n": f"{n_a}+{n_b}",
                "catalog": cat_auc,
                "lr": lr_scores.mean(),
                "xgb": xgb_scores.mean() if isinstance(xgb_scores, np.ndarray) else 0,
            })

            print(f"\n  {cat_a} vs {cat_b}  (n={n_a}+{n_b})")
            print(f"    Catalog (Fisher centroid):  AUC = {cat_auc:.3f}")
            print(f"    LogisticRegression:         AUC = {lr_scores.mean():.3f} ± {lr_scores.std():.3f}")
            print(f"    XGBoost:                    AUC = {xgb_auc}")

    # 5. 전체 멀티클래스 비교
    print("\n" + "=" * 70)
    print("Multi-class Classification: 전체 카테고리 구분 능력")
    print("=" * 70)

    # 충분한 샘플이 있는 카테고리만
    cat_counts = {name: sum(1 for a in assignments if a == name) for name in cat_names}
    valid_cats = [name for name, count in cat_counts.items() if count >= 20]
    print(f"\n  카테고리별 프레임 수: {cat_counts}")
    print(f"  유효 카테고리 (≥20): {valid_cats}")

    if len(valid_cats) >= 2:
        # 유효 카테고리의 프레임만 사용
        mask_valid = np.array([a in valid_cats for a in assignments])
        X_multi = X_daily[mask_valid]
        y_multi_str = [a for a, m in zip(assignments, mask_valid) if m]

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_multi = le.fit_transform(y_multi_str)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # LogisticRegression multi-class
        lr_multi = cross_val_score(
            LogisticRegression(max_iter=1000, multi_class="ovr"),
            X_multi, y_multi, cv=cv, scoring="accuracy",
        )

        # XGBoost multi-class
        try:
            from xgboost import XGBClassifier
            xgb_multi = cross_val_score(
                XGBClassifier(n_estimators=100, max_depth=4, verbosity=0,
                              use_label_encoder=False, eval_metric="mlogloss"),
                X_multi, y_multi, cv=cv, scoring="accuracy",
            )
            xgb_acc = f"{xgb_multi.mean():.3f}±{xgb_multi.std():.3f}"
        except ImportError:
            xgb_acc = "N/A"

        print(f"\n  Samples: {len(X_multi)} ({len(valid_cats)} classes)")
        print(f"  LogisticRegression accuracy: {lr_multi.mean():.3f} ± {lr_multi.std():.3f}")
        print(f"  XGBoost accuracy:            {xgb_acc}")

    # 6. Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    if results:
        avg_cat = np.mean([r["catalog"] for r in results])
        avg_lr = np.mean([r["lr"] for r in results])
        avg_xgb = np.mean([r["xgb"] for r in results])
        print(f"  Average pairwise AUC:")
        print(f"    Catalog:  {avg_cat:.3f}")
        print(f"    LR:       {avg_lr:.3f}  (Δ={avg_lr - avg_cat:+.3f})")
        print(f"    XGBoost:  {avg_xgb:.3f}  (Δ={avg_xgb - avg_cat:+.3f})")


if __name__ == "__main__":
    main()
