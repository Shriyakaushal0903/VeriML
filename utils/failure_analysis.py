"""
utils/failure_analysis.py
--------------------------
Segment-level failure diagnostics:
  • Find slices (feature × bin) where error rate spikes
  • Compute confusion metrics per segment
  • Identify high-uncertainty / high-error clusters
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              accuracy_score, confusion_matrix)
from sklearn.cluster import KMeans


# ══════════════════════════════════════════════════════════════════════
# 1.  Segment-level performance analysis
# ══════════════════════════════════════════════════════════════════════

def segment_performance(X: np.ndarray, y_true: np.ndarray,
                         y_prob: np.ndarray, feature_names: list[str],
                         n_bins: int = 4) -> pd.DataFrame:
    """
    For every feature, bin samples and compute per-bin metrics.

    Returns a DataFrame with columns:
      feature, bin_label, n_samples, accuracy, error_rate, mean_prob,
      actual_rate, auc (when feasible)
    """
    y_pred = (y_prob >= 0.5).astype(int)
    rows = []

    for i, feat in enumerate(feature_names):
        col = X[:, i]
        try:
            quantiles = np.nanquantile(col, np.linspace(0, 1, n_bins + 1))
            quantiles = np.unique(quantiles)          # handle low-cardinality
            if len(quantiles) < 2:
                continue
            bins = np.digitize(col, quantiles[1:-1])
        except Exception:
            continue

        for b in np.unique(bins):
            mask = bins == b
            if mask.sum() < 20:          # skip tiny segments
                continue
            lo = quantiles[b - 1] if b > 0 else col.min()
            hi = quantiles[b]     if b < len(quantiles) else col.max()
            label = f"[{lo:.2f}, {hi:.2f})"

            yt = y_true[mask]; yp = y_pred[mask]; yprob = y_prob[mask]
            acc       = accuracy_score(yt, yp)
            err_rate  = 1 - acc
            mean_prob = yprob.mean()
            act_rate  = yt.mean()
            try:
                auc = roc_auc_score(yt, yprob) if yt.nunique() == 2 else np.nan
            except Exception:
                auc = np.nan
            if hasattr(yt, 'nunique'):
                pass
            else:
                auc = roc_auc_score(yt, yprob) if len(np.unique(yt)) == 2 else np.nan

            rows.append({
                "feature":     feat,
                "bin_label":   label,
                "n_samples":   int(mask.sum()),
                "accuracy":    round(acc, 4),
                "error_rate":  round(err_rate, 4),
                "mean_prob":   round(mean_prob, 4),
                "actual_rate": round(act_rate, 4),
                "auc":         round(auc, 4) if not np.isnan(auc) else None,
            })

    return pd.DataFrame(rows).sort_values("error_rate", ascending=False)


# ══════════════════════════════════════════════════════════════════════
# 2.  Worst-performing slices (top-k)
# ══════════════════════════════════════════════════════════════════════

def worst_slices(seg_df: pd.DataFrame, k: int = 10,
                 min_samples: int = 50) -> pd.DataFrame:
    """Return k worst-performing feature-bin combinations."""
    return (seg_df[seg_df["n_samples"] >= min_samples]
            .nlargest(k, "error_rate")
            .reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════
# 3.  Uncertainty vs. error analysis
# ══════════════════════════════════════════════════════════════════════

def uncertainty_error_correlation(y_true: np.ndarray, y_prob: np.ndarray,
                                   uncertainty: np.ndarray,
                                   n_bins: int = 10) -> pd.DataFrame:
    """
    Bin samples by uncertainty level and compute error rate per bin.
    Ideal: high-uncertainty samples should have higher error rates.
    """
    edges = np.percentile(uncertainty, np.linspace(0, 100, n_bins + 1))
    rows  = []
    y_pred = (y_prob >= 0.5).astype(int)

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (uncertainty >= lo) & (uncertainty <= hi)
        if mask.sum() < 5:
            continue
        err = (y_pred[mask] != y_true[mask]).mean()
        rows.append({
            "uncertainty_range": f"[{lo:.3f},{hi:.3f}]",
            "mean_uncertainty": float(uncertainty[mask].mean()),
            "n_samples":        int(mask.sum()),
            "error_rate":       float(err),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# 4.  Error cluster discovery (KMeans on misclassified samples)
# ══════════════════════════════════════════════════════════════════════

def discover_error_clusters(X: np.ndarray, y_true: np.ndarray,
                              y_pred: np.ndarray, feature_names: list[str],
                              n_clusters: int = 5) -> pd.DataFrame:
    """
    Run KMeans on misclassified samples to identify common error patterns.

    Returns a DataFrame with per-cluster feature centroids.
    """
    errors = (y_pred != y_true)
    if errors.sum() < n_clusters:
        return pd.DataFrame()

    X_err = X[errors]
    km    = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    km.fit(X_err)
    labels = km.labels_

    rows = []
    for c in range(n_clusters):
        mask = labels == c
        row  = {"cluster": c, "n_samples": int(mask.sum())}
        for j, feat in enumerate(feature_names):
            row[feat] = round(float(X_err[mask, j].mean()), 4)
        rows.append(row)

    return (pd.DataFrame(rows)
            .sort_values("n_samples", ascending=False)
            .reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════
# 5.  Full diagnostic report
# ══════════════════════════════════════════════════════════════════════

def full_failure_report(X: np.ndarray, y_true: np.ndarray,
                         y_prob: np.ndarray, uncertainty: np.ndarray,
                         feature_names: list[str]) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n═══ Failure Diagnostics ═══")
    print(f"Overall Error Rate : {(y_pred != y_true).mean():.4f}")
    print(f"Overall AUC        : {roc_auc_score(y_true, y_prob):.4f}")

    seg_df = segment_performance(X, y_true, y_prob, feature_names)
    ws     = worst_slices(seg_df, k=5)
    print("\nTop-5 Worst Slices:")
    print(ws[["feature", "bin_label", "n_samples", "error_rate", "actual_rate"]].to_string(index=False))

    unc_df = uncertainty_error_correlation(y_true, y_prob, uncertainty)
    print("\nUncertainty ↔ Error Rate:")
    print(unc_df.to_string(index=False))

    ec_df = discover_error_clusters(X, y_true, y_pred, feature_names)
    print(f"\nError Clusters: {len(ec_df)} found")

    return {
        "segment_df":           seg_df,
        "worst_slices":         ws,
        "uncertainty_error_df": unc_df,
        "error_clusters_df":    ec_df,
    }
