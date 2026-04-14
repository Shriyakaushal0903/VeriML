"""
utils/visualizer.py
--------------------
Matplotlib / seaborn charts for:
  • Reliability (calibration) diagram
  • Confidence distribution
  • Coverage–accuracy tradeoff
  • Uncertainty vs. error rate
  • Feature importance
  • Error cluster heatmap
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

PALETTE = {
    "bg":       "#0F1117",
    "panel":    "#1A1D27",
    "accent1":  "#4F8EF7",   # blue
    "accent2":  "#F7994F",   # orange
    "accent3":  "#4FF79E",   # green
    "accent4":  "#F74F6E",   # red
    "text":     "#E8EAF0",
    "muted":    "#6B7280",
}


def _style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["panel"],
        "axes.edgecolor":    PALETTE["muted"],
        "axes.labelcolor":   PALETTE["text"],
        "xtick.color":       PALETTE["muted"],
        "ytick.color":       PALETTE["muted"],
        "text.color":        PALETTE["text"],
        "grid.color":        "#2D3748",
        "grid.linestyle":    "--",
        "grid.alpha":        0.6,
        "font.family":       "monospace",
        "axes.titlesize":    11,
        "axes.labelsize":    9,
        "legend.framealpha": 0.2,
        "legend.edgecolor":  PALETTE["muted"],
    })


# ──────────────────────────────────────────────
# 1.  Reliability diagram
# ──────────────────────────────────────────────

def plot_calibration_curves(y_true: np.ndarray,
                             probs_dict: dict,
                             n_bins: int = 10,
                             save_path: str = "outputs/calibration.png"):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    colors = [PALETTE["accent1"], PALETTE["accent2"],
              PALETTE["accent3"], PALETTE["accent4"]]

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "w--", lw=1.5, alpha=0.5, label="Perfect")
    for (name, probs), color in zip(probs_dict.items(), colors):
        frac_pos, mean_pred = calibration_curve(y_true, probs,
                                                n_bins=n_bins, strategy="uniform")
        ax.plot(mean_pred, frac_pos, "o-", color=color, lw=2, ms=5, label=name)
    ax.set_title("Reliability Diagram", color=PALETTE["text"], pad=10)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend(fontsize=8)
    ax.grid(True)

    # Histogram of predicted probs (uncalibrated vs best calibrated)
    ax2 = axes[1]
    names = list(probs_dict.keys())
    ax2.hist(probs_dict[names[0]], bins=40, alpha=0.6,
             color=PALETTE["accent1"], label=names[0], density=True)
    if len(names) > 1:
        ax2.hist(probs_dict[names[1]], bins=40, alpha=0.6,
                 color=PALETTE["accent2"], label=names[1], density=True)
    ax2.set_title("Predicted Probability Distribution", color=PALETTE["text"], pad=10)
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=8)
    ax2.grid(True)

    plt.tight_layout(pad=2)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"[Viz] Saved → {save_path}")


# ──────────────────────────────────────────────
# 2.  Confidence distribution + reject region
# ──────────────────────────────────────────────

def plot_confidence_distribution(probs: np.ndarray,
                                  uncertainty: np.ndarray,
                                  reject_threshold: float = 0.15,
                                  save_path: str = "outputs/confidence.png"):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    margin = np.abs(probs - 0.5)
    accepted = margin >= reject_threshold

    ax = axes[0]
    ax.hist(probs[accepted],  bins=40, alpha=0.7, color=PALETTE["accent3"],
            label=f"Accepted ({accepted.mean():.1%})", density=True)
    ax.hist(probs[~accepted], bins=40, alpha=0.7, color=PALETTE["accent4"],
            label=f"Rejected ({(~accepted).mean():.1%})", density=True)
    ax.axvline(0.5 - reject_threshold, color="white", ls=":", lw=1.5)
    ax.axvline(0.5 + reject_threshold, color="white", ls=":", lw=1.5)
    ax.set_title("Confidence Distribution & Reject Region", pad=10)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True)

    ax2 = axes[1]
    sc = ax2.scatter(probs, uncertainty, c=margin, cmap="plasma",
                     alpha=0.4, s=8, vmin=0, vmax=0.5)
    plt.colorbar(sc, ax=ax2, label="Margin |p − 0.5|")
    ax2.axhline(uncertainty.mean(), color="white", ls="--", lw=1, alpha=0.6,
                label=f"Mean unc = {uncertainty.mean():.3f}")
    ax2.set_title("Uncertainty vs. Predicted Probability", pad=10)
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Ensemble Uncertainty (std)")
    ax2.legend(fontsize=8)
    ax2.grid(True)

    plt.tight_layout(pad=2)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"[Viz] Saved → {save_path}")


# ──────────────────────────────────────────────
# 3.  Coverage–accuracy tradeoff
# ──────────────────────────────────────────────

def plot_coverage_accuracy(tradeoff: list[dict],
                            save_path: str = "outputs/coverage_accuracy.png"):
    _style()
    df = pd.DataFrame(tradeoff).dropna(subset=["accuracy"])

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax.plot(df["coverage"], df["accuracy"],
            color=PALETTE["accent1"], lw=2.5, marker="o", ms=4)
    ax.fill_between(df["coverage"], df["accuracy"],
                    alpha=0.15, color=PALETTE["accent1"])
    ax.set_title("Coverage–Accuracy Tradeoff (Reject Option)", pad=10)
    ax.set_xlabel("Coverage (fraction accepted)")
    ax.set_ylabel("Accuracy on Accepted Samples")
    ax.grid(True)

    # Annotate a few points
    for _, row in df.iloc[::8].iterrows():
        ax.annotate(f'{row["abstain_rate"]:.0%} rejected',
                    xy=(row["coverage"], row["accuracy"]),
                    fontsize=7, color=PALETTE["muted"],
                    xytext=(5, -12), textcoords="offset points")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"[Viz] Saved → {save_path}")


# ──────────────────────────────────────────────
# 4.  Feature importance
# ──────────────────────────────────────────────

def plot_feature_importance(importances: np.ndarray, feature_names: list[str],
                             top_k: int = 15,
                             save_path: str = "outputs/importance.png"):
    _style()
    idx = np.argsort(importances)[-top_k:]
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(PALETTE["bg"])

    colors = [PALETTE["accent1"] if v > np.median(importances[idx]) else PALETTE["muted"]
              for v in importances[idx]]
    ax.barh([feature_names[i] for i in idx], importances[idx],
            color=colors, edgecolor="none", height=0.7)
    ax.set_title("Feature Importance (XGBoost)", pad=10)
    ax.set_xlabel("Importance Score")
    ax.grid(True, axis="x")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"[Viz] Saved → {save_path}")


# ──────────────────────────────────────────────
# 5.  Uncertainty × error heatmap
# ──────────────────────────────────────────────

def plot_uncertainty_error(unc_df: pd.DataFrame,
                            save_path: str = "outputs/uncertainty_error.png"):
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    x = range(len(unc_df))
    bars = ax.bar(x, unc_df["error_rate"], color=PALETTE["accent4"],
                  alpha=0.8, edgecolor="none", label="Error Rate")
    ax2 = ax.twinx()
    ax2.plot(x, unc_df["mean_uncertainty"], color=PALETTE["accent1"],
             lw=2.5, marker="o", ms=5, label="Mean Uncertainty")
    ax2.set_ylabel("Mean Uncertainty", color=PALETTE["accent1"])

    ax.set_xticks(list(x))
    ax.set_xticklabels(unc_df["uncertainty_range"], rotation=45, ha="right", fontsize=7)
    ax.set_title("Error Rate vs. Uncertainty Bucket", pad=10)
    ax.set_xlabel("Uncertainty Bin")
    ax.set_ylabel("Error Rate")
    ax.grid(True, axis="y")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"[Viz] Saved → {save_path}")


# ──────────────────────────────────────────────
# 6.  Full diagnostic dashboard (composite)
# ──────────────────────────────────────────────

def plot_dashboard(y_true, y_prob_raw, y_prob_cal, uncertainty,
                   unc_df, tradeoff, importances, feature_names,
                   save_path: str = "outputs/dashboard.png"):
    _style()
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(PALETTE["bg"])
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0: Calibration ──
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.plot([0, 1], [0, 1], "w--", lw=1.5, alpha=0.5, label="Perfect")
    for name, probs, color in [("Raw", y_prob_raw, PALETTE["accent1"]),
                                ("Calibrated", y_prob_cal, PALETTE["accent2"])]:
        fp, mp = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
        ax0.plot(mp, fp, "o-", color=color, lw=2, ms=5, label=name)
    ax0.set_title("Reliability Diagram"); ax0.legend(fontsize=8); ax0.grid(True)
    ax0.set_xlabel("Mean Predicted Prob"); ax0.set_ylabel("Fraction of Positives")

    ax1 = fig.add_subplot(gs[0, 2:])
    ax1.hist(y_prob_raw, bins=40, alpha=0.6, color=PALETTE["accent1"],
             label="Raw", density=True)
    ax1.hist(y_prob_cal, bins=40, alpha=0.6, color=PALETTE["accent2"],
             label="Calibrated", density=True)
    ax1.set_title("Probability Distribution"); ax1.legend(fontsize=8); ax1.grid(True)
    ax1.set_xlabel("Predicted Probability"); ax1.set_ylabel("Density")

    # ── Row 1: Uncertainty ──
    ax2 = fig.add_subplot(gs[1, :2])
    if unc_df is not None and len(unc_df):
        x = range(len(unc_df))
        ax2.bar(x, unc_df["error_rate"], color=PALETTE["accent4"], alpha=0.8, label="Error Rate")
        ax2b = ax2.twinx()
        ax2b.plot(x, unc_df["mean_uncertainty"], color=PALETTE["accent1"], lw=2, marker="o", ms=4, label="Uncertainty")
        ax2b.set_ylabel("Uncertainty", color=PALETTE["accent1"])
        ax2.set_xticks(list(x)); ax2.set_xticklabels(unc_df["uncertainty_range"], rotation=45, ha="right", fontsize=6)
        ax2.set_title("Error Rate vs. Uncertainty"); ax2.grid(True, axis="y")

    ax3 = fig.add_subplot(gs[1, 2:])
    df_tr = pd.DataFrame(tradeoff).dropna(subset=["accuracy"])
    ax3.plot(df_tr["coverage"], df_tr["accuracy"], color=PALETTE["accent3"], lw=2.5, marker="o", ms=4)
    ax3.fill_between(df_tr["coverage"], df_tr["accuracy"], alpha=0.15, color=PALETTE["accent3"])
    ax3.set_title("Coverage–Accuracy Tradeoff"); ax3.grid(True)
    ax3.set_xlabel("Coverage"); ax3.set_ylabel("Accuracy")

    # ── Row 2: Feature importance + scatter ──
    ax4 = fig.add_subplot(gs[2, :2])
    if importances is not None and len(importances):
        top_k = min(12, len(importances))
        idx   = np.argsort(importances)[-top_k:]
        colors_bar = [PALETTE["accent1"] if v > np.median(importances[idx]) else PALETTE["muted"]
                      for v in importances[idx]]
        ax4.barh([feature_names[i] for i in idx], importances[idx],
                 color=colors_bar, edgecolor="none", height=0.7)
        ax4.set_title("Feature Importance"); ax4.grid(True, axis="x")

    ax5 = fig.add_subplot(gs[2, 2:])
    margin = np.abs(y_prob_raw - 0.5)
    sc = ax5.scatter(y_prob_raw, uncertainty, c=margin, cmap="plasma",
                     alpha=0.3, s=6, vmin=0, vmax=0.5)
    plt.colorbar(sc, ax=ax5, label="|p − 0.5|")
    ax5.set_title("Uncertainty vs. Probability"); ax5.grid(True)
    ax5.set_xlabel("Predicted Probability"); ax5.set_ylabel("Uncertainty (std)")

    fig.suptitle("Reliable ML — Confidence-Aware Prediction System",
                 fontsize=15, color=PALETTE["text"], y=1.01, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"[Viz] Dashboard saved → {save_path}")
