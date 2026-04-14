"""
models/reject_option.py
-----------------------
Reject-option classifier: abstain from predicting when confidence is too
low (high uncertainty or ambiguous probability).

Two complementary criteria:
  1. Margin-based  – |P(y=1) - 0.5| < margin_threshold
  2. Entropy-based – H(p) > entropy_threshold
  3. Ensemble-std  – σ(members) > std_threshold
"""

import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              accuracy_score, classification_report)


# ══════════════════════════════════════════════════════════════════════
# Reject-option wrapper
# ══════════════════════════════════════════════════════════════════════

class RejectOptionClassifier:
    """
    Wraps any probability-outputting classifier and adds an abstention layer.

    Parameters
    ----------
    margin_threshold : float
        Abstain when |p - 0.5| < margin_threshold.
        E.g. 0.15 → abstain when p ∈ (0.35, 0.65).
    uncertainty_threshold : float | None
        If ensemble std is provided at predict-time, abstain when std > this.
    positive_threshold : float
        Decision boundary for the accepted predictions (default 0.5).
    """

    ABSTAIN = -1   # label for rejected samples

    def __init__(self, margin_threshold: float = 0.15,
                 uncertainty_threshold: float | None = 0.10,
                 positive_threshold: float = 0.50):
        self.margin_threshold      = margin_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.positive_threshold    = positive_threshold

    # ------------------------------------------------------------------
    def predict(self, probs: np.ndarray,
                uncertainty: np.ndarray | None = None) -> np.ndarray:
        """
        Parameters
        ----------
        probs       : shape (N,) – positive-class probability
        uncertainty : shape (N,) – ensemble std or MC-dropout std (optional)

        Returns
        -------
        labels : shape (N,) – 0 / 1 / ABSTAIN(-1)
        """
        labels = np.where(probs >= self.positive_threshold, 1, 0)

        # Margin criterion
        margin = np.abs(probs - 0.5)
        reject = margin < self.margin_threshold

        # Uncertainty criterion
        if uncertainty is not None and self.uncertainty_threshold is not None:
            reject |= uncertainty > self.uncertainty_threshold

        labels[reject] = self.ABSTAIN
        return labels

    # ------------------------------------------------------------------
    def coverage_accuracy_tradeoff(self, probs: np.ndarray, y_true: np.ndarray,
                                    uncertainty: np.ndarray | None = None,
                                    thresholds: np.ndarray | None = None):
        """
        Sweep margin_threshold values to produce coverage–accuracy curve.
        Returns list of dicts {threshold, coverage, accuracy, abstain_rate}.
        """
        if thresholds is None:
            thresholds = np.linspace(0.00, 0.45, 46)

        results = []
        original_thresh = self.margin_threshold
        for t in thresholds:
            self.margin_threshold = t
            preds   = self.predict(probs, uncertainty)
            accepted = preds != self.ABSTAIN
            coverage = accepted.mean()
            if coverage == 0:
                results.append({"threshold": t, "coverage": 0,
                                 "accuracy": np.nan, "abstain_rate": 1.0})
                continue
            acc = accuracy_score(y_true[accepted], preds[accepted])
            results.append({
                "threshold":    float(t),
                "coverage":     float(coverage),
                "accuracy":     float(acc),
                "abstain_rate": float(1 - coverage),
            })
        self.margin_threshold = original_thresh
        return results

    # ------------------------------------------------------------------
    def evaluate(self, probs: np.ndarray, y_true: np.ndarray,
                 uncertainty: np.ndarray | None = None,
                 label: str = "") -> dict:
        preds    = self.predict(probs, uncertainty)
        accepted = preds != self.ABSTAIN

        n_total    = len(preds)
        n_accepted = accepted.sum()
        n_rejected = n_total - n_accepted
        coverage   = n_accepted / n_total

        print(f"\n[RejectOption{' '+label if label else ''}]")
        print(f"  Total={n_total}  Accepted={n_accepted} ({coverage:.1%})  "
              f"Rejected={n_rejected} ({1-coverage:.1%})")

        metrics = {"coverage": coverage, "n_rejected": int(n_rejected)}
        if n_accepted > 0:
            acc = accuracy_score(y_true[accepted], preds[accepted])
            print(f"  Accuracy on accepted: {acc:.4f}")
            print(classification_report(y_true[accepted], preds[accepted],
                                        target_names=["No Default", "Default"],
                                        digits=4))
            try:
                auc = roc_auc_score(y_true[accepted], probs[accepted])
                print(f"  AUC on accepted: {auc:.4f}")
                metrics["auc_accepted"] = auc
            except Exception:
                pass
            metrics["accuracy_accepted"] = acc

        return metrics


# ══════════════════════════════════════════════════════════════════════
# Helper: uncertainty-based sample ranking
# ══════════════════════════════════════════════════════════════════════

def rank_by_confidence(probs: np.ndarray,
                        uncertainty: np.ndarray | None = None) -> np.ndarray:
    """
    Return indices sorted from MOST to LEAST confident.
    Confidence = |p - 0.5| (primary) penalised by uncertainty (secondary).
    """
    margin = np.abs(probs - 0.5)
    if uncertainty is not None:
        score = margin - uncertainty          # higher = more confident
    else:
        score = margin
    return np.argsort(-score)                 # descending


def confidence_bins(probs: np.ndarray, uncertainty: np.ndarray | None = None,
                    n_bins: int = 5) -> list[dict]:
    """
    Partition samples into n_bins confidence buckets and return statistics.
    """
    margin = np.abs(probs - 0.5)
    edges  = np.percentile(margin, np.linspace(0, 100, n_bins + 1))
    bins   = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (margin >= lo) & (margin <= hi)
        entry = {
            "confidence_range": (round(float(lo + 0.5), 4),
                                  round(float(hi + 0.5), 4)),
            "n_samples": int(mask.sum()),
            "mean_prob":  float(probs[mask].mean()) if mask.sum() else np.nan,
        }
        if uncertainty is not None and mask.sum():
            entry["mean_uncertainty"] = float(uncertainty[mask].mean())
        bins.append(entry)
    return bins
