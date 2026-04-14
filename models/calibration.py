"""
models/calibration.py
---------------------
Probability calibration:
  • Platt Scaling  (logistic regression on raw scores)
  • Isotonic Regression
  • Temperature Scaling (for neural networks)

Also provides ECE / MCE / Brier-score evaluation helpers.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
# 1.  Calibration wrappers
# ══════════════════════════════════════════════════════════════════════

class PlattScaler:
    """
    Platt Scaling: fit a logistic regression on top of the raw model
    scores using a held-out validation set.
    """
    def __init__(self):
        self._lr = LogisticRegression(C=1.0, solver="lbfgs")

    def fit(self, raw_probs: np.ndarray, y: np.ndarray) -> "PlattScaler":
        """raw_probs: 1-D array of positive-class probabilities."""
        self._lr.fit(raw_probs.reshape(-1, 1), y)
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

    def fit_transform(self, raw_probs, y):
        return self.fit(raw_probs, y).transform(raw_probs)


class IsotonicCalibrator:
    """
    Isotonic Regression calibration – more flexible than Platt but
    needs more calibration data to avoid overfitting.
    """
    def __init__(self):
        self._iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")

    def fit(self, raw_probs: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self._iso.fit(raw_probs, y)
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        return self._iso.predict(raw_probs)

    def fit_transform(self, raw_probs, y):
        return self.fit(raw_probs, y).transform(raw_probs)


class TemperatureScaler:
    """
    Temperature Scaling: divide logits by a scalar T > 0 (learned on val set).
    Particularly useful for neural networks.
    We approximate by optimising T over probability outputs via NLL.
    """
    def __init__(self):
        self.T = 1.0

    def _nll(self, T: float, logits: np.ndarray, y: np.ndarray) -> float:
        scaled = 1 / (1 + np.exp(-logits / T))
        eps = 1e-9
        return -np.mean(y * np.log(scaled + eps) + (1 - y) * np.log(1 - scaled + eps))

    def fit(self, raw_probs: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        """raw_probs: predicted probabilities (converted internally to logits)."""
        eps = 1e-6
        logits = np.log(np.clip(raw_probs, eps, 1 - eps) /
                        np.clip(1 - raw_probs, eps, 1))
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(self._nll, args=(logits, y), bounds=(0.1, 10),
                              method="bounded")
        self.T = float(res.x)
        print(f"[TempScaling] Optimal T = {self.T:.4f}")
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        eps = 1e-6
        logits = np.log(np.clip(raw_probs, eps, 1 - eps) /
                        np.clip(1 - raw_probs, eps, 1))
        return 1 / (1 + np.exp(-logits / self.T))

    def fit_transform(self, raw_probs, y):
        return self.fit(raw_probs, y).transform(raw_probs)


# ══════════════════════════════════════════════════════════════════════
# 2.  Calibration metrics
# ══════════════════════════════════════════════════════════════════════

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                                n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
    Lower is better (0 = perfect calibration).
    """
    bins   = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    n      = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def max_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                           n_bins: int = 10) -> float:
    """Maximum Calibration Error (MCE)."""
    bins   = np.linspace(0, 1, n_bins + 1)
    errors = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        errors.append(abs(y_true[mask].mean() - y_prob[mask].mean()))
    return float(max(errors)) if errors else 0.0


def calibration_report(y_true: np.ndarray,
                        raw_probs: np.ndarray,
                        calibrated_probs: dict[str, np.ndarray],
                        n_bins: int = 10) -> dict:
    """
    Compare ECE / MCE / Brier score before and after calibration.

    Parameters
    ----------
    calibrated_probs : dict  {method_name: probs_array}
    """
    results = {}

    for name, probs in {"Uncalibrated": raw_probs, **calibrated_probs}.items():
        ece    = expected_calibration_error(y_true, probs, n_bins)
        mce    = max_calibration_error(y_true, probs, n_bins)
        brier  = brier_score_loss(y_true, probs)
        results[name] = {"ECE": ece, "MCE": mce, "Brier": brier}
        print(f"  [{name:20s}]  ECE={ece:.4f}  MCE={mce:.4f}  Brier={brier:.4f}")

    return results


def calibration_curve_data(y_true: np.ndarray, probs_dict: dict,
                            n_bins: int = 10) -> dict:
    """
    Compute (fraction_of_positives, mean_predicted_value) for each method.
    Useful for reliability diagram plotting.
    """
    curves = {}
    for name, probs in probs_dict.items():
        frac_pos, mean_pred = calibration_curve(y_true, probs,
                                                n_bins=n_bins, strategy="uniform")
        curves[name] = {"frac_pos": frac_pos.tolist(),
                        "mean_pred": mean_pred.tolist()}
    return curves
