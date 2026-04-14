"""
train.py
--------
End-to-end training pipeline for the Reliable-ML system.

Steps:
  1. Load & preprocess data
  2. Train XGBoost baseline
  3. Train Deep Ensemble
  4. Calibrate probabilities (Platt + Isotonic + Temperature)
  5. Configure reject-option classifier
  6. Failure analysis & diagnostics
  7. Visualise everything
  8. Persist artefacts for the API

Usage:
  python train.py
"""

import sys, os, pickle, json, time
from pathlib import Path
import numpy as np

# ── Ensure project root is on path ────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from data.pipeline          import load_credit_dataset, DataPipeline
from models.classifiers     import XGBoostBaseline, DeepEnsemble
from models.calibration     import (PlattScaler, IsotonicCalibrator,
                                     TemperatureScaler, calibration_report,
                                     calibration_curve_data)
from models.reject_option   import RejectOptionClassifier
from utils.failure_analysis import full_failure_report
from utils.visualizer       import (plot_calibration_curves,
                                     plot_confidence_distribution,
                                     plot_coverage_accuracy,
                                     plot_feature_importance,
                                     plot_uncertainty_error,
                                     plot_dashboard)

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    t_start = time.time()
    print("\n" + "═"*60)
    print("  RELIABLE ML — Confidence-Aware Prediction System")
    print("═"*60 + "\n")

    # ──────────────────────────────────────────────
    # STEP 1 – Data
    # ──────────────────────────────────────────────
    print("── STEP 1: Data Loading & Preprocessing ──")
    df  = load_credit_dataset()
    dp  = DataPipeline(test_size=0.20, val_size=0.10, random_state=42)
    X_tr, X_val, X_te, y_tr, y_val, y_te = dp.fit_transform(df)

    # ──────────────────────────────────────────────
    # STEP 2 – XGBoost Baseline
    # ──────────────────────────────────────────────
    print("\n── STEP 2: XGBoost Baseline ──")
    xgb_model = XGBoostBaseline(n_estimators=400, max_depth=5,
                                 learning_rate=0.05, subsample=0.8)
    xgb_model.fit(X_tr, y_tr, X_val, y_val)
    xgb_model.evaluate(X_te, y_te, label="Test")

    xgb_raw_prob_val = xgb_model.predict_proba(X_val)[:, 1]
    xgb_raw_prob_te  = xgb_model.predict_proba(X_te)[:, 1]

    # ──────────────────────────────────────────────
    # STEP 3 – Deep Ensemble (uncertainty estimation)
    # ──────────────────────────────────────────────
    print("\n── STEP 3: Deep Ensemble (Uncertainty Estimation) ──")
    ensemble = DeepEnsemble(K=5, input_dim=X_tr.shape[1],
                             epochs=40, lr=1e-3, batch=512, random_state=42)
    ensemble.fit(X_tr, y_tr, X_val, y_val)
    ensemble.evaluate(X_te, y_te, label="Test")

    ens_prob_te, unc_te = ensemble.predict_with_uncertainty(X_te)
    ens_prob_val, unc_val_arr = ensemble.predict_with_uncertainty(X_val)

    # ──────────────────────────────────────────────
    # STEP 4 – Calibration
    # ──────────────────────────────────────────────
    print("\n── STEP 4: Probability Calibration ──")

    platt  = PlattScaler()
    iso    = IsotonicCalibrator()
    temp   = TemperatureScaler()

    # Fit on validation set using raw XGBoost probs
    platt.fit(xgb_raw_prob_val, y_val)
    iso.fit(xgb_raw_prob_val, y_val)
    temp.fit(xgb_raw_prob_val, y_val)

    # Calibrated probs on test
    platt_te = platt.transform(xgb_raw_prob_te)
    iso_te   = iso.transform(xgb_raw_prob_te)
    temp_te  = temp.transform(xgb_raw_prob_te)

    print("\nCalibration Metrics (Test set):")
    cal_results = calibration_report(
        y_te, xgb_raw_prob_te,
        {"Platt": platt_te, "Isotonic": iso_te, "Temperature": temp_te}
    )

    cal_curves = calibration_curve_data(
        y_te,
        {"Uncalibrated": xgb_raw_prob_te,
         "Platt": platt_te, "Isotonic": iso_te}
    )

    # ──────────────────────────────────────────────
    # STEP 5 – Reject Option
    # ──────────────────────────────────────────────
    print("\n── STEP 5: Reject-Option Classifier ──")
    ro = RejectOptionClassifier(margin_threshold=0.15,
                                 uncertainty_threshold=0.12,
                                 positive_threshold=0.50)
    ro.evaluate(platt_te, y_te, uncertainty=unc_te, label="Test (Platt)")

    tradeoff = ro.coverage_accuracy_tradeoff(platt_te, y_te, uncertainty=unc_te)

    # ──────────────────────────────────────────────
    # STEP 6 – Failure Analysis
    # ──────────────────────────────────────────────
    print("\n── STEP 6: Failure Diagnostics ──")
    fail_report = full_failure_report(
        X_te, y_te, platt_te, unc_te, dp.feature_names
    )

    # ──────────────────────────────────────────────
    # STEP 7 – Visualisations
    # ──────────────────────────────────────────────
    print("\n── STEP 7: Visualisations ──")

    plot_calibration_curves(
        y_te,
        {"Uncalibrated": xgb_raw_prob_te, "Platt": platt_te,
         "Isotonic": iso_te, "Temperature": temp_te},
        save_path=str(OUT / "calibration.png"),
    )

    plot_confidence_distribution(
        platt_te, unc_te,
        reject_threshold=ro.margin_threshold,
        save_path=str(OUT / "confidence.png"),
    )

    plot_coverage_accuracy(tradeoff, save_path=str(OUT / "coverage_accuracy.png"))

    importances = xgb_model.feature_importances()
    plot_feature_importance(importances, dp.feature_names,
                             save_path=str(OUT / "importance.png"))

    unc_df = fail_report["uncertainty_error_df"]
    if len(unc_df):
        plot_uncertainty_error(unc_df,
                                save_path=str(OUT / "uncertainty_error.png"))

    plot_dashboard(
        y_true=y_te,
        y_prob_raw=xgb_raw_prob_te,
        y_prob_cal=platt_te,
        uncertainty=unc_te,
        unc_df=unc_df if len(unc_df) else None,
        tradeoff=tradeoff,
        importances=importances,
        feature_names=dp.feature_names,
        save_path=str(OUT / "dashboard.png"),
    )

    # ──────────────────────────────────────────────
    # STEP 8 – Persist artefacts
    # ──────────────────────────────────────────────
    print("\n── STEP 8: Saving Artefacts ──")
    artefacts = {
        "pipeline": dp,
        "xgb":      xgb_model,
        "ensemble": ensemble,
        "platt":    platt,
        "reject":   ro,
        "loaded":   True,
    }
    with open(OUT / "artefacts.pkl", "wb") as f:
        pickle.dump(artefacts, f)
    print(f"[Train] Artefacts → {OUT / 'artefacts.pkl'}")

    # Save calibration metrics to JSON
    summary = {
        "calibration_metrics": {
            k: {m: round(v, 6) for m, v in metrics.items()}
            for k, metrics in cal_results.items()
        },
        "calibration_curves": cal_curves,
        "tradeoff": tradeoff,
        "worst_slices": fail_report["worst_slices"].to_dict(orient="records"),
        "feature_names": dp.feature_names,
        "importances": importances.tolist(),
        "ensemble_mean_uncertainty": float(unc_te.mean()),
        "reject_coverage": float((ro.predict(platt_te, unc_te) != ro.ABSTAIN).mean()),
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Train] Summary  → {OUT / 'summary.json'}")

    elapsed = time.time() - t_start
    print(f"\n{'═'*60}")
    print(f"  ✓ Training complete in {elapsed:.1f}s")
    print(f"  Outputs in: {OUT}")
    print("═"*60)


if __name__ == "__main__":
    main()
