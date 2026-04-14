# VeriML
Confidence-Aware Prediction System: When models know when they might be wrong.

## Overview
Production-ready ML system that improves prediction reliability using:
- Uncertainty Estimation (Deep Ensembles)
- Probability Calibration (Platt, Isotonic, Temperature Scaling)
- Reject Option (abstains on low-confidence predictions)
- Failure Diagnostics (segment-level error analysis)
- FastAPI deployment with confidence-aware outputs

---

## Results (Credit Default Dataset)

| Metric | Value |
|--------|-------|
| XGBoost AUC | 0.800 |
| Ensemble AUC | 0.831 |
| Raw Accuracy | 73.9% |
| Accepted Accuracy | 80.5% (+6.6pp) |
| Coverage | 72.4% |

---

## Project Structure
'''bash
VeriML/
├── data/
│   └── pipeline.py          # Data loading, feature engineering, preprocessing
├── models/
│   ├── classifiers.py       # XGBoost baseline + Deep Ensemble
│   ├── calibration.py       # Platt / Isotonic / Temperature scaling
│   └── reject_option.py     # Abstention mechanism + coverage-accuracy tradeoff
├── utils/
│   ├── failure_analysis.py  # Segment-level diagnostics + error clusters
│   └── visualizer.py        # All plots (calibration, confidence, dashboard)
├── api/
│   └── app.py               # FastAPI service
├── outputs/                 # Generated artefacts + plots
│   ├── dashboard.png
│   ├── calibration.png
│   ├── confidence.png
│   ├── coverage_accuracy.png
│   ├── importance.png
│   ├── uncertainty_error.png
│   ├── artefacts.pkl        
│   └── summary.json        
├── train.py                 # Full pipeline runner
└── README.md
'''

---

## Pipeline

1. **Data Processing**  
   Feature engineering (utilization, payment ratio, delinquency score), scaling, train/val/test split  

2. **Modeling**  
   XGBoost baseline + Deep Ensemble (uncertainty via prediction variance)

3. **Calibration**  
   Platt / Isotonic / Temperature scaling for reliable probabilities  

4. **Reject Option**  
   Abstains on uncertain predictions → improves accuracy on accepted samples  

5. **Failure Analysis**  
   Identifies high-error segments & uncertainty-error correlation  

6. **Deployment**  
   FastAPI service with prediction, confidence, and rejection output  

---

## Run Locally

```bash
pip install -r requirements.txt
python train.py
uvicorn api.app:app --reload
