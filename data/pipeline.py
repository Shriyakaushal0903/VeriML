"""
data/pipeline.py
----------------
Data loading, preprocessing, and feature engineering pipeline.
Uses the UCI Credit Card Default dataset (fetched via sklearn-compatible API).
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 1.  Dataset loader
# ──────────────────────────────────────────────

def load_credit_dataset() -> pd.DataFrame:
    """
    Load the 'Default of Credit Card Clients' dataset from OpenML.
    Falls back to a fully-synthetic dataset when network is unavailable.
    
    Target column: 'default' (1 = default next month, 0 = no default)
    """
    try:
        print("[DataPipeline] Fetching dataset from OpenML …")
        ds = fetch_openml(name="default-of-credit-card-clients", version=1,
                          as_frame=True, parser="auto")
        df = ds.frame.copy()
        df.rename(columns={"default.payment.next.month": "default"}, inplace=True)
        df["default"] = df["default"].astype(int)
        print(f"[DataPipeline] Loaded real dataset: {df.shape}")
        return df
    except Exception as e:
        print(f"[DataPipeline] OpenML unavailable ({e}). Generating synthetic data …")
        return _make_synthetic_credit_data(n=10_000, seed=42)


def _make_synthetic_credit_data(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic credit-default dataset."""
    rng = np.random.default_rng(seed)

    age          = rng.integers(21, 70, n)
    limit_bal    = rng.choice([10_000, 20_000, 50_000, 100_000, 200_000], n)
    pay_0        = rng.integers(-2, 9, n)          # repayment status last month
    pay_2        = rng.integers(-2, 9, n)
    pay_3        = rng.integers(-2, 9, n)
    bill_amt1    = rng.uniform(0, 80_000, n)
    bill_amt2    = rng.uniform(0, 80_000, n)
    pay_amt1     = rng.uniform(0, 50_000, n)
    pay_amt2     = rng.uniform(0, 50_000, n)
    education    = rng.choice([1, 2, 3, 4], n)
    marriage     = rng.choice([1, 2, 3], n)
    sex          = rng.choice([1, 2], n)

    # logistic default probability
    log_odds = (
        -3.0
        + 0.40 * pay_0
        + 0.20 * pay_2
        + 0.15 * pay_3
        - 0.005 * (limit_bal / 10_000)
        + 0.02  * (bill_amt1 / 10_000)
        - 0.03  * (pay_amt1  / 10_000)
        + 0.01  * (age - 40)
        + rng.normal(0, 0.5, n)
    )
    prob    = 1 / (1 + np.exp(-log_odds))
    default = (rng.uniform(0, 1, n) < prob).astype(int)

    return pd.DataFrame({
        "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education,
        "MARRIAGE": marriage, "AGE": age,
        "PAY_0": pay_0, "PAY_2": pay_2, "PAY_3": pay_3,
        "BILL_AMT1": bill_amt1, "BILL_AMT2": bill_amt2,
        "PAY_AMT1": pay_amt1, "PAY_AMT2": pay_amt2,
        "default": default,
    })


# ──────────────────────────────────────────────
# 2.  Feature engineering
# ──────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-relevant features to the credit dataset."""
    df = df.copy()

    # Utilisation ratio
    if "BILL_AMT1" in df.columns and "LIMIT_BAL" in df.columns:
        df["utilization_ratio"] = (
            df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1e-6)
        ).clip(0, 5)

    # Payment momentum: how much of bill was paid?
    if "PAY_AMT1" in df.columns and "BILL_AMT1" in df.columns:
        df["payment_ratio_1"] = (
            df["PAY_AMT1"] / (df["BILL_AMT1"] + 1e-6)
        ).clip(0, 5)

    if "PAY_AMT2" in df.columns and "BILL_AMT2" in df.columns:
        df["payment_ratio_2"] = (
            df["PAY_AMT2"] / (df["BILL_AMT2"] + 1e-6)
        ).clip(0, 5)

    # Delinquency score: sum of positive payment statuses
    pay_cols = [c for c in df.columns if c.startswith("PAY_") and c != "PAY_AMT1"
                and c not in ("PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6")]
    if pay_cols:
        df["delinquency_score"] = df[pay_cols].clip(lower=0).sum(axis=1)

    # Log-transform skewed monetary columns
    for col in ["LIMIT_BAL", "BILL_AMT1", "BILL_AMT2", "PAY_AMT1", "PAY_AMT2"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    return df


# ──────────────────────────────────────────────
# 3.  Full preprocessing pipeline
# ──────────────────────────────────────────────

class DataPipeline:
    """
    End-to-end data pipeline:
      load → engineer → impute → scale → split
    """

    def __init__(self, test_size: float = 0.20, val_size: float = 0.10,
                 random_state: int = 42):
        self.test_size    = test_size
        self.val_size     = val_size
        self.random_state = random_state

        self.imputer   = SimpleImputer(strategy="median")
        self.scaler    = StandardScaler()
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame):
        """
        Returns
        -------
        X_train, X_val, X_test : np.ndarray  (scaled)
        y_train, y_val, y_test : np.ndarray  (int)
        """
        df = engineer_features(df)

        target = "default"
        X = df.drop(columns=[target])
        y = df[target].values.astype(int)

        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])
        self.feature_names = list(X.columns)

        # Train / (val+test) split
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=self.test_size + self.val_size,
            random_state=self.random_state, stratify=y
        )
        val_frac = self.val_size / (self.test_size + self.val_size)
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=1 - val_frac,
            random_state=self.random_state, stratify=y_tmp
        )

        # Impute → Scale (fit only on train)
        X_tr  = self.scaler.fit_transform(self.imputer.fit_transform(X_tr))
        X_val = self.scaler.transform(self.imputer.transform(X_val))
        X_te  = self.scaler.transform(self.imputer.transform(X_te))

        print(f"[DataPipeline] Train={X_tr.shape}, Val={X_val.shape}, Test={X_te.shape}")
        print(f"[DataPipeline] Default rate – train: {y_tr.mean():.3f} | "
              f"val: {y_val.mean():.3f} | test: {y_te.mean():.3f}")

        return X_tr, X_val, X_te, y_tr, y_val, y_te

    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data (after fit_transform has been called)."""
        X = X.select_dtypes(include=[np.number])
        # Align columns
        X = X.reindex(columns=self.feature_names, fill_value=0)
        return self.scaler.transform(self.imputer.transform(X))
