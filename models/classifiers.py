"""
models/classifiers.py
---------------------
Baseline XGBoost model + Deep Ensemble (5 neural nets) for
uncertainty estimation via prediction variance.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── optional heavy imports ─────────────────────────────────────────────
try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


# ══════════════════════════════════════════════════════════════════════
# 1.  XGBoost Baseline (falls back to sklearn GBM if xgb not installed)
# ══════════════════════════════════════════════════════════════════════

class XGBoostBaseline:
    """
    Strong gradient-boosted tree baseline with early stopping.
    Uses sklearn's GradientBoostingClassifier as a fallback.
    """

    def __init__(self, n_estimators: int = 400, max_depth: int = 5,
                 learning_rate: float = 0.05, subsample: float = 0.8,
                 random_state: int = 42):
        self.params = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            random_state=random_state,
        )
        self.model = None
        self._using_xgb = _XGB_AVAILABLE

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self._using_xgb:
            self.model = xgb.XGBClassifier(
                **self.params,
                use_label_encoder=False,
                eval_metric="logloss",
                early_stopping_rounds=30 if X_val is not None else None,
            )
            eval_set = [(X_val, y_val)] if X_val is not None else None
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            print(f"[XGBoost] Best iteration: {self.model.best_iteration}")
        else:
            print("[XGBoost] xgboost not found – using sklearn GradientBoosting")
            self.model = GradientBoostingClassifier(
                n_estimators=min(self.params["n_estimators"], 200),
                max_depth=self.params["max_depth"],
                learning_rate=self.params["learning_rate"],
                subsample=self.params["subsample"],
                random_state=self.params["random_state"],
            )
            self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)          # shape (N, 2)

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def evaluate(self, X, y, label: str = ""):
        proba = self.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)
        auc   = roc_auc_score(y, proba)
        ap    = average_precision_score(y, proba)
        acc   = (preds == y).mean()
        print(f"[XGBoost{' '+label if label else ''}] "
              f"AUC={auc:.4f}  AP={ap:.4f}  Acc={acc:.4f}")
        return {"auc": auc, "ap": ap, "accuracy": acc}


# ══════════════════════════════════════════════════════════════════════
# 2.  Neural Network (PyTorch) with MC-Dropout
# ══════════════════════════════════════════════════════════════════════

if _TORCH_AVAILABLE:
    class _MLPBlock(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, dropout: float):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        def forward(self, x):
            return self.block(x)

    class MCDropoutNet(nn.Module):
        """
        3-layer MLP with dropout everywhere (used in train AND inference).
        MC-Dropout: run T forward passes at inference to get uncertainty.
        """
        def __init__(self, input_dim: int, hidden: tuple = (256, 128, 64),
                     dropout: float = 0.30):
            super().__init__()
            dims = [input_dim] + list(hidden)
            layers = []
            for i in range(len(dims) - 1):
                layers.append(_MLPBlock(dims[i], dims[i+1], dropout))
            layers.append(nn.Linear(dims[-1], 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return torch.sigmoid(self.net(x)).squeeze(-1)

        def mc_predict(self, x: torch.Tensor, T: int = 50) -> tuple:
            """
            Returns (mean_prob, std_prob) over T stochastic forward passes.
            Dropout stays active (model.train() mode).
            """
            self.train()          # keep dropout ON
            with torch.no_grad():
                preds = torch.stack([self(x) for _ in range(T)], dim=0)  # (T, N)
            return preds.mean(0).cpu().numpy(), preds.std(0).cpu().numpy()


class DeepEnsemble:
    """
    Deep Ensemble of K independently-trained MLPs.
    Uncertainty = std across member predictions.
    Falls back to sklearn RF if PyTorch is unavailable.
    """

    def __init__(self, K: int = 5, input_dim: int = 20,
                 epochs: int = 50, lr: float = 1e-3, batch: int = 512,
                 device: str = "cpu", random_state: int = 42):
        self.K      = K
        self.epochs = epochs
        self.lr     = lr
        self.batch  = batch
        self.device = device
        self.seed   = random_state
        self.input_dim = input_dim
        self.members   = []
        self._using_torch = _TORCH_AVAILABLE

    # ------------------------------------------------------------------
    def _train_one(self, X_tr, y_tr, X_val, y_val, seed: int):
        if not self._using_torch:
            return None
        torch.manual_seed(seed)
        model = MCDropoutNet(self.input_dim).to(self.device)
        opt   = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = nn.BCELoss()

        Xt = torch.FloatTensor(X_tr).to(self.device)
        yt = torch.FloatTensor(y_tr).to(self.device)
        ds = TensorDataset(Xt, yt)
        dl = DataLoader(ds, batch_size=self.batch, shuffle=True)

        best_val, best_state = np.inf, None
        Xv = torch.FloatTensor(X_val).to(self.device)
        yv = torch.FloatTensor(y_val).to(self.device)

        model.train()
        for ep in range(self.epochs):
            for xb, yb in dl:
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
            sched.step()
            with torch.no_grad():
                model.eval()
                vl = loss_fn(model(Xv), yv).item()
                model.train()
            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        return model

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train, X_val, y_val):
        self.input_dim = X_train.shape[1]
        if self._using_torch:
            print(f"[DeepEnsemble] Training {self.K} PyTorch members …")
            self.members = []
            for k in range(self.K):
                m = self._train_one(X_train, y_train, X_val, y_val,
                                    seed=self.seed + k)
                self.members.append(m)
                val_auc = roc_auc_score(
                    y_val,
                    m.eval()(torch.FloatTensor(X_val).to(self.device))
                     .detach().cpu().numpy()
                )
                print(f"  Member {k+1}/{self.K} – val AUC: {val_auc:.4f}")
        else:
            print("[DeepEnsemble] PyTorch not found – using Random Forest ensemble")
            rng = np.random.default_rng(self.seed)
            self.members = [
                RandomForestClassifier(
                    n_estimators=200, max_features="sqrt",
                    random_state=int(rng.integers(0, 10000))
                ).fit(X_train, y_train)
                for _ in range(self.K)
            ]
        return self

    # ------------------------------------------------------------------
    def predict_with_uncertainty(self, X) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (mean_prob, uncertainty) where uncertainty = std of members.
        """
        if self._using_torch:
            Xt = torch.FloatTensor(X)
            preds = []
            for m in self.members:
                m.eval()
                with torch.no_grad():
                    preds.append(m(Xt).cpu().numpy())
            preds = np.stack(preds, axis=0)          # (K, N)
        else:
            preds = np.stack(
                [m.predict_proba(X)[:, 1] for m in self.members], axis=0
            )

        mean_prob    = preds.mean(axis=0)
        uncertainty  = preds.std(axis=0)
        return mean_prob, uncertainty

    def predict_proba(self, X) -> np.ndarray:
        mean_prob, _ = self.predict_with_uncertainty(X)
        return np.column_stack([1 - mean_prob, mean_prob])

    def evaluate(self, X, y, label: str = ""):
        proba, unc = self.predict_with_uncertainty(X)
        auc = roc_auc_score(y, proba)
        ap  = average_precision_score(y, proba)
        acc = ((proba >= 0.5).astype(int) == y).mean()
        print(f"[Ensemble{' '+label if label else ''}] "
              f"AUC={auc:.4f}  AP={ap:.4f}  Acc={acc:.4f}  "
              f"Mean-Uncertainty={unc.mean():.4f}")
        return {"auc": auc, "ap": ap, "accuracy": acc,
                "mean_uncertainty": float(unc.mean())}
