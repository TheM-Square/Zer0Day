"""
Autoencoder anomaly detection on NK Field SRP telemetry.

Loss: L = || x - x_hat ||^2
Threshold: 95th percentile of training reconstruction error.

Uses sklearn MLPRegressor in autoencoder mode (input = target) to avoid a
heavy torch dependency. The methodology matches the reference:
IEEE OJIM 2026, DOI 10.1109/OJIM.2026.3670416 (cited only, not a dataset).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .config import SEED, TRAIN_FRAC
from .production_model import FEATURES


@dataclass
class AnomalyDetector:
    ae: MLPRegressor
    scaler: StandardScaler
    threshold: float
    percentile: float = 95.0
    feature_names: list = field(default_factory=lambda: list(FEATURES))
    train_err_mean: float = 0.0
    train_err_std: float = 0.0
    test_flag_rate: float = 0.0

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(X)
        Zhat = self.ae.predict(Z)
        return ((Z - Zhat) ** 2).sum(axis=1)

    def flag(self, X: np.ndarray) -> np.ndarray:
        return self.score(X) > self.threshold


def train_anomaly_detector(wide: pd.DataFrame,
                           hidden=(8, 3, 8), max_iter=500) -> AnomalyDetector | None:
    cols = [c for c in FEATURES if c in wide.columns]
    if len(cols) != len(FEATURES):
        return None
    X = wide[FEATURES].to_numpy(dtype=float)
    X = X[~np.isnan(X).any(axis=1)]
    if len(X) < 200:
        return None

    cut = int(len(X) * TRAIN_FRAC)
    X_tr, X_te = X[:cut], X[cut:]

    scaler = StandardScaler().fit(X_tr)
    Z_tr = scaler.transform(X_tr)
    Z_te = scaler.transform(X_te)

    ae = MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                      solver="adam", max_iter=max_iter, random_state=SEED)
    ae.fit(Z_tr, Z_tr)

    err_tr = ((Z_tr - ae.predict(Z_tr)) ** 2).sum(axis=1)
    err_te = ((Z_te - ae.predict(Z_te)) ** 2).sum(axis=1)
    threshold = float(np.percentile(err_tr, 95.0))
    flag_rate = float((err_te > threshold).mean())

    return AnomalyDetector(
        ae=ae, scaler=scaler, threshold=threshold,
        train_err_mean=float(err_tr.mean()),
        train_err_std=float(err_tr.std()),
        test_flag_rate=flag_rate,
    )