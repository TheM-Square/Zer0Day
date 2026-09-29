"""
ML model: next-step maximum rod load prediction.

Task (real ML, on REAL NK Field data when available):
    Input  : window of [spm, fillage, min_load, max_load]
    Target : next-step max_rod_load_kg
    Split  : TEMPORAL per well (first 80% train, last 20% test)

Why this target:
    NK Field does not contain production rates, so we predict the mechanical
    state that we DO have. This is a genuine, defensible regression task.
    We do NOT pretend to predict Baghewala oil rate from NK data.

Model:
    sklearn GradientBoostingRegressor with quantile losses for uncertainty.

Metrics reported:
    MAE, RMSE, R², MAPE (on held-out test set), plus 10/90 quantile widths.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import (
    SEQ_LEN, TRAIN_FRAC, SEED, GB_N_ESTIMATORS, GB_MAX_DEPTH, GB_LEARNING_RATE,
)

FEATURES = ["spm", "pump_fillage_pct", "min_rod_load_kg", "max_rod_load_kg"]
TARGET = "max_rod_load_kg"


@dataclass
class ProductionModel:
    q50: GradientBoostingRegressor
    q10: GradientBoostingRegressor
    q90: GradientBoostingRegressor
    feature_names: list = field(default_factory=lambda: list(FEATURES))
    metrics: dict = field(default_factory=dict)
    n_train: int = 0
    n_test: int = 0
    wells: list = field(default_factory=list)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.q10.predict(X), self.q50.predict(X), self.q90.predict(X)

    def permutation_importance_mae(self, X_test: np.ndarray, y_test: np.ndarray,
                                   n_repeats: int = 5) -> dict:
        rng = np.random.default_rng(SEED)
        base_mae = mean_absolute_error(y_test, self.q50.predict(X_test))
        importances = {}
        for i, name in enumerate(self.feature_names):
            deltas = []
            for _ in range(n_repeats):
                Xp = X_test.copy()
                rng.shuffle(Xp[:, i])
                deltas.append(mean_absolute_error(y_test, self.q50.predict(Xp)) - base_mae)
            importances[name] = float(np.mean(deltas))
        return importances


def _build_windows(wide: pd.DataFrame, seq_len: int = SEQ_LEN):
    Xs, ys, flags, wells = [], [], [], []
    for well_id, g in wide.groupby("well_id", sort=False):
        g = g.sort_values("timestamp")
        missing = [c for c in FEATURES if c not in g.columns]
        if missing:
            continue
        arr = g[FEATURES].to_numpy(dtype=float)
        ok = ~np.isnan(arr).any(axis=1)
        arr = arr[ok]
        if len(arr) <= seq_len + 1:
            continue
        n_windows = len(arr) - seq_len
        cut = int(n_windows * TRAIN_FRAC)
        for i in range(n_windows):
            window = arr[i:i + seq_len]
            # Flat feature vector: mean + last value + std across the window
            feats = np.concatenate([
                window.mean(axis=0),
                window[-1],
                window.std(axis=0),
            ])
            Xs.append(feats)
            ys.append(arr[i + seq_len, FEATURES.index(TARGET)])
            flags.append(0 if i < cut else 1)
            wells.append(str(well_id))
    if not Xs:
        return None
    X = np.asarray(Xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    split = np.asarray(flags)
    return X, y, split, wells


def train_production_model(wide: pd.DataFrame) -> ProductionModel | None:
    built = _build_windows(wide)
    if built is None:
        return None
    X, y, split, wells = built
    if (split == 0).sum() < 50 or (split == 1).sum() < 20:
        return None

    tr = split == 0
    te = split == 1
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]

    # Enrich feature names to match the concatenated vector
    names = ([f"{c}_mean" for c in FEATURES]
             + [f"{c}_last" for c in FEATURES]
             + [f"{c}_std" for c in FEATURES])

    common = dict(n_estimators=GB_N_ESTIMATORS, max_depth=GB_MAX_DEPTH,
                  learning_rate=GB_LEARNING_RATE, random_state=SEED)
    q50 = GradientBoostingRegressor(loss="squared_error", **common).fit(X_tr, y_tr)
    q10 = GradientBoostingRegressor(loss="quantile", alpha=0.10, **common).fit(X_tr, y_tr)
    q90 = GradientBoostingRegressor(loss="quantile", alpha=0.90, **common).fit(X_tr, y_tr)

    pred = q50.predict(X_te)
    mae = float(mean_absolute_error(y_te, pred))
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    r2 = float(r2_score(y_te, pred))
    # MAPE guarded against near-zero targets
    nz = np.abs(y_te) > 1e-3
    mape = float(np.mean(np.abs((y_te[nz] - pred[nz]) / y_te[nz])) * 100.0) if nz.any() else float("nan")

    metrics = {
        "MAE_kg": mae,
        "RMSE_kg": rmse,
        "R2": r2,
        "MAPE_pct": mape,
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "target_mean_kg": float(y.mean()),
        "target_std_kg": float(y.std()),
    }

    return ProductionModel(q50=q50, q10=q10, q90=q90,
                           feature_names=names, metrics=metrics,
                           n_train=int(tr.sum()), n_test=int(te.sum()),
                           wells=sorted(set(wells)))


def predict_from_wide(model: ProductionModel, recent_window: np.ndarray) -> dict:
    """
    recent_window: (SEQ_LEN, 4) raw feature window in the same order as
    config.FEATURES. Returns dict with prediction + interval.
    """
    feats = np.concatenate([
        recent_window.mean(axis=0),
        recent_window[-1],
        recent_window.std(axis=0),
    ]).reshape(1, -1).astype(np.float32)
    lo = float(model.q10.predict(feats)[0])
    mid = float(model.q50.predict(feats)[0])
    hi = float(model.q90.predict(feats)[0])
    return {"p10_kg": lo, "p50_kg": mid, "p90_kg": hi}