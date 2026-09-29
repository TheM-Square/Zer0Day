"""
Data loading and transformation.

Real data:
  Source A — NK Field SRP telemetry
    https://huggingface.co/datasets/Arailym-tleubayeva/NK-Oil-Well-Sensor-Monitoring
    Long-format columns: well_id, well_type, timestamp, parameter, value, unit
    Parameters: SPM, pump_fillage, min_rod_weight, max_rod_weight, dynamometer_area
    License: Apache 2.0

If the download fails (no network / HF unavailable), a synthetic NK-like
dataset is generated and CACHED WITH A FLAG so the UI can clearly label it.
The synthetic generator uses physically plausible relationships and is
fully documented below.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from .config import (
    NK_PARQUET, NK_WIDE, NK_DATASET_ID, SEED,
    DEFAULT_ROD_DIA_IN, DEFAULT_PLUNGER_DIA_IN, DEFAULT_PUMP_DEPTH_FT,
    DEFAULT_FLUID_SG, STEEL_DENSITY_LB_FT3,
)


# ---------------------------------------------------------------------------
# Synthetic fallback — clearly labeled and documented
# ---------------------------------------------------------------------------

def _synthetic_nk(n_wells: int = 6, n_hours: int = 2000) -> pd.DataFrame:
    """
    Generate synthetic SRP telemetry with plausible physics.

    NOT REAL FIELD DATA. Used only when the NK Field download is unavailable.

    Model:
      SPM           random walk in [3, 11]
      fillage_pct   beta-ish in [40, 100], inversely correlated with SPM
      min_load_kg   from a static rod-weight baseline minus inertial term
      max_load_kg   static + fluid load + inertial term, plus noise
      dynamometer_area  monotone with (max - min) load (as in real dyno data)
    """
    rng = np.random.default_rng(SEED)
    rows = []
    t0 = pd.Timestamp("2024-01-01")

    rod_w_lbf = 2.668 * DEFAULT_ROD_DIA_IN ** 2 * DEFAULT_PUMP_DEPTH_FT
    Ap = np.pi / 4.0 * DEFAULT_PLUNGER_DIA_IN ** 2
    fluid_load_lbf = 0.433 * DEFAULT_FLUID_SG * DEFAULT_PUMP_DEPTH_FT * Ap

    for w in range(n_wells):
        well_id = f"NK-{w+1:02d}"
        spm = float(rng.uniform(4.0, 9.0))
        fillage = float(rng.uniform(60.0, 95.0))
        for h in range(n_hours):
            spm += rng.normal(0.0, 0.05)
            spm = float(np.clip(spm, 3.0, 11.0))
            fillage += rng.normal(0.0, 0.30)
            fillage = float(np.clip(fillage, 35.0, 100.0))

            S = 100.0
            inertial = 2.668 * DEFAULT_ROD_DIA_IN ** 2 * DEFAULT_PUMP_DEPTH_FT \
                * (S * spm ** 2) / 70471.2
            # fillage reduces effective fluid load on upstroke (pump-off)
            eff_fluid = fluid_load_lbf * (fillage / 100.0)
            max_lbf = rod_w_lbf + eff_fluid + inertial + rng.normal(0, 200)
            min_lbf = rod_w_lbf * 0.95 - inertial * 0.6 + rng.normal(0, 200)

            max_kg = max_lbf / 2.20462
            min_kg = min_lbf / 2.20462
            dyno_area = float((max_kg - min_kg) * rng.uniform(0.9, 1.1))

            for param, val, unit in [
                ("SPM", spm, "strokes/min"),
                ("pump_fillage", fillage, "%"),
                ("min_rod_weight", min_kg, "kg"),
                ("max_rod_weight", max_kg, "kg"),
                ("dynamometer_area", dyno_area, "kg"),
            ]:
                rows.append({
                    "well_id": well_id,
                    "well_type": "SRP",
                    "timestamp": t0 + pd.Timedelta(hours=h),
                    "parameter": param,
                    "value": float(val),
                    "unit": unit,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Real NK download
# ---------------------------------------------------------------------------

def _try_download_nk(cache_path: str) -> pd.DataFrame | None:
    try:
        from datasets import load_dataset
        ds = load_dataset(NK_DATASET_ID)
        frames = []
        if hasattr(ds, "keys"):
            for split in ds.keys():
                frames.append(ds[split].to_pandas())
        else:
            frames.append(ds.to_pandas())
        df = pd.concat(frames, ignore_index=True)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path, index=False)
        return df
    except Exception as e:
        print(f"[data_loader] NK Field download failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_PARAM_ALIASES = {
    "spm": "spm",
    "pump_fillage": "pump_fillage_pct",
    "pump_fillage_pct": "pump_fillage_pct",
    "min_rod_weight": "min_rod_load_kg",
    "min_rod_load": "min_rod_load_kg",
    "max_rod_weight": "max_rod_load_kg",
    "max_rod_load": "max_rod_load_kg",
    "dynamometer_area": "dyno_area",
    "dyno_area": "dyno_area",
}


def pivot_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    df = df_long.copy()
    if "parameter" in df.columns:
        key = (df["parameter"].astype(str).str.strip().str.lower()
               .str.replace(r"\s+", "_", regex=True))
        df["parameter"] = key.map(_PARAM_ALIASES).fillna(key)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    wide = df.pivot_table(
        index=["well_id", "timestamp"],
        columns="parameter",
        values="value",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None

    wide = wide.rename(columns={
        "SPM": "spm",
        "pump_fillage": "pump_fillage_pct",
        "min_rod_weight": "min_rod_load_kg",
        "max_rod_weight": "max_rod_load_kg",
        "dynamometer_area": "dyno_area",
    })

    for col in ("min_rod_load_kg", "max_rod_load_kg"):
        if col in wide.columns:
            wide[col.replace("_kg", "_lbf")] = wide[col] * 2.20462

    return wide.sort_values(["well_id", "timestamp"]).reset_index(drop=True)


def load_nk(force_download: bool = False) -> tuple[pd.DataFrame, str]:
    """
    Returns (wide_df, provenance) where provenance is 'real_nk' or 'synthetic_nk'.
    """
    if os.path.exists(NK_WIDE) and not force_download:
        wide = pd.read_parquet(NK_WIDE)
        prov = wide.attrs.get("provenance", "cached")
        # attrs don't survive parquet round-trip reliably; infer from file
        marker = os.path.join(os.path.dirname(NK_WIDE), ".nk_synthetic")
        return wide, ("synthetic_nk" if os.path.exists(marker) else "real_nk")

    if os.path.exists(NK_PARQUET) and not force_download:
        long = pd.read_parquet(NK_PARQUET)
        wide = pivot_to_wide(long)
        os.makedirs(os.path.dirname(NK_WIDE), exist_ok=True)
        wide.to_parquet(NK_WIDE, index=False)
        return wide, "real_nk"

    # Try real download
    long = _try_download_nk(NK_PARQUET)
    if long is not None and "parameter" in long.columns:
        wide = pivot_to_wide(long)
        os.makedirs(os.path.dirname(NK_WIDE), exist_ok=True)
        wide.to_parquet(NK_WIDE, index=False)
        marker = os.path.join(os.path.dirname(NK_WIDE), ".nk_synthetic")
        if os.path.exists(marker):
            os.remove(marker)
        return wide, "real_nk"

    # Fallback to synthetic
    long = _synthetic_nk()
    os.makedirs(os.path.dirname(NK_PARQUET), exist_ok=True)
    long.to_parquet(NK_PARQUET, index=False)
    wide = pivot_to_wide(long)
    os.makedirs(os.path.dirname(NK_WIDE), exist_ok=True)
    wide.to_parquet(NK_WIDE, index=False)
    marker = os.path.join(os.path.dirname(NK_WIDE), ".nk_synthetic")
    with open(marker, "w") as f:
        f.write("synthetic fallback - NK Field download unavailable\n")
    return wide, "synthetic_nk"