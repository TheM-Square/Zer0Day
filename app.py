"""
OIL TWIN — Streamlit dashboard.

Run:  streamlit run app.py

Advisory only. Not a digital twin. No live SCADA. No autonomous control.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import (
    BAGHEWALA, B_K, DANGER_MU_CP, SEED,
    DEFAULT_ROD_DIA_IN, DEFAULT_PLUNGER_DIA_IN, DEFAULT_STROKE_IN,
    DEFAULT_PUMP_DEPTH_FT, DEFAULT_FLUID_SG,
    DEFAULT_SIGMA_LIMIT_PSI, DEFAULT_GEARBOX_RATING_IN_LBF,
    DEFAULT_TORQUE_FACTOR_IN,
    T_INJ_DEFAULT_C, HALF_LIFE_DEFAULT_DAYS, ANCHOR_ENVELOPE_C,
)
from src.viscosity_model import ViscosityModel
from src.trajectory import CSSTrajectory, lambda_from_half_life
from src.srp_physics import evaluate
from src.data_loader import load_nk
from src.production_model import train_production_model, FEATURES, TARGET
from src.anomaly import train_anomaly_detector
from src.optimizer import optimize_spm, pump_rate_bpd, spm_band_from_viscosity


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="OIL TWIN — Baghewala CSS + SRP Advisor",
    page_icon="🛢",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_data(force: bool):
    return load_nk(force_download=force)


@st.cache_resource(show_spinner=False)
def _train_production(_wide_id: str, wide: pd.DataFrame):
    return train_production_model(wide)


@st.cache_resource(show_spinner=False)
def _train_anomaly(_wide_id: str, wide: pd.DataFrame):
    return train_anomaly_detector(wide)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("OIL TWIN")
st.caption(
    "Physics-constrained decision support for Cyclic Steam Stimulation (CSS) "
    "and Sucker Rod Pumping (SRP) — Baghewala Field prototype. "
    "**Advisory only.** Not a digital twin. No live SCADA."
)

# Load data
with st.spinner("Loading NK Field SRP telemetry..."):
    try:
        wide, provenance = _load_data(force=False)
    except Exception as e:
        st.error(f"Failed to load telemetry: {e}")
        st.stop()

synthetic = provenance == "synthetic_nk"

# Data provenance badge
cols = st.columns([3, 1, 1])
with cols[0]:
    if synthetic:
        st.warning(
            "**Data source: SYNTHETIC NK-LIKE TELEMETRY.** "
            "The public NK Field dataset could not be downloaded in this environment. "
            "The synthetic generator is documented in `src/data_loader.py`. "
            "All ML metrics below reflect synthetic data — they demonstrate the "
            "pipeline works, not field performance."
        )
    else:
        st.info(
            "**Data source: NK Field SRP telemetry (real, public, Apache 2.0).** "
            "Hugging Face: Arailym-tleubayeva/NK-Oil-Well-Sensor-Monitoring. "
            "NK Field is a conventional oil field — its data is used for SRP "
            "mechanical dynamics only, not thermal behavior."
        )
with cols[1]:
    if st.button("Reload data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
with cols[2]:
    st.metric("Wells loaded", wide["well_id"].nunique() if "well_id" in wide.columns else 0)


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Operating inputs")

    days_since_inj = st.number_input(
        "Days since end of steam injection",
        min_value=0.0, max_value=2000.0, value=30.0, step=1.0,
    )

    current_spm = st.number_input(
        "Current SPM (strokes/min)",
        min_value=0.5, max_value=20.0, value=5.0, step=0.1,
    )

    stroke_in = st.number_input(
        "Stroke length (in)",
        min_value=20.0, max_value=300.0, value=DEFAULT_STROKE_IN, step=5.0,
    )

    fillage = st.slider(
        "Pump fillage (%)", 20.0, 100.0, 75.0, 1.0,
    )

    st.divider()
    st.subheader("CSS model")
    t_inj_c = st.number_input(
        "Injection temperature (degC)",
        min_value=100.0, max_value=350.0, value=T_INJ_DEFAULT_C, step=5.0,
        help="ASSUMPTION — not a Baghewala measurement.",
    )
    half_life = st.number_input(
        "Cooling half-life (days)",
        min_value=1.0, max_value=365.0, value=HALF_LIFE_DEFAULT_DAYS, step=1.0,
        help="Refit from Bao/Wang/Gates (2016) once digitised.",
    )

    st.divider()
    st.subheader("Well / rod geometry")
    d_rod_in = st.number_input("Rod diameter (in)", 0.5, 1.5, DEFAULT_ROD_DIA_IN, 0.125)
    d_plunger_in = st.number_input("Plunger diameter (in)", 1.0, 4.0, DEFAULT_PLUNGER_DIA_IN, 0.25)
    depth_m = st.number_input(
        "Pump depth (m)", 500.0, 2000.0,
        (BAGHEWALA["depth_min_m"] + BAGHEWALA["depth_max_m"]) / 2.0, 25.0,
    )
    sf_fluid = st.number_input("Fluid SG", 0.80, 1.10, DEFAULT_FLUID_SG, 0.01)

    st.divider()
    st.subheader("Guardrail limits")
    sigma_limit = st.number_input(
        "Rod allowable stress (psi)", 5_000.0, 140_000.0,
        DEFAULT_SIGMA_LIMIT_PSI, 1_000.0,
    )
    gearbox = st.number_input(
        "Gearbox rating (in-lbf)", 20_000.0, 1_000_000.0,
        DEFAULT_GEARBOX_RATING_IN_LBF, 1_000.0,
    )
    torque_factor = st.number_input(
        "Torque factor %S (in)", 1.0, 120.0, DEFAULT_TORQUE_FACTOR_IN, 1.0,
    )

# Convert depth to feet for the equations
depth_ft = float(depth_m) * 3.28084


# ---------------------------------------------------------------------------
# Build models
# ---------------------------------------------------------------------------

visc = ViscosityModel.from_baghewala()
traj = CSSTrajectory(
    t_res_c=0.5 * (BAGHEWALA["t_res_min_c"] + BAGHEWALA["t_res_max_c"]),
    t_inj_c=float(t_inj_c),
    lam_per_day=lambda_from_half_life(float(half_life)),
)

T_now = float(traj.T(float(days_since_inj)))
mu_now = float(visc.mu(T_now))
mu_lo, mu_hi = visc.mu_band(T_now)
is_extrap = visc.is_extrapolated(T_now)

# Train ML models
with st.spinner("Training ML models on telemetry..."):
    prod_model = _train_production("v1", wide)
    anom = _train_anomaly("v1", wide)

# Current guardrail check
guard_now = evaluate(
    spm=float(current_spm), stroke_in=float(stroke_in),
    d_rod_in=float(d_rod_in), rod_length_ft=depth_ft,
    d_plunger_in=float(d_plunger_in), pump_depth_ft=depth_ft,
    sf_fluid=float(sf_fluid),
    sigma_limit_psi=float(sigma_limit),
    gearbox_rating_in_lbf=float(gearbox),
    torque_factor_in=float(torque_factor),
)

# Run optimization
opt = optimize_spm(
    mu_cp=mu_now, current_spm=float(current_spm), fillage_pct=float(fillage),
    stroke_in=float(stroke_in), d_plunger_in=float(d_plunger_in),
    d_rod_in=float(d_rod_in), pump_depth_ft=depth_ft, sf_fluid=float(sf_fluid),
    sigma_limit_psi=float(sigma_limit),
    gearbox_rating_in_lbf=float(gearbox),
    torque_factor_in=float(torque_factor),
    spm_min=1.0, spm_max=12.0,
)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tabs = st.tabs([
    "Overview",
    "CSS / Thermal",
    "SRP / Optimization",
    "ML & Validation",
    "Data Quality",
    "Limitations",
])


# ===========================================================================
# Tab 1 — Overview
# ===========================================================================

with tabs[0]:
    st.subheader("Current state")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperature", f"{T_now:.1f} °C",
              delta=("extrapolated" if is_extrap else "in anchor envelope"),
              delta_color="off")
    c2.metric("Predicted viscosity", f"{mu_now:,.0f} cP",
              delta=f"band {mu_lo:,.0f}–{mu_hi:,.0f}",
              delta_color="off")
    c3.metric("Current SPM", f"{current_spm:.1f}")
    c4.metric("Guardrail",
              "PASS" if not guard_now.veto else "VETO",
              delta=f"σ util {guard_now.stress_utilization*100:.0f}%",
              delta_color="off")

    if guard_now.veto:
        st.error("Current operating point violates constraints:")
        for r in guard_now.reasons:
            st.write(f"- {r}")

    if is_extrap:
        st.info(
            f"Temperature {T_now:.1f} °C is outside the ±{ANCHOR_ENVELOPE_C:.0f} °C "
            f"anchor envelope around {BAGHEWALA['mu_ref_temp_c']:.0f} °C. "
            "Viscosity is an **extrapolation** of the Arrhenius functional form, not "
            "a validated prediction."
        )

    st.divider()
    st.subheader("Recommendation")
    if opt.feasible and opt.recommended_spm is not None:
        d1, d2, d3 = st.columns(3)
        d1.metric("Recommended SPM", f"{opt.recommended_spm:.1f}",
                  delta=f"current {current_spm:.1f}")
        d2.metric("Predicted liquid rate",
                  f"{opt.recommended_rate:.1f} bbl/d",
                  delta=f"{opt.delta_rate:+.1f} vs current",
                  delta_color="off")
        d3.metric("Guardrail at recommendation",
                  opt.guardrail_at_recommendation.status,
                  delta=f"σ util {opt.guardrail_at_recommendation.stress_utilization*100:.0f}%",
                  delta_color="off")
        st.caption(f"Rationale: {opt.reason}")
    else:
        st.error(f"No feasible recommendation. Reason: {opt.reason}")


# ===========================================================================
# Tab 2 — CSS / Thermal
# ===========================================================================

with tabs[1]:
    st.subheader("CSS cooling trajectory & viscosity forecast")
    st.caption(visc.describe())

    horizon = st.slider("Forecast horizon (days)", 30.0, 720.0, 180.0, 10.0)
    days = np.linspace(0.0, horizon, 400)
    T_series = traj.T_array(days)
    mu_series = visc.mu_array(T_series)
    band_lo = np.array([visc.mu_band(t)[0] for t in T_series])
    band_hi = np.array([visc.mu_band(t)[1] for t in T_series])
    extrap_mask = (T_series < BAGHEWALA["mu_ref_temp_c"] - ANCHOR_ENVELOPE_C) | \
                  (T_series > BAGHEWALA["mu_ref_temp_c"] + ANCHOR_ENVELOPE_C)

    fig = go.Figure()
    fig.add_hrect(y0=DANGER_MU_CP, y1=mu_series.max() * 1.4,
                  fillcolor="rgba(220,50,50,0.10)", line_width=0,
                  annotation_text=f"danger zone (> {DANGER_MU_CP:,.0f} cP)",
                  annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([band_hi, band_lo[::-1]]),
        fill="toself", fillcolor="rgba(70,130,200,0.15)",
        line=dict(width=0), name="published band (10k–13k cP @ 45 °C)",
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=days, y=mu_series, mode="lines",
                             name="predicted viscosity",
                             line=dict(color="#1f4e79", width=3)))
    fig.add_trace(go.Scatter(x=[days_since_inj], y=[mu_now], mode="markers",
                             name="now",
                             marker=dict(size=12, color="#d62728", symbol="diamond")))
    if extrap_mask.any():
        fig.add_trace(go.Scatter(x=days[extrap_mask], y=mu_series[extrap_mask],
                                 mode="markers", marker=dict(size=4, color="orange"),
                                 name="extrapolated region"))
    fig.add_hline(y=DANGER_MU_CP, line_dash="dash", line_color="#d62728")
    fig.update_layout(xaxis_title="days since injection end",
                      yaxis_title="viscosity (cP, log scale)",
                      yaxis_type="log", height=420,
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("μ(T) anchor view")
    t_grid = np.linspace(20.0, 260.0, 400)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t_grid, y=visc.mu_array(t_grid), mode="lines",
                              name="Arrhenius fit", line=dict(color="#1f4e79", width=3)))
    fig2.add_trace(go.Scatter(x=[BAGHEWALA["mu_ref_temp_c"]] * 2,
                              y=[BAGHEWALA["mu_low_cp"], BAGHEWALA["mu_high_cp"]],
                              mode="markers+lines", name="published anchor range",
                              marker=dict(size=10, color="#d62728"),
                              line=dict(color="#d62728", width=4)))
    fig2.add_vrect(x0=BAGHEWALA["mu_ref_temp_c"] - ANCHOR_ENVELOPE_C,
                   x1=BAGHEWALA["mu_ref_temp_c"] + ANCHOR_ENVELOPE_C,
                   fillcolor="rgba(0,160,0,0.10)", line_width=0,
                   annotation_text="anchor envelope", annotation_position="top left")
    fig2.update_layout(xaxis_title="temperature (°C)",
                       yaxis_title="viscosity (cP, log scale)",
                       yaxis_type="log", height=380,
                       margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)


# ===========================================================================
# Tab 3 — SRP / Optimization
# ===========================================================================

with tabs[2]:
    st.subheader("Constrained SPM optimization")

    if opt.feasible:
        st.success(
            f"**Recommended SPM: {opt.recommended_spm:.1f}** "
            f"(current {current_spm:.1f}, Δ {opt.recommended_spm-current_spm:+.1f})  \n"
            f"Predicted liquid rate: **{opt.recommended_rate:.1f} bbl/d** "
            f"({opt.delta_rate:+.1f} vs current {opt.current_rate:.1f})"
        )
    else:
        st.error(opt.reason)

    st.divider()
    st.subheader("Feasible region & objective")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=opt.grid_spm, y=opt.grid_rate, mode="lines",
                              name="predicted liquid rate",
                              line=dict(color="#1f4e79", width=3)))
    fig3.add_trace(go.Scatter(x=opt.grid_spm[opt.grid_feasible],
                              y=opt.grid_rate[opt.grid_feasible],
                              mode="markers", marker=dict(size=5, color="green"),
                              name="feasible (guardrail PASS)"))
    fig3.add_trace(go.Scatter(x=opt.grid_spm[~opt.grid_feasible],
                              y=opt.grid_rate[~opt.grid_feasible],
                              mode="markers", marker=dict(size=5, color="red"),
                              name="infeasible (guardrail VETO)"))
    fig3.add_vline(x=current_spm, line_dash="dot", line_color="orange",
                   annotation_text="current", annotation_position="top right")
    if opt.recommended_spm is not None:
        fig3.add_vline(x=opt.recommended_spm, line_dash="dash", line_color="green",
                       annotation_text="recommended", annotation_position="top left")
    fig3.update_layout(xaxis_title="SPM (strokes/min)",
                       yaxis_title="predicted liquid rate (bbl/d)",
                       height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Constraints along the SPM grid")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=opt.grid_spm, y=opt.grid_sigma, mode="lines",
                              name="σ_max (psi)", line=dict(color="#d62728")))
    fig4.add_hline(y=sigma_limit, line_dash="dash", line_color="#d62728",
                   annotation_text="rod stress limit")
    fig4.update_layout(xaxis_title="SPM", yaxis_title="σ_max (psi)",
                       height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=opt.grid_spm, y=opt.grid_torque, mode="lines",
                              name="T_peak (in-lbf)", line=dict(color="#1f77b4")))
    fig5.add_hline(y=gearbox, line_dash="dash", line_color="#1f77b4",
                   annotation_text="gearbox rating")
    fig5.update_layout(xaxis_title="SPM", yaxis_title="T_peak (in-lbf)",
                       height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig5, use_container_width=True)

    st.divider()
    st.subheader("Guardrail state at current SPM")
    g = pd.DataFrame([
        ["PRLmax",                  f"{guard_now.prl_max_lbf:,.0f} lbf"],
        ["PRLmin",                  f"{guard_now.prl_min_lbf:,.0f} lbf"],
        ["Peak torque",             f"{guard_now.peak_torque_in_lbf:,.0f} in-lbf"],
        ["σ_max",                   f"{guard_now.sigma_max_psi:,.0f} psi"],
        ["σ limit",                 f"{guard_now.sigma_limit_psi:,.0f} psi"],
        ["σ utilization",           f"{guard_now.stress_utilization*100:.1f} %"],
        ["Torque utilization",      f"{guard_now.torque_utilization*100:.1f} %"],
        ["Scaled load ratio",       f"{guard_now.slr:.3f}"],
        ["Max safe SPM (analytic)", f"{guard_now.max_safe_spm}"],
        ["Status",                  guard_now.status],
    ], columns=["Quantity", "Value"])
    st.dataframe(g, hide_index=True, use_container_width=True)


# ===========================================================================
# Tab 4 — ML & Validation
# ===========================================================================

with tabs[3]:
    st.subheader("Production / rod-load model")
    st.caption(
        "The ML task is genuine: predict next-step max rod load from a window of "
        "surface SRP telemetry. Split is **temporal per well** (first 80 % train, "
        "last 20 % test) to prevent leakage."
    )

    if prod_model is None:
        st.warning("Not enough data to train. Increase the number of rows or wells.")
    else:
        st.markdown(f"**Target:** `{TARGET}`  •  **Features:** {len(prod_model.feature_names)} "
                    f"(mean, last, std of {len(FEATURES)} base signals)")
        m = prod_model.metrics
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("MAE", f"{m['MAE_kg']:.1f} kg")
        k2.metric("RMSE", f"{m['RMSE_kg']:.1f} kg")
        k3.metric("R²", f"{m['R2']:.3f}")
        k4.metric("MAPE", f"{m['MAPE_pct']:.1f} %"
                  if not np.isnan(m['MAPE_pct']) else "n/a")
        st.caption(
            f"Train N = {m['n_train']:,}  •  Test N = {m['n_test']:,}  •  "
            f"Target μ±σ = {m['target_mean_kg']:.0f} ± {m['target_std_kg']:.0f} kg  •  "
            f"Wells: {len(prod_model.wells)}"
        )

        st.divider()
        st.subheader("Permutation importance (MAE increase on test set)")
        # Rebuild test set for importance computation
        from src.production_model import _build_windows
        built = _build_windows(wide)
        if built is not None:
            X, y, split, _ = built
            X_te, y_te = X[split == 1], y[split == 1]
            imp = prod_model.permutation_importance_mae(X_te[:2000], y_te[:2000])
            imp_df = (pd.DataFrame(list(imp.items()), columns=["feature", "ΔMAE_kg"])
                      .sort_values("ΔMAE_kg", ascending=False).head(12))
            fig6 = go.Figure(go.Bar(x=imp_df["ΔMAE_kg"], y=imp_df["feature"],
                                    orientation="h", marker_color="#1f4e79"))
            fig6.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                               xaxis_title="increase in MAE when feature is shuffled (kg)",
                               yaxis_title="")
            fig6.update_yaxes(autorange="reversed")
            st.plotly_chart(fig6, use_container_width=True)

    st.divider()
    st.subheader("Anomaly detector (autoencoder)")
    if anom is None:
        st.warning("Not enough data to train the anomaly detector.")
    else:
        st.markdown(
            f"- **Reconstruction threshold (95th pct of train):** "
            f"{anom.threshold:.4f}  \n"
            f"- **Train error mean / std:** "
            f"{anom.train_err_mean:.4f} / {anom.train_err_std:.4f}  \n"
            f"- **Test flag rate:** {anom.test_flag_rate*100:.2f} %"
        )

    st.divider()
    st.subheader("Methodology notes")
    st.markdown("""
- **Temporal split per well** — no future data is mixed into training.
- **No feature leakage** — target is next-step max rod load; features are the
  preceding window's summary statistics only.
- **Model choice**: Gradient Boosting. It performs well on tabular SRP data
  in published benchmarks and provides clean permutation importances.
- **Uncertainty**: quantile GB (10th/90th) gives prediction intervals without
  assuming Gaussian residuals.
- **What is *not* claimed**: These metrics are on **NK Field geometry**. They
  are not validated on Baghewala. Retraining with Baghewala telemetry would be
  required before operational use.
""")


# ===========================================================================
# Tab 5 — Data Quality
# ===========================================================================

with tabs[4]:
    st.subheader("Data quality report")
    st.markdown(f"**Provenance:** `{provenance}`  •  **Rows:** {len(wide):,}  •  "
                f"**Wells:** {wide['well_id'].nunique() if 'well_id' in wide.columns else 0}")

    present = [c for c in FEATURES if c in wide.columns]
    missing = [c for c in FEATURES if c not in wide.columns]
    st.markdown(f"**Present feature columns:** `{present}`  \n"
                f"**Missing feature columns:** `{missing}`")

    if present:
        stats = wide[present].describe().T
        stats["missing"] = wide[present].isna().sum()
        stats["missing_pct"] = (stats["missing"] / len(wide) * 100).round(2)
        st.dataframe(stats, use_container_width=True)

    st.divider()
    st.subheader("Long → Wide pivot (spec form)")
    st.code(
        'df.pivot_table(index=["well_id", "timestamp"],\n'
        '               columns="parameter", values="value").reset_index()\n'
        'df.rename(columns={"SPM": "spm",\n'
        '                   "pump_fillage": "pump_fillage_pct",\n'
        '                   "min_rod_weight": "min_rod_load_kg",\n'
        '                   "max_rod_weight": "max_rod_load_kg",\n'
        '                   "dynamometer_area": "dyno_area"})\n'
        'df["min_rod_load_lbf"] = df["min_rod_load_kg"] * 2.20462\n'
        'df["max_rod_load_lbf"] = df["max_rod_load_kg"] * 2.20462',
        language="python",
    )


# ===========================================================================
# Tab 6 — Limitations
# ===========================================================================

with tabs[5]:
    st.subheader("Honest limitations")
    st.markdown("""
1. **No Baghewala operational dataset exists publicly.** Nothing in this tool is
   validated against Baghewala field measurements.

2. **NK Field is not a thermal heavy-oil field.** Its telemetry is used for SRP
   mechanical dynamics only. It says nothing about thermal behavior of heavy
   crude at Baghewala.

3. **The Arrhenius fit is anchored at one temperature (45 °C).** Two parameters
   (A, B) cannot be identified from a single temperature. `B` is taken from
   external heavy-crude literature (60 kJ/mol activation energy shape); `A` is
   calibrated so `μ(45 °C) = 11,500 cP` (midpoint of published range). Any
   prediction away from ~45 °C is **extrapolated** and flagged in the UI.

4. **CSS trajectory `λ` and `T_inj` are assumptions.** Refit from the
   Bao, Wang & Gates (2016) curve shape or field thermal history before use.

5. **The SPM recommendation rule table is heuristic.** It encodes the standard
   SRP rule (higher viscosity → lower safe SPM) but is not field-calibrated.

6. **Production rate formula uses a heuristic viscosity-efficiency penalty.**
   `η(μ) = 1 / (1 + (μ/1000)^0.25)`. This is engineering judgement, not a
   measured correlation.

7. **The rod allowable stress (30,000 psi) and gearbox rating (228,000 in-lbf)
   are configuration defaults** — replace with actual equipment data.

8. **NK rod-load predictions do not transfer to Baghewala** without retraining.

9. **This is advisory only.** Not a digital twin (no live bidirectional sync).
   Not autonomous control. No SCADA integration.

10. **`gamma_s` interpretation note.** The `62.4 * Sf * (Wr/gamma_s)` term in
    the PRLmax equation only balances dimensionally if `gamma_s` is steel
    *density* in lb/ft³ (490), not specific gravity (7.85). Using SG makes the
    buoyancy term ~60× too large and drives PRLmax negative.
    See `src/srp_physics.py`.

### What is scientifically defensible today
- The Arrhenius form for heavy-crude μ(T).
- The exponential cooling form for CSS.
- API RP 11L rod-load, torque and stress equations.
- The Scaled Load Ratio from US Patent 11,898,552 B2.
- Temporal-split validation of the ML model on real NK data (or clearly
  flagged synthetic NK-like data).
- Hard guardrail veto on physically unsafe SPM.

### What is NOT defensible today
- Any claim of field-validated production improvement.
- Any claim of Baghewala-specific economics.
- Any claim this is a "digital twin."
- Any claim the ML model has been calibrated for Baghewala crude.
""")

st.divider()
st.caption(
    "Advisory only. No autonomous control. No SCADA integration. "
    "Operational decisions require qualified engineering review."
)