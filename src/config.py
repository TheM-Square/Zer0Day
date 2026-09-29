"""
Central configuration: units, anchors, thresholds, seeds.

Every number here is either:
  (a) a published/sourced value (marked with source)
  (b) an engineering assumption (marked ASSUMPTION)
  (c) a unit conversion constant (marked CONST)
"""

# ---------------------------------------------------------------------------
# Baghewala published anchors — Source: OIL India tender document (public)
# ---------------------------------------------------------------------------
BAGHEWALA = {
    "api_low":            14.0,          # deg API
    "api_high":           17.0,
    "mu_low_cp":          10_000.0,      # cP @ mu_ref_temp_c
    "mu_high_cp":         13_000.0,
    "mu_ref_temp_c":      45.0,          # degC  — the anchor temperature
    "depth_min_m":        1050.0,
    "depth_max_m":        1300.0,
    "t_res_min_c":        46.0,
    "t_res_max_c":        48.0,
}

# ---------------------------------------------------------------------------
# Arrhenius shape parameter B (K)
# ASSUMPTION: Ea ~ 60 kJ/mol for heavy-crude viscous flow (mid-range literature
# value for bitumen-type fluids). This fixes the SHAPE of mu(T); the LEVEL A
# is calibrated to Baghewala. See viscosity_model.py docstring.
# ---------------------------------------------------------------------------
R_GAS = 8.314462618                            # CONST  J/(mol K)
EA_ASSUMED_J_PER_MOL = 60_000.0                # ASSUMPTION
B_K = EA_ASSUMED_J_PER_MOL / R_GAS             # ~ 7217 K

# Anchor-envelope width for Arrhenius validity flagging (degC around 45).
ANCHOR_ENVELOPE_C = 5.0

# ---------------------------------------------------------------------------
# CSS trajectory defaults (all ASSUMPTIONS — refit from field / literature)
# ---------------------------------------------------------------------------
T_INJ_DEFAULT_C = 250.0            # ASSUMPTION: conventional CSS steam temp
HALF_LIFE_DEFAULT_DAYS = 45.0      # ASSUMPTION: cooling half-life
# Methodology reference: Bao, Wang & Gates (2016) Energy 115:969-985.

# ---------------------------------------------------------------------------
# SRP physics constants (API RP 11L; US Patent 11,898,552 B2)
# ---------------------------------------------------------------------------
WATER_DENSITY_LB_FT3 = 62.4        # CONST
STEEL_DENSITY_LB_FT3 = 490.0       # CONST  (NOT 7.85 — see srp_physics.py)
STEEL_SG = 7.85                    # reference only

# Rod / unit defaults — ASSUMPTIONS, replace with actual equipment data
DEFAULT_ROD_DIA_IN = 1.0
DEFAULT_PLUNGER_DIA_IN = 2.25
DEFAULT_STROKE_IN = 100.0
DEFAULT_PUMP_DEPTH_FT = 3937.0     # ~1200 m, midpoint of Baghewala range
DEFAULT_FLUID_SG = 0.96            # 14-17 API -> SG 0.95-0.97
DEFAULT_C_OVER_H = 0.0

# Guardrail limits — CONFIGURATION, replace with actual rod grade / gearbox
DEFAULT_SIGMA_LIMIT_PSI = 30_000.0
DEFAULT_GEARBOX_RATING_IN_LBF = 228_000.0
DEFAULT_TORQUE_FACTOR_IN = 30.0    # API RP 11L torque factor %S

# ---------------------------------------------------------------------------
# SPM recommendation rule table (HEURISTIC — not a published correlation)
# (mu_upper_cP, spm_min, spm_max). Replace with field-validated limits.
# ---------------------------------------------------------------------------
SPM_RULE_TABLE = [
    (500.0,          6.0, 10.0),
    (2_000.0,        5.0,  8.0),
    (5_000.0,        4.0,  6.0),
    (10_000.0,       3.0,  5.0),
    (20_000.0,       2.0,  4.0),
    (float("inf"),   1.0,  3.0),
]
DANGER_MU_CP = BAGHEWALA["mu_low_cp"]
FILLAGE_OK_PCT = 80.0
FILLAGE_LOW_PCT = 60.0
FILLAGE_LOW_CAP_SPM = 3.0

# ---------------------------------------------------------------------------
# ML config
# ---------------------------------------------------------------------------
SEED = 42
SEQ_LEN = 8                 # window length for the GB tabular features
TRAIN_FRAC = 0.80           # temporal split within each well
GB_N_ESTIMATORS = 200
GB_MAX_DEPTH = 4
GB_LEARNING_RATE = 0.05

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(REPO_ROOT, "data", "raw")
DATA_PROC = os.path.join(REPO_ROOT, "data", "processed")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
NK_PARQUET = os.path.join(DATA_RAW, "nk_field.parquet")
NK_WIDE = os.path.join(DATA_PROC, "telemetry_wide.parquet")
NK_DATASET_ID = "Arailym-tleubayeva/NK-Oil-Well-Sensor-Monitoring"