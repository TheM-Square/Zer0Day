"""
SRP (sucker-rod pumping) mechanical dynamics + hard guardrail.

Equations (API RP 11L, spec form):

    PRLmax = Wf - 62.4*Sf*(Wr/gamma_s) + Wr
             + Wr*(S*N^2*(1 +/- c/h)) / 70471.2

    T_peak = %S * (Wf + (2*S*N^2*Wr) / 707471.2)

    sigma_max = PRLmax / Ar

    SLR = F_min / F_max               (US Patent 11,898,552 B2)

    if sigma_max > sigma_limit  ->  VETO

UNITS
    Wf        lbf          fluid load on plunger
    Sf        -            fluid SG (water = 1)
    Wr        lbf          rod string weight in air
    gamma_s   lb/ft^3      490 (steel density — NOT SG 7.85; see note)
    S         in           polished-rod stroke
    N         strokes/min  pumping speed
    c/h       -            rod acceleration ratio
    %S        in           API RP 11L torque factor
    Ar        in^2         rod cross-section
    sigma     psi

NOTE on gamma_s: the 62.4 factor only balances dimensionally if gamma_s is
density in lb/ft^3 (490). Using SG (7.85) makes the buoyancy term ~60x too
large and drives PRLmax negative.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

from .config import (
    WATER_DENSITY_LB_FT3, STEEL_DENSITY_LB_FT3,
    DEFAULT_SIGMA_LIMIT_PSI, DEFAULT_GEARBOX_RATING_IN_LBF,
    DEFAULT_TORQUE_FACTOR_IN,
)


def rod_area_in2(d_rod_in: float) -> float:
    return math.pi / 4.0 * float(d_rod_in) ** 2


def rod_weight_air_lbf(d_rod_in: float, length_ft: float) -> float:
    # Wr = 2.668 * d^2 * L   (steel density 490 lb/ft^3)
    return 2.668 * float(d_rod_in) ** 2 * float(length_ft)


def fluid_load_lbf(sf_fluid: float, pump_depth_ft: float, d_plunger_in: float) -> float:
    ap = math.pi / 4.0 * float(d_plunger_in) ** 2
    return 0.433 * float(sf_fluid) * float(pump_depth_ft) * ap


def sg_from_api(api: float) -> float:
    return 141.5 / (float(api) + 131.5)


def prl_max_lbf(Wf, Sf, Wr, gamma_s, S, N, c_over_h=0.0, sign=+1):
    return (Wf
            - WATER_DENSITY_LB_FT3 * Sf * (Wr / gamma_s)
            + Wr
            + Wr * (S * N ** 2 * (1 + sign * c_over_h)) / 70471.2)


def peak_torque_in_lbf(pct_S, Wf, S, N, Wr):
    return pct_S * (Wf + (2.0 * S * N ** 2 * Wr) / 707471.2)


def max_rod_stress_psi(prl_max, Ar):
    return prl_max / Ar


def scaled_load_ratio(F_min, F_max):
    return F_min / F_max if F_max else float("nan")


@dataclass
class GuardrailResult:
    prl_max_lbf: float
    prl_min_lbf: float
    peak_torque_in_lbf: float
    sigma_max_psi: float
    sigma_limit_psi: float
    stress_utilization: float
    torque_utilization: float
    slr: float
    veto: bool
    status: str
    reasons: list
    max_safe_spm: float | None


def max_safe_spm(*, stroke_in, d_rod_in, rod_length_ft, d_plunger_in,
                 pump_depth_ft, sf_fluid, c_over_h,
                 sigma_limit_psi, gearbox_rating_in_lbf, torque_factor_in):
    """
    Largest SPM satisfying both the stress and torque limits.

    PRLmax(N) = C1 + C2 * N^2    -> N <= sqrt((sigma_limit*Ar - C1) / C2)
    T_peak(N) = %S * (Wf + C3*N^2)  -> N <= sqrt((T_lim/%S - Wf) / C3)
    Returns None if no positive SPM is feasible.
    """
    Ar = rod_area_in2(d_rod_in)
    Wr = rod_weight_air_lbf(d_rod_in, rod_length_ft)
    Wf = fluid_load_lbf(sf_fluid, pump_depth_ft, d_plunger_in)

    C1 = Wf - WATER_DENSITY_LB_FT3 * sf_fluid * (Wr / STEEL_DENSITY_LB_FT3) + Wr
    C2 = Wr * stroke_in * (1.0 + c_over_h) / 70471.2
    C3 = 2.0 * stroke_in * Wr / 707471.2

    candidates = []
    if C2 > 0:
        num = sigma_limit_psi * Ar - C1
        candidates.append(math.sqrt(num / C2) if num > 0 else 0.0)
    elif C1 > sigma_limit_psi * Ar:
        return None

    if C3 > 0 and torque_factor_in > 0:
        num = (gearbox_rating_in_lbf / torque_factor_in) - Wf
        candidates.append(math.sqrt(num / C3) if num > 0 else 0.0)

    if not candidates:
        return None
    return math.floor(min(candidates) * 10.0) / 10.0


def evaluate(*, spm, stroke_in,
             d_rod_in=1.0, rod_length_ft=3937.0,
             d_plunger_in=2.25, pump_depth_ft=3937.0,
             sf_fluid=0.96, c_over_h=0.0,
             sigma_limit_psi=DEFAULT_SIGMA_LIMIT_PSI,
             gearbox_rating_in_lbf=DEFAULT_GEARBOX_RATING_IN_LBF,
             torque_factor_in=DEFAULT_TORQUE_FACTOR_IN) -> GuardrailResult:
    Ar = rod_area_in2(d_rod_in)
    Wr = rod_weight_air_lbf(d_rod_in, rod_length_ft)
    Wf = fluid_load_lbf(sf_fluid, pump_depth_ft, d_plunger_in)

    F_max = prl_max_lbf(Wf, sf_fluid, Wr, STEEL_DENSITY_LB_FT3,
                        stroke_in, spm, c_over_h, sign=+1)
    F_min = prl_max_lbf(Wf, sf_fluid, Wr, STEEL_DENSITY_LB_FT3,
                        stroke_in, spm, c_over_h, sign=-1)

    T_peak = peak_torque_in_lbf(torque_factor_in, Wf, stroke_in, spm, Wr)
    sigma = max_rod_stress_psi(F_max, Ar)

    reasons = []
    if sigma > sigma_limit_psi:
        reasons.append(f"max rod stress {sigma:,.0f} psi > limit {sigma_limit_psi:,.0f} psi")
    if T_peak > gearbox_rating_in_lbf:
        reasons.append(f"peak torque {T_peak:,.0f} in-lbf > rating {gearbox_rating_in_lbf:,.0f}")

    return GuardrailResult(
        prl_max_lbf=F_max,
        prl_min_lbf=F_min,
        peak_torque_in_lbf=T_peak,
        sigma_max_psi=sigma,
        sigma_limit_psi=sigma_limit_psi,
        stress_utilization=sigma / sigma_limit_psi,
        torque_utilization=T_peak / gearbox_rating_in_lbf,
        slr=scaled_load_ratio(F_min, F_max),
        veto=bool(reasons),
        status="VETO" if reasons else "PASS",
        reasons=reasons,
        max_safe_spm=max_safe_spm(
            stroke_in=stroke_in, d_rod_in=d_rod_in, rod_length_ft=rod_length_ft,
            d_plunger_in=d_plunger_in, pump_depth_ft=pump_depth_ft,
            sf_fluid=sf_fluid, c_over_h=c_over_h,
            sigma_limit_psi=sigma_limit_psi,
            gearbox_rating_in_lbf=gearbox_rating_in_lbf,
            torque_factor_in=torque_factor_in,
        ),
    )