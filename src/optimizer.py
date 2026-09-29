"""
Constraint-aware SPM optimizer.

Formulation:
    max   Q_oil(SPM)                     [volumetric SRP pump equation]
    s.t.  sigma_max(SPM) <= sigma_limit  [hard, API RP 11L]
          T_peak(SPM)    <= gearbox      [hard, API RP 11L]
          SPM_min <= SPM <= SPM_max      [equipment]
          fillage >= fillage_min (if measured)

Solution: grid search over SPM (0.1 strokes/min resolution).
    - The feasible set is filtered by the guardrail on every grid point.
    - Objective is the SRP volumetric pump rate scaled by a viscosity-
      dependent efficiency factor.
    - The pump-rate formula and efficiency heuristic are documented below.

This is genuine constrained optimization. The feasible set can be empty,
in which case the optimizer returns a clear reason and no recommendation.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

from .config import SPM_RULE_TABLE, DANGER_MU_CP
from .srp_physics import evaluate, GuardrailResult


def pump_rate_bpd(spm: float, stroke_in: float, d_plunger_in: float,
                  fillage_pct: float, mu_cp: float) -> float:
    """
    Volumetric SRP pump rate (bbl/day) with a viscosity-efficiency factor.

    Q = 0.1484 * Ap * S * N * (fillage/100) * eta(mu)

    Ap in in^2, S in in, N in strokes/min -> Q in bbl/day at 100% efficiency.
    eta(mu): heuristic efficiency penalty; heavy oil reduces fillage and
    increases slip. Functional form:
        eta = 1 / (1 + (mu / mu_ref)^0.25)      with mu_ref = 1000 cP
    This is an ENGINEERING HEURISTIC, not a measured correlation.
    """
    Ap = math.pi / 4.0 * d_plunger_in ** 2
    base = 0.1484 * Ap * stroke_in * float(spm) * (float(fillage_pct) / 100.0)
    mu_ref = 1_000.0
    eta = 1.0 / (1.0 + (float(mu_cp) / mu_ref) ** 0.25)
    return float(base * eta)


@dataclass
class OptimizationResult:
    current_spm: float
    recommended_spm: float | None
    feasible: bool
    reason: str
    grid_spm: np.ndarray
    grid_rate: np.ndarray
    grid_sigma: np.ndarray
    grid_torque: np.ndarray
    grid_feasible: np.ndarray
    current_rate: float
    recommended_rate: float | None
    delta_rate: float | None
    guardrail_at_recommendation: GuardrailResult | None


def optimize_spm(
    mu_cp: float,
    current_spm: float,
    fillage_pct: float,
    stroke_in: float,
    d_plunger_in: float,
    d_rod_in: float,
    pump_depth_ft: float,
    sf_fluid: float,
    sigma_limit_psi: float,
    gearbox_rating_in_lbf: float,
    torque_factor_in: float,
    spm_min: float = 1.0,
    spm_max: float = 12.0,
) -> OptimizationResult:

    grid = np.arange(spm_min, spm_max + 1e-9, 0.1)

    grid_rate = np.array([
        pump_rate_bpd(s, stroke_in, d_plunger_in, fillage_pct, mu_cp)
        for s in grid
    ])
    grid_sigma = np.zeros_like(grid)
    grid_torque = np.zeros_like(grid)
    grid_feasible = np.zeros_like(grid, dtype=bool)

    for i, s in enumerate(grid):
        g = evaluate(
            spm=float(s), stroke_in=stroke_in,
            d_rod_in=d_rod_in, rod_length_ft=pump_depth_ft,
            d_plunger_in=d_plunger_in, pump_depth_ft=pump_depth_ft,
            sf_fluid=sf_fluid,
            sigma_limit_psi=sigma_limit_psi,
            gearbox_rating_in_lbf=gearbox_rating_in_lbf,
            torque_factor_in=torque_factor_in,
        )
        grid_sigma[i] = g.sigma_max_psi
        grid_torque[i] = g.peak_torque_in_lbf
        grid_feasible[i] = not g.veto

    current_rate = pump_rate_bpd(current_spm, stroke_in, d_plunger_in,
                                 fillage_pct, mu_cp)

    if not grid_feasible.any():
        return OptimizationResult(
            current_spm=current_spm, recommended_spm=None, feasible=False,
            reason=("No feasible SPM in the search grid — rod stress or "
                    "gearbox torque limits are exceeded at every candidate."),
            grid_spm=grid, grid_rate=grid_rate,
            grid_sigma=grid_sigma, grid_torque=grid_torque,
            grid_feasible=grid_feasible,
            current_rate=current_rate, recommended_rate=None,
            delta_rate=None, guardrail_at_recommendation=None,
        )

    feasible_idx = np.where(grid_feasible)[0]
    best_i = feasible_idx[np.argmax(grid_rate[feasible_idx])]
    recommended_spm = float(grid[best_i])
    recommended_rate = float(grid_rate[best_i])

    guard = evaluate(
        spm=recommended_spm, stroke_in=stroke_in,
        d_rod_in=d_rod_in, rod_length_ft=pump_depth_ft,
        d_plunger_in=d_plunger_in, pump_depth_ft=pump_depth_ft,
        sf_fluid=sf_fluid,
        sigma_limit_psi=sigma_limit_psi,
        gearbox_rating_in_lbf=gearbox_rating_in_lbf,
        torque_factor_in=torque_factor_in,
    )

    return OptimizationResult(
        current_spm=current_spm,
        recommended_spm=recommended_spm,
        feasible=True,
        reason=f"Found feasible SPM maximizing pump rate within constraints.",
        grid_spm=grid, grid_rate=grid_rate,
        grid_sigma=grid_sigma, grid_torque=grid_torque,
        grid_feasible=grid_feasible,
        current_rate=current_rate,
        recommended_rate=recommended_rate,
        delta_rate=recommended_rate - current_rate,
        guardrail_at_recommendation=guard,
    )


def spm_band_from_viscosity(mu_cp: float) -> tuple[float, float]:
    """Rule lookup on the heuristic table in config.py."""
    for mu_upper, lo, hi in SPM_RULE_TABLE:
        if mu_cp < mu_upper:
            return lo, hi
    return SPM_RULE_TABLE[-1][1], SPM_RULE_TABLE[-1][2]