"""
Arrhenius viscosity model for Baghewala heavy crude.

    mu(T) = A * exp(B / T)      T in Kelvin, mu in cP

Anchor (Source C — OIL tender):
    mu = 10,000-13,000 cP @ 45 degC

Identifiability: with one temperature and a range, A and B cannot both be
identified from data. We therefore take B from external literature (shape)
and calibrate A so that mu(45 degC) = 11,500 cP (midpoint). The published
range is carried as an uncertainty band. Away from ~45 degC, results are
EXTRAPOLATED and flagged.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np

from .config import (
    BAGHEWALA, B_K, ANCHOR_ENVELOPE_C, R_GAS,
)


def c_to_k(t_c: float) -> float:
    return float(t_c) + 273.15


def arrhenius_mu(t_c, A, B):
    return float(A) * math.exp(float(B) / c_to_k(t_c))


def arrhenius_mu_array(t_c, A, B):
    t_k = np.asarray(t_c, dtype=float) + 273.15
    return float(A) * np.exp(float(B) / t_k)


def calibrate_A(B: float, t_c: float, mu_cp: float) -> float:
    return float(mu_cp) / math.exp(float(B) / c_to_k(t_c))


def fit_from_curve(t_c, mu_cp):
    """Least-squares fit of ln(mu) = ln(A) + B*(1/T). Needs >= 2 distinct T."""
    t_c = np.asarray(t_c, dtype=float)
    mu = np.asarray(mu_cp, dtype=float)
    if np.unique(t_c).size < 2:
        raise ValueError("Need at least two distinct temperatures.")
    inv_t = 1.0 / (t_c + 273.15)
    slope, intercept = np.polyfit(inv_t, np.log(mu), 1)
    return float(math.exp(intercept)), float(slope)


@dataclass
class ViscosityModel:
    A: float
    B: float
    mu_low_cp: float = BAGHEWALA["mu_low_cp"]
    mu_high_cp: float = BAGHEWALA["mu_high_cp"]
    mu_ref_temp_c: float = BAGHEWALA["mu_ref_temp_c"]
    B_source: str = field(
        default=f"external shape (Ea={60.0:.0f} kJ/mol, B={B_K:.1f} K)"
    )

    @classmethod
    def from_baghewala(cls, B: float = B_K) -> "ViscosityModel":
        mu_mid = 0.5 * (BAGHEWALA["mu_low_cp"] + BAGHEWALA["mu_high_cp"])
        A = calibrate_A(B, BAGHEWALA["mu_ref_temp_c"], mu_mid)
        return cls(A=A, B=B)

    @classmethod
    def from_curve(cls, t_c, mu_cp, reanchor: bool = True) -> "ViscosityModel":
        A_fit, B_fit = fit_from_curve(t_c, mu_cp)
        if reanchor:
            mu_mid = 0.5 * (BAGHEWALA["mu_low_cp"] + BAGHEWALA["mu_high_cp"])
            A_fit = calibrate_A(B_fit, BAGHEWALA["mu_ref_temp_c"], mu_mid)
        return cls(A=A_fit, B=B_fit, B_source="fitted from supplied mu(T) curve")

    def mu(self, t_c: float) -> float:
        return arrhenius_mu(t_c, self.A, self.B)

    def mu_array(self, t_c) -> np.ndarray:
        return arrhenius_mu_array(t_c, self.A, self.B)

    def mu_band(self, t_c: float) -> tuple[float, float]:
        A_lo = calibrate_A(self.B, self.mu_ref_temp_c, self.mu_low_cp)
        A_hi = calibrate_A(self.B, self.mu_ref_temp_c, self.mu_high_cp)
        return (arrhenius_mu(t_c, A_lo, self.B),
                arrhenius_mu(t_c, A_hi, self.B))

    def is_extrapolated(self, t_c: float) -> bool:
        lo = self.mu_ref_temp_c - ANCHOR_ENVELOPE_C
        hi = self.mu_ref_temp_c + ANCHOR_ENVELOPE_C
        return not (lo <= float(t_c) <= hi)

    def describe(self) -> str:
        ea = self.B * R_GAS / 1000.0
        return (
            f"Arrhenius mu(T) = A*exp(B/T)\n"
            f"  A = {self.A:.6e} cP\n"
            f"  B = {self.B:.2f} K   (Ea = {ea:.1f} kJ/mol)\n"
            f"  B source: {self.B_source}\n"
            f"  Anchored to: {self.mu_low_cp:,.0f}-{self.mu_high_cp:,.0f} cP "
            f"@ {self.mu_ref_temp_c:.0f} degC\n"
            f"  Extrapolation envelope: +/- {ANCHOR_ENVELOPE_C:.0f} degC around anchor"
        )