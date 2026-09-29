"""
CSS cooling trajectory.

    T(t) = T_res + (T_inj - T_res) * exp(-lambda * t)     t in DAYS

Provenance:
  T_res  : Baghewala 46-48 degC (Source C)
  T_inj  : ASSUMPTION, conventional CSS (250 degC default)
  lambda : ASSUMPTION, from a cooling half-life (45 days default)

Methodology: Bao, Wang & Gates (2016) Energy 115:969-985 (shape).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

from .config import (
    BAGHEWALA, T_INJ_DEFAULT_C, HALF_LIFE_DEFAULT_DAYS, ANCHOR_ENVELOPE_C,
)


def lambda_from_half_life(half_life_days: float) -> float:
    if half_life_days <= 0:
        raise ValueError("half_life_days must be > 0")
    return math.log(2.0) / float(half_life_days)


@dataclass
class CSSTrajectory:
    t_res_c: float
    t_inj_c: float
    lam_per_day: float

    @classmethod
    def default(cls) -> "CSSTrajectory":
        t_res = 0.5 * (BAGHEWALA["t_res_min_c"] + BAGHEWALA["t_res_max_c"])
        return cls(
            t_res_c=t_res,
            t_inj_c=T_INJ_DEFAULT_C,
            lam_per_day=lambda_from_half_life(HALF_LIFE_DEFAULT_DAYS),
        )

    def T(self, t_days) -> float:
        t = np.asarray(t_days, dtype=float)
        out = self.t_res_c + (self.t_inj_c - self.t_res_c) * np.exp(-self.lam_per_day * t)
        return float(out) if np.ndim(out) == 0 else out

    def T_array(self, t_days) -> np.ndarray:
        return np.asarray(self.T(t_days), dtype=float)

    def half_life_days(self) -> float:
        return math.log(2.0) / self.lam_per_day

    def describe(self) -> str:
        return (
            f"CSS cooling: T(t) = {self.t_res_c:.1f} + "
            f"({self.t_inj_c:.1f} - {self.t_res_c:.1f}) * exp(-{self.lam_per_day:.5f} * t)\n"
            f"  T_res  = {self.t_res_c:.1f} degC   [Source C, 46-48 degC midpoint]\n"
            f"  T_inj  = {self.t_inj_c:.1f} degC   [ASSUMPTION]\n"
            f"  lambda = {self.lam_per_day:.5f} /day (half-life {self.half_life_days():.1f} d)  [ASSUMPTION]"
        )


def fit_lambda(t_days, t_c, t_res_c, t_inj_c) -> float:
    t = np.asarray(t_days, dtype=float)
    T = np.asarray(t_c, dtype=float)
    ratio = (T - t_res_c) / (t_inj_c - t_res_c)
    if np.any(ratio <= 0):
        raise ValueError("Cooling curve has samples at or below T_res.")
    slope, _ = np.polyfit(t, np.log(ratio), 1)
    lam = -float(slope)
    if lam <= 0:
        raise ValueError("Fitted lambda is not positive.")
    return lam