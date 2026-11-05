"""Numerical parameter retention, separate from identification and fit QC."""

import numpy as np

from pmriprep.utils.dce_constants import (
    FP_BOUNDS, PS_BOUNDS, KTRANS_BOUNDS, VE_BOUNDS, VP_BOUNDS, VP_VE_TOTAL_MAX,
)

# Numerical endpoint tolerance for strictly feasible optimizer nudges.
_RATE_BOUND_RTOL: float = 1e-10


def rate_ceiling_hits(model_name: str, params: dict[str, float]) -> tuple[str, ...]:
    """Return finite rates touching their numerical ceiling, including tolerance."""
    ceilings = (
        (("Fp", FP_BOUNDS.max), ("PS", PS_BOUNDS.max))
        if model_name == "2cxm" else (("Ktrans", KTRANS_BOUNDS.max),)
    )
    return tuple(
        name for name, ceiling in ceilings
        if np.isfinite(params.get(name, np.nan))
        and params[name] >= ceiling - _RATE_BOUND_RTOL * max(1.0, abs(ceiling))
    )


def is_degenerate_fit(
    model_name: str,
    params: dict[str, float],
) -> bool:
    """Return whether a DCE fit violates the parameter-retention rules."""
    ktrans = params.get("Ktrans", 0.0)
    ve = params.get("ve", 0.0)
    vp = params.get("vp", 0.0)
    rate_ceilings = (
        (("Fp", FP_BOUNDS.max), ("PS", PS_BOUNDS.max))
        if model_name == "2cxm" else (("Ktrans", KTRANS_BOUNDS.max),)
    )
    if rate_ceiling_hits(model_name, params) or any(
        not np.isfinite(params.get(name, np.nan)) for name, _ in rate_ceilings
    ):
        return True
    if model_name != "patlak":
        if not np.isfinite(ve) or not 0 < ve <= VE_BOUNDS.max:
            return True
    if model_name in ("patlak", "extended_tofts", "2cxm"):
        if not np.isfinite(vp) or not VP_BOUNDS.min <= vp <= VP_BOUNDS.max:
            return True
        if model_name != "patlak" and ve + vp > VP_VE_TOTAL_MAX + 1e-12:
            return True
    if model_name == "2cxm":
        fp = params.get("Fp", 0.0)
        ps = params.get("PS", 0.0)
        return fp <= 0 or ps <= 0
    if model_name == "patlak":
        return ktrans < 0
    return ktrans <= 0
