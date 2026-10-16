"""Derived DCE metrics independent of fit-quality classification."""

import numpy as np


def compute_kep(ktrans: float, ve: float) -> float:
    """Return a nonnegative ratio representable in DCE float32 maps.

    Parameters
    ----------
    ktrans : float
        Volume transfer constant in inverse minutes.
    ve : float
        Extravascular extracellular volume fraction.

    Returns
    -------
    float
        Ktrans/ve in inverse minutes, or NaN for invalid inputs or map overflow.
    """
    if not np.isfinite(ktrans) or ktrans < 0 or not np.isfinite(ve) or ve <= 0:
        return float("nan")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratio = np.float64(ktrans) / np.float64(ve)
    return (
        float(ratio)
        if np.isfinite(ratio) and ratio <= np.finfo(np.float32).max
        else float("nan")
    )
