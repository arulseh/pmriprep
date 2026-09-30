"""Explicit optional oSVD substitution, separate from CPI numerical acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np

from pmriprep.utils.dsc_constants import EPSILON_CONCENTRATION
from pmriprep.utils.dsc_deconvolution import svd_deconvolve_oscillation

from .acquisition import CPIAcquisition
from .connection import GuidedVoxel


@dataclass(frozen=True)
class CPIOutputVoxel:
    """Native-unit map measurements with the rejected CPI record kept separate.

    The fallback mask is true only for source osvd, never merely because
    fallback was requested or attempted. A substituted result has no CPI
    uncertainty or fitted leakage coefficient. The original record remains
    available for fit diagnostics, including its convergence and R-squared.
    """

    fit: GuidedVoxel
    source: Literal["cpi", "osvd", "missing", "outside_support"]
    cbf: float = np.nan
    cbv: float = np.nan
    mtt: float = np.nan
    delay: float = np.nan
    cbf_std: float = np.nan
    fitted_k2: float = np.nan
    fallback_attempted: bool = False
    fallback_reason: str | None = None


def resolve_fallbacks(
    acquisition: CPIAcquisition, rows: Iterator[GuidedVoxel],
    *, raw_concentration: np.ndarray, corrected_concentration: np.ndarray | None,
    oi_target: float,
) -> Iterator[CPIOutputVoxel]:
    """Resolve rejected fits against the declared leakage policy's concentration.

    Parameters
    ----------
    acquisition : CPIAcquisition
        Prepared signal acquisition and optional fallback policy.
    rows : iterator of GuidedVoxel
        Assessed ordinary or spatial CPI outputs in image-flat order or a subset.
    raw_concentration : ndarray
        Uncorrected, baseline-centred delta-R2-star on the signal image grid.
        Used by none mode regardless of any supplied corrected array.
    corrected_concentration : ndarray or None
        Exact upstream BSW-corrected, baseline-centred delta-R2-star. Required
        for enabled BSW-mode substitution; never reconstructed from fitted K2.
    oi_target : float
        Configured positive oscillation-index threshold for the existing oSVD.

    Yields
    ------
    CPIOutputVoxel
        Accepted CPI, actual oSVD replacement, or missing measurements with
        distinct reasons. Numerical nonconvergence alone never triggers oSVD.

    Notes
    -----
    This does not compute or replace primary area-based CBV. Unexpected
    programming errors propagate; recognized numerical solver failures leave
    the already-rejected voxel missing and record the failure.
    """
    a = acquisition.branches[0]
    enabled = acquisition.options.fallback == "osvd"
    curves = raw_concentration if a.leakage_method == "none" else corrected_concentration
    if enabled:
        expected_shape = (*a.spatial_shape, a.time.size)
        if curves is None or curves.shape != expected_shape:
            raise ValueError("fallback requires the declared leakage policy's concentration on the signal grid")
        if not np.isfinite(oi_target) or oi_target <= 0:
            raise ValueError("positive finite fallback oscillation threshold required")
        curves = np.asarray(curves, dtype=float).reshape(a.signals.shape)
    seen = set()
    for row in rows:
        i = row.flat_index
        if (isinstance(i, bool) or not isinstance(i, (int, np.integer))
                or not 0 <= i < len(a.signals) or i in seen):
            raise ValueError("unique in-range output voxel indices required")
        seen.add(i)
        if row.status == "outside_support":
            yield CPIOutputVoxel(row, "outside_support")
        elif row.status in ("retained", "provisional"):
            yield CPIOutputVoxel(row, "cpi", row.cbf, row.cbv, row.mtt, row.delay,
                                 row.cbf_std, row.fitted_k2)
        elif not enabled:
            yield CPIOutputVoxel(row, "missing", fallback_reason="disabled")
        else:
            assert curves is not None
            tissue = curves[i]
            if not np.all(np.isfinite(tissue)) or np.max(np.abs(tissue)) < EPSILON_CONCENTRATION:
                yield CPIOutputVoxel(row, "missing", fallback_reason="inadmissible_concentration")
                continue
            try:
                fit = svd_deconvolve_oscillation(tissue, a.aif, float(a.time[1] - a.time[0]),
                                                 oi_target=oi_target)
            except (np.linalg.LinAlgError, FloatingPointError):
                yield CPIOutputVoxel(row, "missing", fallback_attempted=True,
                                     fallback_reason="numerical_failure")
                continue
            if (not np.all(np.isfinite([fit.cbf, fit.cbv, fit.mtt, fit.delay]))
                    or min(fit.cbf, fit.cbv, fit.mtt) <= 0):
                yield CPIOutputVoxel(row, "missing", fallback_attempted=True,
                                     fallback_reason="invalid_osvd_measurement")
                continue
            yield CPIOutputVoxel(row, "osvd", fit.cbf, fit.cbv, fit.mtt, np.nan,
                                 fallback_attempted=True)
