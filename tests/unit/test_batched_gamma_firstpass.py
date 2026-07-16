"""Unit tests for ``batched_gamma_firstpass_auc`` (first-pass gamma area).

The estimator fits a gamma-variate to each voxel's first pass and takes the
analytic area. Voxels whose fit is rejected are NaN (no truncating fallback)
and counted in ``unfittable_fraction``. The key properties: the area scales
with amplitude, fitted voxels are not biased low (the removed windowed fallback
was the source of the ~30% under-read), and the unfittable rate rises at low
CNR rather than the area drifting.
"""
from __future__ import annotations

import numpy as np
import pytest

from pmriprep.utils.dsc_aif import batched_gamma_firstpass_auc


def _gamma(t, amplitude, t0, alpha, beta):
    dt = t - t0
    c = np.zeros_like(t, dtype=np.float64)
    m = dt > 0
    c[m] = amplitude * dt[m] ** alpha * np.exp(-dt[m] / beta)
    return c


def test_clean_gamma_fits_and_is_positive():
    T = 80
    t = np.arange(T, dtype=np.float64)
    curve = _gamma(t, 2.0, 8.0, 3.0, 4.0)
    res = batched_gamma_firstpass_auc(curve[None, :], tr=1.0, baseline_frames=5)
    assert res.n_fit == 1
    assert res.unfittable_fraction == 0.0
    assert np.isfinite(res.auc[0]) and res.auc[0] > 0.0


def test_area_scales_with_amplitude():
    """Fixed shape, doubled amplitude -> doubled analytic area (ratio use)."""
    T = 80
    t = np.arange(T, dtype=np.float64)
    c1 = _gamma(t, 1.0, 8.0, 3.0, 4.0)
    c2 = _gamma(t, 2.0, 8.0, 3.0, 4.0)
    res = batched_gamma_firstpass_auc(np.stack([c1, c2]), tr=1.0, baseline_frames=5)
    assert res.n_fit == 2
    assert np.isclose(res.auc[1] / res.auc[0], 2.0, rtol=0.02)


def test_no_bolus_is_unfittable():
    """Pure baseline noise (no first pass) is rejected -> NaN, high unfittable."""
    rng = np.random.default_rng(0)
    flat = rng.normal(0.0, 0.01, size=(200, 60))
    res = batched_gamma_firstpass_auc(flat, tr=1.0, baseline_frames=5)
    assert res.unfittable_fraction > 0.9
    assert np.isnan(res.auc).mean() > 0.9


def test_unfittable_rises_at_low_cnr():
    """The CNR limitation shows up as more unfittable voxels, not a biased area."""
    rng = np.random.default_rng(1)
    T = 80
    t = np.arange(T, dtype=np.float64)
    base = _gamma(t, 3.0, 8.0, 3.0, 4.0)
    peak = float(base.max())

    def unfit(cnr):
        noisy = base[None, :] + rng.normal(0.0, peak / cnr, size=(400, T))
        return batched_gamma_firstpass_auc(noisy, tr=1.0, baseline_frames=5).unfittable_fraction

    assert unfit(5.0) > unfit(30.0)


def test_fitted_area_not_biased_low():
    """Among fitted voxels the mean area tracks the noiseless area.

    Guards the reason the fallback was removed: the ~30% under-read was a
    truncated-tail artifact of the windowed fallback, not the gamma area. A
    reintroduced truncating fallback would pull this mean down.
    """
    rng = np.random.default_rng(2)
    T = 80
    t = np.arange(T, dtype=np.float64)
    base = _gamma(t, 3.0, 8.0, 3.0, 4.0)
    peak = float(base.max())
    noiseless = float(
        batched_gamma_firstpass_auc(base[None, :], tr=1.0, baseline_frames=5).auc[0]
    )
    noisy = base[None, :] + rng.normal(0.0, peak / 20.0, size=(3000, T))
    res = batched_gamma_firstpass_auc(noisy, tr=1.0, baseline_frames=5)
    fitted = res.auc[np.isfinite(res.auc)]
    assert fitted.size > 200
    assert abs(float(np.mean(fitted)) - noiseless) / noiseless < 0.15


def test_requires_2d():
    with pytest.raises(ValueError, match="2D"):
        batched_gamma_firstpass_auc(np.zeros(10), tr=1.0, baseline_frames=5)


def test_nan_inf_do_not_crash_or_propagate():
    T = 80
    t = np.arange(T, dtype=np.float64)
    curve = _gamma(t, 2.0, 8.0, 3.0, 4.0)
    curve[20] = np.nan
    curve[25] = np.inf
    res = batched_gamma_firstpass_auc(curve[None, :], tr=1.0, baseline_frames=5)
    # No inf/garbage: the voxel either fits (finite) or is cleanly rejected (NaN).
    assert np.isfinite(res.auc[0]) or np.isnan(res.auc[0])
