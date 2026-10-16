"""Test Tofts-family recovery with a sampled Parker population AIF.

Synthetic tissue and fitting use the same forward model, so these tests
isolate initialization and optimization behavior across the stated parameter
and noise fixtures. They do not establish recovery for arbitrary acquisitions.
"""
from __future__ import annotations

import numpy as np
import pytest

from pmriprep.models.dce.extended_tofts import ExtendedToftsModel
from pmriprep.models.dce.tofts import ToftsModel


def _parker_aif(
    n: int = 80, dt_min: float = 0.0667, arrival_min: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a Parker population AIF and normalize its peak."""
    t = np.arange(n) * dt_min
    tt = np.maximum(t - arrival_min, 0.0)
    a = (0.809, 0.330)
    t_ctr = (0.17046, 0.365)
    sig = (0.0563, 0.132)
    alpha, beta, s, tau = 1.050, 0.1685, 38.078, 0.483
    curve = np.zeros_like(tt)
    for a_n, t_n, s_n in zip(a, t_ctr, sig):
        curve += a_n / (s_n * np.sqrt(2 * np.pi)) * np.exp(
            -((tt - t_n) ** 2) / (2 * s_n**2)
        )
    curve += alpha * np.exp(-beta * tt) / (1 + np.exp(-s * (tt - tau)))
    curve[t <= arrival_min] = 0.0
    return curve * (5.0 / curve.max()), t


@pytest.mark.parametrize("ve", [0.05, 0.1, 0.15, 0.2, 0.4])
def test_tofts_recovers_ktrans_across_kep(ve: float) -> None:
    """Ktrans is recovered within 2% across kep = 0.75 to 6.0 in these fixtures."""
    aif, t = _parker_aif()
    model = ToftsModel()
    ktrans = 0.3
    gt = model.evaluate([ktrans, ve], aif, t)
    recovered = model.fit(gt, aif, t).parameters["Ktrans"]
    assert recovered == pytest.approx(ktrans, rel=0.02), (
        f"kep={ktrans / ve:.1f}: Ktrans {recovered:.3f} vs truth {ktrans}"
    )


@pytest.mark.parametrize("ve,vp", [(0.5, 0.25), (0.7, 0.2), (0.6, 0.3)])
def test_extended_tofts_recovers_ktrans_high_volume(ve: float, vp: float) -> None:
    """Ktrans is recovered within 2% in the tested high-ve/high-vp fixtures."""
    aif, t = _parker_aif()
    model = ExtendedToftsModel()
    ktrans = 0.3
    gt = model.evaluate([ktrans, ve, vp], aif, t)
    recovered = model.fit(gt, aif, t).parameters["Ktrans"]
    assert recovered == pytest.approx(ktrans, rel=0.02)


def test_tofts_high_kep_stable_under_noise() -> None:
    """Median Ktrans across 50 noise realizations is within 5% of fixture truth."""
    aif, t = _parker_aif()
    model = ToftsModel()
    ktrans, ve = 0.3, 0.1  # kep = 3
    gt = model.evaluate([ktrans, ve], aif, t)
    rng = np.random.default_rng(0)
    ests = [
        model.fit(gt + rng.normal(0, 0.02 * gt.max(), len(gt)), aif, t).parameters["Ktrans"]
        for _ in range(50)
    ]
    assert float(np.median(ests)) == pytest.approx(ktrans, rel=0.05)


def test_tofts_kep_boundary_no_runaway() -> None:
    """Recover Ktrans within 10% for the noiseless kep=10, ve=0.05 fixture."""
    aif, t = _parker_aif()
    model = ToftsModel()
    ktrans, ve = 0.5, 0.05  # kep = 10
    gt = model.evaluate([ktrans, ve], aif, t)
    recovered = model.fit(gt, aif, t).parameters["Ktrans"]
    assert recovered == pytest.approx(ktrans, rel=0.10)


def test_zero_ktrans_not_inflated() -> None:
    """The pure-vascular fixture returns Ktrans below 0.015."""
    aif, t = _parker_aif()
    model = ExtendedToftsModel()
    gt = model.evaluate([0.0, 0.2, 0.05], aif, t)
    recovered = model.fit(gt, aif, t).parameters["Ktrans"]
    assert recovered < 0.015, f"zero-Ktrans inflated to {recovered:.4f}"


def test_kep_reporting_preserves_finite_value() -> None:
    """Derived reporting preserves finite kep values beyond the QC range."""
    from pmriprep.interfaces.dce_compute import _compute_derived_params

    railed = {"Ktrans": 5.0, "ve": 0.1}  # raw kep = 50 >> cap
    _compute_derived_params(railed, "tofts")
    assert railed["kep"] == pytest.approx(50.0)

    ok = {"Ktrans": 0.3, "ve": 0.1}  # raw kep = 3, physiological
    _compute_derived_params(ok, "tofts")
    assert ok["kep"] == pytest.approx(3.0)


def test_models_report_high_finite_kep() -> None:
    """Both Tofts models report finite ratios without upper clipping."""
    assert ToftsModel().compute_derived_parameters([5.0, 0.1])["kep"] == pytest.approx(50)
    assert ExtendedToftsModel().compute_derived_parameters([5.0, 0.1, 0.05])["kep"] == pytest.approx(50)
