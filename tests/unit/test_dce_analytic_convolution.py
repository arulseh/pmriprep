"""Can-fail tests for the exact analytic convolution (Plan 1).

The production DCE forward uses exact piecewise-linear-AIF (*) exponential
convolution (the reference ``expconv``), replacing the left-Riemann
``np.convolve`` that under-read high-kep tumor Ktrans by up to -29%. These
validate recovery against a self-consistent fine-grid ground truth,
scalar<->batched parity, the rate->0 (trapezoid) limit, and the 2CXM
bi-exponential. Tolerances are pinned to the measured analytic recovery floor
(self-consistent dt/100 GT): -0.4% at kep=0.6 rising to +3% at kep=6.
"""
from __future__ import annotations

import numpy as np
import pytest

from pmriprep.models.dce._convolution import analytic_exp_convolution
from pmriprep.models.dce.batched_models import (
    batched_2cxm_model,
    batched_tofts_model,
)
from pmriprep.models.dce.extended_tofts import ExtendedToftsModel
from pmriprep.models.dce.patlak import PatlakModel
from pmriprep.models.dce.tofts import ToftsModel
from pmriprep.models.dce.two_compartment_exchange import (
    TwoCompartmentExchangeModel,
)

_DT = 0.0667
_N = 80


def _parker(t: np.ndarray, arrival: float = 0.5) -> np.ndarray:
    tt = np.maximum(t - arrival, 0.0)
    a = (0.809, 0.330)
    tc = (0.17046, 0.365)
    sc = (0.0563, 0.132)
    al, be, s, ta = 1.050, 0.1685, 38.078, 0.483
    c = np.zeros_like(tt)
    for a_n, t_n, s_n in zip(a, tc, sc):
        c += a_n / (s_n * np.sqrt(2 * np.pi)) * np.exp(-((tt - t_n) ** 2) / (2 * s_n**2))
    c += al * np.exp(-be * tt) / (1 + np.exp(-s * (tt - ta)))
    c[t <= arrival] = 0.0
    return c * (5.0 / c.max())


def _coarse() -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(_N) * _DT
    return _parker(t), t


def _fine_gt_tofts(ktrans: float, ve: float, oversample: int = 100) -> np.ndarray:
    """Self-consistent fine-grid GT: exact convolution of a dt/100
    piecewise-linear Parker AIF (converges to the continuous convolution),
    downsampled to coarse frames. Not the biased left-Riemann dt/20 reference."""
    kep = ktrans / ve
    ft = np.arange(_N * oversample) * (_DT / oversample)
    return ktrans * analytic_exp_convolution(_parker(ft), kep, _DT / oversample)[::oversample]


@pytest.mark.parametrize(
    "ve,tol",
    [(0.5, 0.03), (0.2, 0.03), (0.1, 0.03), (0.05, 0.05), (0.03, 0.08)],
)
def test_tofts_recovers_across_kep(ve: float, tol: float) -> None:
    """Ktrans recovered within the analytic floor across kep 0.6-10 (Standard
    Tofts, no vp alias). Left-Riemann under-read up to -29% in this regime."""
    aif, t = _coarse()
    ktrans = 0.3
    gt = _fine_gt_tofts(ktrans, ve)
    recovered = ToftsModel().fit(gt, aif, t).parameters["Ktrans"]
    assert recovered == pytest.approx(ktrans, rel=tol), (
        f"kep={ktrans / ve:.1f}: Ktrans {recovered:.4f} vs {ktrans}"
    )


def test_matches_independent_expconv_reference() -> None:
    """Independent oracle for the g0/g1 exponential math.

    The recovery/AUC tests build ground truth with analytic_exp_convolution
    itself, so a sign/coefficient error in g0/g1 would corrupt forward and GT
    identically and still pass. Pin the helper here against the canonical
    expconv arrangement (E0=1-e, E1=x-E0, da=dCp/x) — algebraically the same
    integral but different code — so that class of error cannot hide.
    """
    aif, _ = _coarse()

    def _expconv_ref(cp: np.ndarray, rate: float) -> np.ndarray:
        t_const = 1.0 / rate
        x = _DT / t_const
        e = np.exp(-x)
        e0 = 1.0 - e
        e1 = x - e0
        da = np.diff(cp) / x
        add = cp[:-1] * e0 + da * e1
        f = np.zeros(len(cp))
        for i in range(len(cp) - 1):
            f[i + 1] = e * f[i] + add[i]
        return f * t_const

    for rate in (0.6, 1.5, 3.0, 6.0, 10.0):
        assert np.allclose(
            analytic_exp_convolution(aif, rate, _DT), _expconv_ref(aif, rate), atol=1e-13
        )


def test_forward_auc_matches_fine_grid() -> None:
    """The coarse analytic forward AUC is within 1% of the fine-grid truth
    (localizes forward-curve error independent of the optimizer)."""
    aif, _ = _coarse()
    for ve in (0.5, 0.1, 0.05):
        kep = 0.3 / ve
        fwd = 0.3 * analytic_exp_convolution(aif, kep, _DT)
        gt = _fine_gt_tofts(0.3, ve)
        assert abs(fwd.sum() - gt.sum()) / gt.sum() < 0.01


def test_scalar_batched_parity() -> None:
    """Scalar evaluate == batched forward (single shared helper → exact)."""
    aif, t = _coarse()
    kt = np.array([0.05, 0.1, 0.3, 0.5])
    ve = np.array([0.4, 0.1, 0.05, 0.2])
    bat = batched_tofts_model(kt, ve, aif, t).tissue_curves
    mt = ToftsModel()
    for i in range(4):
        assert np.allclose(bat[i], mt.evaluate([kt[i], ve[i]], aif, t), atol=1e-10)

    fp = np.array([80.0, 50.0])
    ps = np.array([20.0, 5.0])
    ve2 = np.array([0.4, 0.2])
    vp = np.array([0.1, 0.05])
    b2 = batched_2cxm_model(fp, ps, ve2, vp, aif, t).tissue_curves
    m2 = TwoCompartmentExchangeModel()
    for i in range(2):
        assert np.allclose(
            b2[i], m2.evaluate([fp[i], ps[i], ve2[i], vp[i]], aif, t), atol=1e-10
        )


def test_rate_zero_trapezoid_limit() -> None:
    """As rate->0 the convolution -> the cumulative trapezoid of Cp; stays
    finite (guards the 2CXM slow-eigenvalue underflow), no NaN/0-collapse."""
    aif, _ = _coarse()
    trap = np.concatenate([[0.0], np.cumsum(0.5 * (aif[1:] + aif[:-1]) * _DT)])
    for rate in (0.0, 1e-11, 1e-6):
        conv = analytic_exp_convolution(aif, rate, _DT)
        assert np.all(np.isfinite(conv))
        assert np.max(np.abs(conv - trap)) < 1e-4  # ~ rate * second moment


def test_2cxm_biexp_is_sum_of_convolutions() -> None:
    """2CXM evaluate == a1*conv(K1) + a2*conv(K2) (linear bi-exponential IRF)."""
    aif, t = _coarse()
    m = TwoCompartmentExchangeModel()
    fp, ps, ve, vp = 80.0, 20.0, 0.4, 0.1
    k1, k2, a1, a2 = m._compute_eigenvalues_and_coefficients(fp, ps, ve, vp)
    expected = a1 * analytic_exp_convolution(aif, k1, _DT) + a2 * analytic_exp_convolution(
        aif, k2, _DT
    )
    assert np.allclose(m.evaluate([fp, ps, ve, vp], aif, t), expected, atol=1e-12)


def test_patlak_integral_is_kep_zero_limit() -> None:
    """Patlak's cumulative-trapezoid integral(Cp) is exactly the kep->0 limit of
    the analytic convolution — the two schemes are internally consistent."""
    aif, _ = _coarse()
    cum_trap = np.concatenate([[0.0], np.cumsum(0.5 * (aif[1:] + aif[:-1]) * _DT)])
    assert np.allclose(analytic_exp_convolution(aif, 0.0, _DT), cum_trap, atol=1e-12)


def test_extended_tofts_recovers_with_vp() -> None:
    """Extended Tofts recovers Ktrans at a moderate kep with a vascular term."""
    aif, t = _coarse()
    ktrans, ve, vp = 0.3, 0.2, 0.05
    kep = ktrans / ve
    ft = np.arange(_N * 100) * (_DT / 100)
    gt = vp * _parker(np.arange(_N) * _DT) + ktrans * analytic_exp_convolution(
        _parker(ft), kep, _DT / 100
    )[::100]
    recovered = ExtendedToftsModel().fit(gt, aif, t).parameters["Ktrans"]
    assert recovered == pytest.approx(ktrans, rel=0.04)


def _fine_gt_model(model, params: list[float], oversample: int = 100) -> np.ndarray:
    """Neutral fine-grid GT: evaluate the model's exact forward on a dt/100
    piecewise-linear Parker AIF (converges to the continuous forward) and
    downsample to coarse frames. Fitting at coarse dt then tests the model's
    own quadrature (∫Cp for Patlak, bi-exp expconv for 2CXM) against a truth it
    did not generate at the fit resolution."""
    ft = np.arange(_N * oversample) * (_DT / oversample)
    return np.asarray(model.evaluate(list(params), _parker(ft), ft))[::oversample]


@pytest.mark.parametrize(
    "ktrans,vp,tol",
    [(0.05, 0.02, 0.05), (0.10, 0.05, 0.05), (0.02, 0.01, 0.08)],
)
def test_patlak_recovers_from_fine_grid(ktrans: float, vp: float, tol: float) -> None:
    """Patlak Ktrans/vp recovered from a fine-grid (dt/100) Patlak forward.

    Validates the trapezoidal ∫₀ᵗCp regressor against a continuous truth the
    coarse fit did not generate; prior Patlak coverage was structural /
    inverse-crime only. Recovery is limited by the coarse-vs-fine ∫Cp floor.
    """
    aif, t = _coarse()
    model = PatlakModel()
    gt = _fine_gt_model(model, [ktrans, vp])
    fit = model.fit(gt, aif, t).parameters
    assert fit["Ktrans"] == pytest.approx(ktrans, rel=tol), (
        f"Patlak Ktrans {fit['Ktrans']:.4f} vs {ktrans}"
    )
    assert fit["vp"] == pytest.approx(vp, abs=0.01), (
        f"Patlak vp {fit['vp']:.4f} vs {vp}"
    )


@pytest.mark.parametrize(
    "fp,ps,ve,vp",
    [(50.0, 4.0, 0.15, 0.03), (60.0, 12.0, 0.30, 0.04)],
)
def test_2cxm_recovers_from_fine_grid(
    fp: float, ps: float, ve: float, vp: float
) -> None:
    """2CXM parameter recovery from a fine-grid (dt/100) bi-exponential forward.

    The only end-to-end neutral 2CXM accuracy test: the IRF coefficients are
    validated against an ODE elsewhere, and the shared expconv is validated for
    Tofts, but the full bi-exp-IRF ⊗ AIF → fit composition was inverse-crime
    only. Asserts on the robust quantities (derived Ktrans, ve, vp) at zero
    noise; PS carries a wider IQR (F3) and is not pinned tightly here.
    """
    aif, t = _coarse()
    model = TwoCompartmentExchangeModel()
    gt = _fine_gt_model(model, [fp, ps, ve, vp])
    fit = model.fit(gt, aif, t).parameters

    ktrans_true = (ps / (fp + ps)) * fp / 100.0
    assert fit["Ktrans"] == pytest.approx(ktrans_true, rel=0.12), (
        f"2CXM derived Ktrans {fit['Ktrans']:.4f} vs {ktrans_true:.4f}"
    )
    assert fit["ve"] == pytest.approx(ve, abs=0.04), f"2CXM ve {fit['ve']:.3f} vs {ve}"
    assert fit["vp"] == pytest.approx(vp, abs=0.02), f"2CXM vp {fit['vp']:.3f} vs {vp}"
