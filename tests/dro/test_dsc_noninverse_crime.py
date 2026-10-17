"""Deconvolution regression checks with a frame-reconstructed AIF.

The tissue is integrated on a fine grid using a linearly interpolated sampled
AIF and a continuous exponential residue. This construction shares the AIF
interpolation assumption with some fitted operators. Fully continuous
reference-generation checks are in ``test_dsc_reference.py``.
"""
import numpy as np
import pytest
from pmriprep.interfaces.dsc_compute import ComputeDSCMaps

from tests.dsc_accuracy import OSVD_RESIDUE_MTT, P2_NO_CORNER

from pmriprep.utils.dsc_deconvolution import (
    svd_deconvolve_oscillation,
    svd_deconvolve_standard,
    tikhonov_deconvolve,
)


def _gamma_aif(t, t0=10.0, alpha=3.0, beta=1.5, peak=4.0):
    a = np.zeros_like(t)
    m = t > t0
    a[m] = ((t[m] - t0) ** alpha) * np.exp(-(t[m] - t0) / beta)
    return a / a.max() * peak if a.max() > 0 else a


def _independent_tissue(aif, cbf, mtt, dt, oversample=20):
    """C(t) = CBF * int_0^t AIF(tau) exp(-(t-tau)/MTT) dtau on a fine grid.

    The AIF is interpolated from frames; only the residue is evaluated directly
    on the fine grid. This shares the piecewise-linear AIF assumption with
    fitted operators that use the same interpolation.
    """
    n = len(aif)
    fine_dt = dt / oversample
    n_fine = n * oversample
    t_fine = np.arange(n_fine) * fine_dt
    t_frame = np.arange(n) * dt
    aif_fine = np.interp(t_fine, t_frame, aif)
    residue_fine = np.exp(-t_fine / mtt)  # R(0) = 1
    conv = np.convolve(aif_fine, residue_fine)[:n_fine] * fine_dt
    return cbf * conv[::oversample]


TISSUES = [("GM", 60.0, 4.0), ("WM", 25.0, 4.8), ("Tumor", 100.0, 6.0)]


@pytest.mark.parametrize("name,cbf_true,mtt_true", TISSUES)
@pytest.mark.parametrize("metric", ["cbf", pytest.param("mtt", marks=OSVD_RESIDUE_MTT)])
def test_osvd_recovers_independent_tissue(
    name: str, cbf_true: float, mtt_true: float, metric: str,
) -> None:
    """oSVD CBF and helper residue-MTT stay within this fixture's regression bounds."""
    dt = 1.5
    n = 60
    t = np.arange(n) * dt
    aif = _gamma_aif(t)
    tissue = _independent_tissue(aif, cbf_true, mtt_true, dt)

    res = svd_deconvolve_oscillation(tissue, aif, dt=dt)

    if metric == "cbf":
        cbf_err = (res.cbf - cbf_true) / cbf_true
        assert -0.25 < cbf_err < 0.10, f"{name} oSVD CBF error {cbf_err:.1%}"
    else:
        mtt_err = (res.mtt - mtt_true) / mtt_true
        assert -0.20 < mtt_err < 0.10, f"{name} oSVD residue-MTT error {mtt_err:.1%}"


@pytest.mark.parametrize("name,cbf_true,mtt_true", TISSUES)
def test_ssvd_recovers_independent_tissue(name, cbf_true, mtt_true):
    """sSVD CBF and helper residue-MTT stay within the sampled-AIF regression bounds."""
    dt, n = 1.5, 60
    t = np.arange(n) * dt
    aif = _gamma_aif(t)
    tissue = _independent_tissue(aif, cbf_true, mtt_true, dt)

    res = svd_deconvolve_standard(tissue, aif, dt=dt)

    cbf_err = (res.cbf - cbf_true) / cbf_true
    assert -0.30 < cbf_err < 0.10, f"{name} sSVD CBF error {cbf_err:.1%}"
    mtt_err = (res.mtt - mtt_true) / mtt_true
    assert -0.15 < mtt_err < 0.40, f"{name} sSVD residue-MTT error {mtt_err:.1%}"


@P2_NO_CORNER
@pytest.mark.parametrize("name,cbf_true,mtt_true", TISSUES)
def test_tikhonov_recovers_independent_tissue(name, cbf_true, mtt_true):
    """Tikhonov CBF stays within the sampled-AIF, noise-free regression bounds."""
    dt, n = 1.5, 60
    t = np.arange(n) * dt
    aif = _gamma_aif(t)
    tissue = _independent_tissue(aif, cbf_true, mtt_true, dt)

    res = tikhonov_deconvolve(tissue, aif, dt=dt)

    cbf_err = (res.cbf - cbf_true) / cbf_true
    assert -0.15 < cbf_err < 0.10, f"{name} Tikhonov CBF error {cbf_err:.1%}"


@pytest.mark.parametrize("name,cbf_true,mtt_true", TISSUES)
def test_auc_cbv_recovers_independent_tissue(name, cbf_true, mtt_true, tmp_path):
    """Check the convolution-area identity within finite-sampling error.

    Full continuous integrals obey ∫tissue/∫AIF = CBF·MTT. These sampled
    finite-window integrals approximate it over the full sampled curves.
    """
    dt, n = 1.5, 60
    t = np.arange(n) * dt
    aif = _gamma_aif(t)
    tissue = _independent_tissue(aif, cbf_true, mtt_true, dt)

    cbv_true = cbf_true * mtt_true
    aif_file = tmp_path / "aif.txt"
    np.savetxt(aif_file, aif)
    curves = tissue.reshape(1, 1, 1, -1)
    maps = ComputeDSCMaps(measured_aif_file=str(aif_file))._compute_auc_maps(
        curves, curves, n_baseline=6, detected_baseline=6,
        detection_info={"arrival_index": 6}, dt=dt,
    )
    cbv_est = float(maps.cbv.item())
    rel = (cbv_est - cbv_true) / cbv_true
    assert abs(rel) < 0.02, (
        f"{name} AUC-CBV {cbv_est:.1f} vs true {cbv_true:.1f} ({rel:+.1%})"
    )
