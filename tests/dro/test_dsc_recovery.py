"""DSC parameter recovery tests using Digital Reference Objects.

Tests that deconvolution methods can recover known ground-truth CBF, CBV,
and MTT from synthetic concentration-time curves at various noise levels.
"""

from itertools import product
from typing import Any, Tuple

import numpy as np
import numpy.typing as npt
import pytest
from scipy.special import gammainc

from tests.dsc_accuracy import P2_NO_CORNER, P2_UPPER_CORNER

from pmriprep.utils.dsc_constants import DEFAULT_LAMBDA_RANGE
from pmriprep.utils.dsc_deconvolution import (
    _build_modified_first_derivative_matrix,
    _tikhonov_solve,
    svd_deconvolve_circulant,
    svd_deconvolve_oscillation,
    svd_deconvolve_standard,
    tikhonov_deconvolve,
)

from .conftest import (
    DSC_DRO_VTF_MODERATE,
    add_gaussian_noise,
    apply_recirculation,
    apply_vtf_dispersion,
    delay_aif,
    dsc_forward_model_circulant,
    dsc_forward_model_volterra,
    generate_dsc_aif_dro,
    make_boxcar_residue,
    make_cbf_mtt_grid,
    make_exponential_residue,
    make_gamma_residue,
    make_triangle_residue,
    score_percent_repeatability,
)


# =============================================================================
# Residue builders
# =============================================================================


class TestResidueBuilders:
    """Box-car and triangle residue families (each with integral CBF*MTT)."""

    def test_boxcar_shape(self) -> None:
        r = make_boxcar_residue(2.0, 4.0, 20, 1.0)
        assert r[0] == 2.0
        assert np.all(r[:5] == 2.0)   # t = 0..4 <= MTT
        assert np.all(r[5:] == 0.0)   # t > MTT

    def test_triangle_shape(self) -> None:
        r = make_triangle_residue(2.0, 4.0, 20, 1.0)
        assert r[0] == 2.0
        assert r[4] == pytest.approx(1.0)   # R(MTT) = CBF/2
        assert r[8] == pytest.approx(0.0)   # R(2*MTT) = 0
        assert np.all(r[9:] == 0.0)

    def test_central_volume_theorem(self) -> None:
        # integral R(t) == CBF*MTT (triangle exact; box-car has a sharp-edge
        # discretization artifact, hence the looser tolerance).
        cbf, mtt, dt, n = 2.0, 4.0, 0.25, 160
        tri = np.trapezoid(make_triangle_residue(cbf, mtt, n, dt), dx=dt)
        box = np.trapezoid(make_boxcar_residue(cbf, mtt, n, dt), dx=dt)
        assert tri == pytest.approx(cbf * mtt, rel=1e-3)
        assert box == pytest.approx(cbf * mtt, rel=0.1)

    def test_gamma_residue_central_volume_and_monotone(self) -> None:
        # gamma (CTH) survival residue: R(0)=CBF, monotone decreasing,
        # integral == CBF*MTT for every shape alpha.
        cbf, mtt, dt, n = 2.0, 4.0, 0.05, 4000
        for alpha in [1.0, 2.0, 5.0, 20.0]:
            r = make_gamma_residue(cbf, mtt, alpha, n, dt)
            assert r[0] == pytest.approx(cbf)
            assert np.all(np.diff(r) <= 1e-9)  # monotone non-increasing
            assert np.trapezoid(r, dx=dt) == pytest.approx(cbf * mtt, rel=1e-3)

    def test_gamma_alpha1_equals_exponential(self) -> None:
        # alpha=1 gamma survival reduces exactly to CBF*exp(-t/MTT).
        cbf, mtt, dt, n = 2.0, 4.0, 0.1, 400
        gamma1 = make_gamma_residue(cbf, mtt, 1.0, n, dt)
        expo = make_exponential_residue(cbf, mtt, n, dt)
        assert np.allclose(gamma1, expo, atol=1e-12)


# =============================================================================
# AIF effects (recirculation, dispersion)
# =============================================================================


class TestAifEffects:
    """Recirculation (delayed exponential) and dispersion (log-normal VTF)."""

    def test_recirculation_adds_unit_area_second_pass(self) -> None:
        aif, _ = generate_dsc_aif_dro(n_timepoints=200, dt_sec=1.0)
        out = apply_recirculation(aif, 1.0, delay_sec=8.0, tau_sec=30.0)
        recirc = out - aif
        # unit-area kernel -> recirculation area equals the first-pass area
        assert np.trapezoid(recirc, dx=1.0) == pytest.approx(
            np.trapezoid(aif, dx=1.0), rel=0.05
        )
        # total mass strictly increases
        assert np.trapezoid(out, dx=1.0) > np.trapezoid(aif, dx=1.0)

    def test_recirculation_lands_on_the_tail(self) -> None:
        aif, _ = generate_dsc_aif_dro(n_timepoints=200, dt_sec=1.0)
        recirc = apply_recirculation(aif, 1.0, delay_sec=8.0, tau_sec=30.0) - aif
        # the returning bolus is later than, and lower than, the first pass
        assert int(np.argmax(recirc)) > int(np.argmax(aif))
        assert recirc.max() < aif.max()

    def test_dispersion_conserves_mass(self) -> None:
        aif, _ = generate_dsc_aif_dro(n_timepoints=200, dt_sec=1.0)
        for sigma, mu in [(1.0, -1.0), (0.75, -0.15), (0.78, 0.59)]:
            out = apply_vtf_dispersion(aif, 1.0, sigma, mu)
            assert np.trapezoid(out, dx=1.0) == pytest.approx(
                np.trapezoid(aif, dx=1.0), rel=0.05
            )

    def test_dispersion_shifts_centroid_by_mean_transit_time(self) -> None:
        # Convolution adds first moments: dispersed centroid - original centroid
        # == E(VTF) = exp(mu + sigma^2/2). Verifies the implemented log-normal
        # against the analytical mean for mild/moderate/severe severities.
        aif, _ = generate_dsc_aif_dro(n_timepoints=800, dt_sec=0.25)
        dt = 0.25
        t = np.arange(len(aif), dtype=np.float64) * dt
        c0 = float(np.sum(t * aif) / np.sum(aif))
        cases = [((1.0, -1.0), 0.61), ((0.75, -0.15), 1.14), ((0.78, 0.59), 2.45)]
        shifts = []
        for (sigma, mu), e_vtf in cases:
            out = apply_vtf_dispersion(aif, dt, sigma, mu)
            shift = float(np.sum(t * out) / np.sum(out)) - c0
            shifts.append(shift)
            assert shift == pytest.approx(e_vtf, abs=0.1)
        # worse dispersion -> later centroid (mild < moderate < severe)
        assert shifts[0] < shifts[1] < shifts[2]

    def test_dispersion_lowers_peak(self) -> None:
        aif, _ = generate_dsc_aif_dro(n_timepoints=200, dt_sec=1.0)
        peak0 = aif.max()
        peaks = [
            apply_vtf_dispersion(aif, 1.0, sigma, mu).max()
            for sigma, mu in [(1.0, -1.0), (0.75, -0.15), (0.78, 0.59)]
        ]
        # every severity spreads the bolus, lowering the peak vs undispersed.
        # On this broad AIF mild/moderate are near-tied (sub-second E(VTF)); the
        # severity ordering is checked properly via the centroid test above.
        assert all(p < peak0 for p in peaks)
        assert peaks[2] < peaks[0] and peaks[2] < peaks[1]


# =============================================================================
# DRO scaffolding (factorial grid, delay, repeatability scoring)
# =============================================================================


class TestDroScaffolding:
    """CBF x MTT grid, AIF delay, and %RC repeatability helpers."""

    def test_grid_keeps_only_physiological_cbv(self) -> None:
        grid = make_cbf_mtt_grid([10, 30, 50, 70], [2, 4, 8, 16, 24])
        assert grid  # non-empty
        for cbf, mtt, cbv in grid:
            assert cbv == pytest.approx(cbf * mtt / 60.0)
            assert 1.0 <= cbv <= 14.0

    def test_grid_drops_supraphysiological_combo(self) -> None:
        # CBF=70, MTT=24 -> CBV=28% (out of range) must be excluded;
        # CBF=60, MTT=4 -> CBV=4% (GM-like) must be kept.
        grid = make_cbf_mtt_grid([60, 70], [4, 24])
        assert (60.0, 4.0, pytest.approx(4.0)) in grid
        assert not any(cbf == 70 and mtt == 24 for cbf, mtt, _ in grid)

    def test_delay_shifts_aif_later(self) -> None:
        aif, time_sec = generate_dsc_aif_dro(n_timepoints=120, dt_sec=1.0)
        delayed = delay_aif(aif, time_sec, 5.0)
        assert int(np.argmax(delayed)) - int(np.argmax(aif)) == 5
        # zero delay is a no-op
        assert np.allclose(delay_aif(aif, time_sec, 0.0), aif)

    def test_percent_repeatability_zero_for_identical(self) -> None:
        v = np.array([10.0, 20.0, 30.0])
        assert score_percent_repeatability(v, v) == pytest.approx(0.0)

    def test_percent_repeatability_scales_with_spread(self) -> None:
        v1 = np.array([10.0, 20.0, 30.0])
        small = score_percent_repeatability(v1, v1 * 1.02)
        large = score_percent_repeatability(v1, v1 * 1.10)
        assert 0.0 < small < large

    @staticmethod
    def _fwhm_seconds(aif: np.ndarray, dt: float) -> float:
        idx = np.where(aif >= aif.max() / 2.0)[0]
        return float((idx[-1] - idx[0]) * dt)

    def test_aif_default_is_broad(self) -> None:
        # default alpha=2.5/beta=3.0 is the broad regression-guard operating
        # point: FWHM ~10 s, analytic peak ~7.5 s after arrival (idx 10).
        aif, _ = generate_dsc_aif_dro(n_timepoints=120, dt_sec=1.0)
        assert self._fwhm_seconds(aif, 1.0) == pytest.approx(10.0, abs=1.0)
        assert int(np.argmax(aif)) - 10 == pytest.approx(8, abs=1)

    def test_aif_alpha_beta_narrows_to_arterial(self) -> None:
        broad, _ = generate_dsc_aif_dro(n_timepoints=120, dt_sec=1.0)
        arterial, _ = generate_dsc_aif_dro(
            n_timepoints=120, dt_sec=1.0, alpha=3.0, beta=1.5
        )
        assert self._fwhm_seconds(arterial, 1.0) < self._fwhm_seconds(broad, 1.0)
        assert self._fwhm_seconds(arterial, 1.0) == pytest.approx(5.0, abs=1.0)

    def test_aif_oversample_preserves_auc_and_is_noop_at_one(self) -> None:
        direct, _ = generate_dsc_aif_dro(
            n_timepoints=120, dt_sec=1.5, alpha=3.0, beta=1.5
        )
        fine, _ = generate_dsc_aif_dro(
            n_timepoints=120, dt_sec=1.5, alpha=3.0, beta=1.5, oversample=15
        )
        # fine-grid build conserves AUC (it only re-bins the within-TR shape)
        assert np.trapezoid(fine, dx=1.5) == pytest.approx(
            np.trapezoid(direct, dx=1.5), rel=0.05
        )
        # oversample=1 is the plain coarse evaluation
        os1, _ = generate_dsc_aif_dro(
            n_timepoints=120, dt_sec=1.5, alpha=3.0, beta=1.5, oversample=1
        )
        assert np.array_equal(os1, direct)


# =============================================================================
# Helpers
# =============================================================================


def _assert_param_recovery(
    recovered: float,
    truth: float,
    rel_tol: float,
    abs_tol: float = 0.5,
    param_name: str = "",
) -> None:
    """Assert DSC parameter recovery within tolerance."""
    if abs(truth) < 1e-3:
        assert abs(recovered - truth) < abs_tol, (
            f"{param_name}: |{recovered:.4f} - {truth:.4f}| = "
            f"{abs(recovered - truth):.4f} >= {abs_tol}"
        )
    else:
        rel_err = abs(recovered - truth) / abs(truth)
        assert rel_err < rel_tol, (
            f"{param_name}: |{recovered:.4f} - {truth:.4f}| / |{truth:.4f}| = "
            f"{rel_err:.4f} >= {rel_tol}"
        )


def _make_dsc_test_data(
    cbf: float,
    mtt: float,
    noise_frac: float = 0.0,
    n_timepoints: int = 60,
    dt_sec: float = 1.5,
    baseline_frames: int = 10,
    seed: int = 42,
    forward_model: Any = None,
) -> Tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    float,
    npt.NDArray[np.floating],
]:
    """Generate DSC test data with known parameters."""
    if forward_model is None:
        # sSVD/Tikhonov deconvolve with the Volterra (product-integration) operator,
        # so the matched GT default is Volterra; circulant borrowers (cSVD/oSVD)
        # pass dsc_forward_model_circulant explicitly.
        forward_model = dsc_forward_model_volterra
    aif, time_sec = generate_dsc_aif_dro(
        n_timepoints=n_timepoints,
        dt_sec=dt_sec,
        baseline_frames=baseline_frames,
        peak_delta_r2=40.0,
    )

    residue = make_exponential_residue(cbf, mtt, n_timepoints, dt_sec)
    tissue = forward_model(aif, residue, dt_sec)

    if noise_frac > 0:
        peak = float(np.max(np.abs(tissue)))
        if peak > 0:
            tissue = add_gaussian_noise(tissue, noise_frac * peak, seed=seed)

    return tissue, aif, dt_sec, time_sec


# =============================================================================
# DSC parameter combinations
# =============================================================================

DSC_PARAMS = [
    pytest.param(0.10, 4.0, id="normal_gm"),
    pytest.param(0.04, 6.0, id="normal_wm"),
    pytest.param(0.16, 8.0, id="hyperperfused_tumor"),
    pytest.param(0.015, 3.0, id="low_flow"),
    pytest.param(0.13, 12.0, id="long_mtt"),
]


# =============================================================================
# sSVD Recovery
# =============================================================================

# (noise_fraction, svd_threshold, cbf_tol, mtt_tol)
# SVD threshold must scale with noise to balance bias vs variance.
# At zero noise, threshold=0.001 gives ~2% CBF error (MTT floor ~9%).
# At higher noise, higher thresholds control noise amplification at
# the cost of increased regularization bias (underestimated CBF).
#
# These tolerance bands are Monte-Carlo-derived REGRESSION GUARDS (mean +
# 3*sigma over N=100 realizations) for the Volterra (product-integration) operator,
# pinning the numerical behavior of this deconvolution on the broad (default
# alpha=2.5/beta=3.0, FWHM ~10 s) DRO AIF. Zero-noise rows are the per-tissue
# max (deterministic, n=1/tissue).
SSVD_NOISE_LEVELS = [
    pytest.param(0.0, 0.001, 0.04, 0.11, id="zero"),
    pytest.param(0.02, 0.02, 0.33, 1.15, id="low"),
    pytest.param(0.05, 0.05, 0.40, 1.45, id="medium"),
    pytest.param(0.10, 0.10, 0.50, 1.60, id="high"),
]


class TestSSVDRecovery:
    """Test standard SVD deconvolution parameter recovery."""

    @pytest.mark.parametrize("noise_frac,threshold,cbf_tol,mtt_tol", SSVD_NOISE_LEVELS)
    @pytest.mark.parametrize("cbf,mtt", DSC_PARAMS)
    def test_ssvd(
        self,
        cbf: float,
        mtt: float,
        noise_frac: float,
        threshold: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test sSVD recovery of CBF and MTT."""
        # SVD regularization systematically inflates MTT for low-CBF residues;
        # the loose mtt_tol absorbs it (MC N=100: low_flow mtt mean+3sigma =
        # 1.22-1.66), so the bound is asserted rather than xfailed.
        tissue, aif, dt, time_sec = _make_dsc_test_data(cbf, mtt, noise_frac)

        result = svd_deconvolve_standard(tissue, aif, dt, threshold=threshold)

        _assert_param_recovery(result.cbf, cbf, cbf_tol, param_name="CBF")
        _assert_param_recovery(result.mtt, mtt, mtt_tol, param_name="MTT")


# =============================================================================
# cSVD Recovery — delay detection only
# =============================================================================


class TestCSVDRecovery:
    """Test circulant SVD deconvolution delay detection.

    cSVD uses a block-circulant matrix. Forward model uses
    the matched circulant matrix to isolate algorithm behavior from
    discretization mismatch.
    """

    @pytest.mark.parametrize(
        "delay_frames",
        [
            pytest.param(2, id="delay_2"),
            pytest.param(5, id="delay_5"),
        ],
    )
    def test_csvd_delay_preserves_shape(self, delay_frames: int) -> None:
        """Test that cSVD recovers a reasonable residue with delayed data."""
        cbf, mtt = 0.10, 4.0
        tissue, aif, dt, time_sec = _make_dsc_test_data(
            cbf, mtt, noise_frac=0.0,
            forward_model=dsc_forward_model_circulant,
        )

        # Shift tissue curve to simulate delay
        tissue_shifted = np.zeros_like(tissue)
        tissue_shifted[delay_frames:] = tissue[:-delay_frames]

        result = svd_deconvolve_circulant(tissue_shifted, aif, dt)

        assert result.cbf > 0, f"CBF should be positive, got {result.cbf}"
        assert np.max(result.residue) > 0
        assert abs(result.delay_frames - delay_frames) <= 3, (
            f"Expected delay ~{delay_frames}, got {result.delay_frames}"
        )


# =============================================================================
# oSVD Recovery
# =============================================================================


class TestOSVDRecovery:
    """Test oscillation-index SVD deconvolution parameter recovery.

    oSVD uses a block-circulant matrix. The forward model uses
    the matched circulant matrix to isolate algorithm behavior from
    discretization mismatch.
    """

    @pytest.mark.parametrize("cbf,mtt", DSC_PARAMS)
    def test_osvd_zero_noise(self, cbf: float, mtt: float) -> None:
        """oSVD recovers the sampled circulant residue's CBF and area/peak."""
        tissue, aif, dt, time_sec = _make_dsc_test_data(
            cbf, mtt, 0.0, forward_model=dsc_forward_model_circulant,
        )
        result = svd_deconvolve_oscillation(
            tissue, aif, dt, threshold_min=1e-6
        )
        assert result.cbf > 0, f"CBF should be positive, got {result.cbf}"
        assert result.mtt > 0, f"MTT should be positive, got {result.mtt}"
        _assert_param_recovery(result.cbf, cbf, 0.10, param_name="CBF")
        sampled_residue = make_exponential_residue(cbf, mtt, len(aif), dt)
        periodic_mtt = float(dt * np.sum(sampled_residue) / np.max(sampled_residue))
        _assert_param_recovery(result.mtt, periodic_mtt, 0.10, param_name="Residue MTT")

    # MC-derived regression guards (mean + 3*sigma, N=100) on the broad-AIF
    # operating point; the wide MTT bands
    # characterize recovery on this synthetic input and noise construction.
    OSVD_NOISY_LEVELS = [
        pytest.param(0.02, 0.005, 0.50, 1.45, id="low"),
        pytest.param(0.05, 0.01, 0.65, 2.25, id="medium"),
        pytest.param(0.10, 0.01, 0.72, 3.20, id="high"),
    ]

    @pytest.mark.parametrize("noise_frac,thresh_min,cbf_tol,mtt_tol", OSVD_NOISY_LEVELS)
    @pytest.mark.parametrize("cbf,mtt", DSC_PARAMS)
    def test_osvd_noisy(
        self,
        cbf: float,
        mtt: float,
        noise_frac: float,
        thresh_min: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test oSVD recovery of CBF and MTT with noise."""
        # SVD MTT inflation at low CBF is absorbed by the loose mtt_tol (MC
        # mean+3sigma), so the bound is asserted rather than xfailed.
        tissue, aif, dt, time_sec = _make_dsc_test_data(
            cbf, mtt, noise_frac,
            forward_model=dsc_forward_model_circulant,
        )
        result = svd_deconvolve_oscillation(
            tissue, aif, dt, threshold_min=thresh_min
        )
        _assert_param_recovery(result.cbf, cbf, cbf_tol, param_name="CBF")
        _assert_param_recovery(result.mtt, mtt, mtt_tol, param_name="MTT")

    @pytest.mark.parametrize(
        "delay_frames",
        [
            pytest.param(2, id="delay_2"),
            pytest.param(5, id="delay_5"),
        ],
    )
    def test_osvd_delay_preserves_shape(self, delay_frames: int) -> None:
        """Test that oSVD recovers a reasonable residue with delayed data."""
        cbf, mtt = 0.10, 4.0
        tissue, aif, dt, time_sec = _make_dsc_test_data(
            cbf, mtt, noise_frac=0.0,
            forward_model=dsc_forward_model_circulant,
        )

        tissue_shifted = np.zeros_like(tissue)
        tissue_shifted[delay_frames:] = tissue[:-delay_frames]

        result = svd_deconvolve_oscillation(
            tissue_shifted, aif, dt, threshold_min=0.001
        )
        assert result.cbf > 0, f"CBF should be positive, got {result.cbf}"
        assert np.max(result.residue) > 0
        assert abs(result.delay_frames - delay_frames) <= 2, (
            f"Expected delay ~{delay_frames}, got {result.delay_frames}"
        )


# =============================================================================
# Tikhonov Recovery — zero noise only
# =============================================================================


class TestTikhonovRecovery:
    """Tikhonov recovery on the DISPERSED DSC residue regime (production).

    Production deconvolves tissue against an auto-detected arterial AIF, but
    the tissue is fed by that artery through downstream transport, so the
    effective residue the deconvolver must recover is dispersed — it rises
    from zero to an interior peak, where the
    production modified penalty operator (identity block on the leading
    samples + first-difference) belongs.

    The DRO models this honestly: the forward tissue is generated from a
    VTF-dispersed AIF (``apply_vtf_dispersion`` at the pre-committed moderate
    severity ``DSC_DRO_VTF_MODERATE``), and the deconvolution uses the
    un-dispersed AIF — the production mismatch. Ground truth is the effective
    residue R_eff = VTF ⊗ R_true: CBF = max(R_eff) (below the nominal CBF,
    since dispersion lowers the peak) and AUC-MTT = ∫R_eff / max(R_eff).
    AUC-CBV is dispersion-invariant, so AUC-MTT carries only the peak (CBF)
    error and is the reported MTT here; the internal residue-MTT is too
    dispersion/noise sensitive to bound usefully.

    The fixed regression bounds cover CBF and AUC-MTT recovery. That the
    modified operator beats pure L₁ on this regime is asserted
    directly in ``test_modified_penalty_beats_l1_on_dispersed``.
    """

    # (cbf, alpha, beta) — gamma survival-function residue; cbf is
    # the un-dispersed peak. The dispersed ground truth is recomputed per case.
    GAMMA_PARAMS = [
        pytest.param(0.10, 3.0, 1.5, id="normal_gm"),
        pytest.param(0.04, 4.0, 1.5, id="normal_wm"),
        pytest.param(0.16, 2.0, 3.0, id="tumor"),
        pytest.param(0.13, 3.0, 4.0, id="long_mtt"),
    ]

    # Bounds per (noise_frac, cbf_tol, aucmtt_tol), across all four tissues.
    NOISE_LEVELS = [
        pytest.param(0.0,  0.16, 0.13, id="zero"),
        pytest.param(0.02, 0.42, 0.45, id="low"),
        pytest.param(0.05, 0.56, 0.85, id="medium"),
    ]

    @pytest.mark.parametrize(
        "cbf,alpha,beta,noise_frac,cbf_tol,aucmtt_tol",
        [
            pytest.param(
                *tissue.values, *noise.values,
                id=f"{noise.id}-{tissue.id}",
                marks=P2_UPPER_CORNER
                if (noise.id, tissue.id) == ("zero", "normal_gm") else (),
            )
            for tissue, noise in product(GAMMA_PARAMS, NOISE_LEVELS)
        ],
    )
    def test_tikhonov(
        self,
        cbf: float,
        alpha: float,
        beta: float,
        noise_frac: float,
        cbf_tol: float,
        aucmtt_tol: float,
    ) -> None:
        """L₁,mod recovers CBF and AUC-MTT on the dispersed residue regime."""
        aif, time_sec = generate_dsc_aif_dro(
            n_timepoints=60, dt_sec=1.5, baseline_frames=10,
            peak_delta_r2=40.0,
        )
        dt = 1.5
        residue = cbf * (1.0 - gammainc(alpha, time_sec / beta))
        # Dispersed forward: tissue from the VTF-dispersed AIF; deconvolve with
        # the un-dispersed AIF (the production mismatch that makes the effective
        # residue rise from zero).
        aif_disp = apply_vtf_dispersion(aif, dt, *DSC_DRO_VTF_MODERATE)
        tissue = dsc_forward_model_volterra(aif_disp, residue, dt)
        if noise_frac > 0:
            peak = float(np.max(np.abs(tissue)))
            tissue = add_gaussian_noise(tissue, noise_frac * peak, seed=42)

        # Ground truth = effective residue R_eff = VTF ⊗ R_true.
        r_eff = apply_vtf_dispersion(residue, dt, *DSC_DRO_VTF_MODERATE)
        cbf_gt = float(np.max(r_eff))
        aucmtt_gt = float(np.trapezoid(r_eff, dx=dt)) / cbf_gt

        result = tikhonov_deconvolve(tissue, aif, dt)
        cbv_auc = float(np.trapezoid(tissue, dx=dt)) / float(
            np.trapezoid(aif, dx=dt)
        )
        auc_mtt = cbv_auc / result.cbf if result.cbf > 0 else 0.0

        _assert_param_recovery(result.cbf, cbf_gt, cbf_tol, param_name="CBF")
        _assert_param_recovery(
            auc_mtt, aucmtt_gt, aucmtt_tol, param_name="AUC-MTT"
        )

    def test_modified_penalty_beats_l1_on_dispersed(self) -> None:
        """Compare modified and pure first-difference penalties on dispersed fixtures.

        The modified operator penalizes leading-sample magnitudes and later
        differences. Count CBF-error wins over the four sampled tissue cases.
        """
        aif, time_sec = generate_dsc_aif_dro(
            n_timepoints=60, dt_sec=1.5, baseline_frames=10, peak_delta_r2=40.0,
        )
        dt = 1.5
        aif_disp = apply_vtf_dispersion(aif, dt, *DSC_DRO_VTF_MODERATE)
        # Both operators run through the same private solve on the same
        # aif/dt (the core builds the identical Volterra A), so the only
        # difference is the penalty. L_mod is the production modified
        # first-difference; l1 is the pure first-difference (non-dispersed).
        n = len(aif)
        L_mod = _build_modified_first_derivative_matrix(n)
        l1 = np.zeros((n - 1, n), dtype=np.float64)
        di = np.arange(n - 1)
        l1[di, di] = -1.0
        l1[di, di + 1] = 1.0
        lo, hi = DEFAULT_LAMBDA_RANGE.min, DEFAULT_LAMBDA_RANGE.max
        losses = 0
        for cbf, alpha, beta in [
            (0.10, 3.0, 1.5), (0.04, 4.0, 1.5),
            (0.16, 2.0, 3.0), (0.13, 3.0, 4.0),
        ]:
            residue = cbf * (1.0 - gammainc(alpha, time_sec / beta))
            tissue = dsc_forward_model_volterra(aif_disp, residue, dt)
            tissue = add_gaussian_noise(
                tissue, 0.02 * float(np.max(np.abs(tissue))), seed=7
            )
            cbf_gt = float(np.max(apply_vtf_dispersion(
                residue, dt, *DSC_DRO_VTF_MODERATE
            )))
            err_modified = abs(
                _tikhonov_solve(tissue, aif, dt, lo, hi, L_mod).cbf - cbf_gt
            )
            err_l1 = abs(
                _tikhonov_solve(tissue, aif, dt, lo, hi, l1).cbf - cbf_gt
            )
            if err_modified >= err_l1:
                losses += 1
        assert losses == 0, (
            f"modified penalty lost to pure L₁ on {losses}/4 dispersed "
            f"tissues at 2% noise"
        )


# =============================================================================
# Gamma-residue test data
# =============================================================================


def _make_gamma_test_data(
    cbf: float,
    alpha: float,
    beta: float,
    noise_frac: float = 0.0,
    n_timepoints: int = 60,
    dt_sec: float = 1.5,
    baseline_frames: int = 10,
    seed: int = 42,
    forward_model: Any = None,
) -> Tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    float,
]:
    """Generate test data with gamma-variate residue.

    Creates the gamma-variate residue R(t) = CBF * [1 - gammainc(alpha, t/beta)],
    then convolves
    with the DRO AIF via the specified forward model.

    Returns
    -------
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], float]
        Tuple of (tissue_concentration, aif, time_sec, true_mtt).
    """
    if forward_model is None:
        # Default to the piecewise-linear product-integration operator.
        forward_model = dsc_forward_model_volterra
    aif, time_sec = generate_dsc_aif_dro(
        n_timepoints=n_timepoints,
        dt_sec=dt_sec,
        baseline_frames=baseline_frames,
    )

    residue = cbf * (1.0 - gammainc(alpha, time_sec / beta))
    tissue = forward_model(aif, residue, dt_sec)

    if noise_frac > 0:
        peak = float(np.max(np.abs(tissue)))
        if peak > 0:
            tissue = add_gaussian_noise(tissue, noise_frac * peak, seed=seed)

    true_mtt = alpha * beta
    return tissue, aif, time_sec, true_mtt


# =============================================================================
# SVD Recovery on Gamma-Variate Residues (cross-residue validation)
# =============================================================================

# SVD methods are model-free and should generalize across residue shapes.
# Gamma-variate residues have a plateau before decay (R'(0)=0 for alpha>1),
# which SVD regularization can truncate, causing higher CBF overestimation
# than on exponential residues.

SVD_GAMMA_PARAMS = [
    pytest.param(0.10, 3.0, 1.5, id="normal_gm"),    # MTT=4.5s
    pytest.param(0.04, 4.0, 1.5, id="normal_wm"),     # MTT=6.0s
    pytest.param(0.16, 2.0, 3.0, id="tumor"),         # MTT=6.0s
    pytest.param(0.13, 3.0, 4.0, id="long_mtt"),      # MTT=12.0s
]

# Tolerances wider than exponential: SVD regularization truncates the
# gamma-variate plateau, causing systematic CBF overestimation.
# Zero-noise: deterministic — sSVD long_mtt has 19% CBF, 15% MTT from
# threshold truncating the gamma plateau. MTT 15.1% sets the floor.
# 5% noise: MC N=20, mean+3σ. Binding: oSVD long_mtt CBF 56%,
# oSVD normal_gm MTT 85%. Large MTT errors are inherent to SVD on
# non-exponential residues.
SVD_GAMMA_NOISE_LEVELS = [
    pytest.param(0.0, 0.20, 0.16, id="zero"),     # CBF 19%, MTT 15.1% (deterministic)
    pytest.param(0.05, 0.57, 0.86, id="medium"),   # MC: CBF m+3σ=56%, MTT m+3σ=85%
]


class TestSVDGammaRecovery:
    """Test SVD deconvolution on gamma-variate residues.

    Validates that model-free SVD methods generalize beyond the exponential
    residues they are traditionally benchmarked on. This matches fig2
    panel b data.
    """

    @pytest.mark.parametrize("noise_frac,cbf_tol,mtt_tol", SVD_GAMMA_NOISE_LEVELS)
    @pytest.mark.parametrize("cbf,alpha,beta", SVD_GAMMA_PARAMS)
    def test_ssvd_gamma(
        self,
        cbf: float,
        alpha: float,
        beta: float,
        noise_frac: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test sSVD recovery of CBF and MTT from gamma-variate residue."""
        tissue, aif, time_sec, true_mtt = _make_gamma_test_data(
            cbf, alpha, beta, noise_frac,
            forward_model=dsc_forward_model_volterra,
        )

        dt = float(time_sec[1] - time_sec[0])
        result = svd_deconvolve_standard(tissue, aif, dt, threshold=0.05)

        _assert_param_recovery(result.cbf, cbf, cbf_tol, param_name="CBF")
        _assert_param_recovery(result.mtt, true_mtt, mtt_tol, param_name="MTT")

    OSVD_GAMMA_NOISE_LEVELS = [
        pytest.param(0.0, 0.10, 0.30, id="zero"),
        pytest.param(0.05, 0.57, 0.86, id="medium"),
    ]

    @pytest.mark.parametrize("noise_frac,cbf_tol,mtt_tol", OSVD_GAMMA_NOISE_LEVELS)
    @pytest.mark.parametrize("cbf,alpha,beta", SVD_GAMMA_PARAMS)
    def test_osvd_gamma(
        self,
        cbf: float,
        alpha: float,
        beta: float,
        noise_frac: float,
        cbf_tol: float,
        mtt_tol: float,
    ) -> None:
        """Test oSVD recovery of CBF and MTT from gamma-variate residue."""
        tissue, aif, time_sec, true_mtt = _make_gamma_test_data(
            cbf, alpha, beta, noise_frac,
            forward_model=dsc_forward_model_circulant,
        )

        dt = float(time_sec[1] - time_sec[0])
        result = svd_deconvolve_oscillation(tissue, aif, dt, threshold_min=0.01)

        _assert_param_recovery(result.cbf, cbf, cbf_tol, param_name="CBF")
        _assert_param_recovery(result.mtt, true_mtt, mtt_tol, param_name="MTT")


# =============================================================================
# Edge Cases
# =============================================================================


class TestDSCEdgeCases:
    """Test edge cases for DSC deconvolution methods."""

    def test_delta_residue(self) -> None:
        """A delta-function residue should recover CBF correctly."""
        n = 60
        dt = 1.5
        aif, time_sec = generate_dsc_aif_dro(n, dt)

        cbf = 0.08
        residue = np.zeros(n, dtype=np.float64)
        residue[0] = cbf

        tissue = dsc_forward_model_volterra(aif, residue, dt)
        result = svd_deconvolve_standard(tissue, aif, dt, threshold=0.001)

        _assert_param_recovery(result.cbf, cbf, 0.10, param_name="CBF")

    def test_zero_tissue(self) -> None:
        """Zero tissue concentration should give near-zero CBF."""
        n = 60
        dt = 1.5
        aif, time_sec = generate_dsc_aif_dro(n, dt)

        tissue = np.zeros(n, dtype=np.float64)
        result = svd_deconvolve_standard(tissue, aif, dt, threshold=0.001)

        assert result.cbf < 1e-4, f"CBF for zero tissue: {result.cbf:.6f}"

    def test_short_mtt(self) -> None:
        """Short MTT (near-delta) should be recoverable."""
        tissue, aif, dt, time_sec = _make_dsc_test_data(
            cbf=0.10, mtt=1.0, noise_frac=0.0
        )
        result = svd_deconvolve_standard(tissue, aif, dt, threshold=0.001)
        _assert_param_recovery(result.cbf, 0.10, 0.10, param_name="CBF")

    def test_long_mtt(self) -> None:
        """Long MTT should be recoverable."""
        tissue, aif, dt, time_sec = _make_dsc_test_data(
            cbf=0.065, mtt=15.0, noise_frac=0.0
        )
        result = svd_deconvolve_standard(tissue, aif, dt, threshold=0.001)
        _assert_param_recovery(result.cbf, 0.065, 0.10, param_name="CBF")
        _assert_param_recovery(result.mtt, 15.0, 0.15, param_name="MTT")

    @P2_NO_CORNER
    def test_ssvd_tikhonov_agreement(self) -> None:
        """Evaluate the 25% CBF agreement target for two causal estimators."""
        tissue, aif, dt, time_sec = _make_dsc_test_data(
            cbf=0.10, mtt=4.0, noise_frac=0.0
        )

        r_ssvd = svd_deconvolve_standard(tissue, aif, dt, threshold=0.001)
        r_tikh = tikhonov_deconvolve(tissue, aif, dt)

        cbf_values = [r_ssvd.cbf, r_tikh.cbf]
        mean_cbf = float(np.mean(cbf_values))

        for cbf_val in cbf_values:
            rel_diff = abs(cbf_val - mean_cbf) / mean_cbf
            assert rel_diff < 0.25, (
                f"Method disagreement: CBFs = {cbf_values}, "
                f"mean = {mean_cbf:.2f}"
            )
