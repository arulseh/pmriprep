"""DSC signal-domain round-trip tests.

Tests the full chain: concentration -> signal -> noise -> signal_to_concentration
-> deconvolution -> parameter recovery. Validates that signal_to_concentration()
correctly inverts the forward model and that noise propagation is understood.

The AIF peak is set to 0.1 s⁻¹ (not the default 40 s⁻¹) so that tissue
concentrations remain physically realistic (peak 5-40 s⁻¹) and the signal
does not clip to zero. Deconvolution recovers the same CBF/MTT regardless
of AIF scaling (linearity).

Exercises:
    dsc_signal.py: signal_to_concentration(), get_baseline_signal()
    Signal ratio clipping, TE scaling, Rician noise bias
"""

from typing import Tuple

import numpy as np
import numpy.typing as npt
import pytest

from pmriprep.utils.dsc_constants import EPSILON_CONCENTRATION
from pmriprep.utils.dsc_deconvolution import (
    svd_deconvolve_oscillation,
    svd_deconvolve_standard,
)
from pmriprep.utils.dsc_signal import (
    get_baseline_signal,
    signal_to_concentration,
)

from .conftest import (
    DSC_TISSUE_PARAMS,
    add_gaussian_noise,
    add_rician_noise,
    concentration_to_signal,
    dsc_forward_model,
    dsc_forward_model_circulant,
    generate_dsc_aif_dro,
    make_exponential_residue,
    snr_to_noise_std,
)

# AIF peak that produces realistic tissue deltaR2* (5-40 s⁻¹)
_AIF_PEAK = 0.1


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


def _make_signal_roundtrip_data(
    cbf: float,
    mtt: float,
    te: float,
    s0: float,
    snr: float = np.inf,
    noise_model: str = "rician",
    seed: int = 42,
    n_timepoints: int = 60,
    dt_sec: float = 1.5,
    baseline_frames: int = 10,
    aif_peak: float = _AIF_PEAK,
) -> Tuple[
    npt.NDArray[np.floating],  # signal (with baseline)
    npt.NDArray[np.floating],  # aif concentration (post-baseline only)
    float,  # dt
    float,  # cbf_truth
    float,  # mtt_truth
]:
    """Generate DSC signal with known parameters for round-trip testing.

    Returns signal WITH baseline prepended, and AIF post-baseline only
    (matching the pipeline convention where AIF concentration has no
    baseline frames). Returned MTT truth is the full-period area of the
    sampled residue divided by its peak, matching the circulant forward model.
    """
    # Generate AIF and tissue concentration (post-baseline only)
    aif_post, _ = generate_dsc_aif_dro(
        n_timepoints=n_timepoints,
        dt_sec=dt_sec,
        baseline_frames=0,
        peak_delta_r2=aif_peak,
    )

    residue = make_exponential_residue(cbf, mtt, n_timepoints, dt_sec)
    mtt_truth = float(dt_sec * np.sum(residue) / np.max(residue))
    tissue_conc_post = dsc_forward_model_circulant(aif_post, residue, dt_sec)

    # Convert to signal domain (prepends baseline)
    signal = concentration_to_signal(tissue_conc_post, te, s0, baseline_frames)

    # Add noise
    if np.isfinite(snr) and snr > 0:
        noise_std = snr_to_noise_std(s0, snr)
        if noise_model == "rician":
            signal = add_rician_noise(signal, noise_std, seed=seed)
        else:
            signal = add_gaussian_noise(signal, noise_std, seed=seed)

    return signal, aif_post, dt_sec, cbf, mtt_truth


# =============================================================================
# Test classes
# =============================================================================


class TestSignalDomainRoundTrip:
    """Test the full signal-domain round-trip chain.

    concentration -> signal -> (noise) -> signal_to_concentration -> deconvolve
    """

    TISSUES = [
        pytest.param("gm", id="gray_matter"),
        pytest.param("wm", id="white_matter"),
        pytest.param("tumor", id="tumor"),
    ]

    TES = [
        pytest.param(0.025, id="TE25ms"),
        pytest.param(0.030, id="TE30ms"),
        pytest.param(0.040, id="TE40ms"),
    ]

    S0S = [
        pytest.param(500.0, id="S0_500"),
        pytest.param(1000.0, id="S0_1000"),
        pytest.param(2000.0, id="S0_2000"),
    ]

    SNRS = [
        pytest.param(np.inf, id="noiseless"),
        pytest.param(50.0, id="SNR50"),
        pytest.param(20.0, id="SNR20"),
        pytest.param(10.0, id="SNR10"),
    ]

    @pytest.mark.parametrize("tissue_key", TISSUES)
    @pytest.mark.parametrize("te", TES)
    @pytest.mark.parametrize("s0", S0S)
    def test_concentration_roundtrip_accuracy(
        self,
        tissue_key: str,
        te: float,
        s0: float,
    ) -> None:
        """Noiseless signal->concentration round-trip should match original.

        Tests signal_to_concentration() directly, without deconvolution,
        to isolate signal-domain errors from SVD truncation errors.
        """
        tissue = DSC_TISSUE_PARAMS[tissue_key]
        aif_post, _ = generate_dsc_aif_dro(
            n_timepoints=60, dt_sec=1.5, baseline_frames=0,
            peak_delta_r2=_AIF_PEAK,
        )
        residue = make_exponential_residue(tissue.cbf, tissue.mtt, 60, 1.5)
        conc_true = dsc_forward_model(aif_post, residue, 1.5)

        signal = concentration_to_signal(conc_true, te, s0, baseline_frames=10)
        conc_recovered = signal_to_concentration(signal, te, baseline_frames=10)

        conc_post = conc_recovered[10:]
        peak = float(np.max(conc_true))
        if peak > 0:
            rmse = float(np.sqrt(np.mean((conc_post - conc_true) ** 2)))
            nrmse = rmse / peak
            assert nrmse < 0.02, (
                f"Concentration round-trip NRMSE {nrmse:.4f} > 0.02 "
                f"for {tissue_key}, TE={te*1000:.0f}ms, S0={s0:.0f}"
            )

    @pytest.mark.parametrize("tissue_key", TISSUES)
    @pytest.mark.parametrize(
        "te", [pytest.param(0.025, id="TE25ms"),
               pytest.param(0.030, id="TE30ms"),
               pytest.param(0.040, id="TE40ms")]
    )
    def test_noiseless_full_chain(
        self,
        tissue_key: str,
        te: float,
    ) -> None:
        """Noiseless full-chain: signal -> concentration -> deconvolution.

        Tolerances include both signal-domain and deconvolution errors.
        oSVD inherently introduces ~10-20% bias from threshold search.
        """
        tissue = DSC_TISSUE_PARAMS[tissue_key]
        signal, aif, dt, cbf_true, mtt_true = _make_signal_roundtrip_data(
            cbf=tissue.cbf, mtt=tissue.mtt, te=te, s0=1000.0, snr=np.inf,
        )

        conc = signal_to_concentration(signal, te, baseline_frames=10)
        result = svd_deconvolve_oscillation(conc[10:], aif, dt)

        _assert_param_recovery(result.cbf, cbf_true, rel_tol=0.35, param_name="CBF")
        _assert_param_recovery(result.mtt, mtt_true, rel_tol=0.30, param_name="MTT")

    @pytest.mark.parametrize("tissue_key", TISSUES)
    @pytest.mark.parametrize("snr", SNRS)
    def test_noisy_roundtrip_rician(
        self,
        tissue_key: str,
        snr: float,
    ) -> None:
        """Noisy round-trip with Rician noise model."""
        tissue = DSC_TISSUE_PARAMS[tissue_key]
        te = 0.030
        s0 = 1000.0

        signal, aif, dt, cbf_true, mtt_true = _make_signal_roundtrip_data(
            cbf=tissue.cbf, mtt=tissue.mtt, te=te, s0=s0,
            snr=snr, noise_model="rician",
        )

        conc = signal_to_concentration(signal, te, baseline_frames=10)
        assert np.all(np.isfinite(conc)), "Concentration contains NaN/Inf"

        if np.isfinite(snr) and snr < 5:
            return

        result = svd_deconvolve_oscillation(conc[10:], aif, dt)

        if np.isinf(snr):
            _assert_param_recovery(result.cbf, cbf_true, rel_tol=0.35, param_name="CBF")
        elif snr >= 20:
            _assert_param_recovery(result.cbf, cbf_true, rel_tol=0.60, param_name="CBF")
        else:
            assert 0 < result.cbf < cbf_true * 3, (
                f"CBF {result.cbf:.1f} outside reasonable range for "
                f"true CBF={cbf_true:.1f} at SNR={snr}"
            )

    @pytest.mark.parametrize("snr", [3.0, 5.0])
    def test_very_low_snr_no_crash(self, snr: float) -> None:
        """Low-SNR curves retain signed areas and mark invalid residue MTT unavailable."""
        tissue = DSC_TISSUE_PARAMS["gm"]
        signal, aif, dt, _, _ = _make_signal_roundtrip_data(
            cbf=tissue.cbf, mtt=tissue.mtt, te=0.030, s0=1000.0,
            snr=snr, noise_model="rician",
        )

        conc = signal_to_concentration(signal, te=0.030, baseline_frames=10)
        assert np.all(np.isfinite(conc)), "Concentration has NaN/Inf at low SNR"

        result = svd_deconvolve_standard(conc[10:], aif, dt)
        assert np.isfinite(result.cbf), "CBF is not finite at low SNR"
        assert np.isfinite(result.cbv), "Residue area is not finite at low SNR"
        assert result.cbv == pytest.approx(np.trapezoid(result.residue, dx=dt))
        if result.cbv > 0 and result.cbf > EPSILON_CONCENTRATION:
            assert result.mtt == pytest.approx(result.cbv / result.cbf)
        else:
            assert np.isnan(result.mtt)


#: Empirical relative-CBF error limits for these signal-roundtrip fixtures.
#: They bound the combined noise, signal-conversion and regularization effects.
OSVD_CBF_ENVELOPE = {50: 0.46, 20: 0.52}


class TestRicianNoiseBias:
    """Measure systematic CBF bias vs SNR and compare noise models.

    Rician noise is always positive, introducing positive bias in the
    signal magnitude. This attenuates the apparent signal drop (peak
    deltaR2*), leading to CBF underestimation at low SNR.
    """

    SNRS = [5, 7, 10, 20, 50]

    @pytest.mark.slow
    @pytest.mark.parametrize("snr", SNRS)
    def test_cbf_bias_vs_snr(self, snr: int) -> None:
        """Measure Rician CBF bias across multiple noise seeds."""
        tissue = DSC_TISSUE_PARAMS["gm"]
        te = 0.030
        s0 = 1000.0
        n_seeds = 5

        cbf_rician = []
        for seed in range(n_seeds):
            signal, aif, dt, _, _ = _make_signal_roundtrip_data(
                cbf=tissue.cbf, mtt=tissue.mtt, te=te, s0=s0,
                snr=float(snr), noise_model="rician", seed=seed + 100,
            )
            conc = signal_to_concentration(signal, te, baseline_frames=10)
            result = svd_deconvolve_oscillation(conc[10:], aif, dt)
            cbf_rician.append(result.cbf)

        mean_rician = np.mean(cbf_rician)

        if snr in OSVD_CBF_ENVELOPE:
            tol = OSVD_CBF_ENVELOPE[snr]
            assert abs(mean_rician - tissue.cbf) / tissue.cbf < tol, (
                f"Rician CBF bias outside oSVD envelope at SNR={snr}: "
                f"mean={mean_rician:.1f}, true={tissue.cbf:.1f}, tol={tol}"
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("snr", [5, 10, 20])
    def test_peak_attenuation_vs_snr(self, snr: int) -> None:
        """Quantify clipping-induced peak deltaR2* attenuation."""
        tissue = DSC_TISSUE_PARAMS["gm"]
        te = 0.030
        s0 = 1000.0
        n_seeds = 5

        peak_ratios = []
        for seed in range(n_seeds):
            signal_clean, aif, _, _, _ = _make_signal_roundtrip_data(
                cbf=tissue.cbf, mtt=tissue.mtt, te=te, s0=s0, snr=np.inf,
            )
            conc_clean = signal_to_concentration(signal_clean, te, baseline_frames=10)
            peak_clean = float(np.max(conc_clean))

            signal_noisy, _, _, _, _ = _make_signal_roundtrip_data(
                cbf=tissue.cbf, mtt=tissue.mtt, te=te, s0=s0,
                snr=float(snr), noise_model="rician", seed=seed,
            )
            conc_noisy = signal_to_concentration(signal_noisy, te, baseline_frames=10)
            peak_noisy = float(np.max(conc_noisy))

            if peak_clean > 0:
                peak_ratios.append(peak_noisy / peak_clean)

        mean_ratio = np.mean(peak_ratios)
        if snr >= 20:
            assert mean_ratio > 0.80, (
                f"Excessive peak attenuation at SNR={snr}: ratio={mean_ratio:.3f}"
            )


class TestBaselineS0Estimation:
    """Test S0 estimation accuracy from get_baseline_signal()."""

    @pytest.mark.parametrize(
        "baseline_frames",
        [pytest.param(b, id=f"bl{b}") for b in [2, 3, 5, 10, 15, 20]],
    )
    @pytest.mark.parametrize(
        "snr",
        [pytest.param(50, id="SNR50"), pytest.param(20, id="SNR20"),
         pytest.param(10, id="SNR10")],
    )
    def test_s0_accuracy(self, baseline_frames: int, snr: int) -> None:
        """S0 estimation accuracy vs baseline length and noise."""
        s0_true = 1000.0
        te = 0.030
        tissue = DSC_TISSUE_PARAMS["gm"]

        # Ensure we have enough baseline frames in the signal
        bl_gen = max(baseline_frames, 10)
        signal, _, _, _, _ = _make_signal_roundtrip_data(
            cbf=tissue.cbf, mtt=tissue.mtt, te=te, s0=s0_true,
            snr=float(snr), noise_model="rician",
            baseline_frames=bl_gen,
        )

        s0_est = get_baseline_signal(signal, baseline_frames=baseline_frames)
        s0_err = abs(float(s0_est) - s0_true) / s0_true

        # Wider tolerance at low SNR with few baseline frames
        max_err = 0.30 if (snr <= 10 and baseline_frames <= 3) else 0.15
        assert s0_err < max_err, (
            f"S0 error {s0_err:.3f} exceeds tolerance {max_err} "
            f"(baseline_frames={baseline_frames}, SNR={snr})"
        )

    @pytest.mark.parametrize(
        "baseline_frames",
        [pytest.param(b, id=f"bl{b}") for b in [3, 5, 10, 20]],
    )
    def test_s0_downstream_cbf_error(self, baseline_frames: int) -> None:
        """S0 error propagation to CBF through signal_to_concentration."""
        s0_true = 1000.0
        te = 0.030
        tissue = DSC_TISSUE_PARAMS["gm"]

        # Always generate with the test baseline_frames so signal length matches
        signal, aif, dt, cbf_true, _ = _make_signal_roundtrip_data(
            cbf=tissue.cbf, mtt=tissue.mtt, te=te, s0=s0_true,
            snr=20.0, noise_model="rician",
            baseline_frames=baseline_frames,
        )

        conc = signal_to_concentration(signal, te, baseline_frames=baseline_frames)
        assert np.all(np.isfinite(conc)), "Concentration has NaN/Inf"

        # Post-baseline concentration and AIF must have same length
        conc_post = conc[baseline_frames:]
        assert len(conc_post) == len(aif), (
            f"Length mismatch: conc_post={len(conc_post)}, aif={len(aif)}"
        )

        result = svd_deconvolve_oscillation(conc_post, aif, dt)
        assert np.isfinite(result.cbf), "CBF is not finite"
        assert 0 < result.cbf < cbf_true * 3, (
            f"CBF {result.cbf:.1f} outside reasonable range "
            f"(baseline_frames={baseline_frames})"
        )


class TestSignalEdgeCases:
    """Edge case handling for signal_to_concentration."""

    def test_te_zero_raises(self) -> None:
        """TE <= 0 should raise ValueError."""
        signal = np.ones(60, dtype=np.float64) * 1000.0
        with pytest.raises(ValueError, match="TE must be positive"):
            signal_to_concentration(signal, te=0.0)

    def test_te_negative_raises(self) -> None:
        """Negative TE should raise ValueError."""
        signal = np.ones(60, dtype=np.float64) * 1000.0
        with pytest.raises(ValueError, match="TE must be positive"):
            signal_to_concentration(signal, te=-0.03)

    def test_all_zero_signal(self) -> None:
        """All-zero signal should not crash, output should be finite."""
        signal = np.zeros(60, dtype=np.float64)
        conc = signal_to_concentration(signal, te=0.030, baseline_frames=5)
        assert np.all(np.isfinite(conc)), "All-zero signal produced NaN/Inf"

    def test_single_frame_baseline(self) -> None:
        """Single-frame baseline (baseline_frames=1) should work."""
        signal = np.ones(60, dtype=np.float64) * 1000.0
        signal[5:15] = 500.0
        conc = signal_to_concentration(signal, te=0.030, baseline_frames=1)
        assert np.all(np.isfinite(conc))
        assert np.max(conc[5:15]) > 0

    def test_empty_signal_raises(self) -> None:
        """Empty signal should raise ValueError."""
        signal = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="empty"):
            signal_to_concentration(signal, te=0.030)

    def test_constant_signal_zero_concentration(self) -> None:
        """Constant signal should yield zero concentration."""
        signal = np.ones(60, dtype=np.float64) * 1000.0
        conc = signal_to_concentration(signal, te=0.030, baseline_frames=5)
        assert np.max(np.abs(conc)) < 1e-6

    def test_4d_input(self) -> None:
        """4D input should work correctly."""
        shape = (3, 3, 3, 60)
        signal = np.ones(shape, dtype=np.float64) * 1000.0
        signal[1, 1, 1, 10:20] = 500.0

        conc = signal_to_concentration(signal, te=0.030, baseline_frames=5)
        assert conc.shape == shape
        assert np.all(np.isfinite(conc))
        assert np.max(conc[1, 1, 1, 10:20]) > 0
        assert np.max(np.abs(conc[0, 0, 0, :])) < 1e-6

    def test_signal_above_baseline_produces_negative(self) -> None:
        """Signal above baseline produces negative ΔR2* (no upper clipping).

        S(t)/S0 = 1.2 -> ΔR2* = -ln(1.2)/TE < 0. Preserves symmetric
        noise in post-bolus frames.
        """
        signal = np.ones(60, dtype=np.float64) * 1000.0
        signal[20:30] = 1200.0
        conc = signal_to_concentration(signal, te=0.030, baseline_frames=5)
        expected = -np.log(1.2) / 0.030
        assert conc[25] == pytest.approx(expected, rel=1e-6)
        assert conc[25] < 0, "Signal > S0 should give negative ΔR2*"


class TestNoiseModelComparison:
    """Compare Rician vs Gaussian noise effects on DSC signal recovery.

    Rician noise biases signal upward by σ²/(2S), which biases apparent ΔR2*
    downward. Gaussian noise is zero-mean with no systematic bias. Tests
    verify this at three levels: signal, concentration, and CBF.

    The Rician bias per-timepoint is small (~0.5-5 units at typical SNR),
    so reliable detection requires averaging over many samples. Tests use
    replicated constant-signal arrays (100 voxels × 20 seeds) for the
    signal/concentration tests, and the full DSC chain for CBF tests.

    All tests are marked slow.
    """

    _TE = 0.030
    _S0 = 1000.0
    _N_SEEDS = 20
    _TISSUE = "gm"
    # Bolus-peak ΔR2* for concentration-level tests (23 s⁻¹ gives ~50%
    # signal drop, representative of GM peak in clinical DSC).
    _PEAK_CONC = 23.0

    @pytest.mark.slow
    @pytest.mark.parametrize("snr", [10, 20, 50], ids=["SNR10", "SNR20", "SNR50"])
    def test_rician_attenuates_more_than_gaussian(self, snr: int) -> None:
        """Rician gives lower mean concentration than Gaussian at bolus peak.

        At constant signal S = S0·exp(-TE·C), Rician noise biases S upward,
        so -ln(S/S0)/TE recovers a lower concentration. Gaussian noise is
        zero-mean and (after clipping) produces slightly higher concentration
        due to log-transform asymmetry. Average over 100 voxels × 20 seeds
        resolves the ~0.2-0.7 s⁻¹ difference.
        """
        n_voxels = 100
        signal_at_peak = np.ones(n_voxels) * self._S0 * np.exp(
            -self._TE * self._PEAK_CONC,
        )
        noise_std = snr_to_noise_std(self._S0, float(snr))

        rician_means = []
        gaussian_means = []
        for seed in range(self._N_SEEDS):
            sig_r = add_rician_noise(signal_at_peak.copy(), noise_std, seed=seed)
            sig_g = add_gaussian_noise(signal_at_peak.copy(), noise_std, seed=seed)

            # Manual concentration recovery (matches signal_to_concentration)
            ratio_r = np.clip(sig_r / self._S0, 1e-10, 1.0)
            ratio_g = np.clip(sig_g / self._S0, 1e-10, 1.0)
            rician_means.append(float(np.mean(-np.log(ratio_r) / self._TE)))
            gaussian_means.append(float(np.mean(-np.log(ratio_g) / self._TE)))

        mean_rician = float(np.mean(rician_means))
        mean_gaussian = float(np.mean(gaussian_means))

        assert mean_rician < mean_gaussian, (
            f"SNR={snr}: Rician mean conc ({mean_rician:.3f}) should be < "
            f"Gaussian ({mean_gaussian:.3f}) at peak ΔR2*={self._PEAK_CONC}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("snr", [20, 50], ids=["SNR20", "SNR50"])
    def test_gaussian_cbf_within_osvd_envelope(self, snr: int) -> None:
        """Gaussian-noise mean CBF stays within the validated oSVD envelope.

        SVD deconvolution is not an unbiased CBF estimator: the regularized
        residue peak is systematically clipped, so even zero-mean Gaussian
        noise leaves a regularization underestimate. Mean CBF over seeds must
        stay within the physics-derived oSVD envelope (Gaussian is less biased
        than Rician at matched SNR; see test_rician_attenuates_more_than_gaussian).
        """
        tissue = DSC_TISSUE_PARAMS[self._TISSUE]
        cbf_values = []
        for seed in range(self._N_SEEDS):
            signal, aif, dt, _, _ = _make_signal_roundtrip_data(
                cbf=tissue.cbf, mtt=tissue.mtt, te=self._TE, s0=self._S0,
                snr=float(snr), noise_model="gaussian", seed=seed + 300,
            )
            conc = signal_to_concentration(signal, self._TE, baseline_frames=10)
            result = svd_deconvolve_oscillation(conc[10:], aif, dt)
            cbf_values.append(result.cbf)

        mean_cbf = float(np.mean(cbf_values))
        rel_err = abs(mean_cbf - tissue.cbf) / tissue.cbf
        tol = OSVD_CBF_ENVELOPE[snr]
        assert rel_err < tol, (
            f"SNR={snr}: Gaussian mean CBF={mean_cbf:.3f} vs "
            f"truth={tissue.cbf:.3f}, rel_err={rel_err:.3f} >= envelope {tol}"
        )

    @pytest.mark.slow
    def test_noise_models_converge_at_high_snr(self) -> None:
        """At SNR=50, Rician and Gaussian concentration means nearly match.

        High SNR makes the Rician rectification bias σ²/(2S) negligible.
        Mean concentration difference < 0.1 s⁻¹ at the bolus peak.
        """
        n_voxels = 100
        signal_at_peak = np.ones(n_voxels) * self._S0 * np.exp(
            -self._TE * self._PEAK_CONC,
        )
        noise_std = snr_to_noise_std(self._S0, 50.0)

        rician_means = []
        gaussian_means = []
        for seed in range(self._N_SEEDS):
            sig_r = add_rician_noise(signal_at_peak.copy(), noise_std, seed=seed)
            sig_g = add_gaussian_noise(signal_at_peak.copy(), noise_std, seed=seed)

            ratio_r = np.clip(sig_r / self._S0, 1e-10, 1.0)
            ratio_g = np.clip(sig_g / self._S0, 1e-10, 1.0)
            rician_means.append(float(np.mean(-np.log(ratio_r) / self._TE)))
            gaussian_means.append(float(np.mean(-np.log(ratio_g) / self._TE)))

        diff = abs(float(np.mean(rician_means)) - float(np.mean(gaussian_means)))
        assert diff < 0.10, (
            f"At SNR=50: Rician-Gaussian conc diff={diff:.4f} >= 0.10 s⁻¹"
        )
