"""K2 leakage correction round-trip tests.

Tests K2 estimation and correction in dsc_preload.py against known ground
truth K2 values. Uses a multi-voxel DRO phantom where the forward leakage
model uses C̄(t) (mean brain reference), matching what the estimator recovers.

Exercises:
    dsc_preload.py: estimate_k2_per_voxel(), apply_k2_correction()
"""

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from pmriprep.utils.dsc_preload import (
    K2CorrectionResult,
    _compute_k2_qc_metrics,
    _compute_bsw_reference,
    apply_k2_correction,
    estimate_k2_per_voxel,
)

from .conftest import (
    fine_grid_forward,
    generate_dsc_aif_dro,
    generate_k2_dro_phantom,
    make_exponential_residue,
)


# =============================================================================
# Test classes
# =============================================================================


class TestK2EstimationRecovery:
    """Test K2 estimation accuracy against known ground truth."""

    def test_reference_selection_excludes_late_signal_enhancement(self) -> None:
        """The conventional BSW screen excludes the negative Delta-R2* tail."""
        n_timepoints = 24
        baseline = np.array([-0.1, 0.1, -0.1, 0.1, 0.0])
        reference_curve = np.zeros(n_timepoints, dtype=np.float64)
        reference_curve[: baseline.size] = baseline
        reference_curve[6:10] = [1.0, 3.0, 2.0, 0.5]
        curves = np.repeat(reference_curve[None, :], 300, axis=0)
        curves[-2, -10:] = 1.0
        curves[-1, -10:] = -1.0

        concentration = curves.reshape(15, 20, 1, n_timepoints)
        signal = np.exp(-0.03 * concentration)
        observed = _compute_bsw_reference(
            concentration,
            signal,
            np.ones((15, 20, 1), dtype=bool),
            baseline_frames=baseline.size,
        )

        assert observed is not None
        expected = (298.0 * reference_curve + curves[-2]) / 299.0
        np.testing.assert_allclose(observed, expected)

    def test_reference_membership_is_decided_in_raw_signal_domain(self) -> None:
        """Raw signal, rather than Delta-R2*, determines reference support."""
        n_timepoints = 24
        reference_curve = np.zeros(n_timepoints, dtype=np.float64)
        reference_curve[6:10] = [1.0, 3.0, 2.0, 0.5]
        curves = np.repeat(reference_curve[None, :], 300, axis=0)
        curves[-2, -10:] = -2.0
        curves[-1, -10:] = 2.0

        signal = np.full_like(curves, 1000.0)
        signal[:, :5] += np.array([-2.0, 2.0, -2.0, 2.0, 0.0])
        signal[-2, -10:] = 1005.0

        observed = _compute_bsw_reference(
            curves.reshape(15, 20, 1, n_timepoints),
            signal.reshape(15, 20, 1, n_timepoints),
            np.ones((15, 20, 1), dtype=bool),
            baseline_frames=5,
        )

        assert observed is not None
        expected = (298.0 * reference_curve + curves[-1]) / 299.0
        np.testing.assert_allclose(observed, expected)

    def test_reference_excludes_zero_signal_voxels(self) -> None:
        """Physically invalid zero-signal voxels cannot dilute the reference."""
        n_timepoints = 24
        reference_curve = np.zeros(n_timepoints, dtype=np.float64)
        reference_curve[6:10] = [1.0, 3.0, 2.0, 0.5]
        curves = np.repeat(reference_curve[None, :], 300, axis=0)
        curves[-1] = 0.0
        signal = 1000.0 * np.exp(-0.03 * curves)
        signal[-1] = 0.0

        observed = _compute_bsw_reference(
            curves.reshape(15, 20, 1, n_timepoints),
            signal.reshape(15, 20, 1, n_timepoints),
            np.ones((15, 20, 1), dtype=bool),
            baseline_frames=5,
        )

        assert observed is not None
        np.testing.assert_allclose(observed, reference_curve)

    def test_signed_k2_is_not_rectified(self) -> None:
        """The fitted full-curve coefficient retains either valid sign."""
        tr = 1.0
        reference = np.zeros(30, dtype=np.float64)
        reference[5:12] = [0.5, 1.5, 3.0, 2.0, 1.0, 0.5, 0.2]
        integral = cumulative_trapezoid(reference, dx=tr, initial=0.0)
        curves = np.repeat(reference[None, :], 20, axis=0)
        curves[0] = 1.2 * reference + 0.1 * integral
        curves[1] = 1.2 * reference - 0.1 * integral

        observed = estimate_k2_per_voxel(
            curves.reshape(4, 5, 1, 30),
            brain_mask=np.ones((4, 5, 1), dtype=bool),
            tr=tr,
            reference_curve=reference,
        ).reshape(-1)

        assert observed[0] == pytest.approx(-0.1)
        assert observed[1] == pytest.approx(0.1)

    def test_canonical_bsw_does_not_censor_large_finite_k2(self) -> None:
        """Canonical BSW retains finite coefficients without a magnitude cutoff."""
        tr = 1.0
        reference = np.zeros(30, dtype=np.float64)
        reference[5:12] = [0.5, 1.5, 3.0, 2.0, 1.0, 0.5, 0.2]
        integral = cumulative_trapezoid(reference, dx=tr, initial=0.0)
        curves = np.repeat(reference[None, :], 20, axis=0)
        expected_k2 = 0.22
        curves[0] = reference - expected_k2 * integral

        observed = estimate_k2_per_voxel(
            curves.reshape(4, 5, 1, 30),
            brain_mask=np.ones((4, 5, 1), dtype=bool),
            tr=tr,
            reference_curve=reference,
        ).reshape(-1)

        assert observed[0] == pytest.approx(expected_k2)
        assert np.all(np.isfinite(observed[1:]))

    def test_late_signal_enhancement_failure_is_identified(self) -> None:
        """A deterministic late signal-enhancement tail fails explicitly."""
        curves = np.zeros((300, 1, 1, 25), dtype=np.float64)
        curves[..., 5:15] = 1.0
        curves[..., 15:] = -0.1
        qc: dict[str, object] = {}

        signal = np.exp(-0.03 * curves)
        reference = _compute_bsw_reference(
            curves,
            signal,
            np.ones(curves.shape[:3], dtype=bool),
            baseline_frames=5,
            reference_qc=qc,
        )

        assert reference is None
        assert qc["k2_reference_failure_reason"] == "insufficient_selected_support"

    @pytest.mark.parametrize(
        "k2_true",
        [
            pytest.param(0.0, id="K2_0.00"),
            pytest.param(0.02, id="K2_0.02"),
            pytest.param(0.05, id="K2_0.05"),
            pytest.param(0.1, id="K2_0.10"),
            pytest.param(0.2, id="K2_0.20"),
        ],
    )
    def test_k2_recovery(self, k2_true: float) -> None:
        """Recovered K2 should be within tolerance of ground truth.

        Uses multi-voxel phantom with mean brain reference matching
        the estimator's reference curve.
        """
        phantom = generate_k2_dro_phantom(k2_enhancing=k2_true)

        k2_map = estimate_k2_per_voxel(
            concentration=phantom.concentration,
            signal=phantom.signal,
            brain_mask=phantom.brain_mask,
            tr=phantom.dt,
        )

        if k2_true == 0.0:
            # Non-enhancing voxels should have near-zero K2
            assert np.median(k2_map[phantom.brain_mask]) < 0.02, (
                f"K2 at zero should be near 0, got median "
                f"{np.median(k2_map[phantom.brain_mask]):.4f}"
            )
        else:
            # Enhancing voxels should recover K2
            k2_enh = float(np.median(k2_map[phantom.enhancing_mask]))
            assert k2_enh == pytest.approx(k2_true, rel=0.5), (
                f"K2={k2_enh:.4f} outside 50% tolerance for true K2={k2_true}"
            )
            # Non-enhancing voxels should remain near zero
            non_enhancing = phantom.brain_mask & ~phantom.enhancing_mask
            assert np.median(k2_map[non_enhancing]) < 0.02, (
                f"Non-enhancing K2 median should be <0.02, got "
                f"{np.median(k2_map[non_enhancing]):.4f}"
            )


class TestK2CorrectionEffectiveness:
    """Test that K2 correction restores clean tissue curves."""

    @pytest.mark.parametrize(
        "k2_true",
        [
            pytest.param(0.05, id="K2_0.05"),
            pytest.param(0.1, id="K2_0.10"),
            pytest.param(0.2, id="K2_0.20"),
        ],
    )
    def test_correction_reduces_error(self, k2_true: float) -> None:
        """RMSE on enhancing voxels: corrected < uncorrected * 0.8."""
        phantom = generate_k2_dro_phantom(k2_enhancing=k2_true)
        bl = 10  # baseline frames

        result = apply_k2_correction(
            concentration=phantom.concentration,
            signal=phantom.signal,
            brain_mask=phantom.brain_mask,
            tr=phantom.dt,
        )

        # Compute RMSE for enhancing voxels
        enh_idx = np.where(phantom.enhancing_mask)
        c_corr = result.concentration_corrected[
            enh_idx[0], enh_idx[1], enh_idx[2], bl:
        ]
        c_leak = phantom.concentration[enh_idx[0], enh_idx[1], enh_idx[2], bl:]
        c_true = phantom.clean[enh_idx[0], enh_idx[1], enh_idx[2], bl:]

        rmse_uncorrected = float(np.sqrt(np.mean((c_leak - c_true) ** 2)))
        rmse_corrected = float(np.sqrt(np.mean((c_corr - c_true) ** 2)))

        assert rmse_corrected < rmse_uncorrected * 0.8, (
            f"Correction did not improve enough: "
            f"RMSE_corrected={rmse_corrected:.4f} vs "
            f"RMSE_uncorrected={rmse_uncorrected:.4f}"
        )


class TestK2EdgeCases:
    """Edge cases for K2 estimation and correction."""

    def test_bsw_requires_raw_signal_without_explicit_reference(self) -> None:
        """The public BSW path must not substitute a concentration selector."""
        phantom = generate_k2_dro_phantom(k2_enhancing=0.1)

        with pytest.raises(ValueError, match="Raw signal is required"):
            apply_k2_correction(
                concentration=phantom.concentration,
                brain_mask=phantom.brain_mask,
                tr=phantom.dt,
            )

    def test_k2_zero_everywhere(self) -> None:
        """K2=0: corrected should approximately equal input."""
        phantom = generate_k2_dro_phantom(k2_enhancing=0.0)

        result = apply_k2_correction(
            concentration=phantom.concentration,
            signal=phantom.signal,
            brain_mask=phantom.brain_mask,
            tr=phantom.dt,
        )

        # With K2 ≈ 0, correction should be minimal
        diff = np.max(np.abs(
            result.concentration_corrected - phantom.concentration
        ))
        peak = float(np.max(np.abs(phantom.concentration)))
        if peak > 0:
            assert diff / peak < 0.10, (
                f"K2=0 correction changed signal too much: "
                f"max_diff/peak = {diff/peak:.4f}"
            )

    def test_high_k2_no_overcorrection(self) -> None:
        """Very high K2 (0.4) should not over-correct wildly."""
        phantom = generate_k2_dro_phantom(k2_enhancing=0.4)

        result = apply_k2_correction(
            concentration=phantom.concentration,
            signal=phantom.signal,
            brain_mask=phantom.brain_mask,
            tr=phantom.dt,
        )

        # Peak of corrected should not exceed 3x peak of clean
        peak_corrected = float(np.max(result.concentration_corrected))
        peak_clean = float(np.max(phantom.clean))
        if peak_clean > 0:
            assert peak_corrected < peak_clean * 3.0, (
                f"Over-correction: corrected peak {peak_corrected:.2f} > "
                f"3x clean peak {peak_clean:.2f}"
            )

    def test_short_time_series(self) -> None:
        """Short time series (10 frames) should degrade gracefully."""
        phantom = generate_k2_dro_phantom(
            k2_enhancing=0.1, n_timepoints=10, baseline_frames=3,
        )

        # Should not raise
        k2_map = estimate_k2_per_voxel(
            concentration=phantom.concentration,
            signal=phantom.signal,
            brain_mask=phantom.brain_mask,
            tr=phantom.dt,
        )

        assert np.all(np.isnan(k2_map)), "Unavailable short-series K2 should be NaN"

    def test_k2_correction_result_fields(self) -> None:
        """K2CorrectionResult should have all expected fields."""
        phantom = generate_k2_dro_phantom(k2_enhancing=0.1)

        result = apply_k2_correction(
            concentration=phantom.concentration,
            signal=phantom.signal,
            brain_mask=phantom.brain_mask,
            tr=phantom.dt,
        )

        assert isinstance(result, K2CorrectionResult)
        assert result.concentration_corrected.shape == phantom.concentration.shape
        assert result.k2_map.shape == phantom.concentration.shape[:3]
        assert result.auc_tissue.shape == phantom.concentration.shape[:3]
        assert result.correction_applied is True
        assert result.qc_metrics is not None
        assert result.qc_metrics["k2_signed"] is True
        assert result.qc_metrics["k2_reference_available"] is True
        assert result.qc_metrics["k2_reference_selection"] == (
            "raw_signal_final10_mean_1_baseline_sample_sd"
        )
        assert result.qc_metrics["k2_reference_selection_domain"] == "raw_signal"
        assert 0.0 < result.qc_metrics["k2_reference_selected_fraction"] <= 1.0

    def test_precomputed_k2_map(self) -> None:
        """Providing a pre-computed K2 map should bypass estimation."""
        phantom = generate_k2_dro_phantom(k2_enhancing=0.1)

        # Pre-computed K2 map with known value
        k2_precomputed = np.full(
            phantom.concentration.shape[:3], 0.1, dtype=np.float64,
        )

        result = apply_k2_correction(
            concentration=phantom.concentration,
            signal=phantom.signal,
            brain_mask=phantom.brain_mask,
            tr=phantom.dt,
            k2_map=k2_precomputed,
        )

        # Should use the pre-computed value, not estimate
        assert result.correction_applied is True
        # Corrected should differ from input (K2=0.1 is above gate threshold)
        diff = float(np.max(np.abs(
            result.concentration_corrected - phantom.concentration
        )))
        assert diff > 0, "Pre-computed K2=0.1 should produce a correction"

    def test_brain_mask_restricts_estimation(self) -> None:
        """Brain mask should restrict which voxels get K2 estimated.

        Uses a phantom with enough masked voxels (≥10) to compute
        the mean brain reference.
        """
        phantom = generate_k2_dro_phantom(
            k2_enhancing=0.1,
            n_brain_voxels=20,
            n_enhancing_voxels=5,
        )

        # Create a sub-mask: only first 15 brain voxels
        sub_mask = np.zeros_like(phantom.brain_mask)
        brain_idx = np.where(phantom.brain_mask)
        for i in range(min(15, len(brain_idx[0]))):
            sub_mask[brain_idx[0][i], brain_idx[1][i], brain_idx[2][i]] = True

        k2_map = estimate_k2_per_voxel(
            concentration=phantom.concentration,
            signal=phantom.signal,
            brain_mask=sub_mask,
            tr=phantom.dt,
        )

        # Voxels outside the fitted support are undefined.
        outside_mask = ~sub_mask
        assert np.all(np.isnan(k2_map[outside_mask]))

    def test_insufficient_brain_voxels_skips_correction(self) -> None:
        """A mask below the validated support floor skips correction gracefully."""
        phantom = generate_k2_dro_phantom(k2_enhancing=0.1)

        # Create tiny mask with only 3 voxels
        tiny_mask = np.zeros_like(phantom.brain_mask)
        brain_idx = np.where(phantom.brain_mask)
        for i in range(min(3, len(brain_idx[0]))):
            tiny_mask[brain_idx[0][i], brain_idx[1][i], brain_idx[2][i]] = True

        result = apply_k2_correction(
            concentration=phantom.concentration,
            signal=phantom.signal,
            brain_mask=tiny_mask,
            tr=phantom.dt,
        )

        assert result.correction_applied is False
        assert result.qc_metrics is not None
        assert result.qc_metrics["k2_signed"] is True
        assert result.qc_metrics["k2_reference_available"] is False
        expected = phantom.concentration - np.mean(
            phantom.concentration[..., :5], axis=-1, keepdims=True
        )
        np.testing.assert_array_equal(result.concentration_corrected, expected)


# =============================================================================
# Bidirectional K2 Tests (Leu et al. 2016)
# =============================================================================


class TestK2RecoveryNeutral:
    """K2 recovery on independent (non-inverse-crime) tissue.

    The brain and tumor curves are built by the dt/20 fine-grid forward, while
    the fitted instantaneous term remains a single scaled reference. The known
    clean reference isolates regression sign and scale while retaining mismatch
    between the tissue forward model and the fitted instantaneous term. The
    absolute tolerance covers that model-mismatch floor.

    This deliberately does NOT probe the rectangular->trapezoidal cumulative
    switch: the BSW K2 estimate is provably quadrature-invariant. The rect-trap
    difference of the cumulative integral equals 0.5*dt*C_bar, which is collinear
    with the C_bar column of the [C_bar, integral(C_bar)] design matrix, so it is
    absorbed into that coefficient and leaves the K2 (integral) coefficient
    unchanged. The quadrature choice therefore has no measurable effect on K2,
    and no test can be can-fail for it here.
    """

    @pytest.mark.parametrize("k2_true", [0.02, 0.05, 0.1])
    def test_k2_recovers_from_independent_leakage(self, k2_true: float) -> None:
        dt = 1.5
        n_bl, n_post = 8, 52
        n = n_bl + n_post
        aif, _ = generate_dsc_aif_dro(
            n_timepoints=n, dt_sec=dt, baseline_frames=n_bl, peak_delta_r2=40.0
        )
        rng = np.random.default_rng(0)

        # Non-enhancing brain: varied CBF/MTT, tissue via the fine-grid forward
        # (an independent forward the estimator does not invert).
        n_brain, n_enh = 200, 8
        brain = np.empty((n_brain, n), dtype=np.float64)
        for i in range(n_brain):
            residue = make_exponential_residue(
                float(rng.uniform(20.0, 60.0)),
                float(rng.uniform(3.0, 6.0)),
                n,
                dt,
            )
            brain[i] = fine_grid_forward(aif, residue, dt)
        # Leakage is driven by the CLEAN mean reference; the estimator will
        # instead see a mean contaminated by the enhancing voxels below.
        c_bar_clean = brain.mean(axis=0)

        # Enhancing (tumor) leaky curves: clean forward minus the BSW leakage
        # integral K2 * int_0^t C_bar of the clean reference.
        tumor_clean = fine_grid_forward(
            aif, make_exponential_residue(100.0, 6.0, n, dt), dt
        )
        leakage = k2_true * cumulative_trapezoid(c_bar_clean, dx=dt, initial=0.0)
        tumor_leaky = tumor_clean - leakage

        # Assemble a single-slice 4D phantom + masks.
        nx = 16
        ny = (n_brain + n_enh + nx - 1) // nx + 1
        conc = np.zeros((nx, ny, 1, n), dtype=np.float64)
        brain_mask = np.zeros((nx, ny, 1), dtype=bool)
        enh_mask = np.zeros((nx, ny, 1), dtype=bool)
        for idx in range(n_brain):
            conc[idx % nx, idx // nx, 0, :] = brain[idx]
            brain_mask[idx % nx, idx // nx, 0] = True
        for j in range(n_enh):
            idx = n_brain + j
            conc[idx % nx, idx // nx, 0, :] = tumor_leaky
            brain_mask[idx % nx, idx // nx, 0] = True
            enh_mask[idx % nx, idx // nx, 0] = True

        k2_map = estimate_k2_per_voxel(
            concentration=conc,
            brain_mask=brain_mask,
            tr=dt,
            baseline_frames=n_bl,
            reference_curve=c_bar_clean,
        )
        k2_enh = float(np.median(k2_map[enh_mask]))

        # A matched zero-leak control isolates the injected leakage coefficient
        # from the BSW scaled-reference model's tissue-shape mismatch.
        conc_control = conc.copy()
        conc_control[enh_mask] = tumor_clean
        k2_control_map = estimate_k2_per_voxel(
            concentration=conc_control,
            brain_mask=brain_mask,
            tr=dt,
            baseline_frames=n_bl,
            reference_curve=c_bar_clean,
        )
        k2_control = float(np.median(k2_control_map[enh_mask]))
        recovered_increment = k2_enh - k2_control
        assert recovered_increment == pytest.approx(k2_true, abs=1e-10)


class TestK2QCMetrics:
    """Signed K2 distribution summaries use a common finite support."""

    def test_unmasked_qc_retains_signed_distribution(self) -> None:
        """QC without a mask must not censor negative fitted coefficients."""
        k2_map = np.array([[[-0.03], [-0.01]], [[0.01], [0.03]]])
        auc = np.ones_like(k2_map)

        metrics = _compute_k2_qc_metrics(k2_map, None, auc)

        assert metrics["n_brain_voxels"] == 4
        assert metrics["k2_min"] == pytest.approx(-0.03)
        assert metrics["k2_p05"] < 0.0
        assert metrics["percent_negative_k2"] == pytest.approx(50.0)
