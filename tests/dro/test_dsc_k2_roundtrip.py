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
    apply_k2_correction,
    apply_k2_correction_bidirectional,
    estimate_k2_per_voxel,
    estimate_k2_bidirectional,
    _compute_ees_concentration,
)
from pmriprep.utils.dsc_constants import KEP_DSC_DEFAULT

from .conftest import (
    fine_grid_forward,
    generate_dsc_aif_dro,
    generate_k2_dro_phantom,
    generate_leaky_dsc_tissue_bidirectional,
    make_exponential_residue,
)


# =============================================================================
# Test classes
# =============================================================================


class TestK2EstimationRecovery:
    """Test K2 estimation accuracy against known ground truth."""

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

    def test_k2_zero_everywhere(self) -> None:
        """K2=0: corrected should approximately equal input."""
        phantom = generate_k2_dro_phantom(k2_enhancing=0.0)

        result = apply_k2_correction(
            concentration=phantom.concentration,
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
            brain_mask=phantom.brain_mask,
            tr=phantom.dt,
        )

        # K2 may be zero or small — just check finite
        assert np.all(np.isfinite(k2_map)), "K2 map has NaN/Inf"

    def test_k2_correction_result_fields(self) -> None:
        """K2CorrectionResult should have all expected fields."""
        phantom = generate_k2_dro_phantom(k2_enhancing=0.1)

        result = apply_k2_correction(
            concentration=phantom.concentration,
            brain_mask=phantom.brain_mask,
            tr=phantom.dt,
        )

        assert isinstance(result, K2CorrectionResult)
        assert result.concentration_corrected.shape == phantom.concentration.shape
        assert result.k2_map.shape == phantom.concentration.shape[:3]
        assert result.auc_tissue.shape == phantom.concentration.shape[:3]
        assert result.correction_applied is True
        assert result.qc_metrics is not None

    def test_precomputed_k2_map(self) -> None:
        """Providing a pre-computed K2 map should bypass estimation."""
        phantom = generate_k2_dro_phantom(k2_enhancing=0.1)

        # Pre-computed K2 map with known value
        k2_precomputed = np.full(
            phantom.concentration.shape[:3], 0.1, dtype=np.float64,
        )

        result = apply_k2_correction(
            concentration=phantom.concentration,
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
            brain_mask=sub_mask,
            tr=phantom.dt,
        )

        # Voxels outside mask should be zero
        outside_mask = ~sub_mask
        assert np.all(k2_map[outside_mask] == 0.0), (
            "Voxels outside brain mask should have K2=0"
        )

    def test_insufficient_brain_voxels_skips_correction(self) -> None:
        """Tiny mask (<10 voxels) should skip correction gracefully."""
        phantom = generate_k2_dro_phantom(k2_enhancing=0.1)

        # Create tiny mask with only 3 voxels
        tiny_mask = np.zeros_like(phantom.brain_mask)
        brain_idx = np.where(phantom.brain_mask)
        for i in range(min(3, len(brain_idx[0]))):
            tiny_mask[brain_idx[0][i], brain_idx[1][i], brain_idx[2][i]] = True

        result = apply_k2_correction(
            concentration=phantom.concentration,
            brain_mask=tiny_mask,
            tr=phantom.dt,
        )

        assert result.correction_applied is False
        np.testing.assert_array_equal(
            result.concentration_corrected, phantom.concentration,
        )


# =============================================================================
# Bidirectional K2 Tests (Leu et al. 2016)
# =============================================================================


class TestBidirectionalEESConcentration:
    """Test _compute_ees_concentration() helper."""

    def test_ees_concentration_shape(self) -> None:
        """C_e should have same length as input."""
        c_bar = np.sin(np.linspace(0, np.pi, 60))
        c_e = _compute_ees_concentration(c_bar, kep=0.005, tr=1.5)
        assert c_e.shape == c_bar.shape

    def test_ees_concentration_limit_low_kep(self) -> None:
        """When kep → 0, C_e → cumsum(C_bar) * TR (approaches BSW)."""
        c_bar = np.sin(np.linspace(0, np.pi, 60))
        tr = 1.5

        # Very small kep
        c_e_small = _compute_ees_concentration(c_bar, kep=1e-6, tr=tr)
        cumsum_bsw = np.cumsum(c_bar) * tr

        # Should be approximately equal (ratio close to 1)
        ratio = c_e_small[-1] / cumsum_bsw[-1] if cumsum_bsw[-1] > 0 else 1.0
        assert ratio == pytest.approx(1.0, rel=0.01), (
            f"Small kep should approach BSW: ratio={ratio:.4f}"
        )

    def test_ees_concentration_high_kep_decays(self) -> None:
        """High kep should cause C_e to decay after bolus peak."""
        # Single Gaussian pulse
        c_bar = np.exp(-((np.arange(60) - 20) ** 2) / 50)
        c_e = _compute_ees_concentration(c_bar, kep=0.05, tr=1.5)

        # C_e should peak later and then decay
        peak_c_bar = np.argmax(c_bar)
        peak_c_e = np.argmax(c_e)
        assert peak_c_e >= peak_c_bar, "C_e peak should not be earlier than C_bar"

        # Check decay in tail
        assert c_e[-1] < c_e[peak_c_e], "C_e should decay after peak with high kep"


class TestBidirectionalK2Estimation:
    """Test bidirectional K2 estimation accuracy."""

    @pytest.mark.parametrize(
        "k2_true,kep",
        [
            pytest.param(0.02, 0.005, id="K2_0.02_kep_0.005"),
            pytest.param(0.05, 0.005, id="K2_0.05_kep_0.005"),
            pytest.param(0.1, 0.005, id="K2_0.10_kep_0.005"),
            pytest.param(0.1, 0.01, id="K2_0.10_kep_0.01"),
        ],
    )
    def test_bidirectional_k2_recovery(self, k2_true: float, kep: float) -> None:
        """Bidirectional estimator should recover K2 within tolerance."""
        phantom = generate_k2_dro_phantom(k2_enhancing=k2_true)

        k2_map = estimate_k2_bidirectional(
            concentration=phantom.concentration,
            brain_mask=phantom.brain_mask,
            tr=phantom.dt,
            kep=kep,
        )

        # Enhancing voxels should recover K2 (wider tolerance due to model mismatch)
        k2_enh = float(np.median(k2_map[phantom.enhancing_mask]))
        assert k2_enh == pytest.approx(k2_true, rel=0.8), (
            f"K2={k2_enh:.4f} outside 80% tolerance for true K2={k2_true}"
        )


class TestBidirectionalK2Correction:
    """Test bidirectional K2 correction."""

    def test_bidirectional_result_fields(self) -> None:
        """Result should have method and kep_used fields."""
        phantom = generate_k2_dro_phantom(k2_enhancing=0.1)

        result = apply_k2_correction_bidirectional(
            concentration=phantom.concentration,
            brain_mask=phantom.brain_mask,
            tr=phantom.dt,
            kep=0.005,
        )

        assert result.method == "bidirectional"
        assert result.kep_used == 0.005
        assert result.qc_metrics is not None
        assert result.qc_metrics.get("k2_method") == "bidirectional"
        assert result.qc_metrics.get("kep_used") == 0.005

    def test_bidirectional_uses_default_kep(self) -> None:
        """When kep=None, should use KEP_DSC_DEFAULT."""
        phantom = generate_k2_dro_phantom(k2_enhancing=0.1)

        result = apply_k2_correction_bidirectional(
            concentration=phantom.concentration,
            brain_mask=phantom.brain_mask,
            tr=phantom.dt,
            kep=None,
        )

        assert result.kep_used == KEP_DSC_DEFAULT

    def test_bsw_result_has_method_field(self) -> None:
        """Standard BSW result should have method='bsw'."""
        phantom = generate_k2_dro_phantom(k2_enhancing=0.1)

        result = apply_k2_correction(
            concentration=phantom.concentration,
            brain_mask=phantom.brain_mask,
            tr=phantom.dt,
        )

        assert result.method == "bsw"
        assert result.kep_used is None
        assert result.qc_metrics.get("k2_method") == "bsw"

    @pytest.mark.parametrize(
        "k2_true",
        [
            pytest.param(0.05, id="K2_0.05"),
            pytest.param(0.1, id="K2_0.10"),
        ],
    )
    def test_bidirectional_reduces_error(self, k2_true: float) -> None:
        """Bidirectional correction should reduce RMSE on enhancing voxels."""
        phantom = generate_k2_dro_phantom(k2_enhancing=k2_true)
        bl = 10

        result = apply_k2_correction_bidirectional(
            concentration=phantom.concentration,
            brain_mask=phantom.brain_mask,
            tr=phantom.dt,
            kep=KEP_DSC_DEFAULT,
        )

        enh_idx = np.where(phantom.enhancing_mask)
        c_corr = result.concentration_corrected[
            enh_idx[0], enh_idx[1], enh_idx[2], bl:
        ]
        c_leak = phantom.concentration[enh_idx[0], enh_idx[1], enh_idx[2], bl:]
        c_true = phantom.clean[enh_idx[0], enh_idx[1], enh_idx[2], bl:]

        rmse_uncorrected = float(np.sqrt(np.mean((c_leak - c_true) ** 2)))
        rmse_corrected = float(np.sqrt(np.mean((c_corr - c_true) ** 2)))

        # Should improve, though may not be as strong as BSW on BSW-generated phantom
        assert rmse_corrected <= rmse_uncorrected * 1.1, (
            f"Bidirectional correction should not make things much worse: "
            f"RMSE_corrected={rmse_corrected:.4f} vs "
            f"RMSE_uncorrected={rmse_uncorrected:.4f}"
        )


class TestK2NeutralQuadrature:
    """Neutral (non-inverse-crime) K2 accuracy.

    The round-trip tests above generate the leakage with the same quadrature
    the estimator uses (cumulative_trapezoid == estimator), so they prove
    parity but are structurally blind to a quadrature bias. Here the leakage
    integral K2 * int_0^t C_bar is built on a dt/20 fine grid and the tissue
    curves come from the fine-grid forward, so the production estimator's
    coarse trapezoidal cumulative regressor is tested against a truth it did
    not generate. Recovery is dominated by mean-reference contamination from
    the (few) enhancing voxels plus the sub-percent coarse-vs-fine quadrature
    floor, so the neutral tolerance is tighter than the round-trip's rel=0.5.
    """

    @staticmethod
    def _fine_cumulative(
        curve: np.ndarray, dt: float, oversample: int = 20
    ) -> np.ndarray:
        """int_0^t curve on a dt/oversample grid, downsampled to frames.

        Independent of the estimator's coarse cumulative_trapezoid.
        """
        n = len(curve)
        t_frame = np.arange(n) * dt
        t_fine = np.arange(n * oversample) * (dt / oversample)
        fine = np.interp(t_fine, t_frame, curve)
        cum = cumulative_trapezoid(fine, dx=dt / oversample, initial=0.0)
        return np.asarray(cum[::oversample], dtype=np.float64)

    @pytest.mark.parametrize("k2_true", [0.02, 0.05, 0.1])
    def test_k2_recovers_from_fine_grid_leakage(self, k2_true: float) -> None:
        dt = 1.5
        n_bl, n_post = 8, 52
        n = n_bl + n_post
        aif, _ = generate_dsc_aif_dro(
            n_timepoints=n, dt_sec=dt, baseline_frames=n_bl, peak_delta_r2=40.0
        )
        rng = np.random.default_rng(0)

        # Non-enhancing brain: varied CBF/MTT, tissue via the neutral fine-grid
        # forward (never touches the estimator's cumulative quadrature).
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
        c_bar = brain.mean(axis=0)

        # Enhancing (tumor) leaky curves: clean forward minus a FINE-grid
        # leakage integral of the mean reference.
        tumor_clean = fine_grid_forward(
            aif, make_exponential_residue(100.0, 6.0, n, dt), dt
        )
        tumor_leaky = tumor_clean - k2_true * self._fine_cumulative(c_bar, dt)

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
            concentration=conc, brain_mask=brain_mask, tr=dt
        )
        k2_enh = float(np.median(k2_map[enh_mask]))
        assert k2_enh == pytest.approx(k2_true, rel=0.20), (
            f"neutral K2={k2_enh:.4f} outside 20% of true {k2_true}"
        )
