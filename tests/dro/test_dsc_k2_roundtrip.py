"""K2 leakage correction round-trip tests.

Tests K2 estimation and correction in dsc_preload.py against known ground
truth K2 values. Uses a multi-voxel DRO phantom where the forward leakage
model uses C̄(t) (mean brain reference), matching what the estimator recovers.

Exercises:
    dsc_preload.py: estimate_k2_per_voxel(), apply_k2_correction()
"""

import numpy as np
import pytest

from pmriprep.utils.dsc_preload import (
    K2CorrectionResult,
    apply_k2_correction,
    estimate_k2_per_voxel,
)

from .conftest import (
    generate_k2_dro_phantom,
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
