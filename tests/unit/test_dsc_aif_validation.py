"""Unit tests for DSC AIF post-averaging validation."""

import numpy as np
import pytest

from pmriprep.interfaces.dsc_aif import AIFDetection, _FP_GAP_SECONDS


# =============================================================================
# Helpers
# =============================================================================


def _make_aif_curve(
    n_frames: int = 80,
    baseline_frames: int = 15,
    tr: float = 1.5,
    fp_peak_frame: int = 19,
    fp_peak_val: float = 8.0,
    recirc_peak_val: float = 3.0,
) -> np.ndarray:
    """Create a synthetic AIF with controllable first-pass and recirculation.

    Parameters
    ----------
    n_frames : int
        Total number of timepoints.
    baseline_frames : int
        Number of pre-bolus baseline frames.
    tr : float
        Repetition time in seconds.
    fp_peak_frame : int
        Frame of first-pass peak.
    fp_peak_val : float
        Peak concentration at first pass.
    recirc_peak_val : float
        Peak concentration during recirculation (post first-pass).

    Returns
    -------
    np.ndarray
        1D AIF concentration time course.
    """
    t = np.arange(n_frames, dtype=np.float64)
    curve = np.zeros(n_frames, dtype=np.float64)

    # First-pass Gaussian peak
    sigma_fp = 2.0
    fp = fp_peak_val * np.exp(-0.5 * ((t - fp_peak_frame) / sigma_fp) ** 2)
    curve += fp

    # Recirculation: broad Gaussian centred ~40 frames after baseline
    recirc_center = baseline_frames + int(40.0 / tr)
    sigma_recirc = 8.0
    recirc = recirc_peak_val * np.exp(
        -0.5 * ((t - recirc_center) / sigma_recirc) ** 2
    )
    curve += recirc

    # Zero out pre-baseline
    curve[:baseline_frames] = 0.0
    return curve


def _make_conc_data_from_curve(
    curve: np.ndarray,
    shape: tuple[int, int, int] = (10, 10, 10),
    center: tuple[int, int, int] = (5, 5, 5),
    noise_std: float = 0.0,
) -> np.ndarray:
    """Create 4D concentration data with the AIF curve at a given voxel."""
    n_frames = len(curve)
    data = np.zeros((*shape, n_frames), dtype=np.float64)
    # Set centre voxel to the AIF curve
    data[center[0], center[1], center[2], :] = curve
    # Fill neighbouring voxels with a slightly different version (tissue-like)
    rng = np.random.default_rng(42)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if (i, j, k) != center:
                    # Tissue: lower peak, delayed
                    tissue = curve * 0.3
                    if noise_std > 0:
                        tissue += rng.normal(0, noise_std, n_frames)
                    data[i, j, k, :] = np.clip(tissue, 0.0, None)
    return data


def _make_brain_mask(
    shape: tuple[int, int, int] = (10, 10, 10),
) -> np.ndarray:
    """Create a brain mask (all ones)."""
    return np.ones(shape, dtype=np.uint8)


def _make_candidates(
    cost: float = 1.0,
    coord: tuple[int, int, int] = (5, 5, 5),
    n: int = 1,
) -> list[tuple[float, tuple[int, int, int], dict]]:
    """Create a list of n identical candidates."""
    return [(cost + i * 0.1, coord, {"s0": 100.0}) for i in range(n)]


# =============================================================================
# TestValidateAveragedAIF
# =============================================================================


class TestValidateAveragedAIF:
    """Tests for AIFDetection._validate_averaged_aif."""

    def test_normal_aif_no_retries(self) -> None:
        """Normal AIF (SPR ~50%) passes validation without retries."""
        curve = _make_aif_curve(
            fp_peak_val=8.0,
            recirc_peak_val=2.0,  # SPR well under 100%
        )
        conc = _make_conc_data_from_curve(curve)
        mask = _make_brain_mask()
        candidates = _make_candidates()

        result_curve, result_mask, result_cands, qc = (
            AIFDetection._validate_averaged_aif(
                curve, candidates, conc, mask,
                tr=1.5, baseline_frames=15,
            )
        )

        assert qc["averaged_aif_validation_retries"] == 0
        assert qc["averaged_aif_final_roi_size"] == 3
        assert "aif_recirculation_warning" not in qc
        np.testing.assert_array_equal(result_curve, curve)
        assert len(result_cands) == len(candidates)

    def test_recirc_dominant_triggers_retry(self) -> None:
        """Recirculation-dominant AIF (SPR > 100%) triggers roi_size=1 retry."""
        # First-pass peak = 5, recirculation = 8 → SPR > 100%
        curve = _make_aif_curve(
            fp_peak_val=5.0,
            recirc_peak_val=8.0,
        )
        conc = _make_conc_data_from_curve(curve, center=(5, 5, 5))
        mask = _make_brain_mask()
        candidates = _make_candidates(coord=(5, 5, 5))

        _, _, _, qc = AIFDetection._validate_averaged_aif(
            curve, candidates, conc, mask,
            tr=1.5, baseline_frames=15,
        )

        assert qc["averaged_aif_validation_retries"] >= 1
        assert "averaged_aif_spr_retry1" in qc

    def test_single_voxel_fallback(self) -> None:
        """When roi_size=1 retry also fails, falls back to single best."""
        # Build a curve where even single-voxel has bad recirculation
        curve = _make_aif_curve(
            fp_peak_val=3.0,
            recirc_peak_val=10.0,
        )
        conc = _make_conc_data_from_curve(curve, center=(5, 5, 5))
        mask = _make_brain_mask()
        candidates = _make_candidates(coord=(5, 5, 5), n=3)

        _, _, result_cands, qc = AIFDetection._validate_averaged_aif(
            curve, candidates, conc, mask,
            tr=1.5, baseline_frames=15,
        )

        assert qc["averaged_aif_validation_retries"] == 2
        assert "averaged_aif_spr_retry2" in qc
        # Only 1 candidate used in final fallback
        assert len(result_cands) == 1

    def test_warning_flag_on_all_retries_fail(self) -> None:
        """Warning flags set when all retries still have bad recirculation."""
        curve = _make_aif_curve(
            fp_peak_val=2.0,
            recirc_peak_val=12.0,
        )
        conc = _make_conc_data_from_curve(curve, center=(5, 5, 5))
        mask = _make_brain_mask()
        candidates = _make_candidates(coord=(5, 5, 5))

        _, _, _, qc = AIFDetection._validate_averaged_aif(
            curve, candidates, conc, mask,
            tr=1.5, baseline_frames=15,
        )

        assert qc.get("aif_recirculation_warning") is True
        assert qc.get("aif_confidence") == "reduced"


# =============================================================================
# TestFirstPassGap
# =============================================================================


class TestFirstPassGap:
    """Tests for the _FP_GAP_SECONDS constant and first-pass window."""

    def test_fp_gap_constant_value(self) -> None:
        """_FP_GAP_SECONDS should be 3.0."""
        assert _FP_GAP_SECONDS == 3.0

    def test_fp_start_captures_early_peak(self) -> None:
        """With 3s gap and TR=1.5s, fp_start = baseline + 2 frames."""
        baseline_frames = 15
        tr = 1.5
        fp_start = baseline_frames + max(1, int(_FP_GAP_SECONDS / tr))
        # int(3.0 / 1.5) = 2 → fp_start = 17
        assert fp_start == 17
        # Frame 19 (the actual peak in sub-0005) is within the window
        fp_end = min(baseline_frames + int(40.0 / tr), 80)
        assert 17 <= 19 < fp_end

    def test_fp_start_old_gap_misses_peak(self) -> None:
        """With old 10s gap and TR=1.5s, fp_start=22 would miss frame 19."""
        baseline_frames = 15
        tr = 1.5
        old_fp_start = baseline_frames + max(1, int(10.0 / tr))
        # int(10.0 / 1.5) = 6 → fp_start = 21
        assert old_fp_start == 21
        # Frame 19 would be BEFORE the old window
        assert 19 < old_fp_start


# =============================================================================
# TestPSRLowerBound
# =============================================================================


class TestPSRLowerBound:
    """Tests for PSR lower bound rejection in _score_candidate_cost."""

    def _make_metrics(self, psr: float = 80.0) -> dict:
        """Create minimal candidate metrics dict."""
        return {
            "pre_bolus_frames": 15,
            "dt": 1.5,
            "ttp": 30.0,  # 30s after start → 9s after baseline end
            "baseline_return_ratio": 15.0,
            "peak_to_baseline_ratio": 5.0,
            "secondary_peak_ratio": 40.0,
            "post_peak_monotonicity": 70.0,
            "fwhm": 6.0,
            "ph": 50.0,
            "psr": psr,
            "s0": 50.0,
            "max_allowed_ttp": 100.0,
        }

    def test_psr_below_strict_threshold_rejected(self) -> None:
        """PSR < 40% rejected at strict level."""
        metrics = self._make_metrics(psr=30.0)
        cost = AIFDetection._score_candidate_cost(
            metrics, criteria="strict", median_s0=200.0,
        )
        assert cost == float("inf")

    def test_psr_above_strict_threshold_passes(self) -> None:
        """PSR = 50% passes at strict level."""
        metrics = self._make_metrics(psr=50.0)
        cost = AIFDetection._score_candidate_cost(
            metrics, criteria="strict", median_s0=200.0,
        )
        assert cost < float("inf")

    def test_psr_below_relaxed_threshold_rejected(self) -> None:
        """PSR < 30% rejected at relaxed level."""
        metrics = self._make_metrics(psr=20.0)
        cost = AIFDetection._score_candidate_cost(
            metrics, criteria="relaxed", median_s0=200.0,
        )
        assert cost == float("inf")

    def test_psr_nan_not_rejected(self) -> None:
        """PSR = NaN should not trigger rejection."""
        metrics = self._make_metrics(psr=float("nan"))
        cost = AIFDetection._score_candidate_cost(
            metrics, criteria="strict", median_s0=200.0,
        )
        assert cost < float("inf")

    def test_psr_above_max_still_rejected(self) -> None:
        """PSR > 110% still rejected at strict (existing PSR max check)."""
        metrics = self._make_metrics(psr=120.0)
        cost = AIFDetection._score_candidate_cost(
            metrics, criteria="strict", median_s0=200.0,
        )
        assert cost == float("inf")
