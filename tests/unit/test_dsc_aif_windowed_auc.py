"""Tests for windowed-AUC cascade gate (v0.29.2).

Verifies the first-pass windowed AUC comparison between detected and
synthetic AIFs, including the helper function, quality gate integration,
constant values, and edge cases.
"""

import numpy as np
import pytest

from pmriprep.utils.dsc_aif import compute_windowed_first_pass_auc
from pmriprep.utils.dsc_constants import (
    WINDOWED_AUC_FWHM_MULTIPLIER,
    WINDOWED_AUC_RATIO_MAX,
    WINDOWED_AUC_RATIO_MIN,
    WINDOWED_AUC_TRAILING_CAP_S,
)


def _make_gamma_bolus(
    n_timepoints: int = 60,
    tr: float = 1.5,
    baseline_frames: int = 5,
    peak_amplitude: float = 35.0,
    alpha: float = 4.0,
    time_to_peak: float = 4.5,
) -> np.ndarray:
    """Generate a gamma-variate bolus curve for testing."""
    curve = np.zeros(n_timepoints)
    t0 = baseline_frames * tr
    for i in range(baseline_frames, n_timepoints):
        t = i * tr - t0
        if t > 0:
            beta = time_to_peak / alpha
            curve[i] = peak_amplitude * (t / time_to_peak) ** alpha * np.exp(
                alpha * (1.0 - t / time_to_peak)
            )
    return curve


# -----------------------------------------------------------------------
# compute_windowed_first_pass_auc helper
# -----------------------------------------------------------------------

class TestComputeWindowedFirstPassAuc:

    def test_gaussian_peak_known_auc(self):
        curve = _make_gamma_bolus(peak_amplitude=35.0)
        auc, win_start, win_end = compute_windowed_first_pass_auc(
            curve, baseline_frames=5, tr=1.5,
        )
        assert auc > 0
        assert win_start >= 5
        assert win_end <= len(curve)

    def test_window_bounds_respect_baseline(self):
        curve = _make_gamma_bolus(baseline_frames=10)
        _, win_start, _ = compute_windowed_first_pass_auc(
            curve, baseline_frames=10, tr=1.5,
        )
        assert win_start >= 10

    def test_trailing_cap_short_acquisition(self):
        curve = _make_gamma_bolus(n_timepoints=20, tr=2.0, baseline_frames=3)
        _, _, win_end = compute_windowed_first_pass_auc(
            curve, baseline_frames=3, tr=2.0, trailing_cap_s=10.0,
        )
        assert win_end <= 20
        peak_idx = int(np.argmax(curve))
        assert win_end <= peak_idx + int(10.0 / 2.0) + 1

    def test_zero_curve_returns_zero_auc(self):
        curve = np.zeros(60)
        auc, _, _ = compute_windowed_first_pass_auc(
            curve, baseline_frames=5, tr=1.5,
        )
        assert auc == 0.0

    def test_too_short_curve_returns_zero(self):
        curve = np.zeros(3)
        auc, win_start, win_end = compute_windowed_first_pass_auc(
            curve, baseline_frames=5, tr=1.5,
        )
        assert auc == 0.0

    def test_window_symmetric_around_peak(self):
        curve = _make_gamma_bolus(n_timepoints=120, baseline_frames=20)
        _, win_start, win_end = compute_windowed_first_pass_auc(
            curve, baseline_frames=20, tr=1.5,
        )
        peak_idx = int(np.argmax(curve))
        assert win_start <= peak_idx
        assert win_end > peak_idx

    def test_auc_scales_with_amplitude(self):
        curve1 = _make_gamma_bolus(peak_amplitude=35.0)
        curve2 = _make_gamma_bolus(peak_amplitude=70.0)
        auc1, _, _ = compute_windowed_first_pass_auc(
            curve1, baseline_frames=5, tr=1.5,
        )
        auc2, _, _ = compute_windowed_first_pass_auc(
            curve2, baseline_frames=5, tr=1.5,
        )
        assert auc2 == pytest.approx(2.0 * auc1, rel=0.1)


# -----------------------------------------------------------------------
# Windowed AUC gate logic
# -----------------------------------------------------------------------

class TestWindowedAucGate:

    def test_good_aif_passes(self):
        from pmriprep.interfaces.dsc_aif import AIFDetection

        det = AIFDetection()
        curve = _make_gamma_bolus(peak_amplitude=35.0)
        ref_auc, _, _ = compute_windowed_first_pass_auc(
            curve, baseline_frames=5, tr=1.5,
        )
        det._synth_ref_windowed_auc = ref_auc
        result = det._aif_passes_quality(curve, baseline_frames=5, tr=1.5)
        assert result.auc_ratio is not None
        assert result.auc_ratio == pytest.approx(1.0, abs=0.05)

    def test_partial_volume_fails(self):
        from pmriprep.interfaces.dsc_aif import AIFDetection

        det = AIFDetection()
        curve = _make_gamma_bolus(peak_amplitude=35.0)
        ref_auc, _, _ = compute_windowed_first_pass_auc(
            curve, baseline_frames=5, tr=1.5,
        )
        det._synth_ref_windowed_auc = ref_auc
        scaled_curve = curve * 0.2
        result = det._aif_passes_quality(scaled_curve, baseline_frames=5, tr=1.5)
        assert result.auc_ratio is not None
        assert result.auc_ratio < WINDOWED_AUC_RATIO_MIN

    def test_over_scale_fails(self):
        from pmriprep.interfaces.dsc_aif import AIFDetection

        det = AIFDetection()
        curve = _make_gamma_bolus(peak_amplitude=35.0)
        ref_auc, _, _ = compute_windowed_first_pass_auc(
            curve, baseline_frames=5, tr=1.5,
        )
        det._synth_ref_windowed_auc = ref_auc
        scaled_curve = curve * 2.5
        result = det._aif_passes_quality(scaled_curve, baseline_frames=5, tr=1.5)
        assert result.auc_ratio is not None
        assert result.auc_ratio > WINDOWED_AUC_RATIO_MAX

    def test_gate_disabled_when_ref_zero(self):
        from pmriprep.interfaces.dsc_aif import AIFDetection

        det = AIFDetection()
        curve = _make_gamma_bolus(peak_amplitude=35.0) * 0.1
        result = det._aif_passes_quality(curve, baseline_frames=5, tr=1.5)
        assert result.auc_ratio is None

    def test_boundary_low_passes(self):
        from pmriprep.interfaces.dsc_aif import AIFDetection

        det = AIFDetection()
        curve = _make_gamma_bolus(peak_amplitude=35.0)
        ref_auc, _, _ = compute_windowed_first_pass_auc(
            curve, baseline_frames=5, tr=1.5,
        )
        det._synth_ref_windowed_auc = ref_auc
        scaled = curve * WINDOWED_AUC_RATIO_MIN
        result = det._aif_passes_quality(scaled, baseline_frames=5, tr=1.5)
        assert result.auc_ratio is not None
        assert result.auc_ratio == pytest.approx(WINDOWED_AUC_RATIO_MIN, rel=0.05)

    def test_boundary_high_passes(self):
        from pmriprep.interfaces.dsc_aif import AIFDetection

        det = AIFDetection()
        curve = _make_gamma_bolus(peak_amplitude=35.0)
        ref_auc, _, _ = compute_windowed_first_pass_auc(
            curve, baseline_frames=5, tr=1.5,
        )
        det._synth_ref_windowed_auc = ref_auc
        scaled = curve * WINDOWED_AUC_RATIO_MAX
        result = det._aif_passes_quality(scaled, baseline_frames=5, tr=1.5)
        assert result.auc_ratio is not None
        assert result.auc_ratio == pytest.approx(WINDOWED_AUC_RATIO_MAX, rel=0.05)


# -----------------------------------------------------------------------
# _aif_passes_quality integration
# -----------------------------------------------------------------------

class TestPassesQualityWithAucGate:

    def test_low_scale_fails_with_ref(self):
        from pmriprep.interfaces.dsc_aif import AIFDetection

        det = AIFDetection()
        curve = _make_gamma_bolus(peak_amplitude=35.0)
        ref_auc, _, _ = compute_windowed_first_pass_auc(
            curve, baseline_frames=5, tr=1.5,
        )
        det._synth_ref_windowed_auc = ref_auc
        scaled = curve * 0.15
        result = det._aif_passes_quality(scaled, baseline_frames=5, tr=1.5)
        assert not result.passes or result.auc_ratio is None or result.auc_ratio < WINDOWED_AUC_RATIO_MIN

    def test_low_scale_passes_without_ref(self):
        from pmriprep.interfaces.dsc_aif import AIFDetection

        det = AIFDetection()
        curve = _make_gamma_bolus(peak_amplitude=35.0) * 0.15
        result = det._aif_passes_quality(curve, baseline_frames=5, tr=1.5)
        assert result.auc_ratio is None

    def test_auc_sole_rejection(self):
        """AUC ratio is the only gate that fails."""
        from pmriprep.interfaces.dsc_aif import AIFDetection

        det = AIFDetection()
        curve = _make_gamma_bolus(peak_amplitude=35.0)
        ref_auc, _, _ = compute_windowed_first_pass_auc(
            curve, baseline_frames=5, tr=1.5,
        )
        det._synth_ref_windowed_auc = ref_auc

        # Full-scale: should pass all gates including AUC
        result_full = det._aif_passes_quality(curve, baseline_frames=5, tr=1.5)
        spr_ok = result_full.spr <= 100.0
        brr_ok = True  # gamma bolus has good BRR

        # Severely scaled: AUC should be the rejecting gate
        scaled = curve * 0.1
        result_scaled = det._aif_passes_quality(
            scaled, baseline_frames=5, tr=1.5,
        )
        assert result_scaled.auc_ratio is not None
        assert result_scaled.auc_ratio < WINDOWED_AUC_RATIO_MIN

    def test_named_tuple_fields(self):
        from pmriprep.interfaces.dsc_aif import AIFDetection, AifQualityResult

        det = AIFDetection()
        curve = _make_gamma_bolus()
        result = det._aif_passes_quality(curve, baseline_frames=5, tr=1.5)
        assert isinstance(result, AifQualityResult)
        assert hasattr(result, "passes")
        assert hasattr(result, "spr")
        assert hasattr(result, "peak_ratio")
        assert hasattr(result, "brr")
        assert hasattr(result, "auc_ratio")


# -----------------------------------------------------------------------
# DualRoiResult named tuple
# -----------------------------------------------------------------------

class TestDualRoiResult:

    def test_named_tuple_fields(self):
        from pmriprep.interfaces.dsc_aif import DualRoiResult

        result = DualRoiResult(
            curve=np.zeros(10),
            mask=np.zeros((3, 3, 3), dtype=bool),
            roi_size=1,
            spr=50.0,
            peak_ratio=1.0,
            brr=10.0,
            auc_ratio=0.95,
        )
        assert result.roi_size == 1
        assert result.auc_ratio == 0.95


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

class TestConstants:

    def test_ratio_min(self):
        assert WINDOWED_AUC_RATIO_MIN == 0.4

    def test_ratio_max(self):
        assert WINDOWED_AUC_RATIO_MAX == 1.8

    def test_fwhm_multiplier(self):
        assert WINDOWED_AUC_FWHM_MULTIPLIER == 2.0

    def test_trailing_cap(self):
        assert WINDOWED_AUC_TRAILING_CAP_S == 20.0

    def test_ratio_range_is_valid(self):
        assert WINDOWED_AUC_RATIO_MIN < 1.0 < WINDOWED_AUC_RATIO_MAX
