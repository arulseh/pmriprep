"""Unit tests for DCE physics constants."""

import pytest

from pmriprep.utils.dce_constants import (
    EPSILON_T1,
    EPSILON_VE,
    FAHRAEUS_RATIO,
    HEMATOCRIT,
    HEMATOCRIT_ARTERIAL,
    capillary_hematocrit,
    KTRANS_BOUNDS,
    MAX_KEP,
    VE_BOUNDS,
    VP_BOUNDS,
    compute_fwhm_threshold_frames,
    compute_timing_margin_frames,
    get_aif_thresholds,
    get_b1_thresholds,
    get_blood_t1,
    get_r1_relaxivity,
    get_tissue_t1_ranges,
    is_low_temporal_resolution,
)


class TestR1Relaxivity:
    """Tests for field-strength-dependent R1 relaxivity lookup."""

    def test_r1_relaxivity_known_fields(self) -> None:
        """Known field strengths return catalogued values."""
        assert get_r1_relaxivity(1.5) == 3.9
        assert get_r1_relaxivity(3.0) == 3.8
        assert get_r1_relaxivity(7.0) == 3.5

    def test_r1_relaxivity_interpolation(self) -> None:
        """Midpoint between 1.5T and 3.0T interpolates linearly."""
        midpoint = (1.5 + 3.0) / 2.0  # 2.25T
        result = get_r1_relaxivity(midpoint)
        expected = (3.9 + 3.8) / 2.0  # 3.85
        assert result == pytest.approx(expected)

    def test_r1_relaxivity_clamping(self) -> None:
        """Values outside known range clamp to boundary values."""
        assert get_r1_relaxivity(0.5) == 3.9   # clamped to 1.5T
        assert get_r1_relaxivity(10.0) == 3.5  # clamped to 7.0T


class TestBloodT1:
    """Tests for field-strength-dependent blood T1 lookup."""

    def test_blood_t1_known_fields(self) -> None:
        """Known field strengths return catalogued values."""
        assert get_blood_t1(1.5) == 1.35
        assert get_blood_t1(3.0) == 1.65
        assert get_blood_t1(7.0) == 2.10

    def test_blood_t1_interpolation(self) -> None:
        """Intermediate field strength interpolates between neighbors."""
        result = get_blood_t1(2.25)
        expected = 1.35 + 0.5 * (1.65 - 1.35)  # midpoint = 1.50
        assert result == pytest.approx(expected)


class TestTissueT1Ranges:
    """Tests for tissue T1 QC ranges."""

    def test_tissue_t1_ranges_3t(self) -> None:
        """At 3.0T: WM [800,1000], GM [1200,1600], CSF [3500,5500]."""
        ranges = get_tissue_t1_ranges(3.0)
        assert ranges.white_matter.min == 800
        assert ranges.white_matter.max == 1000
        assert ranges.gray_matter.min == 1200
        assert ranges.gray_matter.max == 1600
        assert ranges.csf.min == 3500
        assert ranges.csf.max == 5500
        # Verify min < max for all tissue types
        assert ranges.white_matter.min < ranges.white_matter.max
        assert ranges.gray_matter.min < ranges.gray_matter.max
        assert ranges.csf.min < ranges.csf.max

    def test_tissue_t1_ranges_nearest(self) -> None:
        """Field between 1.5T and 3.0T returns nearest known ranges."""
        # 2.0T is closer to 1.5T than 3.0T
        ranges_near_1_5 = get_tissue_t1_ranges(2.0)
        ranges_1_5 = get_tissue_t1_ranges(1.5)
        assert ranges_near_1_5 == ranges_1_5

        # 2.5T is closer to 3.0T than 1.5T
        ranges_near_3 = get_tissue_t1_ranges(2.5)
        ranges_3 = get_tissue_t1_ranges(3.0)
        assert ranges_near_3 == ranges_3


class TestAIFThresholds:
    """Tests for AIF quality thresholds."""

    def test_aif_thresholds_strict(self) -> None:
        """Strict criteria returns expected values."""
        t = get_aif_thresholds("strict")
        assert t.washout_min == 0.20
        assert t.fwhm_min_seconds == 8.0
        assert t.fwhm_max_seconds == 25.0
        assert t.baseline_frames == 5

    def test_aif_thresholds_relaxed(self) -> None:
        """Relaxed criteria has looser thresholds than strict."""
        strict = get_aif_thresholds("strict")
        relaxed = get_aif_thresholds("relaxed")
        assert relaxed.washout_min < strict.washout_min
        assert relaxed.fwhm_min_seconds < strict.fwhm_min_seconds
        assert relaxed.fwhm_max_seconds > strict.fwhm_max_seconds

    def test_aif_thresholds_tr_adaptive(self) -> None:
        """TR > 5s adapts FWHM minimum to max(tr*0.8, base_fwhm_min)."""
        t = get_aif_thresholds("strict", tr=10.0)
        # max(10.0 * 0.8, 8.0) = max(8.0, 8.0) = 8.0
        assert t.fwhm_min_seconds == pytest.approx(8.0)

        # TR=20s: max(20*0.8, 8.0) = max(16.0, 8.0) = 16.0
        t2 = get_aif_thresholds("strict", tr=20.0)
        assert t2.fwhm_min_seconds == pytest.approx(16.0)

    def test_aif_thresholds_invalid_criteria(self) -> None:
        """Unknown criteria raises ValueError."""
        with pytest.raises(ValueError, match="Unknown AIF criteria"):
            get_aif_thresholds("unknown")


class TestParameterBounds:
    """Tests for pharmacokinetic parameter bounds."""

    def test_parameter_bounds_valid(self) -> None:
        """All parameter bounds have min < max and reasonable ranges."""
        assert KTRANS_BOUNDS.min < KTRANS_BOUNDS.max
        assert KTRANS_BOUNDS.min >= 0.0

        assert VE_BOUNDS.min < VE_BOUNDS.max
        assert VE_BOUNDS.min >= 0.0
        assert VE_BOUNDS.max <= 1.0

        assert VP_BOUNDS.min < VP_BOUNDS.max
        assert VP_BOUNDS.min >= 0.0
        assert VP_BOUNDS.max <= 1.0


class TestConstantValues:
    """Tests for specific constant values."""

    def test_constants_values(self) -> None:
        """Verify key constant values match expected."""
        assert HEMATOCRIT == 0.42
        assert HEMATOCRIT_ARTERIAL == 0.42
        assert FAHRAEUS_RATIO == 0.85
        assert abs(capillary_hematocrit() - 0.357) < 1e-6
        assert abs(capillary_hematocrit(0.30) - 0.255) < 1e-6  # anemia
        assert EPSILON_T1 == 0.01
        assert MAX_KEP == 1000.0


class TestB1Thresholds:
    """Tests for B1+ field-strength thresholds."""

    def test_b1_thresholds_known_fields(self) -> None:
        """Known field strengths return catalogued B1 thresholds."""
        t = get_b1_thresholds(3.0)
        assert t.clip_low == 0.2
        assert t.clip_high == 2.0
        assert t.dce_clip_low == 0.5
        assert t.dce_clip_high == 1.5

    def test_b1_thresholds_nearest(self) -> None:
        """Intermediate field returns nearest known thresholds."""
        # 2.0T closer to 1.5T
        assert get_b1_thresholds(2.0) == get_b1_thresholds(1.5)
        # 5.5T closer to 7.0T
        assert get_b1_thresholds(5.5) == get_b1_thresholds(7.0)


class TestTRAdaptiveFunctions:
    """Tests for TR-adaptive timing and FWHM functions."""

    def test_compute_timing_margin_frames(self) -> None:
        """Timing margin adapts to TR."""
        # TR=2s: round(4.0/2.0) = 2
        assert compute_timing_margin_frames(2.0) == 2
        # TR=4s: round(4.0/4.0) = 1
        assert compute_timing_margin_frames(4.0) == 1
        # TR=0 or negative: returns 1
        assert compute_timing_margin_frames(0.0) == 1

    def test_compute_fwhm_threshold_frames(self) -> None:
        """FWHM threshold is min of frame-based and seconds-based."""
        # TR=2s strict: min(5, 25/2) = min(5, 12.5) = 5
        assert compute_fwhm_threshold_frames(2.0, strict=True) == pytest.approx(5.0)
        # TR=8s strict: min(5, 25/8) = min(5, 3.125) = 3.125
        assert compute_fwhm_threshold_frames(8.0, strict=True) == pytest.approx(3.125)

    def test_is_low_temporal_resolution(self) -> None:
        """TR > 5s is flagged as low temporal resolution."""
        assert is_low_temporal_resolution(6.0) is True
        assert is_low_temporal_resolution(5.0) is False
        assert is_low_temporal_resolution(2.0) is False
