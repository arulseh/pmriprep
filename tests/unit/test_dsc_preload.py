"""Unit tests for DSC preload detection and classification."""

import numpy as np
import pytest

from pmriprep.utils.dsc_preload import (
    PreloadClass,
    classify_preload,
    get_preload_thresholds,
    _compute_ees_concentration,
)
from pmriprep.utils.dsc_constants import (
    KEP_DSC_DEFAULT,
    KEP_DSC_LOWER,
    KEP_DSC_UPPER,
)


class TestClassifyPreload:
    """Tests for classify_preload."""

    def test_classify_preload_none(self) -> None:
        """Low baseline delta_r2 at 3T classifies as NONE."""
        assert classify_preload(0.1, field_strength=3.0) == PreloadClass.NONE

    def test_classify_preload_minimal(self) -> None:
        """delta_r2=1.0 at 3T classifies as MINIMAL (0.5 <= 1.0 < 2.0)."""
        assert classify_preload(1.0, field_strength=3.0) == PreloadClass.MINIMAL

    def test_classify_preload_half_dose(self) -> None:
        """delta_r2=3.0 at 3T classifies as HALF_DOSE (2.0 <= 3.0 < 5.0)."""
        assert classify_preload(3.0, field_strength=3.0) == PreloadClass.HALF_DOSE

    def test_classify_preload_significant(self) -> None:
        """delta_r2=7.0 at 3T classifies as SIGNIFICANT (5.0 <= 7.0 < 10.0)."""
        assert classify_preload(7.0, field_strength=3.0) == PreloadClass.SIGNIFICANT

    def test_classify_preload_severe(self) -> None:
        """delta_r2=15.0 at 3T classifies as SEVERE (>= 10.0)."""
        assert classify_preload(15.0, field_strength=3.0) == PreloadClass.SEVERE

    def test_classify_preload_boundary(self) -> None:
        """At exact threshold (0.5 at 3T), classify as MINIMAL not NONE."""
        assert classify_preload(0.5, field_strength=3.0) == PreloadClass.MINIMAL

    def test_classify_preload_negative(self) -> None:
        """Negative delta_r2 classifies as NONE (below all thresholds)."""
        assert classify_preload(-0.1, field_strength=3.0) == PreloadClass.NONE


class TestGetPreloadThresholds:
    """Tests for get_preload_thresholds."""

    def test_get_preload_thresholds_3t(self) -> None:
        """3T thresholds match expected values."""
        t = get_preload_thresholds(3.0)
        assert t.none_max == 0.5
        assert t.minimal_max == 2.0
        assert t.half_dose_max == 5.0
        assert t.significant_max == 10.0

    def test_get_preload_thresholds_field_scaling(self) -> None:
        """Thresholds increase monotonically with field strength."""
        t_15 = get_preload_thresholds(1.5)
        t_30 = get_preload_thresholds(3.0)
        t_70 = get_preload_thresholds(7.0)

        assert t_15.none_max < t_30.none_max < t_70.none_max
        assert t_15.minimal_max < t_30.minimal_max < t_70.minimal_max
        assert t_15.half_dose_max < t_30.half_dose_max < t_70.half_dose_max
        assert t_15.significant_max < t_30.significant_max < t_70.significant_max


class TestPreloadClassEnum:
    """Tests for PreloadClass enum values."""

    def test_preload_class_enum_values(self) -> None:
        """PreloadClass enum members have expected string values."""
        assert PreloadClass.NONE.value == "none"
        assert PreloadClass.MINIMAL.value == "minimal"
        assert PreloadClass.HALF_DOSE.value == "half_dose"
        assert PreloadClass.SIGNIFICANT.value == "significant"
        assert PreloadClass.SEVERE.value == "severe"


class TestKepConstants:
    """Tests for KEP DSC constants."""

    def test_kep_default_in_range(self) -> None:
        """KEP_DSC_DEFAULT should be within valid range."""
        assert KEP_DSC_LOWER <= KEP_DSC_DEFAULT <= KEP_DSC_UPPER

    def test_kep_default_matches_leu_2016(self) -> None:
        """KEP_DSC_DEFAULT should match Leu 2016 GBM mean (~0.005 s⁻¹)."""
        assert KEP_DSC_DEFAULT == pytest.approx(0.005, rel=0.01)

    def test_kep_bounds_order(self) -> None:
        """Lower bound should be less than upper bound."""
        assert KEP_DSC_LOWER < KEP_DSC_UPPER


class TestComputeEESConcentration:
    """Tests for _compute_ees_concentration() helper."""

    def test_output_shape(self) -> None:
        """Output should match input shape."""
        c_bar = np.sin(np.linspace(0, np.pi, 60))
        c_e = _compute_ees_concentration(c_bar, kep=0.005, tr=1.5)
        assert c_e.shape == c_bar.shape

    def test_output_dtype(self) -> None:
        """Output should be float64."""
        c_bar = np.sin(np.linspace(0, np.pi, 60))
        c_e = _compute_ees_concentration(c_bar, kep=0.005, tr=1.5)
        assert c_e.dtype == np.float64

    def test_zero_input_gives_zero_output(self) -> None:
        """Zero input should give zero output."""
        c_bar = np.zeros(60)
        c_e = _compute_ees_concentration(c_bar, kep=0.005, tr=1.5)
        assert np.allclose(c_e, 0)

    def test_zero_kep_approaches_cumsum(self) -> None:
        """When kep=0, exp(-kep*t)=1, so convolution gives cumsum."""
        c_bar = np.ones(10)  # Step function
        tr = 1.0

        # kep=0 gives exp_kernel=1 everywhere, so convolution = cumsum * tr
        c_e = _compute_ees_concentration(c_bar, kep=0.0, tr=tr)
        expected = np.cumsum(c_bar) * tr

        np.testing.assert_allclose(c_e, expected, rtol=1e-10)

    def test_positive_input_gives_positive_output(self) -> None:
        """Positive concentration should give positive C_e."""
        c_bar = np.abs(np.sin(np.linspace(0, np.pi, 60)))
        c_e = _compute_ees_concentration(c_bar, kep=0.005, tr=1.5)
        assert np.all(c_e >= 0)

    def test_high_kep_causes_faster_decay(self) -> None:
        """Higher kep should cause faster decay after bolus."""
        # Single pulse input
        c_bar = np.exp(-((np.arange(60) - 20) ** 2) / 50)
        tr = 1.5

        c_e_low = _compute_ees_concentration(c_bar, kep=0.001, tr=tr)
        c_e_high = _compute_ees_concentration(c_bar, kep=0.05, tr=tr)

        # At late timepoints, high kep should decay more
        assert c_e_high[-1] < c_e_low[-1], (
            "High kep should give lower C_e at late timepoints"
        )
