"""Unit tests for DSC preload detection and classification."""

import pytest

from pmriprep.utils.dsc_preload import (
    PreloadClass,
    classify_preload,
    get_preload_thresholds,
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
