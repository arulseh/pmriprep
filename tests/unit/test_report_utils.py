"""Unit tests for report utility functions.

Tests for QC metric classification helpers used in HTML report generation.
"""

import pytest

from pmriprep.utils.report_utils import (
    _classify_b1,
    _classify_metric,
    _classify_metric_inverse,
)


class TestClassifyMetric:
    """Tests for _classify_metric helper function."""

    def test_pass_above_warn_threshold(self) -> None:
        """Values >= warn_threshold return 'pass'."""
        assert _classify_metric(25.0, 20.0, 5.0) == "pass"

    def test_pass_at_warn_threshold(self) -> None:
        """Values exactly at warn_threshold return 'pass'."""
        assert _classify_metric(20.0, 20.0, 5.0) == "pass"

    def test_warn_between_thresholds(self) -> None:
        """Values >= min_threshold but < warn_threshold return 'warn'."""
        assert _classify_metric(10.0, 20.0, 5.0) == "warn"

    def test_warn_at_min_threshold(self) -> None:
        """Values exactly at min_threshold return 'warn'."""
        assert _classify_metric(5.0, 20.0, 5.0) == "warn"

    def test_poor_below_min_threshold(self) -> None:
        """Values < min_threshold return 'poor'."""
        assert _classify_metric(3.0, 20.0, 5.0) == "poor"

    def test_none_returns_na(self) -> None:
        """None values return 'na'."""
        assert _classify_metric(None, 20.0, 5.0) == "na"

    def test_nan_returns_na(self) -> None:
        """NaN values return 'na'."""
        assert _classify_metric(float("nan"), 20.0, 5.0) == "na"


class TestClassifyB1:
    """Tests for _classify_b1 helper function."""

    def test_pass_within_range(self) -> None:
        """Values within [low, high] return 'pass'."""
        assert _classify_b1(1.0, 0.7, 1.3) == "pass"

    def test_pass_at_low_boundary(self) -> None:
        """Values exactly at low boundary return 'pass'."""
        assert _classify_b1(0.7, 0.7, 1.3) == "pass"

    def test_pass_at_high_boundary(self) -> None:
        """Values exactly at high boundary return 'pass'."""
        assert _classify_b1(1.3, 0.7, 1.3) == "pass"

    def test_warn_below_low(self) -> None:
        """Values below low boundary return 'warn'."""
        assert _classify_b1(0.6, 0.7, 1.3) == "warn"

    def test_warn_above_high(self) -> None:
        """Values above high boundary return 'warn'."""
        assert _classify_b1(1.4, 0.7, 1.3) == "warn"

    def test_none_returns_na(self) -> None:
        """None values return 'na'."""
        assert _classify_b1(None, 0.7, 1.3) == "na"

    def test_nan_returns_na(self) -> None:
        """NaN values return 'na'."""
        assert _classify_b1(float("nan"), 0.7, 1.3) == "na"


class TestClassifyMetricInverse:
    """Tests for _classify_metric_inverse helper (lower is better)."""

    def test_pass_below_warn_threshold(self) -> None:
        """Values < warn_threshold return 'pass'."""
        assert _classify_metric_inverse(3.0, 5, 10) == "pass"

    def test_warn_at_warn_threshold(self) -> None:
        """Values exactly at warn_threshold return 'warn'."""
        assert _classify_metric_inverse(5.0, 5, 10) == "warn"

    def test_warn_between_thresholds(self) -> None:
        """Values >= warn but < poor return 'warn'."""
        assert _classify_metric_inverse(7.0, 5, 10) == "warn"

    def test_poor_at_poor_threshold(self) -> None:
        """Values exactly at poor_threshold return 'poor'."""
        assert _classify_metric_inverse(10.0, 5, 10) == "poor"

    def test_poor_above_poor_threshold(self) -> None:
        """Values >= poor_threshold return 'poor'."""
        assert _classify_metric_inverse(15.0, 5, 10) == "poor"

    def test_none_returns_na(self) -> None:
        """None values return 'na'."""
        assert _classify_metric_inverse(None, 5, 10) == "na"

    def test_nan_returns_na(self) -> None:
        """NaN values return 'na'."""
        assert _classify_metric_inverse(float("nan"), 5, 10) == "na"
