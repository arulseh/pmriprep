"""Unit tests for DSC physics constants."""

import pytest

from pmriprep.utils.dsc_constants import (
    CBF_BOUNDS,
    DEFAULT_OI_THRESHOLD,
    DEFAULT_OSVD_GRID_STEPS,
    DEFAULT_SVD_THRESHOLD,
    DSC_DETECTION_LEVELS,
    EPSILON_CONCENTRATION,
    EPSILON_DELTA_R2,
    EPSILON_SIGNAL,
    BIDIRECTIONAL_K2_ABSOLUTE_SAFETY_BOUND,
    MTT_BOUNDS,
    calculate_k_h,
    get_r2_star,
    get_r2_star_tissue,
    is_valid_detection_level,
    split_detection_level,
)


def test_k2_absolute_safety_bound_is_positive() -> None:
    """The signed K2 numerical envelope has a positive absolute limit."""
    assert BIDIRECTIONAL_K2_ABSOLUTE_SAFETY_BOUND == 0.2


class TestR2StarTissue:
    """Tests for field-strength-dependent R2* tissue relaxivity lookup."""

    def test_r2_star_tissue_known_fields(self) -> None:
        """Known field strengths return Kjølby 2006 / Calamante 2007 values."""
        assert get_r2_star_tissue(1.5) == 87.0
        assert get_r2_star_tissue(3.0) == 174.0
        assert get_r2_star_tissue(7.0) == 400.0

    def test_r2_star_tissue_interpolation(self) -> None:
        """Midpoint between 1.5T and 3.0T interpolates linearly."""
        midpoint = (1.5 + 3.0) / 2.0  # 2.25T
        result = get_r2_star_tissue(midpoint)
        expected = 87.0 + (174.0 - 87.0) * (2.25 - 1.5) / (3.0 - 1.5)
        assert result == pytest.approx(expected)

    def test_r2_star_tissue_clamping(self) -> None:
        """Field strengths outside known range clamp to boundary values."""
        # Below minimum → clamp to 1.5T value
        assert get_r2_star_tissue(0.5) == 87.0
        # Above maximum → clamp to 7.0T value
        assert get_r2_star_tissue(10.0) == 400.0

    def test_r2_star_legacy_alias(self) -> None:
        """get_r2_star is a legacy alias for get_r2_star_tissue."""
        assert get_r2_star(3.0) == get_r2_star_tissue(3.0)
        assert get_r2_star(1.5) == get_r2_star_tissue(1.5)
        assert get_r2_star(7.0) == get_r2_star_tissue(7.0)


class TestCalculateKH:
    """Tests for the K_H calibration factor."""

    def test_calculate_k_h_standard(self) -> None:
        """Default parameters produce expected K_H value."""
        expected = (1.04 / 1.05) * (1.0 / (1.0 - 0.42)) * 100.0
        result = calculate_k_h()
        assert result == pytest.approx(expected, rel=1e-6)

    def test_calculate_k_h_custom_hematocrit(self) -> None:
        """Custom hematocrit alters K_H appropriately."""
        expected = (1.04 / 1.05) * (1.0 / (1.0 - 0.45)) * 100.0
        result = calculate_k_h(hematocrit=0.45)
        assert result == pytest.approx(expected, rel=1e-6)


class TestParameterBounds:
    """Tests for physiological parameter bounds."""

    def test_cbf_bounds_valid(self) -> None:
        """CBF bounds are (0.0, 1.0) with min < max (relative s⁻¹ units)."""
        assert CBF_BOUNDS.min == 0.0
        assert CBF_BOUNDS.max == 2.0
        assert CBF_BOUNDS.min < CBF_BOUNDS.max

    def test_mtt_bounds_valid(self) -> None:
        """MTT bounds are (0.0, 30.0) with min < max."""
        assert MTT_BOUNDS.min == 0.0
        assert MTT_BOUNDS.max == 30.0
        assert MTT_BOUNDS.min < MTT_BOUNDS.max


class TestDefaultThresholds:
    """Tests for default deconvolution thresholds."""

    def test_default_thresholds(self) -> None:
        """SVD and OI thresholds have expected values."""
        assert DEFAULT_SVD_THRESHOLD == 0.2
        assert DEFAULT_OI_THRESHOLD == 0.095
        assert DEFAULT_OSVD_GRID_STEPS == 50

class TestEpsilonGuards:
    """Tests for numerical epsilon guards."""

    def test_epsilon_guards_positive(self) -> None:
        """All epsilon guards are positive and non-zero."""
        assert EPSILON_SIGNAL > 0
        assert EPSILON_DELTA_R2 > 0
        assert EPSILON_CONCENTRATION > 0


class TestSplitDetectionLevel:
    """Tests for the centralized compound detection_level splitter."""

    def test_single(self) -> None:
        assert split_detection_level("atlas_strict") == ["atlas_strict"]

    def test_compound(self) -> None:
        assert split_detection_level("atlas_relaxed+atlas_permissive") == [
            "atlas_relaxed", "atlas_permissive",
        ]

    def test_empty_string(self) -> None:
        assert split_detection_level("") == []

    def test_none(self) -> None:
        assert split_detection_level(None) == []  # type: ignore[arg-type]


class TestIsValidDetectionLevel:
    """Tests for the centralized compound detection_level validator."""

    def test_single_valid(self) -> None:
        assert is_valid_detection_level("atlas_strict") is True

    def test_compound_valid(self) -> None:
        assert is_valid_detection_level("atlas_relaxed+atlas_permissive") is True

    def test_all_canonical_levels_valid(self) -> None:
        for level in DSC_DETECTION_LEVELS:
            assert is_valid_detection_level(level) is True, f"{level!r} should be valid"

    def test_invalid(self) -> None:
        assert is_valid_detection_level("bogus") is False

    def test_partial_invalid(self) -> None:
        assert is_valid_detection_level("atlas_strict+bogus") is False

    def test_empty(self) -> None:
        assert is_valid_detection_level("") is False
