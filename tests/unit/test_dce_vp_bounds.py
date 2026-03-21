"""Unit tests for vp_bounds soft constraint in DCE PK model fitting.

Tests that the vp_bounds parameter correctly constrains vp during NLLS
fitting without fixing it (soft constraint behavior).
"""

import numpy as np
import pytest

from pmriprep.models.dce.extended_tofts import ExtendedToftsModel
from pmriprep.models.dce.patlak import PatlakModel
from pmriprep.models.dce.tofts import ToftsModel
from pmriprep.models.dce.two_compartment_exchange import TwoCompartmentExchangeModel
from pmriprep.utils.dce_constants import VP_BOUNDS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _parker_like_aif(time: np.ndarray) -> np.ndarray:
    """Generate a Parker-like double-Gaussian AIF."""
    g1 = 5.0 * np.exp(-((time - 0.5) ** 2) / (2 * 0.15**2))
    g2 = 1.5 * np.exp(-((time - 1.5) ** 2) / (2 * 0.4**2))
    return g1 + g2


@pytest.fixture
def time_aif() -> tuple[np.ndarray, np.ndarray]:
    """Standard time vector and AIF for tests."""
    dt = 0.05
    time = np.arange(200) * dt
    aif = _parker_like_aif(time)
    return time, aif


# ---------------------------------------------------------------------------
# _apply_vp_bounds_override
# ---------------------------------------------------------------------------


class TestApplyVpBoundsOverride:
    """Test the base class helper for overriding vp bounds."""

    def test_overrides_vp_in_etofts(self) -> None:
        model = ExtendedToftsModel()
        bounds = model.get_bounds()
        original_vp = bounds[2]  # Ktrans, ve, vp
        assert original_vp == (VP_BOUNDS.min, VP_BOUNDS.max)

        new_bounds = model._apply_vp_bounds_override(bounds, (0.02, 0.04))
        assert new_bounds[2] == (0.02, 0.04)
        # Original list unchanged
        assert bounds[2] == original_vp

    def test_overrides_vp_in_patlak(self) -> None:
        model = PatlakModel()
        bounds = model.get_bounds()
        new_bounds = model._apply_vp_bounds_override(bounds, (0.01, 0.03))
        vp_idx = model.get_parameter_names().index("vp")
        assert new_bounds[vp_idx] == (0.01, 0.03)

    def test_overrides_vp_in_2cxm(self) -> None:
        model = TwoCompartmentExchangeModel()
        bounds = model.get_bounds()
        new_bounds = model._apply_vp_bounds_override(bounds, (0.005, 0.015))
        vp_idx = model.get_parameter_names().index("vp")
        assert new_bounds[vp_idx] == (0.005, 0.015)

    def test_no_op_for_tofts(self) -> None:
        model = ToftsModel()
        bounds = model.get_bounds()
        new_bounds = model._apply_vp_bounds_override(bounds, (0.02, 0.04))
        # Tofts has no vp — bounds unchanged
        assert new_bounds == bounds

    def test_does_not_mutate_input(self) -> None:
        model = ExtendedToftsModel()
        bounds = model.get_bounds()
        original = [b for b in bounds]
        model._apply_vp_bounds_override(bounds, (0.02, 0.04))
        assert bounds == original


# ---------------------------------------------------------------------------
# fit() with vp_bounds
# ---------------------------------------------------------------------------


class TestFitWithVpBounds:
    """Test that fit(vp_bounds=...) constrains vp within the given range."""

    def test_etofts_vp_within_bounds(
        self, time_aif: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Extended Tofts fit with vp_bounds should produce vp in range."""
        time, aif = time_aif
        model = ExtendedToftsModel()

        # Generate tissue curve with known vp=0.05
        true_params = [0.1, 0.2, 0.05]  # Ktrans, ve, vp
        tissue = model.evaluate(true_params, aif, time)

        # Fit with tight bounds around a DIFFERENT vp value (0.03 ± 20%)
        vp_bounds = (0.024, 0.036)
        result = model.fit(tissue, aif, time, vp_bounds=vp_bounds)

        fitted_vp = result.parameters["vp"]
        assert vp_bounds[0] <= fitted_vp <= vp_bounds[1], (
            f"vp={fitted_vp:.4f} outside bounds [{vp_bounds[0]}, {vp_bounds[1]}]"
        )

    def test_etofts_vp_not_fixed(
        self, time_aif: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """vp_bounds should allow vp to vary, not fix it."""
        time, aif = time_aif
        model = ExtendedToftsModel()

        # Generate tissue curve with vp=0.03
        true_params = [0.1, 0.2, 0.03]
        tissue = model.evaluate(true_params, aif, time)

        # Fit with bounds that include the true value
        vp_bounds = (0.02, 0.04)
        result = model.fit(tissue, aif, time, vp_bounds=vp_bounds)

        fitted_vp = result.parameters["vp"]
        # Should recover near the true value, not be stuck at a boundary
        assert abs(fitted_vp - 0.03) < 0.01, (
            f"vp={fitted_vp:.4f} should be near 0.03"
        )

    def test_etofts_without_vp_bounds_is_unconstrained(
        self, time_aif: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Without vp_bounds, vp should use full default range."""
        time, aif = time_aif
        model = ExtendedToftsModel()

        true_params = [0.1, 0.2, 0.15]  # vp outside typical soft range
        tissue = model.evaluate(true_params, aif, time)

        result = model.fit(tissue, aif, time)
        fitted_vp = result.parameters["vp"]
        # Should recover near 0.15, which would be outside a tight soft bound
        assert fitted_vp > 0.10, (
            f"vp={fitted_vp:.4f} should recover near 0.15 unconstrained"
        )

    def test_patlak_vp_within_bounds(
        self, time_aif: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Patlak fit with vp_bounds should produce vp in range."""
        time, aif = time_aif
        model = PatlakModel()

        true_params = [0.05, 0.04]  # Ktrans, vp
        tissue = model.evaluate(true_params, aif, time)

        vp_bounds = (0.02, 0.03)
        result = model.fit(tissue, aif, time, vp_bounds=vp_bounds)

        fitted_vp = result.parameters["vp"]
        assert vp_bounds[0] <= fitted_vp <= vp_bounds[1], (
            f"vp={fitted_vp:.4f} outside bounds [{vp_bounds[0]}, {vp_bounds[1]}]"
        )

    def test_tofts_ignores_vp_bounds(
        self, time_aif: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Tofts fit should work normally when vp_bounds is passed."""
        time, aif = time_aif
        model = ToftsModel()

        true_params = [0.1, 0.2]  # Ktrans, ve
        tissue = model.evaluate(true_params, aif, time)

        # Should not raise, vp_bounds is ignored
        result = model.fit(tissue, aif, time, vp_bounds=(0.02, 0.04))
        assert result.converged


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestVpBoundsEdgeCases:
    """Edge cases for vp_bounds handling."""

    def test_vp_bounds_at_vp_bounds_limits(self) -> None:
        """vp_bounds at the VP_BOUNDS limits should work."""
        model = ExtendedToftsModel()
        bounds = model.get_bounds()
        new_bounds = model._apply_vp_bounds_override(
            bounds, (VP_BOUNDS.min, VP_BOUNDS.max)
        )
        vp_idx = model.get_parameter_names().index("vp")
        assert new_bounds[vp_idx] == (VP_BOUNDS.min, VP_BOUNDS.max)

    def test_narrow_vp_bounds(
        self, time_aif: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Very narrow vp_bounds should still allow fitting."""
        time, aif = time_aif
        model = ExtendedToftsModel()

        true_params = [0.1, 0.2, 0.03]
        tissue = model.evaluate(true_params, aif, time)

        # Very narrow: ±5%
        vp_bounds = (0.0285, 0.0315)
        result = model.fit(tissue, aif, time, vp_bounds=vp_bounds)

        fitted_vp = result.parameters["vp"]
        assert vp_bounds[0] <= fitted_vp <= vp_bounds[1]
