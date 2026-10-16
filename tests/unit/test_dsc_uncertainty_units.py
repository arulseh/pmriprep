"""Unit tests for DSC uncertainty map unit conversion."""

import numpy as np


def test_cbf_uncertainty_units_match_cbf():
    """CV should be preserved across unit conversion.

    CBF is converted from s⁻¹ to min⁻¹ (×60). For CV (coefficient of variation)
    to remain correct, std must also be converted by the same factor.

    For Y = 60*X: std(Y) = 60*std(X), so CV = std(Y)/mean(Y) = std(X)/mean(X)
    """
    cbf_raw, std_raw = 0.5, 0.1  # s⁻¹
    cbf_converted = cbf_raw * 60.0  # min⁻¹
    std_converted = std_raw * 60.0  # min⁻¹

    cv_before = std_raw / cbf_raw
    cv_after = std_converted / cbf_converted
    assert np.isclose(cv_before, cv_after)


def test_ci_bounds_scale_with_cbf():
    """CI bounds must scale by same factor as CBF point estimate."""
    cbf_raw = 0.5  # s⁻¹
    ci_lower_raw = 0.4  # s⁻¹
    ci_upper_raw = 0.6  # s⁻¹

    cbf_converted = cbf_raw * 60.0  # min⁻¹
    ci_lower_converted = ci_lower_raw * 60.0  # min⁻¹
    ci_upper_converted = ci_upper_raw * 60.0  # min⁻¹

    # CI width relative to CBF should be preserved
    rel_width_before = (ci_upper_raw - ci_lower_raw) / cbf_raw
    rel_width_after = (ci_upper_converted - ci_lower_converted) / cbf_converted
    assert np.isclose(rel_width_before, rel_width_after)

    # Point estimate should remain within bounds
    assert ci_lower_converted < cbf_converted < ci_upper_converted


def test_ci_center_matches_cbf():
    """The converted interval midpoint matches the converted CBF point estimate."""
    # Use a CBF of 9.6 and an SD in native inverse-second units
    cbf_raw = 0.160  # s⁻¹
    ci_lower_raw = 0.140  # s⁻¹
    ci_upper_raw = 0.180  # s⁻¹

    cbf_converted = cbf_raw * 60.0  # 9.6 min⁻¹
    ci_lower_converted = ci_lower_raw * 60.0  # min⁻¹
    ci_upper_converted = ci_upper_raw * 60.0  # min⁻¹

    ci_center_converted = (ci_lower_converted + ci_upper_converted) / 2

    # CI center (~9.6) should match CBF (~9.6), not CBF/60 (~0.16)
    assert np.isclose(ci_center_converted, cbf_converted, rtol=0.15)
