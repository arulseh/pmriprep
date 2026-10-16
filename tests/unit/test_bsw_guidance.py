"""BSW guidance opt-in parity and independent uncertainty algebra checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from pmriprep.utils import dsc_preload as candidate
from pmriprep.utils.bsw_guidance import (
    BSWGuidanceStatus as Status,
    build_bsw_guidance,
)



def inputs(dtype: Any = np.float64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a small reference-selection fixture with unequal signed effects."""
    rng = np.random.default_rng(230923)
    curve = np.zeros(30)
    curve[6:13] = [0.5, 1, 2, 3, 2, 1, 0.5]
    rows = curve + rng.normal(0, 0.015, (360, 30))
    rows += np.linspace(-0.5, 0.5, 360)[:, None]
    integral = cumulative_trapezoid(curve, initial=0)
    rows[-2] += 0.02 * integral
    rows[-1] -= 0.02 * integral
    signal = np.full_like(rows, 1000)
    signal[:, :5] += [-2, 2, -2, 2, 0]
    signal[300:330, -10:] = 1020
    mask = np.ones(360, bool)
    mask[340:345] = False
    return (rows.astype(dtype).reshape(12, 30, 1, 30),
            signal.astype(dtype).reshape(12, 30, 1, 30), mask.reshape(12, 30, 1))


def assert_correction_equal(left: Any, right: Any) -> None:
    """Require identical correction arrays, scalar flags and QC, including missing values."""
    for field in ("concentration_corrected", "k2_map", "auc_tissue"):
        np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
    assert left.correction_applied == right.correction_applied
    assert left.method == right.method
    assert left.qc_metrics.keys() == right.qc_metrics.keys()
    for key in left.qc_metrics:
        np.testing.assert_equal(left.qc_metrics[key], right.qc_metrics[key])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("use_mask", [True, False])
@pytest.mark.parametrize("baseline", [2, 5, 40])
def test_correction_and_qc_are_identical(
    dtype: Any, use_mask: bool, baseline: int,
) -> None:
    """Opt-in guidance leaves correction arrays and existing QC unchanged."""
    concentration, signal, mask = inputs(dtype)
    originals = [value.copy() for value in (concentration, signal, mask)]
    kwargs: dict[str, Any] = dict(brain_mask=mask if use_mask else None, signal=signal,
                  baseline_frames=baseline, tr=1.5)
    before = candidate.apply_k2_correction(concentration, **kwargs)
    plain = candidate.apply_k2_correction(concentration, **kwargs)
    after = candidate.apply_k2_correction(concentration, **kwargs, return_guidance=True)
    assert_correction_equal(before, plain)
    assert_correction_equal(before, after)
    assert plain.guidance is None
    assert after.guidance is not None
    for value, saved in zip((concentration, signal, mask), originals, strict=True):
        np.testing.assert_array_equal(value, saved)
    np.testing.assert_array_equal(after.guidance.k2, before.k2_map)
    assert after.guidance.baseline_frames == min(max(baseline, 4), 28)


def test_exact_membership_and_independent_covariance() -> None:
    """Use the actual selected rows, preserving all temporal covariance terms."""
    concentration, signal, mask = inputs()
    centered = concentration - concentration[..., :5].mean(axis=-1, keepdims=True)
    result = candidate.apply_k2_correction(
        concentration, signal=signal, brain_mask=mask, baseline_frames=5,
        tr=1.3, return_guidance=True,
    )
    guidance = result.guidance
    assert guidance is not None and guidance.reference_basis is not None
    flat_mask = mask.copy().reshape(-1)
    flat_mask[300:330] = False
    expected_mask = flat_mask.reshape(mask.shape)
    np.testing.assert_array_equal(guidance.reference_mask, expected_mask)
    assert guidance.reference_count == 325
    expected = candidate._compute_bsw_reference(centered, signal, mask, 5)
    np.testing.assert_array_equal(guidance.reference_curve, expected)
    samples = centered[expected_mask]
    integrals = cumulative_trapezoid(samples, dx=1.3, axis=1, initial=0)
    expected_covariance = np.cov(integrals, rowvar=False, ddof=1) / len(samples)
    np.testing.assert_allclose(guidance.reference_basis @ guidance.reference_basis.T,
                               expected_covariance, rtol=2e-11, atol=2e-17)
    np.testing.assert_array_equal(guidance.reference_integral,
                                   cumulative_trapezoid(expected, dx=1.3, initial=0))
    np.testing.assert_array_equal(guidance.reference_basis[0], 0)
    assert guidance.reference_reason is None
    assert guidance.available[mask].all()
    assert np.all(guidance.status[~mask] == Status.OUTSIDE_SUPPORT)


def test_standard_error_matches_independent_regression() -> None:
    """Conditional OLS variance is computed without changing the signed mean."""
    concentration, signal, mask = inputs()
    result = candidate.apply_k2_correction(
        concentration, signal=signal, brain_mask=mask, tr=2.3, return_guidance=True,
    )
    g = result.guidance
    assert g is not None
    assert g.reference_curve is not None and g.reference_integral is not None
    y = concentration - concentration[..., :5].mean(axis=-1, keepdims=True)
    design = np.column_stack((g.reference_curve, -g.reference_integral))
    inverse_gram = np.linalg.inv(design.T @ design)
    beta = inverse_gram @ design.T @ y[mask].T
    residual = y[mask].T - design @ beta
    expected_sd = np.sqrt(np.sum(residual ** 2, axis=0) / 28 * inverse_gram[1, 1])
    np.testing.assert_allclose(g.k2_sd[mask], expected_sd, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(g.k2[mask], beta[1], rtol=1e-10, atol=1e-14)
    assert g.k2[-1, -2, 0] < 0 < g.k2[-1, -1, 0]
    assert g.degrees_of_freedom == 28


@pytest.mark.parametrize("tr_scale,amplitude", [(2., 1.), (1., 10.), (0.5, 0.1)])
def test_acquisition_scaling(tr_scale: float, amplitude: float) -> None:
    """Time changes inverse-second coefficients; common amplitude changes covariance."""
    concentration, signal, mask = inputs()
    kwargs: dict[str, Any] = dict(signal=signal, brain_mask=mask, return_guidance=True)
    one = candidate.apply_k2_correction(concentration, tr=1., **kwargs).guidance
    two = candidate.apply_k2_correction(concentration * amplitude, tr=tr_scale, **kwargs).guidance
    assert one is not None and two is not None
    assert one.reference_basis is not None and two.reference_basis is not None
    assert one.reference_integral is not None and two.reference_integral is not None
    np.testing.assert_allclose(two.k2[mask], one.k2[mask] / tr_scale, atol=1e-14)
    np.testing.assert_allclose(two.k2_sd[mask], one.k2_sd[mask] / tr_scale, rtol=1e-11)
    np.testing.assert_allclose(two.reference_integral,
                               one.reference_integral * amplitude * tr_scale, atol=1e-13)
    np.testing.assert_allclose(two.reference_basis @ two.reference_basis.T,
                               (one.reference_basis @ one.reference_basis.T) *
                               (amplitude * tr_scale) ** 2, atol=1e-15)


@pytest.mark.parametrize("failure", ["brain", "selected", "tail"])
def test_missing_reference_preserves_legacy_skip(failure: str) -> None:
    """Skipped guidance is explicit without changing the existing correction skip."""
    concentration, signal, mask = inputs()
    if failure == "brain":
        mask.reshape(-1)[200:] = False
    elif failure == "selected":
        signal.reshape(360, 30)[:200, -10:] = 1100
    else:
        concentration, signal = concentration[..., :12], signal[..., :12]
    kwargs: dict[str, Any] = dict(signal=signal, brain_mask=mask)
    before = candidate.apply_k2_correction(concentration, **kwargs)
    after = candidate.apply_k2_correction(concentration, **kwargs, return_guidance=True)
    assert_correction_equal(before, after)
    assert after.guidance is not None
    assert not after.guidance.available.any()
    assert after.guidance.reference_basis is None
    assert after.guidance.reference_reason == before.qc_metrics["k2_reference_failure_reason"]


def test_supplied_curve_does_not_invent_reference_uncertainty() -> None:
    """A single reference mean still permits correction, but not the guided model."""
    concentration, signal, mask = inputs()
    centered = concentration - concentration[..., :5].mean(axis=-1, keepdims=True)
    curve = candidate._compute_bsw_reference(centered, signal, mask)
    kwargs: dict[str, Any] = dict(brain_mask=mask, reference_curve=curve)
    before = candidate.apply_k2_correction(concentration, **kwargs)
    after = candidate.apply_k2_correction(concentration, **kwargs, return_guidance=True)
    assert_correction_equal(before, after)
    assert after.guidance is not None
    assert after.guidance.reference_reason == "reference_samples_unavailable"
    assert after.guidance.reference_mask is None and after.guidance.reference_basis is None
    assert not after.guidance.available.any()


def test_supplied_coefficients_do_not_acquire_regression_se() -> None:
    """A supplied coefficient map has no declared conditional regression provenance."""
    concentration, signal, mask = inputs()
    coefficients = np.full(mask.shape, 0.1)
    kwargs: dict[str, Any] = dict(signal=signal, brain_mask=mask, k2_map=coefficients)
    before = candidate.apply_k2_correction(concentration, **kwargs)
    after = candidate.apply_k2_correction(concentration, **kwargs, return_guidance=True)
    assert_correction_equal(before, after)
    assert after.guidance is not None
    assert np.all(after.guidance.status[mask] == Status.SUPPLIED_COEFFICIENT)
    assert np.isnan(after.guidance.k2_sd).all()


def test_rank_deficiency_remains_missing() -> None:
    """Reference presence does not make a singular two-coefficient design usable."""
    concentration, signal, mask = inputs()
    concentration[:] = 0
    kwargs: dict[str, Any] = dict(signal=signal, brain_mask=mask)
    before = candidate.apply_k2_correction(concentration, **kwargs)
    after = candidate.apply_k2_correction(concentration, **kwargs, return_guidance=True)
    assert_correction_equal(before, after)
    assert after.guidance is not None
    assert after.guidance.reference_reason == "rank_deficient_reference"
    assert not after.guidance.available.any()


@pytest.mark.parametrize("value,expected", [(0., Status.STANDARD_ERROR_UNAVAILABLE),
                                            (np.nan, Status.NONFINITE_TISSUE),
                                            (np.inf, Status.NONFINITE_TISSUE)])
def test_zero_error_and_nonfinite_tissue_have_no_invented_prior(
    value: float, expected: Status,
) -> None:
    """An exactly zero residual or invalid curve cannot provide a positive SE."""
    concentration, signal, mask = inputs()
    concentration[-1, -1, 0] = value
    signal[-1, -1, 0, -10:] = 1100
    with np.errstate(invalid="ignore"):
        result = candidate.apply_k2_correction(
            concentration, signal=signal, brain_mask=mask, return_guidance=True,
        )
    assert result.guidance is not None
    assert result.guidance.status[-1, -1, 0] == expected
    assert np.isnan(result.guidance.k2_sd[-1, -1, 0])
    assert not result.guidance.available[-1, -1, 0]


def test_reference_membership_cannot_be_substituted() -> None:
    """Reference uncertainty must belong to the exact correction reference."""
    concentration, signal, mask = inputs()
    centered = concentration - concentration[..., :5].mean(axis=-1, keepdims=True)
    result = candidate.apply_k2_correction(
        concentration, signal=signal, brain_mask=mask, return_guidance=True,
    )
    assert result.guidance is not None
    with pytest.raises(ValueError, match="membership"):
        build_bsw_guidance(
            centered, result.k2_map, tr=1.5, baseline_frames=5, brain_mask=mask,
            reference_curve=result.guidance.reference_curve, reference_mask=mask,
        )


@pytest.mark.parametrize("tr", [0., -1., np.nan, np.inf])
def test_invalid_acquisition_is_rejected(tr: float) -> None:
    """Guidance has no arbitrary default for invalid acquisition seconds."""
    concentration, signal, mask = inputs()
    with pytest.raises(ValueError, match="TR"):
        candidate.apply_k2_correction(
            concentration, signal=signal, brain_mask=mask, tr=tr, return_guidance=True,
        )


def test_zero_empirical_variance_is_not_missing_reference() -> None:
    """Identical measured references have a zero-rank covariance, not no samples."""
    concentration = np.zeros((256, 1, 1, 24))
    concentration[..., 6:10] = [1, 3, 2, 1]
    signal = 1000 * np.exp(-0.03 * concentration)
    result = candidate.apply_k2_correction(
        concentration, signal=signal, brain_mask=np.ones((256, 1, 1), bool),
        tr=1., return_guidance=True,
    )
    assert result.guidance is not None
    assert result.guidance.reference_reason is None
    assert result.guidance.reference_count == 256
    assert result.guidance.reference_basis is not None
    assert result.guidance.reference_basis.shape == (24, 0)


def test_repository_dro_correction_remains_identical() -> None:
    """Exercise the existing BSW DRO without adding any CPI fitting."""
    from tests.dro.conftest import generate_k2_dro_phantom

    phantom = generate_k2_dro_phantom()
    kwargs: dict[str, Any] = dict(
        signal=phantom.signal, brain_mask=phantom.brain_mask,
        tr=phantom.dt, baseline_frames=10,
    )
    before = candidate.apply_k2_correction(phantom.concentration, **kwargs)
    after = candidate.apply_k2_correction(phantom.concentration, **kwargs, return_guidance=True)
    assert_correction_equal(before, after)
    assert after.guidance is not None and after.guidance.reference_reason is None
    assert after.guidance.available[phantom.brain_mask].all()


@pytest.mark.parametrize("invalid", ["mask_shape", "mask_type", "coefficient_shape",
                                     "baseline", "membership"])
def test_guidance_input_contract_is_explicit(invalid: str) -> None:
    """Inconsistent geometry/provenance fails rather than silently changing support."""
    concentration, signal, mask = inputs()
    centered = concentration - concentration[..., :5].mean(axis=-1, keepdims=True)
    fitted = candidate.apply_k2_correction(
        concentration, signal=signal, brain_mask=mask, return_guidance=True,
    )
    assert fitted.guidance is not None
    kwargs: dict[str, Any] = dict(
        k2=fitted.k2_map, tr=1.5, baseline_frames=5, brain_mask=mask,
        reference_curve=fitted.guidance.reference_curve,
        reference_mask=fitted.guidance.reference_mask,
    )
    if invalid == "mask_shape":
        kwargs["brain_mask"] = mask[:1]
    elif invalid == "mask_type":
        kwargs["brain_mask"] = mask.astype(np.uint8)
    elif invalid == "coefficient_shape":
        kwargs["k2"] = fitted.k2_map[:1]
    elif invalid == "baseline":
        kwargs["baseline_frames"] = True
    else:
        kwargs["reference_mask"] = mask
    with pytest.raises(ValueError):
        build_bsw_guidance(centered, **kwargs)
