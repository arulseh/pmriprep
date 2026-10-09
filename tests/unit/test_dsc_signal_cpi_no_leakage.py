"""Leakage-disabled signal fitting must not depend on BSW availability."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from pmriprep.models.dsc.local_vi import local_vi_state
from pmriprep.models.dsc.batched_fitting import DSCBatchedFitResult
from pmriprep.models.dsc.vi_adapter import DSCVIPosterior
from pmriprep.models.dsc.signal_cpi import connection, leakage, spatial
from pmriprep.models.dsc.signal_cpi.residue import signal_evaluator
from pmriprep.models.dsc.signal_cpi.geometry import Model
from pmriprep.models.dsc.signal_cpi.policy import automatic_count, model_for
from pmriprep.utils.bsw_guidance import BSWGuidance


@pytest.fixture
def inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, BSWGuidance]:
    """Supply measured signals and optional guidance on the same grid."""
    time = np.arange(30.) * 3
    u = np.maximum(time - 9, 0)
    aif = 10 * u**2 * np.exp(-u / 3)
    reference = .2 * u**2 * np.exp(-u / 5)
    integral = cumulative_trapezoid(reference, time, initial=0)
    te = .03
    signals = (100 * np.exp(-te * np.stack([reference, 1.1 * reference])))[:, None, None]
    guidance = BSWGuidance(reference, integral, .001 * integral[:, None],
        np.ones((2, 1, 1), bool), 2, None, np.array([-.01, .01])[:, None, None],
        np.full((2, 1, 1), .005), np.zeros((2, 1, 1), np.uint8), 4, 3., 28)
    return signals, aif, time, te, guidance


@pytest.mark.parametrize("family", ["cpi", "cpi_vtf"])
def test_none_has_independent_mask_count_and_no_bsw_inputs(inputs: Any, family: Model) -> None:
    """Use measured-signal support/count without reference or leakage dimensions."""
    signals, aif, time, te, guidance = inputs
    mask = np.array([True, False])[:, None, None]
    prepared = connection.prepare_acquisition(
        signals, aif, time, te, leakage_method="none", mask=mask,
        baseline_frames=4, family=family,
    )
    expected = automatic_count(signals[:1].reshape(1, -1), aif, time, te,
                               baseline_frames=4, family=family)
    assert prepared.decision == expected
    assert prepared.leakage_method == "none"
    assert prepared.reference is None and prepared.reference_reason is None
    assert prepared.guidance_status is None
    assert np.all(np.isnan(prepared.k2)) and np.all(np.isnan(prepared.k2_sd))
    np.testing.assert_array_equal(prepared.fit_support, [True, False])
    assert not prepared.fit_support.flags.writeable
    mask[:] = True
    np.testing.assert_array_equal(prepared.fit_support, [True, False])
    # Even unusable supplied guidance cannot alter a leakage-disabled fit.
    unavailable = replace(guidance, reference_reason="no_reference",
                          k2=np.full((1,), np.nan), tr=np.nan)
    other = connection.prepare_acquisition(
        signals, aif, time, te, unavailable, leakage_method="none",
        mask=np.array([True, False])[:, None, None], baseline_frames=4, family=family,
    )
    assert other.decision == prepared.decision
    np.testing.assert_array_equal(other.signals, prepared.signals)


@pytest.mark.parametrize("scale", [1e-8, 1e8])
def test_acquisition_wide_signal_scale_is_removed_before_inference(
    inputs: Any,
    scale: float,
) -> None:
    """Arbitrary magnitude units cannot change the CPI fitting problem."""
    signals, aif, time, te, _ = inputs
    reference = connection.prepare_acquisition(
        signals,
        aif,
        time,
        te,
        leakage_method="none",
        baseline_frames=4,
        control_points=5,
        infer_delay=False,
    )
    scaled = connection.prepare_acquisition(
        signals * scale,
        aif,
        time,
        te,
        leakage_method="none",
        baseline_frames=4,
        control_points=5,
        infer_delay=False,
    )
    assert reference.signal_scale == pytest.approx(1.0)
    assert scaled.signal_scale == pytest.approx(scale)
    assert scaled.decision == reference.decision
    np.testing.assert_allclose(scaled.signals, reference.signals, rtol=2e-15, atol=0)


def test_signal_scale_preserves_fit_and_restores_prediction_units(inputs: Any) -> None:
    """Quantitative readouts are invariant while saved fits use input units."""
    signals, aif, time, te, _ = inputs

    def fit(values: np.ndarray) -> tuple[connection.GuidedAcquisition, connection.GuidedVoxel]:
        prepared = connection.prepare_acquisition(
            values,
            aif,
            time,
            te,
            leakage_method="none",
            baseline_frames=4,
            control_points=5,
            infer_delay=False,
        )
        return prepared, list(connection.fit_voxels(prepared, [0]))[0]

    reference_acquisition, reference = fit(signals)
    scaled_acquisition, scaled = fit(signals * 1e8)
    assert scaled.status == reference.status
    np.testing.assert_allclose(
        [scaled.cbf, scaled.cbv, scaled.mtt, scaled.cbf_std],
        [reference.cbf, reference.cbv, reference.mtt, reference.cbf_std],
        rtol=2e-10,
        atol=1e-12,
    )
    assert reference.fitted_signal is not None and scaled.fitted_signal is not None
    np.testing.assert_allclose(
        scaled.fitted_signal / scaled_acquisition.signal_scale,
        reference.fitted_signal / reference_acquisition.signal_scale,
        rtol=2e-10,
        atol=1e-10,
    )
    assert float(np.median(scaled.fitted_signal[:4])) > 1e9


def test_bsw_still_requires_guidance(inputs: Any) -> None:
    """Missing guidance is not silently interpreted as leakage disabled."""
    with pytest.raises(ValueError, match="guidance"):
        connection.prepare_acquisition(*inputs[:4])


@pytest.mark.parametrize("method", ["off", "auto", None])
def test_unknown_leakage_policy_raises(inputs: Any, method: Any) -> None:
    with pytest.raises(ValueError, match="leakage_method"):
        connection.prepare_acquisition(*inputs, leakage_method=method)


@pytest.mark.parametrize("mask", [np.ones((1, 1, 1), bool), np.ones((2, 1, 1))])
def test_none_validates_mask_geometry_and_type(inputs: Any, mask: np.ndarray) -> None:
    with pytest.raises(ValueError, match="mask"):
        connection.prepare_acquisition(*inputs[:4], leakage_method="none", mask=mask)


@pytest.mark.parametrize("baseline", [0, 30, True, 3.5])
def test_none_validates_baseline_window(inputs: Any, baseline: Any) -> None:
    with pytest.raises(ValueError, match="baseline"):
        connection.prepare_acquisition(*inputs[:4], leakage_method="none",
                                       baseline_frames=baseline)


@pytest.mark.parametrize("bad_value", [0., -1., np.nan, np.inf])
def test_invalid_signal_does_not_enter_fit_or_count(inputs: Any, bad_value: float) -> None:
    signals, aif, time, te, _ = inputs
    signals[0, 0, 0, 8] = bad_value
    prepared = connection.prepare_acquisition(signals, aif, time, te, leakage_method="none")
    assert prepared.valid_signal.tolist() == [False, True]
    output = list(connection.fit_voxels(prepared, [0]))[0]
    assert output.status == "fallback_requested"
    assert output.reasons == ("invalid_measured_signal",)
    assert np.isnan(output.bsw_k2) and np.isnan(output.fitted_k2)


def test_empty_mask_is_not_a_missing_reference_failure(inputs: Any) -> None:
    prepared = connection.prepare_acquisition(*inputs[:4], leakage_method="none",
                                               mask=np.zeros((2, 1, 1), bool))
    assert prepared.decision is None
    assert list(connection.fit_voxels(prepared)) == []
    excluded = list(connection.fit_voxels(prepared, [1]))[0]
    assert excluded.status == "outside_support"
    assert excluded.reasons == ("outside_fit_support",)
    result = spatial.fit_spatial_voxels(replace(prepared, inference_method="svb"))
    assert result.voxels == () and result.local_calls == 0


@pytest.mark.parametrize("family", ["cpi", "cpi_vtf"])
def test_none_dispatches_vascular_model_to_same_engine(
    inputs: Any, family: Model, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No hidden leakage/reference coordinates survive removal of the term."""
    signals, aif, time, te, _ = inputs
    model = model_for(signals[0, 0, 0], aif, time, te, 5, family=family)
    evaluator = signal_evaluator(model)

    class Captured(Exception):
        pass

    def capture(**kwargs: Any) -> None:
        p = 13 if family == "cpi_vtf" else 11
        assert len(kwargs["fitting_bounds"]) == p
        assert kwargs["n_iterations"] == 100
        starts = kwargs["initial_means_transformed"]
        np.testing.assert_array_equal(starts, model.starts())
        np.testing.assert_array_equal(kwargs["prior_means_nat"], model.prior.mean)
        np.testing.assert_array_equal(kwargs["prior_stds_nat"],
                                       np.sqrt(np.diag(model.prior.covariance)))
        np.testing.assert_allclose(kwargs["model_adapter"].predict(starts),
                                   evaluator.predictions(starts), rtol=1e-12)
        np.testing.assert_allclose(kwargs["model_adapter"].jacobian(starts),
                                   evaluator.jacobians(starts), rtol=1e-12)
        assert kwargs["return_full_posterior"]
        raise Captured

    monkeypatch.setattr(leakage, "batched_dsc_vi_fit", capture)
    with pytest.raises(Captured):
        leakage.fit_without_leakage(signals[0, 0, 0], aif, time, te, 5, family=family)


def test_all_bounded_mean_failures_request_fallback(
    inputs: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed restart rows reach CPI policy without evaluating invalid states."""
    signals, aif, time, te, _ = inputs

    def failed_result(**kwargs: Any) -> DSCBatchedFitResult:
        starts = np.asarray(kwargs["initial_means_transformed"])
        n_rows, n_params = starts.shape
        posterior = DSCVIPosterior(
            np.full_like(starts, np.nan),
            np.full((n_rows, n_params, n_params), np.nan),
            np.full(n_rows, np.nan), np.full(n_rows, np.nan),
        )
        return DSCBatchedFitResult(
            params={"cbf": np.full(n_rows, np.nan)},
            r_squared=np.full(n_rows, np.nan),
            converged=np.zeros(n_rows, dtype=bool),
            fitted_curves=np.full((n_rows, len(time)), np.nan),
            n_iterations=1, model_name="signal_cpi",
            full_posterior=posterior,
            failure_reasons=("bounded_mean_failed",) * n_rows,
        )

    monkeypatch.setattr(leakage, "batched_dsc_vi_fit", failed_result)
    diagnostic = leakage.fit_without_leakage(
        signals[0, 0, 0], aif, time, te, 5,
    )
    output = connection.classify_result(0, np.nan, diagnostic)
    assert diagnostic.fits == []
    assert output.status == "fallback_requested"
    assert output.reasons == ("bounded_mean_failed",)


@pytest.mark.parametrize("family", ["cpi", "cpi_vtf"])
def test_no_guidance_dispatch_preserves_voxel_ownership(
    inputs: Any, family: Model, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = connection.prepare_acquisition(*inputs[:4], leakage_method="none", family=family)
    seen = []

    def capture(signal: np.ndarray, aif: np.ndarray, time: np.ndarray, te: float,
                count: int, **kwargs: Any) -> Any:
        seen.append((signal.copy(), count, kwargs["family"]))
        return None

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("BSW fit must not run with leakage disabled")

    monkeypatch.setattr(connection, "fit_without_leakage", capture)
    monkeypatch.setattr(connection, "fit_with_prior", forbidden)
    monkeypatch.setattr(connection, "classify_result", lambda i, k2, result:
                        connection.GuidedVoxel(i, "retained", ()))
    assert [r.flat_index for r in connection.fit_voxels(prepared, [1, 0])] == [1, 0]
    assert prepared.decision is not None
    for row, index in zip(seen, [1, 0], strict=True):
        np.testing.assert_array_equal(row[0], prepared.signals[index])
        assert row[1:] == (prepared.decision.selected, family)


@pytest.mark.parametrize("family", ["cpi", "cpi_vtf"])
@pytest.mark.parametrize("converged", [False, True])
def test_none_readout_keeps_valid_perfusion_without_inventing_k2(
    inputs: Any, family: Model, converged: bool,
) -> None:
    """Covariance checks still apply, but there are no leakage factors to check."""
    signals, aif, time, te, _ = inputs
    model = model_for(signals[0, 0, 0], aif, time, te, 5, family=family)
    mean = model.starts()[0]
    evaluator = signal_evaluator(model)
    prediction = evaluator.predictions(mean[None])
    jacobian = evaluator.jacobians(mean[None])
    observed = prediction[0] + np.random.default_rng(83).normal(0, .01, len(time))
    prior = model.prior
    precision = np.asarray(np.linalg.inv(prior.covariance), dtype=np.float64)
    state = local_vi_state(mean[None], jacobian, observed[None] - prediction,
                           prior.mean, precision, np.ones(len(time)))
    fit, assessment, k2 = leakage.readout_state(
        model, observed, mean, state.covariance[0], state.shape, state.scale[0],
        prediction[0], jacobian[0], 5, 0, converged,
    )
    assert assessment.valid and k2 == {}
    arrays = dict(covariance=state.covariance, fitted_curves=prediction)
    result = leakage.LeakageBatch([fit], [assessment],
        [{"selected": fit, "assessment": assessment}], [], arrays, 0., 1, "none")
    output = connection.classify_result(5, .05, result)
    assert output.status == ("retained" if converged else "provisional")
    assert output.leakage_method == "none"
    assert np.isnan(output.bsw_k2) and np.isnan(output.fitted_k2)
    assert output.cbf == mean[0] and np.isfinite(output.cbf_std)
    assert output.cbv == pytest.approx(output.cbf * output.mtt)
    assert output.assessment is not None and output.assessment.voxel == 5


def test_none_actual_vi_and_spatial_fits_without_reference(inputs: Any) -> None:
    """Exercise real VI and spatial sweeps, not just argument dispatch."""
    weights = np.linspace(.5, 1., len(inputs[2]))
    prepared = connection.prepare_acquisition(
        *inputs[:4], leakage_method="none", temporal_weights=weights,
        baseline_frames=4,
    )
    assert prepared.decision is not None
    n_parameters = 2 * prepared.decision.selected + 1
    ordinary = list(connection.fit_voxels(prepared, [0]))[0]
    assert ordinary.status in ("retained", "provisional")
    assert ordinary.diagnostics is not None
    assert ordinary.diagnostics.arrays["mean"].shape[1] == n_parameters
    assert ordinary.diagnostics.leakage == []
    assert "reference_basis" not in ordinary.diagnostics.arrays
    assert np.isnan(ordinary.fitted_k2) and np.isnan(ordinary.bsw_k2)
    np.testing.assert_array_equal(ordinary.diagnostics.arrays["weights"], weights)
    result = spatial.fit_spatial_voxels(replace(prepared, inference_method="svb"),
                                        keep_diagnostics=True)
    np.testing.assert_array_equal(result.spatial_indices, [0, 1])
    assert result.spatial_sweeps > 0
    assert result.posterior_means is not None
    assert result.posterior_means.shape == (2, n_parameters)
    for row in result.voxels:
        assert row.status in ("retained", "provisional")
        assert row.leakage_method == "none"
        assert np.isnan(row.bsw_k2) and np.isnan(row.fitted_k2)
        assert row.diagnostics is not None and row.diagnostics.leakage == []
        assert row.assessment is not None and row.assessment.valid
        assert row.assessment.voxel == row.flat_index
        assert row.cbf > 0 and row.mtt > 0
        assert row.cbv == pytest.approx(row.cbf * row.mtt)
        assert row.diagnostics.arrays["covariance"].shape == (1, n_parameters, n_parameters)


@pytest.mark.parametrize("field", ["time", "te", "baseline"])
def test_no_bsw_does_not_bypass_acquisition_checks(inputs: Any, field: str) -> None:
    signals, aif, time, te, _ = inputs
    baseline: int | None = None
    if field == "time":
        time = time.copy()
        time[3] += .2
    elif field == "te":
        te = 0.
    else:
        baseline = 30
    with pytest.raises(ValueError):
        connection.prepare_acquisition(signals, aif, time, te, leakage_method="none",
                                       baseline_frames=baseline)


def test_bsw_unavailable_still_requests_fallback(inputs: Any) -> None:
    """Only an explicit none policy removes the reference requirement."""
    signals, aif, time, te, guidance = inputs
    guidance = replace(guidance, status=np.full((2, 1, 1), 2, np.uint8),
                        reference_reason="insufficient_reference_samples",
                        reference_curve=None, reference_integral=None, reference_basis=None)
    prepared = connection.prepare_acquisition(signals, aif, time, te, guidance)
    results = list(connection.fit_voxels(prepared))
    assert len(results) == 2
    for row in results:
        assert row.leakage_method == "bsw" and row.status == "fallback_requested"
        assert row.reasons == ("bsw_reference_unavailable", "insufficient_reference_samples")


def test_bsw_baseline_override_must_agree(inputs: Any) -> None:
    with pytest.raises(ValueError, match="baseline"):
        connection.prepare_acquisition(*inputs, baseline_frames=5)


def test_none_propagates_engine_errors(inputs: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A library defect is not a missing-reference fallback or a missing voxel."""
    prepared = connection.prepare_acquisition(*inputs[:4], leakage_method="none")

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("engine defect")

    monkeypatch.setattr(leakage, "batched_dsc_vi_fit", fail)
    with pytest.raises(RuntimeError, match="engine defect"):
        list(connection.fit_voxels(prepared, [0]))
