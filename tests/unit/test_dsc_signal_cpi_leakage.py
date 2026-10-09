"""Plain/dispersed signal CPI must retain the same signed leakage contract."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from pmriprep.models.dsc.local_vi import LocalVIFactorError, local_vi_state
from pmriprep.models.dsc.signal_cpi import connection, leakage, spatial
from pmriprep.models.dsc.signal_cpi.geometry import Model
from pmriprep.models.dsc.signal_cpi.policy import model_for
from pmriprep.utils.bsw_guidance import BSWGuidance


@pytest.fixture
def acquisition() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, BSWGuidance]:
    time = np.arange(30.) * 3
    u = np.maximum(time - 9, 0)
    aif = 10 * u**2 * np.exp(-u / 3)
    reference = .2 * u**2 * np.exp(-u / 5)
    integral = cumulative_trapezoid(reference, time, initial=0)
    te = .03
    signals = (100 * np.exp(-te * np.stack([reference, 1.1 * reference])))[:, None, None]
    guidance = BSWGuidance(reference, integral, np.zeros((30, 0)),
        np.ones((2, 1, 1), bool), 2, None, np.array([-.01, .01])[:, None, None],
        np.full((2, 1, 1), .005), np.zeros((2, 1, 1), np.uint8), 4, 3., 28)
    return signals, aif, time, te, guidance


@pytest.mark.parametrize("family", ["cpi", "cpi_vtf"])
@pytest.mark.parametrize("k2", [-.01, 0., .01])
def test_signed_leakage_composition_and_derivatives(acquisition: Any, family: Model, k2: float) -> None:
    signals, aif, time, te, guidance = acquisition
    reference = leakage.LeakageReference(time, guidance.reference_integral,
        .01 * guidance.reference_integral[:, None], "empirical_iid_region_mean", 32)
    vascular = model_for(signals[0, 0, 0], aif, time, te, 5, family=family)
    model = leakage.BSWCenteredModel(vascular, reference, leakage.BSWPrior(k2, .004))
    mean = model.starts()[0]
    # Starts remain data-derived; explicitly probe both leakage signs and nuisances.
    mean[model.rho_index] = k2 * te * reference.scale
    mean[-1] = .3
    expected = vascular.predict(mean[:model.rho_index]) * np.exp(
        te * k2 * (reference.mean + .3 * reference.basis[:, 0]))
    np.testing.assert_allclose(model.predict(mean), expected, rtol=1e-14)
    evaluator = leakage.LeakageBatchEvaluator(model)
    np.testing.assert_allclose(evaluator.predictions(mean[None])[0], expected, rtol=1e-9)
    jacobian = evaluator.jacobians(mean[None])[0]
    # Independent two-sided differences include the gamma coordinates, not just K2.
    for i in range(len(mean)):
        step = 1e-5 * max(abs(mean[i]), .001)
        left, right = mean.copy(), mean.copy()
        left[i] -= step
        right[i] += step
        expected_column = (model.predict(right) - model.predict(left)) / (2 * step)
        np.testing.assert_allclose(jacobian[:, i], expected_column, rtol=2e-4, atol=2e-5)
    assert model.rho_index == (13 if family == "cpi_vtf" else 11)


@pytest.mark.parametrize("family", ["cpi", "cpi_vtf"])
def test_automatic_count_and_voxel_dispatch(acquisition: Any, family: Model, monkeypatch: Any) -> None:
    prepared = connection.prepare_acquisition(*acquisition, family=family)
    assert prepared.family == family and prepared.decision is not None
    extra = 4 if family == "cpi_vtf" else 2
    assert all(2 * n + extra < len(prepared.time) for n in prepared.decision.allowed)
    seen = []

    def capture(signal: np.ndarray, aif: np.ndarray, time: np.ndarray, te: float,
                count: int, reference: Any, prior: Any, **kwargs: Any) -> Any:
        seen.append((count, prior.k2, kwargs["family"]))
        return None

    monkeypatch.setattr(connection, "fit_with_prior", capture)
    monkeypatch.setattr(connection, "classify_result", lambda i, k2, result:
                        connection.GuidedVoxel(i, "retained", ()))
    assert [r.flat_index for r in connection.fit_voxels(prepared, [1, 0])] == [1, 0]
    assert seen == [(prepared.decision.selected, .01, family),
                    (prepared.decision.selected, -.01, family)]


def test_bsw_guided_preparation_uses_the_same_acquisition_scale(acquisition: Any) -> None:
    """BSW guidance remains unchanged when arbitrary signal units are removed."""
    signals, aif, time, te, guidance = acquisition
    reference = connection.prepare_acquisition(*acquisition, control_points=5)
    scaled = connection.prepare_acquisition(
        signals * 1e8,
        aif,
        time,
        te,
        guidance,
        control_points=5,
    )
    assert scaled.signal_scale == pytest.approx(reference.signal_scale * 1e8)
    np.testing.assert_allclose(scaled.signals, reference.signals, rtol=2e-15, atol=0)
    np.testing.assert_array_equal(scaled.k2, reference.k2)
    np.testing.assert_array_equal(scaled.k2_sd, reference.k2_sd)
    assert scaled.decision == reference.decision


def test_local_factor_failure_requests_voxel_fallback(
    acquisition: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A data-dependent local SVD failure cannot abort the whole CPI map."""
    prepared = connection.prepare_acquisition(*acquisition)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise LocalVIFactorError("forced local factor failure")

    monkeypatch.setattr(connection, "fit_with_prior", fail)
    output = list(connection.fit_voxels(prepared, [0]))[0]
    assert output.status == "fallback_requested"
    assert output.reasons == ("local_factor_failed",)
    assert output.bsw_k2 == pytest.approx(-0.01)


def test_unexpected_cpi_error_still_propagates(
    acquisition: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback handling must not conceal programming errors."""
    prepared = connection.prepare_acquisition(*acquisition)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise TypeError("broken CPI interface")

    monkeypatch.setattr(connection, "fit_with_prior", fail)
    with pytest.raises(TypeError, match="broken CPI interface"):
        list(connection.fit_voxels(prepared, [0]))


def test_gamma_count_reserves_two_additional_coordinates(acquisition: Any) -> None:
    signals, aif, time, te, _ = acquisition
    plain = leakage.leakage_count(signals.reshape(2, 30)[:, :20], aif[:20], time[:20], te)
    gamma = leakage.leakage_count(signals.reshape(2, 30)[:, :20], aif[:20], time[:20], te,
                                  family="cpi_vtf")
    assert plain.allowed[-1] == 8 and gamma.allowed[-1] == 7


@pytest.mark.parametrize("family", ["bad", "gamma_aif"])
def test_invalid_family_cannot_silently_become_plain(acquisition: Any, family: Any) -> None:
    with pytest.raises(ValueError, match="family"):
        connection.prepare_acquisition(*acquisition, family=family)
    signals, aif, time, te, guidance = acquisition
    reference = leakage.LeakageReference(time, guidance.reference_integral,
                                        np.zeros((len(time), 0)), "conditional_reference", 0)
    with pytest.raises(ValueError, match="family"):
        leakage.fit_with_prior(signals[0, 0, 0], aif, time, te, 5, reference,
                               leakage.BSWPrior(0, .01), family=family)


@pytest.mark.parametrize("family", ["cpi", "cpi_vtf"])
def test_vi_solver_receives_combined_model(acquisition: Any, family: Model, monkeypatch: Any) -> None:
    signals, aif, time, te, guidance = acquisition
    reference = leakage.LeakageReference(time, guidance.reference_integral,
                                        np.zeros((len(time), 0)), "conditional_reference", 0)

    class Captured(Exception):
        pass

    def capture(**kwargs: Any) -> None:
        p = 14 if family == "cpi_vtf" else 12
        assert len(kwargs["fitting_bounds"]) == p
        assert kwargs["prior_means_nat"].shape == (p,)
        starts = kwargs["initial_means_transformed"]
        assert starts.shape[1] == p
        assert kwargs["model_adapter"].predict(starts).shape == (len(starts), len(time))
        assert kwargs["return_full_posterior"]
        raise Captured

    monkeypatch.setattr(leakage, "batched_dsc_vi_fit", capture)
    with pytest.raises(Captured):
        leakage.fit_with_prior(signals[0, 0, 0], aif, time, te, 5, reference,
                               leakage.BSWPrior(.01, .004), family=family)


@pytest.mark.parametrize("family", ["cpi", "cpi_vtf"])
def test_readout_is_intrinsic_and_spatial_model_keeps_family(
    acquisition: Any, family: Model, monkeypatch: Any,
) -> None:
    prepared = connection.prepare_acquisition(*acquisition, family=family, inference_method="svb")
    assert prepared.reference is not None and prepared.decision is not None
    count = prepared.decision.selected
    vascular = model_for(prepared.signals[0], prepared.aif, prepared.time, prepared.te,
                          count, family=family)
    model = leakage.BSWCenteredModel(vascular, prepared.reference, leakage.BSWPrior(.01, .005))
    mean = model.starts()[0]
    evaluator = leakage.LeakageBatchEvaluator(model)
    prediction = evaluator.predictions(mean[None])
    jacobian = evaluator.jacobians(mean[None])
    prior = model.prior
    state = local_vi_state(mean[None], jacobian, prepared.signals[:1] - prediction,
                            prior.mean, np.asarray(np.linalg.inv(prior.covariance), dtype=np.float64),
                            np.ones(len(prepared.time)))
    fit, assessment, k2 = leakage.readout_state(model, prepared.signals[0], mean,
        state.covariance[0], state.shape, state.scale[0], prediction[0], jacobian[0], count, 0, True)
    expected_mtt = model.residue(mean).integrate(
        0, model.duration)
    assert fit.mtt == pytest.approx(expected_mtt)
    assert fit.cbv == pytest.approx(fit.cbf * fit.mtt / 60)
    arrays = dict(prior_mean=prior.mean, prior_covariance=prior.covariance)
    diagnostic = leakage.LeakageBatch([fit], [assessment], [{"selected": fit}], [k2], arrays, 0, 1)
    row = connection.GuidedVoxel(0, "retained", (), diagnostics=diagnostic)
    monkeypatch.setattr(spatial, "fit_voxels", lambda a: iter([row, replace(row, flat_index=1)]))

    class Captured(Exception):
        pass

    def capture(**kwargs: Any) -> None:
        assert len(kwargs["fitting_bounds"]) == len(mean)
        np.testing.assert_allclose(kwargs["model_adapter"].predict(mean[None]), prediction)
        assert kwargs["local_fit"].means.shape == (2, len(mean))
        raise Captured

    monkeypatch.setattr(spatial, "batched_dsc_spatial_vi_fit", capture)
    with pytest.raises(Captured):
        spatial.fit_spatial_voxels(prepared)
