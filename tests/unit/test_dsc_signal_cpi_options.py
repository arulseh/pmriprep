"""Approved CPI choices preserve the model while controlling inferred coordinates."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from pmriprep.models.dsc.signal_cpi import acquisition, connection, leakage
from pmriprep.models.dsc.signal_cpi.conditioned import FixedDelaySignalCPI
from pmriprep.models.dsc.signal_cpi.geometry import Model
from pmriprep.models.dsc.signal_cpi.policy import model_for
from pmriprep.models.dsc.signal_cpi.residue import signal_evaluator
from pmriprep.models.dsc.signal_cpi.settings import CPIOptions
from tests.unit.test_dsc_signal_cpi_connection import selection


@pytest.fixture
def inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Supply a small, acquisition-valid signal field, not an accuracy benchmark."""
    time = np.arange(30.) * 3
    u = np.maximum(time - 9, 0)
    aif = 10 * u**2 * np.exp(-u / 3)
    tissue = .2 * u**2 * np.exp(-u / 5)
    te = .03
    signals = (100 * np.exp(-te * np.stack([tissue, 1.1 * tissue])))[:, None, None]
    return signals, aif, time, te


@pytest.mark.parametrize("kwargs", [
    {"control_points": True}, {"control_points": 5.5}, {"control_points": "6"},
    {"control_points": -1}, {"control_points": 4}, {"control_points": 13},
    {"infer_delay": 0}, {"infer_delay": "false"},
    {"dispersion": "off"}, {"fallback": "auto"},
])
def test_invalid_options_are_not_coerced(kwargs: dict[str, Any]) -> None:
    with pytest.raises((ValueError, TypeError)):
        CPIOptions(**kwargs)


def test_default_uses_plain_transport() -> None:
    assert CPIOptions().dispersion == "none"


@pytest.mark.parametrize("dispersion,families", [
    ("none", ("cpi",)), ("gamma", ("cpi_vtf",)), ("auto", ("cpi", "cpi_vtf")),
])
def test_options_prepare_counts_without_copying_the_image(
    inputs: Any, dispersion: Any, families: tuple[str, ...],
) -> None:
    result = acquisition.prepare_cpi_acquisition(
        *inputs, leakage_method="none", options=CPIOptions(control_points=5,
            infer_delay=False, dispersion=dispersion, fallback="none"),
    )
    assert tuple(a.family for a in result.branches) == families
    for branch in result.branches:
        assert not branch.infer_delay
        assert branch.decision is not None
        assert branch.decision.selected == branch.decision.requested == 5
        assert branch.decision.selection_method == "manual"
        assert branch.signals is result.branches[0].signals
        assert branch.reference is None


def test_manual_count_rejects_infeasible_acquisition(inputs: Any) -> None:
    signals, aif, time, te = inputs
    with pytest.raises(ValueError, match="infeasible"):
        acquisition.prepare_cpi_acquisition(signals[..., :20], aif[:20], time[:20], te,
            leakage_method="none", options=CPIOptions(control_points=12))


@pytest.mark.parametrize("family", ["cpi", "cpi_vtf"])
def test_fixed_delay_projects_prior_forward_model_and_jacobian(inputs: Any, family: Model) -> None:
    signals, aif, time, te = inputs
    full = model_for(signals[0, 0, 0], aif, time, te, 5, family=family)
    fixed = model_for(signals[0, 0, 0], aif, time, te, 5, family=family, infer_delay=False)
    assert isinstance(fixed, FixedDelaySignalCPI)
    reduced = np.asarray(fixed.starts())
    expanded = fixed.expand(reduced)
    assert np.all(expanded[:, full.delay_index] == 0)
    np.testing.assert_array_equal(fixed.prior.mean, full.prior.mean[fixed.keep])
    np.testing.assert_array_equal(fixed.prior.covariance,
        full.prior.covariance[np.ix_(fixed.keep, fixed.keep)])
    for row, restored in zip(reduced, expanded, strict=True):
        assert fixed.valid_mean(row) == full.valid_mean(restored)
        np.testing.assert_array_equal(fixed.residue(row).c, full.residue(restored).c)
        np.testing.assert_array_equal(fixed.predict(row), full.predict(restored))
    evaluator, expected = signal_evaluator(fixed), signal_evaluator(full)
    np.testing.assert_array_equal(evaluator.predictions(reduced), expected.predictions(expanded))
    np.testing.assert_array_equal(evaluator.jacobians(reduced),
                                   expected.jacobians(expanded)[:, :, fixed.keep])


@pytest.mark.parametrize("family", ["cpi", "cpi_vtf"])
def test_fixed_delay_keeps_bsw_and_reference_coordinates(inputs: Any, family: Model) -> None:
    """Removing timing must not shift a gamma, slope, K2 or nuisance coordinate."""
    signals, aif, time, te = inputs
    integral = cumulative_trapezoid(aif*.02, time, initial=0)
    reference = leakage.LeakageReference(time, integral, integral[:, None]*.01,
                                         "empirical_iid_region_mean", 32)
    prior = leakage.BSWPrior(.003, .004)
    full_vascular = model_for(signals[0, 0, 0], aif, time, te, 5, family=family)
    fixed_vascular = model_for(signals[0, 0, 0], aif, time, te, 5, family=family, infer_delay=False)
    full = leakage.BSWCenteredModel(full_vascular, reference, prior)
    fixed = leakage.BSWCenteredModel(fixed_vascular, reference, prior)
    mean = full.starts()[0]
    mean[full_vascular.delay_index] = 0
    mean[full.rho_index] = full.prior.mean[full.rho_index]
    mean[-1] = .3
    keep = np.delete(np.arange(mean.size), full_vascular.delay_index)
    reduced = mean[keep]
    np.testing.assert_array_equal(fixed.prior.mean, full.prior.mean[keep])
    np.testing.assert_array_equal(fixed.prior.covariance, full.prior.covariance[np.ix_(keep, keep)])
    np.testing.assert_array_equal(fixed.predict(reduced), full.predict(mean))
    actual, expected = leakage.LeakageBatchEvaluator(fixed), leakage.LeakageBatchEvaluator(full)
    np.testing.assert_array_equal(actual.predictions(reduced[None]), expected.predictions(mean[None]))
    np.testing.assert_array_equal(actual.jacobians(reduced[None]), expected.jacobians(mean[None])[:, :, keep])
    assert fixed.arrival_delay(reduced) == 0


@pytest.mark.parametrize("family", ["cpi", "cpi_vtf"])
def test_fixed_delay_real_fit_reports_missing_not_zero(inputs: Any, family: Model) -> None:
    signals, aif, time, te = inputs
    result = leakage.fit_without_leakage(signals[0, 0, 0], aif, time, te, 5,
                                         family=family, infer_delay=False)
    output = connection.classify_result(0, np.nan, result)
    assert output.status in ("retained", "provisional")
    assert np.isnan(output.delay)
    assert all(fit.delay == 0 for fit in result.fits)
    assert result.arrays["mean"].shape[1] == (10 if family == "cpi" else 12)
    assert result.arrays["covariance"].shape[1:] == (result.arrays["mean"].shape[1],) * 2


def branch(family: Model, score: float, *, converged: bool = True) -> connection.GuidedVoxel:
    """Create a fully assessed branch with a controlled ordinary-VI score."""
    diagnostic = selection(.01, converged=converged)
    fit = replace(diagnostic.fits[0], score=score)
    diagnostic.fits[0] = fit
    diagnostic.rows[0]["selected"] = fit
    return replace(connection.classify_result(1, .004, diagnostic), family=family)


@pytest.mark.parametrize("plain_score,gamma_score,winner", [(2., 1., "cpi"),
    (1., 2., "cpi_vtf"), (1., 1., "cpi")])
def test_selection_keeps_score_rule_and_nonconvergence_is_not_failure(
    plain_score: float, gamma_score: float, winner: str,
) -> None:
    plain, gamma = branch("cpi", plain_score, converged=False), branch("cpi_vtf", gamma_score)
    result = acquisition.choose_transport([gamma, plain])
    assert result.family == winner
    assert result.diagnostics is (plain if winner == "cpi" else gamma).diagnostics
    assert dict(result.selection_scores) == {"cpi": plain_score, "cpi_vtf": gamma_score}


def test_failed_branches_cannot_win_and_reasons_survive() -> None:
    bad = connection.GuidedVoxel(1, "fallback_requested", ("bad_signal",), family="cpi")
    good = branch("cpi_vtf", -100.)
    assert acquisition.choose_transport([bad, good]).family == "cpi_vtf"
    other = replace(bad, family="cpi_vtf", reasons=("bad_signal", "no_state"))
    result = acquisition.choose_transport([bad, other])
    assert result.family is None and result.status == "fallback_requested"
    assert result.reasons == ("bad_signal", "no_state")
    assert all(value is None for _, value in result.selection_scores)


def test_selection_pairs_voxels_without_spatially_fitting_each_branch(
    inputs: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = acquisition.prepare_cpi_acquisition(*inputs, leakage_method="none",
        inference_method="svb", options=CPIOptions(dispersion="auto"))
    calls = []

    def fit(a: connection.GuidedAcquisition, indices: Any) -> Any:
        assert a.inference_method == "vi"
        calls.append((a.family, indices))
        for index in indices:
            yield replace(branch(a.family, float(index if a.family == "cpi" else 1 - index)),
                          flat_index=index)

    monkeypatch.setattr(acquisition, "fit_voxels", fit)
    result = list(acquisition.fit_selected_voxels(prepared, [1, 0]))
    assert [r.flat_index for r in result] == [1, 0]
    assert [r.family for r in result] == ["cpi", "cpi_vtf"]
    assert calls == [("cpi", [1, 0]), ("cpi_vtf", [1, 0])]
