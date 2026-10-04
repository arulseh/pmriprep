"""Final spatial readouts preserve factors, failures and progress ownership."""

from dataclasses import replace
import logging
from typing import Any

import numpy as np
import pytest

from pmriprep.models.dsc.signal_cpi import field, progress, spatial
from pmriprep.models.dsc.signal_cpi.spatial_readout import SpatialReadout
from tests.unit.test_dsc_signal_cpi_execution import prepared
from tests.unit.test_dsc_signal_cpi_spatial import coordinates, linear_arguments
from pmriprep.models.dsc.spatial_vb import batched_dsc_spatial_vi_fit


def compare_readouts(expected: Any, actual: Any) -> None:
    """Compare local states; the single-family reference has no selection scores."""
    assert len(expected) == len(actual)
    for left, right in zip(expected, actual, strict=True):
        assert (left.flat_index, left.status, left.reasons) == (
            right.flat_index, right.status, right.reasons)
        for name in ("cbf", "cbv", "mtt", "delay", "cbf_std", "fitted_k2"):
            np.testing.assert_allclose(getattr(left, name), getattr(right, name),
                                       rtol=1e-9, atol=1e-11)
        if left.diagnostics is None:
            assert right.diagnostics is None
            continue
        assert right.diagnostics is not None
        for name in ("mean", "covariance", "noise_shape", "noise_scale", "fitted_curves",
                     "prior_mean", "prior_covariance", "jacobian"):
            np.testing.assert_allclose(left.diagnostics.arrays[name], right.diagnostics.arrays[name],
                                       rtol=1e-9, atol=1e-11, err_msg=name)


@pytest.mark.parametrize("leakage", ["none", "bsw"])
def test_final_readout_matches_every_pass_reference(
    leakage: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skipping discarded readouts leaves complete final local factors intact."""
    acquisition = prepared(leakage, inference="svb", sloppy=False)
    expected = spatial.fit_spatial_voxels(acquisition.branches[0], keep_diagnostics=True)
    calls = []
    original = SpatialReadout.__call__

    def counted(self: SpatialReadout, *args: Any, **kwargs: Any) -> Any:
        calls.append(self.flat_index)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SpatialReadout, "__call__", counted)
    actual = field.fit_selected_field(acquisition, keep_diagnostics=True)
    assert actual.spatial_sweeps == expected.spatial_sweeps == 10
    assert calls == [0, 1]
    compare_readouts(expected.voxels, actual.voxels)
    np.testing.assert_allclose(actual.posterior_means, expected.posterior_means[:, :1],
                               rtol=1e-10, atol=1e-12)


def test_no_neighbors_still_reports_final_ordinary_vi() -> None:
    """An isolated voxel has one ordinary update and an available final readout."""
    acquisition = prepared("bsw", inference="svb")
    a = acquisition.branches[0]
    a = replace(a, support=np.array([True, False]))
    acquisition = replace(acquisition, branches=(a,))
    expected = spatial.fit_spatial_voxels(a, keep_diagnostics=True)
    actual = field.fit_selected_field(acquisition, keep_diagnostics=True)
    assert actual.local_calls == 1 and actual.spatial_sweeps == 0
    compare_readouts(expected.voxels, actual.voxels)


@pytest.mark.parametrize("failed_sweep", [1, 10])
def test_spatial_failure_isolation_and_recovery(
    failed_sweep: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intermediate failure can recover; final failure keeps its exact reason."""
    acquisition = prepared("bsw", inference="svb", sloppy=False)
    original = spatial.batched_dsc_vi_fit
    calls = 0

    def fail_one(**kwargs: Any) -> Any:
        nonlocal calls
        result = original(**kwargs)
        calls += 1
        if calls == 2*failed_sweep + 1:
            assert kwargs["n_iterations_override"] == 1
            result = replace(result, failure_reasons=("local_factor_failed",),
                             fitted_curves=np.full_like(result.fitted_curves, np.nan))
        return result

    monkeypatch.setattr(spatial, "batched_dsc_vi_fit", fail_one)
    actual = field.fit_selected_field(acquisition, keep_diagnostics=True)
    assert calls == 22 and actual.spatial_sweeps == 10
    assert actual.voxels[1].status in ("retained", "provisional")
    if failed_sweep == 10:
        assert actual.voxels[0].status == "fallback_requested"
        assert actual.voxels[0].reasons == ("local_factor_failed",)
        assert np.isnan(actual.voxels[0].cbf)
    else:
        assert actual.voxels[0].status in ("retained", "provisional")
    assert np.all(np.isfinite(actual.posterior_means))


@pytest.mark.parametrize("connected", [True, False])
@pytest.mark.parametrize("sloppy", [True, False])
def test_spatial_pass_observer_preserves_schedule(connected: bool, sloppy: bool) -> None:
    """The observer names actual passes without altering their numerical result."""
    args = linear_arguments()
    coords, shape = coordinates(connected)
    args.update(mask_coords=coords, spatial_shape=shape, sloppy=sloppy)
    expected = batched_dsc_spatial_vi_fit(**args)
    seen = []
    actual = batched_dsc_spatial_vi_fit(**args, on_pass_start=lambda i, n: seen.append((i, n)))
    n = 6 if sloppy else 10
    assert seen == ([(i, n) for i in range(n+1)] if connected else [(0, 0)])
    np.testing.assert_array_equal(actual.posterior_means_transformed, expected.posterior_means_transformed)
    np.testing.assert_array_equal(actual.posterior_cov_diag, expected.posterior_cov_diag)


def test_progress_reports_warm_start_and_each_sweep(caplog: pytest.LogCaptureFixture) -> None:
    """Each synchronous pass reports its own voxel count and completion."""
    acquisition = prepared(inference="svb")
    with caplog.at_level(logging.INFO, logger="nipype.interface"):
        result = field.fit_selected_field(acquisition)
    messages = [r.getMessage() for r in caplog.records]
    for label in ["SVB warm start"] + [f"SVB sweep {i}/6" for i in range(1, 7)]:
        assert any(f"{label} progress: 2/2 voxels (100.0%)" in m for m in messages)
    assert any("SVB sweep 2/6" in m and "estimated remaining" in m for m in messages)
    assert result.spatial_sweeps == 6


def test_time_based_progress_between_percentage_marks(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Long sparse completions still report after five minutes."""
    ticks = iter([0., 301., 302.])
    monkeypatch.setattr(progress.time, "monotonic", lambda: next(ticks))
    reporter = progress._CPIProgress(1000, label="SVB sweep 3/10")
    with caplog.at_level(logging.INFO, logger="nipype.interface"):
        reporter.advance()
        reporter.advance()
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1 and "1/1000 voxels" in messages[0]
