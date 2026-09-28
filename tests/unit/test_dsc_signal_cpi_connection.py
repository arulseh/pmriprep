"""Candidate measurement eligibility, independent of numerical convergence."""

from dataclasses import replace

import numpy as np
import pytest

from pmriprep.models.dsc.signal_cpi.connection import classify_result
from pmriprep.models.dsc.signal_cpi.leakage import LeakageBatch
from pmriprep.models.dsc.signal_cpi.policy import FitAssessment
from pmriprep.models.dsc.signal_cpi.probability import SharedFit


def selection(flow: float, *, converged: bool = True) -> LeakageBatch:
    """Build an already assessed selection, with flow in native inverse seconds."""
    mtt = 11.89786370056497
    fit = SharedFit(
        voxel=0, order=6, restart=2, converged=converged, score=-12.0,
        cbf=flow * 6000, cbv=flow * mtt * 100, mtt=mtt, delay=2.0,
        s0=1000.0, noise_sd=1.0, rss=10.0, mean=[flow, 1000.0],
    )
    assessment = FitAssessment(0, 2, True, converged, (), (), 0.9, 1.0, 0.0)
    return LeakageBatch(
        [fit], [assessment], [{"selected": fit, "assessment": assessment}],
        [{"restart": 2, "k2": 0.003}],
        {"fitted_curves": np.full((1, 10), 999.0),
         "covariance": np.array([np.diag([1e-6, 1.0])])},
        seconds=0.0, iterations=3,
    )


@pytest.mark.parametrize("converged", [True, False])
@pytest.mark.parametrize("flow", [
    0.0, np.nextafter(0.0, 1.0), 2.168404344971009e-19,
    np.nextafter(np.finfo(float).eps, 0.0), np.finfo(float).eps,
])
def test_zero_bound_withholds_measurements_preserves_state(
    flow: float, converged: bool,
) -> None:
    """A shape integral is not measured MTT when the flow multiplier vanishes."""
    diagnostic = selection(flow, converged=converged)
    raw = diagnostic.fits[0]
    output = classify_result(19, 0.004, diagnostic)
    assert output.status == "fallback_requested"
    assert "nonpositive_perfusion_measurement" in output.reasons or "flow_at_zero_bound" in output.reasons
    assert all(np.isnan(getattr(output, name)) for name in (
        "cbf", "cbv", "mtt", "delay", "fitted_k2", "cbf_std",
    ))
    assert output.diagnostics is diagnostic and diagnostic.fits[0] is raw
    assert output.assessment is not None and output.assessment.converged == converged
    assert output.assessment.voxel == 19 and output.assessment.valid
    assert raw.mtt > 0 and raw.mean[0] == flow
    assert output.bsw_k2 == 0.004
    np.testing.assert_array_equal(output.fitted_signal, diagnostic.arrays["fitted_curves"][0])
    assert output.fitted_signal is not None and not output.fitted_signal.flags.writeable


@pytest.mark.parametrize("converged", [True, False])
@pytest.mark.parametrize("flow", [np.nextafter(np.finfo(float).eps, np.inf), 1e-12, 1e-6, 0.01])
def test_positive_flow_is_not_subject_to_a_physiological_floor(
    flow: float, converged: bool,
) -> None:
    """The numerical contact tolerance does not impose a low-perfusion QC limit."""
    diagnostic = selection(flow, converged=converged)
    output = classify_result(5, 0.004, diagnostic)
    assert output.status == ("retained" if converged else "provisional")
    assert output.reasons == ()
    assert output.cbf == flow
    assert output.cbv == pytest.approx(flow * output.mtt, rel=1e-14, abs=0)
    assert output.cbf_std == 0.001 and output.fitted_k2 == 0.003


@pytest.mark.parametrize("field,value,reason", [
    ("cbf", -1.0, "nonpositive_perfusion_measurement"),
    ("cbf", np.nan, "nonfinite_required_measurement"),
    ("cbf", np.inf, "nonfinite_required_measurement"),
    ("cbv", 0.0, "nonpositive_perfusion_measurement"),
    ("mtt", 0.0, "nonpositive_perfusion_measurement"),
    ("mtt", np.nan, "nonfinite_required_measurement"),
])
def test_existing_invalid_measurement_rejections_remain(
    field: str, value: float, reason: str,
) -> None:
    """The near-zero check must not replace finite and strictly positive guards."""
    diagnostic = selection(0.01)
    raw = diagnostic.fits[0]
    fit = replace(raw, cbf=value if field == "cbf" else raw.cbf,
                  cbv=value if field == "cbv" else raw.cbv,
                  mtt=value if field == "mtt" else raw.mtt)
    diagnostic.fits[0] = fit
    diagnostic.rows[0]["selected"] = fit
    output = classify_result(3, 0.004, diagnostic)
    assert output.status == "fallback_requested" and reason in output.reasons
    assert np.isnan(output.mtt)
