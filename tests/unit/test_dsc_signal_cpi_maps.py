"""CPI image assembly preserves selected-fit identity and fallback accounting."""

from dataclasses import replace
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pmriprep.models.dsc.signal_cpi import maps, progress
from pmriprep.models.dsc.signal_cpi.acquisition import prepare_cpi_acquisition
from pmriprep.models.dsc.signal_cpi.connection import GuidedVoxel
from pmriprep.models.dsc.signal_cpi.policy import FitAssessment
from pmriprep.models.dsc.signal_cpi.settings import CPIOptions
from pmriprep.utils.dsc_deconvolution import svd_deconvolve_oscillation
from pmriprep.utils.provenance import enrich_provenance_from_qc


@pytest.mark.parametrize("label", ["CPI fitting", "CPI initial fitting"])
def test_progress_reports_count_elapsed_and_remaining(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
) -> None:
    """Long voxel fits report bounded controller-side progress and completion."""
    ticks = iter([0.0, 60.0, 300.0, 600.0])
    monkeypatch.setattr(progress.time, "monotonic", lambda: next(ticks))
    reporter = progress._CPIProgress(20, label=label)
    with caplog.at_level(logging.INFO, logger="nipype.interface"):
        reporter.advance()
        reporter.advance()
        reporter.completed = 19
        reporter.advance()

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        f"{label} progress: 1/20 voxels (5.0%); elapsed 1m 00s; "
        "estimated remaining 19m 00s",
        f"{label} progress: 2/20 voxels (10.0%); elapsed 5m 00s; "
        "estimated remaining 45m 00s",
        f"{label} progress: 20/20 voxels (100.0%); elapsed 10m 00s; "
        "estimated remaining 0s",
    ]


@pytest.mark.parametrize("fallback", ["none", "osvd"])
@pytest.mark.parametrize("oi_target", [0.035, 1e-10])
def test_mixed_map_keeps_provisional_fit_and_only_marks_actual_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fallback: Any,
    oi_target: float,
) -> None:
    """No-convergence alone does not reject CPI; failed fallback remains NaN."""
    time = np.arange(30.0) * 3
    u = np.maximum(time - 9, 0)
    aif = 10 * u**2 * np.exp(-u / 3)
    raw = np.broadcast_to(0.2 * u**2 * np.exp(-u / 5), (5, 1, 1, 30)).copy()
    signal = 100 * np.exp(-0.03 * raw)
    mask = np.ones((5, 1, 1), bool)
    mask[-1] = False
    acquisition = prepare_cpi_acquisition(
        signal,
        aif,
        time,
        0.03,
        leakage_method="none",
        mask=mask,
        options=CPIOptions(dispersion="none", fallback=fallback),
    )
    assessment = FitAssessment(
        0, 0, True, False, (), ("spacing_at_bound", "arrival_at_bound"), 0.99, 1.0, 0.0
    )
    provisional = GuidedVoxel(
        0,
        "provisional",
        (),
        0.01,
        0.04,
        4.0,
        0.0,
        cbf_std=0.001,
        family="cpi",
        assessment=assessment,
        fitted_signal=signal[0, 0, 0],
    )
    retained = replace(
        provisional,
        flat_index=1,
        family="cpi_vtf",
        status="retained",
        assessment=replace(assessment, voxel=1, converged=True, warnings=()),
    )
    rows = [
        provisional,
        retained,
        GuidedVoxel(2, "fallback_requested", ("bad_signal_fit",)),
        GuidedVoxel(3, "fallback_requested", ("bad_signal_fit",)),
        GuidedVoxel(4, "outside_support", ("outside_fit_support",)),
    ]
    raw[3] = np.nan
    monkeypatch.setattr(maps, "fit_selected_voxels", lambda a, **kwargs: iter(rows))
    result = maps.fit_cpi_maps(
        acquisition, raw, raw, oi_target=oi_target, prediction_file=tmp_path / "fits.npy"
    )
    np.testing.assert_array_equal(result.cbf[:2], 0.01)
    np.testing.assert_array_equal(result.cbf_std[:2], 0.001)
    assert not np.any(result.fallback_mask[:2])
    assert np.isnan(result.cbf[3:]).all()
    assert np.isnan(result.cbf_std[2:]).all()
    assert result.qc["n_converged"] == 1 and result.qc["n_cpi_retained"] == 2
    assert result.qc["cpi_gamma_fraction"] == 0.5
    assert result.qc["n_mask_voxels"] == 4
    assert result.qc["n_fallback_requested"] == 2
    assert result.qc["fallback_requested_rate"] == 0.5
    assert result.qc["cpi_spacing_at_bound_fraction"] == 0.5
    assert result.qc["delay_at_bound_fraction"] == 0.5
    assert result.qc["rejection_reasons"]["bad_signal_fit"] == 2
    np.testing.assert_array_equal(np.load(tmp_path / "fits.npy")[0], signal[0])
    if fallback == "none":
        assert np.isnan(result.cbf[2]).all() and not np.any(result.fallback_mask)
        assert result.qc["n_fallback_attempted"] == 0 and result.qc["n_missing"] == 2
    else:
        assert np.isfinite(result.cbf[2]).all() and result.fallback_mask[2, 0, 0] == 1
        assert np.isnan(result.delay[2, 0, 0])
        assert result.qc["n_fallback_attempted"] == 1 and result.qc["n_missing"] == 1
        assert result.qc["fallback_rate"] == 0.25

    target_status = result.qc["fallback_oscillation_target"]
    if fallback == "none":
        assert target_status["met_voxels"] == target_status["unmet_voxels"] == 0
        assert target_status["unmet_fraction"] is None
    else:
        fit = svd_deconvolve_oscillation(
            raw[2, 0, 0], acquisition.branches[0].aif, 3., oi_target=oi_target,
        )
        assert fit.converged == (oi_target == 0.035)
        assert target_status["met_voxels"] == int(fit.converged)
        assert target_status["unmet_voxels"] == int(not fit.converged)
        assert target_status["unmet_fraction"] == float(not fit.converged)
        assert result.qc["n_fallback_failed"] == 0
        assert result.qc["fallback_outcomes"] == {"inadmissible_concentration": 1}
    assert target_status["target"] == oi_target
    assert target_status["fraction_denominator"] == "oSVD replacements"
    assert target_status["unmet_policy"] == "retain_and_flag"
    sidecar = enrich_provenance_from_qc(
        {"PerfusionProcessing": {"Deconvolution": {"method": "cpi"}}},
        {"deconvolution": result.qc}, "DSC",
    )
    assert sidecar["PerfusionProcessing"]["Deconvolution"]["fallback_oscillation_target"] == target_status

    normalization = result.qc["cpi_signal_normalization"]
    assert normalization == {
        "method": "acquisition_median_prebolus",
        "reference_baseline": 100.0,
        "input_units_per_normalized_unit": pytest.approx(acquisition.branches[0].signal_scale),
        "baseline_frames": acquisition.branches[0].decision.baseline_frames,
    }
