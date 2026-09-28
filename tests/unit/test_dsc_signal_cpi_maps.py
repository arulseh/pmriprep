"""CPI image assembly preserves selected-fit identity and fallback accounting."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pmriprep.models.dsc.signal_cpi import maps
from pmriprep.models.dsc.signal_cpi.acquisition import prepare_cpi_acquisition
from pmriprep.models.dsc.signal_cpi.connection import GuidedVoxel
from pmriprep.models.dsc.signal_cpi.policy import FitAssessment
from pmriprep.models.dsc.signal_cpi.settings import CPIOptions


@pytest.mark.parametrize("fallback", ["none", "osvd"])
def test_mixed_map_keeps_provisional_fit_and_only_marks_actual_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fallback: Any,
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
        acquisition, raw, raw, oi_target=0.035, prediction_file=tmp_path / "fits.npy"
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
        assert result.qc["n_fallback_attempted"] == 1 and result.qc["n_missing"] == 1
        assert result.qc["fallback_rate"] == 0.25
