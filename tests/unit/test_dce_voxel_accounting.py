"""Tests for explicit DCE voxel-accounting stages and denominators."""

import numpy as np

from pmriprep.utils.dce_voxel_prep import summarize_regional_voxel_accounting


def test_regional_accounting_uses_brain_intersection_and_actual_stage_masks() -> None:
    """Regional fractions use the tissue-in-brain count, not positive PK maps."""
    brain = np.array([True, True, True, False])
    regions = {
        "enhancing_tumor": np.array([True, True, False, True]),
        "nawm": np.array([False, False, True, False]),
        "missing": None,
    }
    stages = {
        "fitting_eligible": np.array([True, True, True, True]),
        "optimizer_converged": np.array([True, False, True, True]),
        "retained_nondegenerate": np.array([True, True, True, True]),
        "quantitative_qc_accepted": np.array([False, True, True, True]),
    }

    accounting = summarize_regional_voxel_accounting(brain, stages, regions)

    enhancing = accounting["enhancing_tumor"]
    assert enhancing["region_voxel_count"] == 2
    assert enhancing["stages"]["optimizer_converged"] == {
        "count": 1,
        "fraction_of_region": 0.5,
    }
    assert enhancing["stages"]["quantitative_qc_accepted"] == {
        "count": 1,
        "fraction_of_region": 0.5,
    }
    assert accounting["nawm"]["region_voxel_count"] == 1
    assert "missing" not in accounting


def test_regional_accounting_retains_empty_available_region() -> None:
    """An available empty region is reported with guarded zero fractions."""
    accounting = summarize_regional_voxel_accounting(
        np.array([True, True]),
        {"fitting_eligible": np.array([True, False])},
        {"enhancing_tumor": np.array([False, False])},
    )

    assert accounting["enhancing_tumor"] == {
        "region_voxel_count": 0,
        "stages": {
            "fitting_eligible": {
                "count": 0,
                "fraction_of_region": 0.0,
            }
        },
    }
