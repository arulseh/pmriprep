"""Aggregator helpers consume QC* flat keys translated from preproc sidecars.

PR2 collapse: ``_compute_overall_recommendation`` /
``_collect_all_warnings`` / ``_collect_provenance_warnings`` /
``_collect_acquisition_warnings`` operate on the legacy nested shape
returned by ``_load_pipeline_sidecars``. This test feeds those helpers
the dict the loader produces from anchor-sidecar QC* flat keys and
confirms they pick up the recommendation + three warnings flavors.
"""
from __future__ import annotations

import json
from pathlib import Path

from pmriprep.utils.report_utils import (
    _collect_acquisition_warnings,
    _collect_all_warnings,
    _collect_provenance_warnings,
    _compute_overall_recommendation,
    _load_pipeline_sidecars,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def test_recommendation_and_warnings_propagate(tmp_path: Path) -> None:
    ses = tmp_path / "pmriprep" / "sub-01" / "ses-0001"
    prefix = "sub-01_ses-0001"

    _write(
        ses / "perf" / f"{prefix}_desc-dscPreproc_dsc.json",
        {
            "QCRecommendationAction": "review",
            "QCRecommendationReason": "AIF marginal",
            "QCWarnings": ["Brain mask in DSC space not available"],
            "QCProvenanceWarnings": ["DSC preload auto-detected"],
            "QCAcquisitionWarnings": ["AIF fell back to template"],
        },
    )
    _write(
        ses / "perf" / f"{prefix}_desc-dcePreproc_dce.json",
        {
            "QCRecommendationAction": "include",
            "QCRecommendationReason": "AIF good",
            "QCWarnings": ["Lesion mask in DCE space not available"],
        },
    )

    pipeline_data = {
        "anat": None,
        "tumor_seg": None,
        "dsc": {"qc_metrics": _load_pipeline_sidecars(ses, prefix, "dsc")},
        "dce": {"qc_metrics": _load_pipeline_sidecars(ses, prefix, "dce")},
    }

    rec = _compute_overall_recommendation(pipeline_data)
    assert rec is not None
    # DSC review beats DCE include in worst-case aggregation
    assert rec["action"] == "review"
    assert "AIF marginal" in rec["reason"]

    warnings = _collect_all_warnings(pipeline_data)
    assert "DSC Perfusion" in warnings
    assert "Brain mask in DSC space not available" in warnings["DSC Perfusion"]
    assert "DCE Permeability" in warnings
    assert (
        "Lesion mask in DCE space not available" in warnings["DCE Permeability"]
    )

    prov = _collect_provenance_warnings(pipeline_data)
    assert prov.get("DSC Perfusion") == ["DSC preload auto-detected"]

    acq = _collect_acquisition_warnings(pipeline_data)
    assert acq.get("DSC Perfusion") == ["AIF fell back to template"]
