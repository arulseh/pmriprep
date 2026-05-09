"""DSC recommendation flat-key round-trip via anchor sidecar.

Verifies that ``enrich_recommendation_keys`` produces ``QC*`` flat keys
on a synthetic DSC-preproc anchor sidecar, and ``_qc_keys_from_anchor``
reads them back into the legacy nested ``qc_metrics["recommendation"]``
shape consumed by ``_compute_overall_recommendation``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pmriprep.utils.report_utils import (
    _load_pipeline_sidecars,
    _qc_keys_from_anchor,
)
from pmriprep.utils.sidecar_builders import enrich_recommendation_keys


@pytest.fixture
def dsc_qc_metrics() -> dict:
    """Producer-shape DSC qc_metrics with a marginal-motion case that
    should drive ``recommend_dsc_qc`` to action='review'."""
    return {
        "aif_detection": {"detection_method": "atlas_relaxed"},
        "preprocessing": {
            "motion_status": "WARN",
            "fd_mean": 0.18,
            "fd_max": 0.62,
            "percent_outliers": 1.7,
            "mean_cnr": 5.4,
            "fraction_cnr_gt_4": 0.81,
        },
        "preload": {"k2_classification": "none"},
    }


class TestEnrichRecommendationFlatKeys:
    def test_dsc_recommendation_emits_flat_keys(self, dsc_qc_metrics):
        sidecar: dict = {"Description": "DSC preproc"}
        enrich_recommendation_keys(
            sidecar,
            modality="DSC",
            qc_metrics=dsc_qc_metrics,
        )
        assert "QCRecommendationAction" in sidecar
        assert sidecar["QCRecommendationAction"] in {"include", "review", "exclude"}
        assert "QCRecommendationReason" in sidecar
        assert isinstance(sidecar["QCRecommendationReason"], str)
        assert sidecar["QCRecommendationReason"]


class TestQcKeysFromAnchorRoundTrip:
    def test_flat_keys_unflatten_to_nested_recommendation(self, dsc_qc_metrics):
        anchor = {"Description": "DSC preproc"}
        enrich_recommendation_keys(
            anchor, modality="DSC", qc_metrics=dsc_qc_metrics,
        )

        out = _qc_keys_from_anchor(anchor)
        assert "recommendation" in out
        rec = out["recommendation"]
        assert rec["action"] in {"include", "review", "exclude"}
        assert rec["reason"]


class TestLoaderEndToEnd:
    def test_dsc_loader_surfaces_recommendation(self, tmp_path, dsc_qc_metrics):
        ses_dir = tmp_path
        perf_dir = ses_dir / "perf"
        perf_dir.mkdir()
        prefix = "sub-X_ses-Y"

        # Synthesize the dscPreproc anchor sidecar enriched with the
        # recommendation flat keys.
        anchor: dict = {"Description": "DSC preproc"}
        enrich_recommendation_keys(
            anchor, modality="DSC", qc_metrics=dsc_qc_metrics,
        )
        (perf_dir / f"{prefix}_desc-dscPreproc_dsc.json").write_text(
            json.dumps(anchor, indent=2),
        )

        # No resume sidecar — verifies the recommendation comes purely
        # from the anchor's flat-key path.
        qc = _load_pipeline_sidecars(ses_dir, prefix, "dsc")
        assert "recommendation" in qc
        rec = qc["recommendation"]
        assert rec["action"] in {"include", "review", "exclude"}
        assert isinstance(rec["reason"], str)
        assert rec["reason"]
