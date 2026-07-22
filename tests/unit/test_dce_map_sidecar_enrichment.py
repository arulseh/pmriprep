"""PR1 (BIDS arc finishing): DCE per-PK-map sidecar enrichment.

Asserts ``enrich_dce_pk_map_sidecar`` produces flat ``PK*`` keys, attaches
cross-modality fields only on the maps they apply to (Ktrans →
``DSCDCECrossValCorrelation``; vp → ``CBVNegativeFraction``), and accepts
per-map ``LesionStats`` fragments.
"""
from __future__ import annotations

from typing import Any

import pytest

from pmriprep.utils.sidecar_builders import enrich_dce_pk_map_sidecar


@pytest.fixture
def base_sidecar() -> dict[str, Any]:
    return {
        "Description": "Volume transfer constant (1/min)",
        "Units": "1/min",
        "SpatialReference": "bids::sub-X/perf/sub-X_desc-dceMean_stat.nii.gz",
        "KtransConvention": "plasma_Tofts1999",
    }


@pytest.fixture
def pk_qc_metrics() -> dict[str, Any]:
    return {
        "pk_modeling": {
            "model": "auto",
            "candidate_models": ["tofts", "extended_tofts", "patlak"],
            "qc_voxel_rsquared": 0.87,
            "inference": "vi",
            # ComputeDCEMaps emits bat_correction at the top of its node qc;
            # _collect_qc_metrics nests that whole dict under "pk_modeling".
            "bat_correction": {
                "applied": True,
                "reason": "applied",
                "delay_seconds": 7.0,
                "delay_frames": 1.0,
                "granularity": "enhancing_tumor",
                "n_reference_voxels": 142,
            },
        },
        "dsc_dce_validation": {"metrics": {"correlation": 0.72}},
        "cbv_constraint": {"cbv_negative_fraction": 0.012},
    }


class TestCrossModalityFields:
    def test_dsc_dce_correlation_only_on_ktrans(self, base_sidecar, pk_qc_metrics):
        out_ktrans = enrich_dce_pk_map_sidecar(
            dict(base_sidecar), qc_metrics=pk_qc_metrics, map_key="ktrans",
        )
        out_ve = enrich_dce_pk_map_sidecar(
            dict(base_sidecar), qc_metrics=pk_qc_metrics, map_key="ve",
        )
        assert out_ktrans["DSCDCECrossValCorrelation"] == pytest.approx(0.72)
        # ve sidecar must not carry the cross-modality field
        assert "DSCDCECrossValCorrelation" not in out_ve

    def test_cbv_negative_only_on_vp(self, base_sidecar, pk_qc_metrics):
        out_vp = enrich_dce_pk_map_sidecar(
            dict(base_sidecar), qc_metrics=pk_qc_metrics, map_key="vp",
        )
        out_ktrans = enrich_dce_pk_map_sidecar(
            dict(base_sidecar), qc_metrics=pk_qc_metrics, map_key="ktrans",
        )
        assert out_vp["CBVNegativeFraction"] == pytest.approx(0.012)
        assert "CBVNegativeFraction" not in out_ktrans


class TestLesionFragment:
    def test_lesion_fragment_attached(self, base_sidecar, pk_qc_metrics):
        fragment = {
            "Regions": {
                "contrast_enhancing_tumor": {
                    "voxel_count": 30,
                    "volume_ml": 0.030,
                    "stats": {"mean": 0.12, "std": 0.04},
                },
            },
        }
        out = enrich_dce_pk_map_sidecar(
            dict(base_sidecar), qc_metrics=pk_qc_metrics, map_key="ktrans",
            lesion_fragment=fragment,
        )
        assert out["LesionStats"] == fragment

    def test_no_lesion_fragment_skipped(self, base_sidecar, pk_qc_metrics):
        out = enrich_dce_pk_map_sidecar(
            dict(base_sidecar), qc_metrics=pk_qc_metrics, map_key="ktrans",
            lesion_fragment=None,
        )
        assert "LesionStats" not in out


class TestPKMapKeyDriftGuard:
    """Regression: every typed PK-map entry must carry its enrichment key.

    Otherwise a new PK map could silently emit ``map_key=""`` from the sink
    wiring (see workflows/dce/datasinks.py:_add_dce_pk_map_sink).
    """

    def test_pk_map_key_covers_all_pk_maps(self) -> None:
        from pmriprep.workflows.dce.datasinks import _PK_MAPS

        assert {spec.in_attr: spec.map_key for spec in _PK_MAPS} == {
            "ktrans_map": "ktrans",
            "ve_map": "ve",
            "kep_map": "kep",
            "vp_map": "vp",
            "fp_map": "fp",
            "ps_map": "ps",
        }

    def test_all_pk_output_keys_are_registered_for_aif_provenance(self) -> None:
        """Every PK-family sink key must survive strict AIF stamping."""
        from pmriprep.utils.aif_provenance import DCE_POPULATION_AIF_USAGE
        from pmriprep.workflows.dce.datasinks import (
            _CI95_MAPS,
            _CV_MAPS,
            _PK_MAPS,
            _UNCERTAINTY_MAPS,
        )

        emitted = {
            spec.map_key
            for table in (_PK_MAPS, _UNCERTAINTY_MAPS, _CV_MAPS, _CI95_MAPS)
            for spec in table
        }
        assert emitted <= DCE_POPULATION_AIF_USAGE.keys()

        for key in ("fp", "ps", "fp_uncertainty", "ps_uncertainty"):
            assert DCE_POPULATION_AIF_USAGE[key] == "ratio_defensible"
        for key in ("fp_cv", "ps_cv"):
            assert DCE_POPULATION_AIF_USAGE[key] == "safe"


class TestBATCorrectionProvenance:
    """Regression: BATCorrection must surface from the MERGED qc dict.

    ``_collect_qc_metrics`` nests the PK node qc under ``pk_modeling``, so
    ``bat_correction`` lives at ``qc_metrics["pk_modeling"]["bat_correction"]``,
    not the top level. The enricher must read both levels (this is the bug
    that left BATCorrection off every PK-map sidecar in v0.62.0).
    """

    def test_surfaces_from_merged_pipeline_qc(self, base_sidecar, pk_qc_metrics):
        out = enrich_dce_pk_map_sidecar(
            dict(base_sidecar), qc_metrics=pk_qc_metrics, map_key="ktrans",
        )
        bat = out["PerfusionProcessing"]["BATCorrection"]
        assert bat["applied"] is True
        assert bat["reason"] == "applied"
        assert bat["delay_frames"] == 1.0

    def test_surfaces_from_flat_node_qc(self, base_sidecar):
        # Direct node-output shape: bat_correction at the top level.
        flat = {"bat_correction": {"applied": False, "reason": "alignment_off"}}
        out = enrich_dce_pk_map_sidecar(
            dict(base_sidecar), qc_metrics=flat, map_key="vp",
        )
        assert out["PerfusionProcessing"]["BATCorrection"]["reason"] == "alignment_off"

    def test_absent_when_no_bat_correction(self, base_sidecar):
        out = enrich_dce_pk_map_sidecar(
            dict(base_sidecar),
            qc_metrics={"pk_modeling": {"model": "auto"}},
            map_key="ktrans",
        )
        assert "BATCorrection" not in out.get("PerfusionProcessing", {})


class TestEmptyInputs:
    def test_no_qc_metrics(self, base_sidecar):
        out = enrich_dce_pk_map_sidecar(
            dict(base_sidecar), qc_metrics=None, map_key="ktrans",
        )
        assert all(not k.startswith("PK") for k in out
                   if k not in {"KtransConvention",})

    def test_no_qc_with_lesion(self, base_sidecar):
        fragment = {"Regions": {"X": {"voxel_count": 1, "stats": {}}}}
        out = enrich_dce_pk_map_sidecar(
            dict(base_sidecar), qc_metrics=None, map_key="ktrans",
            lesion_fragment=fragment,
        )
        assert out["LesionStats"] == fragment
