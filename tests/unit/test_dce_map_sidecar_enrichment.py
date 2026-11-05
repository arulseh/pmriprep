"""PR1 (BIDS arc finishing): DCE per-PK-map sidecar enrichment.

Asserts ``enrich_dce_pk_map_sidecar`` produces flat ``PK*`` keys, keeps
cross-modal concordance in the QC record, attaches map-specific QC only where
it applies (vp → ``CBVNegativeFraction``), and accepts per-map statistics.
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
        "InputFunctionCompartment": "plasma",
    }


@pytest.fixture
def pk_qc_metrics() -> dict[str, Any]:
    return {
        "pk_modeling": {
            "model": "auto",
            "candidate_models": ["tofts", "extended_tofts", "patlak"],
            "qc_voxel_rsquared": 0.87,
            "inference": "vi",
            # ComputeDCEMaps emits delay_correction at the top of its node qc;
            # _collect_qc_metrics nests that whole dict under "pk_modeling".
            "delay_correction": {
                "enabled": True, "method": "local_profile", "grid_step_seconds": 0.1,
                "maximum_seconds": 12.0,
            },
        },
        "dsc_dce_validation": {"metrics": {"correlation": 0.72}},
        "cbv_constraint": {"cbv_negative_fraction": 0.012},
    }


class TestCrossModalityFields:
    def test_concordance_is_not_duplicated_into_map_sidecars(
        self, base_sidecar, pk_qc_metrics
    ):
        for map_key in ("ktrans", "ve", "vp"):
            out = enrich_dce_pk_map_sidecar(
                dict(base_sidecar), qc_metrics=pk_qc_metrics, map_key=map_key,
            )
            assert "DSCDCECrossValCorrelation" not in out

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
            "delay_map": "delay",
            "ktrans_map": "ktrans",
            "ve_map": "ve",
            "kep_map": "kep",
            "vp_map": "vp",
            "fp_map": "fp",
            "ps_map": "ps",
            "e_map": "e",
        }

    def test_kep_sidecar_records_derived_validity_domain(self) -> None:
        out = enrich_dce_pk_map_sidecar({}, map_key="kep")

        assert out["Estimator"] == "Ktrans / ve"
        assert out["ValueDomain"] == "0 to 10 1/min, or NaN"
        assert "identifiable reporting range" in out["UndefinedValue"]
        assert "EES term is non-identifiable" in out["UndefinedValue"]

    @pytest.mark.parametrize("map_key", ["ve", "vp"])
    def test_volume_fraction_sidecars_record_boundary_handling(
        self, map_key: str,
    ) -> None:
        out = enrich_dce_pk_map_sidecar({}, map_key=map_key)

        assert "BoundaryHandling" in out
        assert "NaN" in out["ValueDomain"]

    def test_all_pk_output_keys_are_registered_for_aif_provenance(self) -> None:
        """Every PK-family sink key must survive strict AIF stamping."""
        from pmriprep.utils.aif_provenance import DCE_AIF_MAP_KEYS
        from pmriprep.workflows.dce.datasinks import (
            _UNCERTAINTY_MAPS,
            _PK_MAPS,
        )

        emitted = {
            spec.map_key
            for table in (_PK_MAPS, _UNCERTAINTY_MAPS)
            for spec in table
        }
        assert emitted <= DCE_AIF_MAP_KEYS

    def test_2cxm_sidecars_record_identifiable_domains(self) -> None:
        for map_key in ("fp", "ps"):
            out = enrich_dce_pk_map_sidecar({}, map_key=map_key)
            assert "BoundaryHandling" in out
            assert "NaN" in out["ValueDomain"]

        extraction = enrich_dce_pk_map_sidecar({}, map_key="e")
        assert extraction["Estimator"] == "PS / (Fp + PS)"
        assert "zero-exchange" in extraction["UndefinedValue"]


class TestDelayCorrectionProvenance:
    """DelayCorrection is read from either top-level or nested PK QC metrics."""

    def test_surfaces_from_merged_pipeline_qc(self, base_sidecar, pk_qc_metrics):
        out = enrich_dce_pk_map_sidecar(
            dict(base_sidecar), qc_metrics=pk_qc_metrics, map_key="ktrans",
        )
        bat = out["PerfusionProcessing"]["DelayCorrection"]
        assert bat["enabled"] is True
        assert bat["method"] == "local_profile"
        assert bat["grid_step_seconds"] == 0.1

    def test_surfaces_from_flat_node_qc(self, base_sidecar):
        # Direct node-output shape: delay_correction at the top level.
        flat = {"delay_correction": {"enabled": False, "method": "local_profile", "maximum_seconds": 0.0}}
        out = enrich_dce_pk_map_sidecar(
            dict(base_sidecar), qc_metrics=flat, map_key="vp",
        )
        assert out["PerfusionProcessing"]["DelayCorrection"]["enabled"] is False

    def test_absent_when_no_delay_correction(self, base_sidecar):
        out = enrich_dce_pk_map_sidecar(
            dict(base_sidecar),
            qc_metrics={"pk_modeling": {"model": "auto"}},
            map_key="ktrans",
        )
        assert "DelayCorrection" not in out.get("PerfusionProcessing", {})


class TestInferenceFallbackProvenance:
    """Bayesian-to-NLLS fallback accounting reaches every PK-map sidecar."""

    def test_surfaces_from_merged_pipeline_qc(
        self, base_sidecar, pk_qc_metrics,
    ) -> None:
        fallback = {
            "requested_inference": "vi",
            "fallback_inference": "nlls",
            "n_retained_voxels": 100,
            "n_nlls_fallback": 4,
            "fallback_fraction_of_retained": 0.04,
        }
        pk_qc_metrics["pk_modeling"]["inference_fallback"] = fallback

        out = enrich_dce_pk_map_sidecar(
            dict(base_sidecar), qc_metrics=pk_qc_metrics, map_key="ve",
        )

        assert out["PerfusionProcessing"]["InferenceFallback"] == fallback


class TestEmptyInputs:
    def test_no_qc_metrics(self, base_sidecar):
        out = enrich_dce_pk_map_sidecar(
            dict(base_sidecar), qc_metrics=None, map_key="ktrans",
        )
        assert all(not k.startswith("PK") for k in out)

    def test_no_qc_with_lesion(self, base_sidecar):
        fragment = {"Regions": {"X": {"voxel_count": 1, "stats": {}}}}
        out = enrich_dce_pk_map_sidecar(
            dict(base_sidecar), qc_metrics=None, map_key="ktrans",
            lesion_fragment=fragment,
        )
        assert out["LesionStats"] == fragment
