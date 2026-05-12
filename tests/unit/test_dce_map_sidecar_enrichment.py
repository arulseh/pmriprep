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
        "SpatialReference": "bids::pmriprep/sub-X/perf/sub-X_desc-dceMean_stat.nii.gz",
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
    """Regression: every entry in _PK_MAPS must have a _PK_MAP_KEY mapping.

    Otherwise a new PK map would silently emit ``map_key=""`` from the
    sink wiring (see workflows/dce/datasinks.py:_add_dce_pk_map_sink).
    """

    def test_pk_map_key_covers_all_pk_maps(self) -> None:
        from pmriprep.workflows.dce.datasinks import _PK_MAP_KEY, _PK_MAPS

        pk_in_attrs = {entry[0] for entry in _PK_MAPS}
        assert set(_PK_MAP_KEY) == pk_in_attrs, (
            f"_PK_MAP_KEY drift: missing {pk_in_attrs - set(_PK_MAP_KEY)}; "
            f"extra {set(_PK_MAP_KEY) - pk_in_attrs}"
        )


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
