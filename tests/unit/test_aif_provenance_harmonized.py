"""Tests for harmonized AIF provenance across DSC and DCE modalities."""
from __future__ import annotations

import pytest

from pmriprep.utils.aif_provenance import (
    AIF_INDEPENDENT_MAPS,
    DCE_AIF_INDEPENDENT_MAPS,
    DCE_AIF_MAP_KEYS,
    DSC_AIF_MAP_KEYS,
    _derive_aif_source,
    _derive_quality_flag,
    _is_valid_detection_level,
    apply_aif_sidecar,
    stamp_aif_dependency,
)

@pytest.mark.parametrize(
    ("qc", "modality", "expected"),
    [
        pytest.param({"detection_method": "unavailable"}, "DSC", "unavailable", id="dsc-unavailable"),
        pytest.param({"detection_method": "atlas_strict"}, "DSC", "measured", id="dsc-measured"),
        pytest.param({"detection_method": "external"}, "DSC", "external", id="dsc-external"),
        pytest.param({"detection_method": "population"}, "DCE", "population", id="dce-population"),
        pytest.param({"detection_method": "atlas"}, "DCE", "measured", id="dce-atlas"),
        pytest.param({"detection_method": "whole_brain"}, "DCE", "measured", id="dce-whole-brain"),
        pytest.param({"detection_method": "external"}, "DCE", "external", id="dce-external"),
    ],
)
def test_derive_aif_source(qc: dict, modality: str, expected: str) -> None:
    assert _derive_aif_source(qc, modality) == expected


@pytest.mark.parametrize(
    ("qc", "source", "modality", "expected"),
    [
        pytest.param(
            {"detection_method": "atlas_strict", "quality_score": 0.14},
            "measured", "DSC", "pass", id="dsc-strict-ignores-ranking-score",
        ),
        pytest.param({"detection_method": "atlas_relaxed"}, "measured", "DSC", "pass", id="dsc-relaxed"),
        pytest.param({"detection_method": "atlas_permissive"}, "measured", "DSC", "warn", id="dsc-permissive"),
        pytest.param(
            {"detection_method": "atlas_relaxed+atlas_permissive"},
            "measured", "DSC", "warn", id="dsc-compound-worst-tier",
        ),
        pytest.param({"quality_score": 0.70}, "measured", "DCE", "pass", id="dce-score-pass"),
        pytest.param({"quality_score": 0.50}, "measured", "DCE", "warn", id="dce-score-warn"),
        pytest.param({"quality_score": 0.49}, "measured", "DCE", "poor", id="dce-score-poor"),
        pytest.param({"quality_score": 0.95}, "unavailable", "DSC", "poor", id="unavailable-forced-poor"),
        pytest.param({"quality_score": 0.80}, "population", "DCE", "poor", id="population-forced-poor"),
        pytest.param({}, "measured", "DSC", "warn", id="missing-score"),
        pytest.param(
            {"detection_method": "external", "quality_score": 0.95},
            "external", "DSC", "warn", id="dsc-external",
        ),
    ],
)
def test_derive_quality_flag(
    qc: dict,
    source: str,
    modality: str,
    expected: str,
) -> None:
    assert _derive_quality_flag(qc, source, modality) == expected


class TestBuildProvenanceBlockDSC:
    def test_measured(self) -> None:
        from pmriprep.utils.aif_provenance import build_aif_provenance_block

        block = build_aif_provenance_block(
            {"aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.8}},
            modality="DSC",
        )
        assert block["Source"] == "measured"
        assert block["QualityFlag"] == "pass"
        assert block["QualityScore"] == pytest.approx(0.8)
        assert block["UsedAsCBVDenominator"] is True

    def test_spatial_policy_is_recorded(self) -> None:
        from pmriprep.utils.aif_provenance import build_aif_provenance_block

        block = build_aif_provenance_block(
            {"aif_detection": {
                "detection_method": "brain_relaxed",
                "brain_edge_exclusion_target_mm": 8.0,
                "whole_brain_symmetry_method": "brain_projection_reflection",
                "whole_brain_symmetry_angle_degrees": -2.5,
                "whole_brain_symmetry_overlap": 0.96,
            }},
            modality="DSC",
        )

        assert block["BrainEdgeExclusionTarget"] == pytest.approx(8.0)
        assert block["BrainEdgeExclusionTargetUnits"] == "mm"
        assert block["WholeBrainSymmetryMethod"] == "brain_projection_reflection"
        assert block["WholeBrainSymmetryAngleDegrees"] == pytest.approx(-2.5)
        assert block["WholeBrainSymmetryOverlap"] == pytest.approx(0.96)

    def test_unavailable(self) -> None:
        from pmriprep.utils.aif_provenance import build_aif_provenance_block

        block = build_aif_provenance_block(
            {
                "aif_detection": {"detection_method": "unavailable", "quality_score": 0.3},

            },
            modality="DSC",
        )
        assert block["Source"] == "unavailable"
        assert block["QualityFlag"] == "poor"
        assert block["UsedAsCBVDenominator"] is False

    def test_gamma_optimizer_failure_is_explicit(self) -> None:
        """A retained measured AIF carries optional gamma-fit failure provenance."""
        from pmriprep.utils.aif_provenance import build_aif_provenance_block

        block = build_aif_provenance_block(
            {"aif_detection": {
                "detection_method": "atlas_strict",
                "quality_score": 0.8,
                "gamma_fit_converged": False,
                "gamma_fit_failure_reason": "optimizer_error",
            }},
            modality="DSC",
        )

        assert block["Source"] == "measured"
        assert block["GammaFitConverged"] is False
        assert block["GammaFitFailureReason"] == "optimizer_error"

    def test_missing_quality_score_omits_key(self) -> None:
        from pmriprep.utils.aif_provenance import build_aif_provenance_block

        block = build_aif_provenance_block(
            {"aif_detection": {"detection_method": "atlas_strict"}},
            modality="DSC",
        )
        assert "QualityScore" not in block
        assert block["Source"] == "measured"


class TestBuildProvenanceBlockDCE:
    def test_detected(self) -> None:
        from pmriprep.utils.aif_provenance import build_aif_provenance_block

        block = build_aif_provenance_block(
            {"aif_detection": {
                "detection_method": "atlas",
                "quality_score": 0.75,
                "aif_confidence": "standard",
                "aif_uncertainty_inflation_factor": 1.0,
                "hematocrit": 0.42,
                "peak_scaling_enabled": True,
                "peak_scaling_applied": True,
                "raw_peak_mm": 4.0,
                "emitted_peak_mm": 8.0,
                "peak_scaling_factor": 2.0,
                "peak_scaling_required_factor": 2.0,
                "peak_scaling_policy": "one_sided_lower_bound",
                "peak_scaling_trigger": "below_expected_minimum",
                "atlas_search_outcome": "selected",
                "atlas_search_support_voxels": 830,
                "whole_brain_search_support_voxels": 4120,
                "whole_brain_symmetry_angle_degrees": -3.25,
                "whole_brain_symmetry_overlap": 0.97,
            }},
            modality="DCE",
        )
        assert block["Source"] == "measured"
        assert block["QualityFlag"] == "pass"
        assert block["Confidence"] == "standard"
        assert block["UncertaintyInflationFactor"] == pytest.approx(1.0)
        assert block["Hematocrit"] == pytest.approx(0.42)
        assert block["AtlasSearchOutcome"] == "selected"
        assert block["AtlasSearchSupportVoxels"] == 830
        assert block["WholeBrainSearchSupportVoxels"] == 4120
        assert block["WholeBrainSymmetryAngleDegrees"] == pytest.approx(-3.25)
        assert block["WholeBrainSymmetryOverlap"] == pytest.approx(0.97)
        assert block["PeakScaling"] == {
            "Enabled": True,
            "Applied": True,
            "RawPeak": 4.0,
            "EmittedPeak": 8.0,
            "Factor": 2.0,
            "RequiredFactor": 2.0,
            "Policy": "one_sided_lower_bound",
            "Trigger": "below_expected_minimum",
            "PeakUnits": "mM",
        }

    def test_population(self) -> None:
        from pmriprep.utils.aif_provenance import build_aif_provenance_block

        block = build_aif_provenance_block(
            {"aif_detection": {
                "detection_method": "population",
                "quality_score": 0.5,
                "auto_fallback_reason": "low_cnr",
            }},
            modality="DCE",
        )
        assert block["Source"] == "population"
        assert block["QualityFlag"] == "poor"
        assert block["PopulationModel"] == "parker"
        assert block["PopulationFallbackReason"] == "low_cnr"

    def test_external(self) -> None:
        from pmriprep.utils.aif_provenance import build_aif_provenance_block

        block = build_aif_provenance_block(
            {"aif_detection": {"detection_method": "external", "quality_score": 0.9}},
            modality="DCE",
        )
        assert block["Source"] == "external"
        assert block["QualityFlag"] == "warn"
        assert block["ExternalAIFVerified"] is False

    def test_invalid_detection_level_returns_empty(self) -> None:
        from pmriprep.utils.aif_provenance import build_aif_provenance_block

        block = build_aif_provenance_block(
            {"aif_detection": {"detection_method": "unknown_tier"}},
            modality="DCE",
        )
        assert block == {}


class TestApplyAifSidecarDCE:
    @pytest.mark.parametrize(
        ("map_key", "expected"),
        [
            pytest.param(
                "ktrans", {"AIFDependent": True},
                id="pk-map",
            ),
            pytest.param(
                "iauc60", {"AIFDependent": False},
                id="iauc",
            ),
            pytest.param(
                "ser", {"AIFDependent": False},
                id="model-free",
            ),
            pytest.param(
                "ktrans_uncertainty",
                {"AIFDependent": True},
                id="absolute-uncertainty",
            ),
        ],
    )
    def test_map_dependency(self, map_key: str, expected: dict) -> None:
        sidecar: dict = {}
        qc = {"aif_detection": {"detection_method": "atlas", "quality_score": 0.8}}
        apply_aif_sidecar(sidecar, map_key, qc, modality="DCE")
        assert expected.items() <= sidecar.items()
        assert "PerfusionProcessing" in sidecar

    def test_pk_map_carries_peak_scaling(self) -> None:
        """Quantitative-map sidecars retain the canonical scaling fragment."""
        sidecar: dict = {}
        qc = {
            "aif_detection": {
                "detection_method": "atlas",
                "peak_scaling_enabled": True,
                "peak_scaling_applied": True,
                "raw_peak_mm": 4.0,
                "emitted_peak_mm": 8.0,
                "peak_scaling_factor": 2.0,
                "peak_scaling_required_factor": 2.0,
                "peak_scaling_policy": "one_sided_lower_bound",
                "peak_scaling_trigger": "below_expected_minimum",
            }
        }
        apply_aif_sidecar(sidecar, "ktrans", qc, modality="DCE")
        scaling = sidecar["PerfusionProcessing"]["AIF"]["PeakScaling"]
        assert scaling["RawPeak"] == pytest.approx(4.0)
        assert scaling["EmittedPeak"] == pytest.approx(8.0)
        assert scaling["Policy"] == "one_sided_lower_bound"

    def test_unknown_map_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown DCE map key"):
            apply_aif_sidecar({}, "nonexistent_map", {}, modality="DCE")


@pytest.mark.parametrize(
    ("independent", "registered"),
    [
        pytest.param(DCE_AIF_INDEPENDENT_MAPS, DCE_AIF_MAP_KEYS, id="dce"),
        pytest.param(AIF_INDEPENDENT_MAPS, DSC_AIF_MAP_KEYS, id="dsc"),
    ],
)
def test_independent_aif_maps_are_registered(
    independent: frozenset[str],
    registered: frozenset[str],
) -> None:
    assert independent <= registered


@pytest.mark.parametrize(
    ("map_key", "expected"),
    [
        pytest.param(
            "ktrans", {"AIFDependent": True},
            id="dependent",
        ),
        pytest.param(
            "ser", {"AIFDependent": False},
            id="independent",
        ),
        pytest.param("nonexistent", {}, id="unregistered"),
        pytest.param("", {}, id="empty"),
    ],
)
def test_stamp_aif_dependency(map_key: str, expected: dict) -> None:
    sidecar: dict = {}
    stamp_aif_dependency(sidecar, map_key, modality="DCE")
    assert sidecar == expected


@pytest.mark.parametrize(
    ("level", "modality", "expected"),
    [
        pytest.param("atlas", "DCE", True, id="dce-atlas"),
        pytest.param("population", "DCE", True, id="dce-population"),
        pytest.param("atlas_strict", "DCE", False, id="dce-rejects-dsc-tier"),
        pytest.param("atlas_strict+atlas_relaxed", "DSC", True, id="dsc-compound"),
    ],
)
def test_detection_level_validation(level: str, modality: str, expected: bool) -> None:
    assert _is_valid_detection_level(level, modality) is expected


class TestDSCRegression:

    def test_dsc_apply_sidecar_preserves_source_and_dependence(self) -> None:
        sidecar: dict = {}
        qc = {"aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.8}}
        apply_aif_sidecar(sidecar, "cbf", qc, modality="DSC")
        assert sidecar["AIFDependent"] is True
        assert sidecar["PerfusionProcessing"]["AIF"]["Source"] == "measured"

class TestPublicArtifactSchema:
    def test_dce_prefers_final_emitted_curve_metrics(self) -> None:
        from pmriprep.utils.aif_provenance import build_aif_provenance_block

        block = build_aif_provenance_block(
            {
                "detection_method": "atlas",
                "peak_mm_mean": 3.0,
                "final_peak_mm": 6.0,
                "fwhm_seconds_mean": 8.0,
                "final_fwhm_seconds": 11.0,
            },
            modality="DCE",
        )
        assert block["PeakAmplitude"] == pytest.approx(6.0)
        assert block["PeakAmplitudeUnits"] == "mM"
        assert block["FWHM"] == pytest.approx(11.0)
        assert block["FWHMUnits"] == "s"

    def test_external_dsc_fwhm_retains_frame_units(self) -> None:
        from pmriprep.utils.aif_provenance import build_aif_provenance_block

        block = build_aif_provenance_block(
            {
                "detection_method": "external",
                "aif_metrics": {"fwhm": 5.0},
            },
            modality="DSC",
        )
        assert block["FWHM"] == pytest.approx(5.0)
        assert block["FWHMUnits"] == "frames"

    def test_curve_and_mask_use_identical_dce_block(self, tmp_path) -> None:
        from pmriprep.utils.sidecar_builders import build_aif_timeseries_sidecar
        from pmriprep.workflows.dce.datasinks import _build_aif_mask_meta

        qc = {
            "detection_method": "atlas",
            "quality_score": 0.65,
            "n_aif_voxels_selected": 7,
            "aif_metrics": {"peak_conc": 4.2, "fwhm": 8.0},
            "hematocrit": 0.42,
            "peak_scaling_enabled": True,
            "peak_scaling_applied": False,
        }
        curve = build_aif_timeseries_sidecar(
            modality="DCE",
            source_paths=[],
            bids_dir=tmp_path,
            subject="01",
            session=None,
            aif_qc=qc,
        )
        mask = _build_aif_mask_meta({}, "AIF mask", {"aif_detection": qc})

        assert (
            curve["PerfusionProcessing"]["AIF"]
            == mask["PerfusionProcessing"]["AIF"]
        )
        assert mask["Type"] == "ROI"
        assert mask["SpatialSupportAvailable"] is True

    @pytest.mark.parametrize("source", ["external", "population"])
    def test_nonspatial_dce_aif_mask_is_explicit(self, source: str) -> None:
        """An empty non-measured AIF mask does not imply failed detection."""
        from pmriprep.workflows.dce.datasinks import _build_aif_mask_meta

        mask = _build_aif_mask_meta(
            {},
            "AIF spatial-support mask",
            {"aif_detection": {"detection_method": source}},
        )

        assert mask["SpatialSupportAvailable"] is False
        assert "No spatial AIF voxels" in mask["EmptyMaskMeaning"]

    def test_curve_mask_and_map_use_identical_dsc_block(self, tmp_path) -> None:
        from pmriprep.utils.sidecar_builders import build_aif_timeseries_sidecar
        from pmriprep.workflows.dsc.datasinks import _build_aif_mask_meta

        qc = {
            "detection_method": "atlas_strict",
            "quality_score": 0.8,
            "n_aif_voxels_selected": 6,
            "gamma_fit_r_squared": 0.95,
        }
        curve = build_aif_timeseries_sidecar(
            modality="DSC",
            source_paths=[],
            bids_dir=tmp_path,
            subject="01",
            session=None,
            aif_qc=qc,
        )
        mask = _build_aif_mask_meta({}, "AIF mask", {"aif_detection": qc})
        quantitative_map: dict = {}
        apply_aif_sidecar(
            quantitative_map,
            "cbf",
            {"aif_detection": qc},
            modality="DSC",
        )

        expected = curve["PerfusionProcessing"]["AIF"]
        assert mask["PerfusionProcessing"]["AIF"] == expected
        assert quantitative_map["PerfusionProcessing"]["AIF"] == expected
        assert mask["SpatialSupportAvailable"] is True

    @pytest.mark.parametrize("source", ["unavailable", "external"])
    def test_nonspatial_dsc_aif_mask_is_explicit(self, source: str) -> None:
        """An empty non-measured AIF mask does not imply failed detection."""
        from pmriprep.workflows.dsc.datasinks import _build_aif_mask_meta

        mask = _build_aif_mask_meta(
            {},
            "AIF spatial-support mask",
            {"aif_detection": {"detection_method": source}},
        )

        assert mask["SpatialSupportAvailable"] is False
        assert "No spatial AIF voxels" in mask["EmptyMaskMeaning"]
