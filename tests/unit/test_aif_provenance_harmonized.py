"""Tests for harmonized AIF provenance across DSC and DCE modalities."""
from __future__ import annotations

import pytest


class TestDeriveAifSource:
    def test_dsc_synthetic(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_aif_source

        assert _derive_aif_source({"detection_method": "synthetic"}, "DSC") == "synthetic"

    def test_dsc_measured(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_aif_source

        assert _derive_aif_source({"detection_method": "atlas_strict"}, "DSC") == "measured"

    def test_dsc_gamma_fit(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_aif_source

        qc = {"detection_method": "atlas_strict", "gamma_fit_applied": True}
        assert _derive_aif_source(qc, "DSC") == "gamma_fit_measured"

    def test_dsc_external(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_aif_source

        assert _derive_aif_source({"detection_method": "external"}, "DSC") == "external"

    def test_dce_population(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_aif_source

        assert _derive_aif_source({"detection_method": "population"}, "DCE") == "population"

    def test_dce_atlas(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_aif_source

        assert _derive_aif_source({"detection_method": "atlas"}, "DCE") == "measured"

    def test_dce_whole_brain(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_aif_source

        assert _derive_aif_source({"detection_method": "whole_brain"}, "DCE") == "measured"

    def test_dce_external(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_aif_source

        assert _derive_aif_source({"detection_method": "external"}, "DCE") == "external"

    def test_dce_ignores_gamma_fit(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_aif_source

        qc = {"detection_method": "atlas", "gamma_fit_applied": True}
        assert _derive_aif_source(qc, "DCE") == "measured"


class TestDeriveQualityFlag:
    def test_dsc_strict_passes_despite_ranking_score(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_quality_flag

        qc = {"detection_method": "atlas_strict", "quality_score": 0.14}
        assert _derive_quality_flag(qc, "measured") == "pass"

    def test_dsc_relaxed_passes(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_quality_flag

        assert _derive_quality_flag(
            {"detection_method": "atlas_relaxed"}, "measured"
        ) == "pass"

    def test_dsc_permissive_warns(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_quality_flag

        assert _derive_quality_flag(
            {"detection_method": "atlas_permissive"}, "measured"
        ) == "warn"

    def test_dsc_compound_tier_uses_worst_tier(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_quality_flag

        qc = {"detection_method": "atlas_relaxed+atlas_permissive"}
        assert _derive_quality_flag(qc, "measured") == "warn"

    def test_dce_score_fallback_thresholds(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_quality_flag

        assert _derive_quality_flag({"quality_score": 0.70}, "measured", "DCE") == "pass"
        assert _derive_quality_flag({"quality_score": 0.50}, "measured", "DCE") == "warn"
        assert _derive_quality_flag({"quality_score": 0.49}, "measured", "DCE") == "poor"

    def test_synthetic_forced_poor(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_quality_flag

        assert _derive_quality_flag({"quality_score": 0.95}, "synthetic") == "poor"

    def test_population_forced_poor(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_quality_flag

        assert _derive_quality_flag({"quality_score": 0.80}, "population") == "poor"

    def test_missing_score_warns(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_quality_flag

        assert _derive_quality_flag({}, "measured") == "warn"

    def test_dsc_external_warns(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_quality_flag

        qc = {"detection_method": "external", "quality_score": 0.95}
        assert _derive_quality_flag(qc, "external") == "warn"


class TestBuildProvenanceBlockDSC:
    def test_measured(self) -> None:
        from pmriprep.utils.aif_provenance import build_aif_provenance_block

        block = build_aif_provenance_block(
            {"aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.8}},
            modality="DSC",
        )
        assert block["Source"] == "measured"
        assert block["QualityFlag"] == "pass"
        assert block["CrossSubjectComparable"] is True
        assert block["GammaFitApplied"] is False
        assert block["QualityScore"] == pytest.approx(0.8)

    def test_synthetic(self) -> None:
        from pmriprep.utils.aif_provenance import build_aif_provenance_block

        block = build_aif_provenance_block(
            {
                "aif_detection": {"detection_method": "synthetic", "quality_score": 0.3},
                "synthetic_fallback_reason": "no_candidates",
            },
            modality="DSC",
        )
        assert block["Source"] == "synthetic"
        assert block["QualityFlag"] == "poor"
        assert block["CrossSubjectComparable"] is False
        assert block["SyntheticFallbackReason"] == "no_candidates"

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
            }},
            modality="DCE",
        )
        assert block["Source"] == "measured"
        assert block["QualityFlag"] == "pass"
        assert block["CrossSubjectComparable"] is True
        assert block["Confidence"] == "standard"
        assert block["UncertaintyInflationFactor"] == pytest.approx(1.0)
        assert block["Hematocrit"] == pytest.approx(0.42)
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
        assert "GammaFitApplied" not in block

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
        assert block["CrossSubjectComparable"] is False
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
    def test_pk_map_dependent(self) -> None:
        from pmriprep.utils.aif_provenance import apply_aif_sidecar

        sidecar: dict = {}
        qc = {"aif_detection": {"detection_method": "atlas", "quality_score": 0.8}}
        apply_aif_sidecar(sidecar, "ktrans", qc, modality="DCE")
        assert sidecar["AIFDependent"] is True
        assert sidecar["TemplateAIFUsage"] == "ratio_defensible"
        assert "PerfusionProcessing" in sidecar

    def test_pk_map_carries_peak_scaling(self) -> None:
        """Quantitative-map sidecars retain the canonical scaling fragment."""
        from pmriprep.utils.aif_provenance import apply_aif_sidecar

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

    def test_iauc_independent(self) -> None:
        from pmriprep.utils.aif_provenance import apply_aif_sidecar

        sidecar: dict = {}
        qc = {"aif_detection": {"detection_method": "atlas", "quality_score": 0.8}}
        apply_aif_sidecar(sidecar, "iauc60", qc, modality="DCE")
        assert sidecar["AIFDependent"] is False
        assert sidecar["TemplateAIFUsage"] == "safe"

    def test_model_free_independent(self) -> None:
        from pmriprep.utils.aif_provenance import apply_aif_sidecar

        sidecar: dict = {}
        qc = {"aif_detection": {"detection_method": "atlas", "quality_score": 0.8}}
        apply_aif_sidecar(sidecar, "ser", qc, modality="DCE")
        assert sidecar["AIFDependent"] is False

    def test_absolute_uncertainty_is_aif_dependent(self) -> None:
        from pmriprep.utils.aif_provenance import apply_aif_sidecar

        sidecar: dict = {}
        qc = {"aif_detection": {"detection_method": "atlas", "quality_score": 0.8}}
        apply_aif_sidecar(sidecar, "ktrans_uncertainty", qc, modality="DCE")
        assert sidecar["TemplateAIFUsage"] == "ratio_defensible"

    def test_unknown_map_raises(self) -> None:
        from pmriprep.utils.aif_provenance import apply_aif_sidecar

        with pytest.raises(KeyError, match="Unknown DCE map key"):
            apply_aif_sidecar({}, "nonexistent_map", {}, modality="DCE")


class TestRegistryInvariants:
    def test_dce_independent_subset_of_usage_registry(self) -> None:
        from pmriprep.utils.aif_provenance import (
            DCE_AIF_INDEPENDENT_MAPS,
            DCE_POPULATION_AIF_USAGE,
        )

        assert DCE_AIF_INDEPENDENT_MAPS <= frozenset(DCE_POPULATION_AIF_USAGE)

    def test_dsc_independent_subset_of_usage_registry(self) -> None:
        from pmriprep.utils.aif_provenance import (
            AIF_INDEPENDENT_MAPS,
            TEMPLATE_AIF_USAGE,
        )

        assert AIF_INDEPENDENT_MAPS <= frozenset(TEMPLATE_AIF_USAGE)

    def test_dce_independent_maps_all_safe(self) -> None:
        from pmriprep.utils.aif_provenance import (
            DCE_AIF_INDEPENDENT_MAPS,
            DCE_POPULATION_AIF_USAGE,
        )

        for key in DCE_AIF_INDEPENDENT_MAPS:
            assert DCE_POPULATION_AIF_USAGE[key] == "safe", (
                f"{key} is independent but not 'safe'"
            )

    def test_dsc_independent_maps_all_safe(self) -> None:
        from pmriprep.utils.aif_provenance import (
            AIF_INDEPENDENT_MAPS,
            TEMPLATE_AIF_USAGE,
        )

        for key in AIF_INDEPENDENT_MAPS:
            assert TEMPLATE_AIF_USAGE[key] == "safe", (
                f"{key} is independent but not 'safe'"
            )


class TestStampAifDependency:
    def test_registered_key_stamps_both_tags(self) -> None:
        from pmriprep.utils.aif_provenance import stamp_aif_dependency

        sidecar: dict = {}
        stamp_aif_dependency(sidecar, "ktrans", modality="DCE")
        assert sidecar["AIFDependent"] is True
        assert sidecar["TemplateAIFUsage"] == "ratio_defensible"

    def test_independent_key_stamps_false(self) -> None:
        from pmriprep.utils.aif_provenance import stamp_aif_dependency

        sidecar: dict = {}
        stamp_aif_dependency(sidecar, "ser", modality="DCE")
        assert sidecar["AIFDependent"] is False
        assert sidecar["TemplateAIFUsage"] == "safe"

    def test_unregistered_key_skips(self) -> None:
        from pmriprep.utils.aif_provenance import stamp_aif_dependency

        sidecar: dict = {}
        stamp_aif_dependency(sidecar, "nonexistent", modality="DCE")
        assert "AIFDependent" not in sidecar

    def test_empty_key_skips_silently(self) -> None:
        from pmriprep.utils.aif_provenance import stamp_aif_dependency

        sidecar: dict = {}
        stamp_aif_dependency(sidecar, "", modality="DCE")
        assert "AIFDependent" not in sidecar


class TestDetectionLevelValidation:
    def test_dce_atlas_valid(self) -> None:
        from pmriprep.utils.aif_provenance import _is_valid_detection_level

        assert _is_valid_detection_level("atlas", "DCE") is True

    def test_dce_population_valid(self) -> None:
        from pmriprep.utils.aif_provenance import _is_valid_detection_level

        assert _is_valid_detection_level("population", "DCE") is True

    def test_dce_dsc_tier_invalid(self) -> None:
        from pmriprep.utils.aif_provenance import _is_valid_detection_level

        assert _is_valid_detection_level("atlas_strict", "DCE") is False

    def test_dsc_compound_valid(self) -> None:
        from pmriprep.utils.aif_provenance import _is_valid_detection_level

        assert _is_valid_detection_level("atlas_strict+atlas_relaxed", "DSC") is True


class TestDSCRegression:
    def test_dsc_block_unchanged(self) -> None:
        from pmriprep.utils.aif_provenance import build_aif_provenance_block

        block = build_aif_provenance_block(
            {"aif_detection": {
                "detection_method": "atlas_strict",
                "quality_score": 0.8,
                "gamma_fit_applied": True,
            }},
            modality="DSC",
        )
        assert block["Source"] == "gamma_fit_measured"
        assert block["GammaFitApplied"] is True
        assert "Confidence" not in block
        assert "PopulationModel" not in block

    def test_dsc_apply_sidecar_unchanged(self) -> None:
        from pmriprep.utils.aif_provenance import apply_aif_sidecar

        sidecar: dict = {}
        qc = {"aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.8}}
        apply_aif_sidecar(sidecar, "cbf", qc, modality="DSC")
        assert sidecar["AIFDependent"] is True
        assert sidecar["TemplateAIFUsage"] == "ratio_defensible"
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
        from pmriprep.utils.aif_provenance import apply_aif_sidecar
        from pmriprep.utils.sidecar_builders import build_aif_timeseries_sidecar
        from pmriprep.workflows.dsc.datasinks import _build_aif_mask_meta

        qc = {
            "detection_method": "atlas_strict",
            "quality_score": 0.8,
            "n_aif_voxels_selected": 6,
            "gamma_fit_applied": True,
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
