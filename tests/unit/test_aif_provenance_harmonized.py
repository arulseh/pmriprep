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

    def test_dce_automatic(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_aif_source

        assert _derive_aif_source({"detection_method": "automatic"}, "DCE") == "measured"

    def test_dce_external(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_aif_source

        assert _derive_aif_source({"detection_method": "external"}, "DCE") == "external"

    def test_dce_ignores_gamma_fit(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_aif_source

        qc = {"detection_method": "atlas", "gamma_fit_applied": True}
        assert _derive_aif_source(qc, "DCE") == "measured"


class TestDeriveQualityFlag:
    def test_pass_threshold(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_quality_flag

        assert _derive_quality_flag({"quality_score": 0.70}, "measured") == "pass"

    def test_warn_threshold(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_quality_flag

        assert _derive_quality_flag({"quality_score": 0.50}, "measured") == "warn"

    def test_poor_threshold(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_quality_flag

        assert _derive_quality_flag({"quality_score": 0.49}, "measured") == "poor"

    def test_boundary_pass_warn(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_quality_flag

        assert _derive_quality_flag({"quality_score": 0.69}, "measured") == "warn"

    def test_synthetic_forced_poor(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_quality_flag

        assert _derive_quality_flag({"quality_score": 0.95}, "synthetic") == "poor"

    def test_population_forced_poor(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_quality_flag

        assert _derive_quality_flag({"quality_score": 0.80}, "population") == "poor"

    def test_missing_score_warns(self) -> None:
        from pmriprep.utils.aif_provenance import _derive_quality_flag

        assert _derive_quality_flag({}, "measured") == "warn"


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
            }},
            modality="DCE",
        )
        assert block["Source"] == "measured"
        assert block["QualityFlag"] == "pass"
        assert block["CrossSubjectComparable"] is True
        assert block["Confidence"] == "standard"
        assert block["UncertaintyInflationFactor"] == pytest.approx(1.0)
        assert block["Hematocrit"] == pytest.approx(0.42)
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

    def test_cv_safe(self) -> None:
        from pmriprep.utils.aif_provenance import apply_aif_sidecar

        sidecar: dict = {}
        qc = {"aif_detection": {"detection_method": "atlas", "quality_score": 0.8}}
        apply_aif_sidecar(sidecar, "ktrans_cv", qc, modality="DCE")
        assert sidecar["TemplateAIFUsage"] == "safe"

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

    def test_dsc_k2_independent(self) -> None:
        from pmriprep.utils.aif_provenance import apply_aif_sidecar

        sidecar: dict = {}
        qc = {"aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.8}}
        apply_aif_sidecar(sidecar, "k2", qc, modality="DSC")
        assert sidecar["AIFDependent"] is False
        assert sidecar["TemplateAIFUsage"] == "safe"
