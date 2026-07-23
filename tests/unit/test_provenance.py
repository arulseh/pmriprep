"""Unit tests for provenance enrichment utilities."""

import pytest

from pmriprep.utils.provenance import enrich_provenance_from_qc


def _make_dce_sidecar() -> dict:
    """Minimal DCE sidecar with PerfusionProcessing from Phase 1."""
    return {
        "PerfusionProcessing": {
            "ContrastAgent": {"GenericName": "gadobutrol", "Source": "bids_sidecar"},
            "Relaxivity": {"r1": 5.0, "units": "s-1.mM-1", "field_strength_T": 3.0},
            "ContrastDose": {"standard_dose_mmol_kg": 0.1, "dose_multiplier": 1.0},
        }
    }


def _make_dsc_sidecar() -> dict:
    """Minimal DSC sidecar with PerfusionProcessing from Phase 1."""
    return {
        "PerfusionProcessing": {
            "ContrastDose": {"standard_dose_mmol_kg": 0.1, "dose_multiplier": 1.0},
            "Deconvolution": {"method": "oSVD"},
            "LeakageCorrection": {"method": "bsw"},
            "PreloadDetection": {"Source": "auto_detected"},
        }
    }


class TestEnrichDCE:
    """Tests for DCE provenance enrichment."""

    def test_full_qc_enrichment(self) -> None:
        """Full QC metrics produce BloodT1, T1Mapping, Hematocrit, AIF, PKModeling."""
        sidecar = _make_dce_sidecar()
        qc = {
            "t1_mapping": {"method": "vfa", "b1_correction": "AFI"},
            "aif_detection": {
                "aif_source": "auto",
                "n_aif_voxels_selected": 8,
                "peak_mm_mean": 6.2,
                "fwhm_seconds_mean": 12.3,
                "quality_score": 0.82,
                "blood_t1_seconds": 1.664,
                "arterial_cascade_tier": 1,
                "hemisphere_coverage": "bilateral",
            },
            "pk_modeling": {
                "model": "extended_tofts",
                "inference": "vi",
                "convergence_rate": 0.98,
                "r2_median": 0.87,
                "qc_voxel_rsquared": 0.95,
            },
        }
        result = enrich_provenance_from_qc(sidecar, qc, "DCE")
        perf = result["PerfusionProcessing"]

        assert "BloodT1" in perf
        assert perf["BloodT1"]["value_s"] == pytest.approx(1.664, abs=0.001)
        assert perf["BloodT1"]["Source"] == "literature_lookup"

        assert "Hematocrit" in perf
        assert perf["Hematocrit"]["value"] == pytest.approx(0.42)

        assert "T1Mapping" in perf
        assert perf["T1Mapping"]["method"] == "vfa"
        assert perf["T1Mapping"]["B1_correction"] == "AFI"

        assert "AIF" in perf
        assert perf["AIF"]["method"] == "auto"
        assert perf["AIF"]["n_voxels"] == 8
        assert perf["AIF"]["arterial_cascade_tier"] == 1

        assert "PKModeling" in perf
        assert perf["PKModeling"]["model"] == "extended_tofts"
        assert perf["PKModeling"]["inference"] == "vi"
        assert perf["PKModeling"]["convergence_rate"] == pytest.approx(0.98, abs=0.001)
        assert perf["PKModeling"]["voxel_rsquared"] == pytest.approx(0.95, abs=0.001)

    def test_signal_conversion_qc(self) -> None:
        """Signal-conversion diagnostics are retained in map provenance."""
        sidecar = _make_dce_sidecar()
        conversion_qc = {
            "t1_invalid_voxels": 12,
            "b1_missing_voxels": 34,
            "e1_bounded_fraction": 0.00125,
        }
        result = enrich_provenance_from_qc(
            sidecar,
            {"signal_conversion": conversion_qc},
            "DCE",
        )
        assert (
            result["PerfusionProcessing"]["SignalConversionQC"]
            == conversion_qc
        )

    def test_population_aif(self) -> None:
        """Population AIF includes model name and dose factor."""
        sidecar = _make_dce_sidecar()
        qc = {
            "aif_detection": {
                "aif_source": "population",
                "population_aif": True,
                "dose_factor": 0.5,
                "n_aif_voxels_selected": 0,
                "blood_t1_seconds": 1.664,
            },
        }
        result = enrich_provenance_from_qc(sidecar, qc, "DCE")
        aif = result["PerfusionProcessing"]["AIF"]
        assert aif["population_model"] == "parker"
        assert aif["dose_factor"] == pytest.approx(0.5)

    def test_assumed_t1_segmented(self) -> None:
        """Assumed T1 with tissue segmentation records WM/GM values."""
        sidecar = _make_dce_sidecar()
        qc = {
            "t1_mapping": {
                "method": "assumed",
                "b1_correction": "none",
                "tissue_t1_used": True,
                "mean_t1_wm_ms": 800.0,
                "mean_t1_gm_ms": 1300.0,
            },
        }
        result = enrich_provenance_from_qc(sidecar, qc, "DCE")
        t1 = result["PerfusionProcessing"]["T1Mapping"]
        assert t1["tissue_segmented"] is True
        assert t1["T1_WM_ms"] == pytest.approx(800.0)
        assert t1["T1_GM_ms"] == pytest.approx(1300.0)

    def test_assumed_t1_uniform(self) -> None:
        """Assumed T1 without segmentation records uniform values."""
        sidecar = _make_dce_sidecar()
        qc = {
            "t1_mapping": {
                "method": "assumed",
                "b1_correction": "none",
                "tissue_t1_used": False,
                "mean_t1_wm_ms": 1200.0,
                "mean_t1_gm_ms": 1200.0,
            },
        }
        result = enrich_provenance_from_qc(sidecar, qc, "DCE")
        t1 = result["PerfusionProcessing"]["T1Mapping"]
        assert t1["tissue_segmented"] is False
        assert t1["T1_WM_ms"] == pytest.approx(1200.0)

    def test_missing_t1_mapping(self) -> None:
        """Missing t1_mapping in QC is silently skipped."""
        sidecar = _make_dce_sidecar()
        qc = {"aif_detection": {"aif_source": "auto", "n_aif_voxels_selected": 5}}
        result = enrich_provenance_from_qc(sidecar, qc, "DCE")
        assert "T1Mapping" not in result["PerfusionProcessing"]

    def test_empty_qc(self) -> None:
        """Empty qc_metrics produces no enrichment."""
        sidecar = _make_dce_sidecar()
        result = enrich_provenance_from_qc(sidecar, {}, "DCE")
        perf = result["PerfusionProcessing"]
        assert "AIF" not in perf
        assert "T1Mapping" not in perf
        assert "Hematocrit" in perf  # always added for DCE

    def test_pk_model_auto_selection(self) -> None:
        """Auto model selection includes candidate models and distribution."""
        sidecar = _make_dce_sidecar()
        qc = {
            "pk_modeling": {
                "model": "auto",
                "inference": "vi",
                "candidate_models": ["patlak", "extended_tofts"],
                "patlak_fraction": 0.85,
                "extended_tofts_fraction": 0.15,
            },
        }
        result = enrich_provenance_from_qc(sidecar, qc, "DCE")
        pk = result["PerfusionProcessing"]["PKModeling"]
        assert pk["model"] == "auto"
        assert pk["model_selection"]["CandidateModels"] == ["patlak", "extended_tofts"]
        assert pk["model_selection"]["ModelDistribution"]["patlak"] == pytest.approx(0.85)


class TestEnrichDSC:
    """Tests for DSC provenance enrichment."""

    def test_full_qc_enrichment(self) -> None:
        """Full QC produces Deconvolution details, LeakageCorrection, AIF, BaselineFrames."""
        sidecar = _make_dsc_sidecar()
        qc = {
            "deconvolution": {
                "threshold_mean": 0.065,
                "oi_mean": 0.08,
            },
            "preload": {
                "preload_correction_applied": True,
                "k2_median": 0.008,
                "preload_classification": "inadequate",
                # Advisory key: preload_baseline_delta_r2_advisory.
                "preload_baseline_delta_r2_advisory": 0.12,
                "preload_classification_source": "k2_primary",
                "preload_k2_classification": "inadequate",
            },
            "aif_detection": {
                "detection_method": "atlas_strict",
                "n_aif_voxels_selected": 6,
                "quality_score": 0.91,
                "aif_metrics": {"fwhm": 4.0},
                "arrival_time_seconds": 3.2,
                "baseline_frames_used": 10,
                "baseline_detection_method": "auto_p99",
            },
            "normalization_factor": 0.023,
        }
        result = enrich_provenance_from_qc(sidecar, qc, "DSC")
        perf = result["PerfusionProcessing"]

        assert perf["Deconvolution"]["threshold_mean"] == pytest.approx(0.065, abs=0.001)
        assert perf["Deconvolution"]["oscillation_index_mean"] == pytest.approx(0.08, abs=0.001)

        assert perf["LeakageCorrection"]["applied"] is True
        assert perf["LeakageCorrection"]["K2_median"] == pytest.approx(0.008, abs=0.001)

        assert perf["PreloadDetection"]["classification"] == "inadequate"
        assert perf["PreloadDetection"]["baseline_delta_r2_advisory"] == pytest.approx(0.12, abs=0.001)
        assert perf["PreloadDetection"]["classification_source"] == "k2_primary"
        assert perf["PreloadDetection"]["k2_classification"] == "inadequate"

        assert perf["AIF"]["method"] == "atlas_strict"
        assert perf["AIF"]["fwhm_frames"] == pytest.approx(4.0)
        assert perf["AIF"]["synthetic"] is False

        assert perf["QuantificationMode"] == "relative"
        assert perf["NormalizationReference"]["tissue"] == "NAWM"
        assert perf["NormalizationReference"]["cbv_factor"] == pytest.approx(0.023, abs=0.001)

        assert perf["BaselineFrames"]["n_frames"] == 10
        assert perf["BaselineFrames"]["detection_method"] == "auto_p99"

    def test_synthetic_aif(self) -> None:
        """Synthetic AIF is derived from detection_method."""
        sidecar = _make_dsc_sidecar()
        qc = {
            "aif_detection": {"detection_method": "synthetic", "n_aif_voxels_selected": 0},
        }
        result = enrich_provenance_from_qc(sidecar, qc, "DSC")
        assert result["PerfusionProcessing"]["AIF"]["synthetic"] is True

    def test_tikhonov_lambda(self) -> None:
        """Tikhonov deconvolution includes lambda_mean."""
        sidecar = _make_dsc_sidecar()
        qc = {"deconvolution": {"lambda_mean": 0.15}}
        result = enrich_provenance_from_qc(sidecar, qc, "DSC")
        assert result["PerfusionProcessing"]["Deconvolution"]["lambda_mean"] == pytest.approx(0.15)

    def test_parametric_fe_plateau_and_fallback(self) -> None:
        """CPI/Vascular runs propagate fe_plateau_rate and fallback_rate."""
        sidecar = _make_dsc_sidecar()
        qc = {
            "deconvolution": {
                "fe_plateau_rate": 0.0001,
                "fallback_rate": 0.07,
            },
        }
        result = enrich_provenance_from_qc(sidecar, qc, "DSC")
        deconv = result["PerfusionProcessing"]["Deconvolution"]
        assert deconv["fe_plateau_rate"] == pytest.approx(0.0001, abs=1e-6)
        assert deconv["fallback_rate"] == pytest.approx(0.07, abs=1e-6)
        # Old key must not have been re-introduced.
        assert "convergence_rate" not in deconv

    def test_no_normalization_factor(self) -> None:
        """NormalizationReference omitted when no normalization_factor."""
        sidecar = _make_dsc_sidecar()
        qc = {"deconvolution": {}}
        result = enrich_provenance_from_qc(sidecar, qc, "DSC")
        assert "NormalizationReference" not in result["PerfusionProcessing"]

    def test_empty_qc(self) -> None:
        """Empty QC still adds QuantificationMode."""
        sidecar = _make_dsc_sidecar()
        result = enrich_provenance_from_qc(sidecar, {}, "DSC")
        assert result["PerfusionProcessing"]["QuantificationMode"] == "relative"

    def test_cross_modality_no_crash(self) -> None:
        """DCE enrichment with DSC-shaped QC doesn't crash."""
        sidecar = _make_dce_sidecar()
        dsc_qc = {
            "deconvolution": {"threshold_mean": 0.065},
            "preload": {"preload_classification": "none"},
        }
        result = enrich_provenance_from_qc(sidecar, dsc_qc, "DCE")
        assert "PerfusionProcessing" in result
