"""Unit tests for provenance enrichment utilities."""

import pytest

from pmriprep.utils.provenance import enrich_provenance_from_qc


def _make_dce_sidecar() -> dict:
    """Minimal DCE sidecar with base PerfusionProcessing metadata."""
    return {
        "PerfusionProcessing": {
            "ContrastAgent": {"GenericName": "gadobutrol", "Source": "bids_sidecar"},
            "Relaxivity": {"r1": 5.0, "units": "s-1.mM-1", "field_strength_T": 3.0},
            "ContrastDose": {"standard_dose_mmol_kg": 0.1, "dose_multiplier": 1.0},
        }
    }


def _make_dsc_sidecar() -> dict:
    """Minimal DSC sidecar with base PerfusionProcessing metadata."""
    return {
        "PerfusionProcessing": {
            "ContrastDose": {"standard_dose_mmol_kg": 0.1, "dose_multiplier": 1.0},
            "Deconvolution": {"method": "oSVD"},
            "LeakageCorrection": {"method": "bsw"},
            "PreloadDetection": {"source": "unknown"},
        }
    }


class TestEnrichDCE:
    """Tests for DCE provenance enrichment."""

    def test_full_qc_enrichment(self) -> None:
        """Full QC metrics produce reference, mapping, AIF, and PK provenance."""
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
                "peak_scaling_enabled": True,
                "peak_scaling_applied": True,
                "raw_peak_mm": 4.0,
                "emitted_peak_mm": 8.0,
                "peak_scaling_factor": 2.0,
                "peak_scaling_required_factor": 2.0,
                "peak_scaling_policy": "one_sided_lower_bound",
                "peak_scaling_trigger": "below_expected_minimum",
            },
            "pk_modeling": {
                "model": "extended_tofts",
                "inference": "vi",
                "voxel_accounting": {
                    "optimizer_converged": {
                        "count": 98,
                        "fraction_of_attempted": 0.98,
                    }
                },
                "qc_acceptance_policy": {
                    "acquisition": {"minimum_tsnr": 10.0},
                    "fit": {
                        "minimum_r_squared_exclusive": 0.0,
                    },
                },
                "qc_component_failures": {
                    "scope": "fitted_voxels",
                    "counts_are_overlapping": True,
                    "r_squared": 12,
                },
                "identification_qc": {
                    "basis": "selected_nlls", "by_model": {"extended_tofts": {"evaluated": 98}},
                },
                "timing_diagnostics": {
                    "basis": "selected_nlls", "report_only": True,
                    "acquisition_frame_interval_seconds": 7.4,
                },
                "r2_median": 0.87,
                "qc_voxel_rsquared": 0.95,
            },
            "model_suitability": {
                "2cxm": {
                    "status": "WARN",
                    "reasons": [
                        "temporal_sampling_outside_auto_2cxm_eligibility"
                    ],
                }
            },
        }
        result = enrich_provenance_from_qc(sidecar, qc, "DCE")
        perf = result["PerfusionProcessing"]

        assert "BloodT1" not in perf
        assert "BloodT1Reference" in perf
        blood_t1 = perf["BloodT1Reference"]
        assert blood_t1["value_s"] == pytest.approx(1.664, abs=0.001)
        assert blood_t1["Source"] == "literature_lookup"
        assert blood_t1["Role"] == "aif_expected_peak_and_qc_reference"
        assert blood_t1["AppliedToArterialSignalConversion"] is False

        assert "Hematocrit" in perf
        assert perf["Hematocrit"]["value"] == pytest.approx(0.42)

        assert "T1Mapping" in perf
        assert perf["T1Mapping"]["method"] == "vfa"
        assert perf["T1Mapping"]["B1_correction"] == "AFI"

        assert "AIF" in perf
        assert perf["AIF"]["DetectionLevel"] == "atlas"
        assert perf["AIF"]["NumberOfVoxelsSelected"] == 8
        assert "ArterialCascadeTier" not in perf["AIF"]
        assert perf["AIF"]["PeakScaling"]["RawPeak"] == pytest.approx(4.0)
        assert perf["AIF"]["PeakScaling"]["EmittedPeak"] == pytest.approx(8.0)
        assert (
            perf["AIF"]["PeakScaling"]["Policy"]
            == "one_sided_lower_bound"
        )

        assert "PKModeling" in perf
        assert perf["PKModeling"]["model"] == "extended_tofts"
        assert perf["PKModeling"]["inference"] == "vi"
        for key in ("IdentificationQC", "TimingDiagnostics", "VoxelAccounting",
                    "QCAcceptancePolicy", "QCComponentFailures"):
            assert key not in perf["PKModeling"]
        assert qc["pk_modeling"]["identification_qc"]
        assert perf["PKModeling"]["voxel_rsquared"] == pytest.approx(0.95, abs=0.001)
        assert "ModelSuitability" not in perf

    def test_signal_conversion_qc(self) -> None:
        """Maps retain applied conversion settings without run-wide counters."""
        sidecar = _make_dce_sidecar()
        conversion_qc = {
            "t1_invalid_voxels": 12,
            "b1_missing_voxels": 34,
            "e1_bounded_fraction": 0.00125,
            "t1_source": "assumed", "assumed_t1_ms": 1400.0,
        }
        result = enrich_provenance_from_qc(
            sidecar,
            {"signal_conversion": conversion_qc},
            "DCE",
        )
        assert result["PerfusionProcessing"]["SignalConversion"] == {
            "t1_source": "assumed", "assumed_t1_ms": 1400.0,
        }
        assert "SignalConversionQC" not in result["PerfusionProcessing"]

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
        assert aif["PopulationModel"] == "parker"
        assert aif["DoseFactor"] == pytest.approx(0.5)

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

    def test_assumed_t1_numeric_policy(self) -> None:
        """Uniform assumed T1 records its value, field strength, and policy."""
        sidecar = _make_dce_sidecar()
        qc = {
            "t1_mapping": {
                "method": "assumed",
                "b1_correction": "precomputed",
                "assumed_t1_ms": 1200.0,
                "assumed_t1_field_strength_t": 3.0,
                "assumed_t1_policy": "uniform_field_strength_reference",
            },
        }
        result = enrich_provenance_from_qc(sidecar, qc, "DCE")
        t1 = result["PerfusionProcessing"]["T1Mapping"]
        assert t1["AssumedT1_ms"] == pytest.approx(1200.0)
        assert t1["FieldStrength_T"] == pytest.approx(3.0)
        assert t1["assumption_policy"] == "uniform_field_strength_reference"
        assert t1["B1_correction"] == "precomputed"

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

    def test_disabled_leakage_correction_is_explicit(self) -> None:
        """Configuration-driven omission is preserved in provenance."""
        result = enrich_provenance_from_qc(
            _make_dsc_sidecar(),
            {
                "preload": {
                    "preload_correction_applied": False,
                    "leakage_correction_disabled": True,
                },
            },
            "DSC",
        )

        leakage = result["PerfusionProcessing"]["LeakageCorrection"]
        assert leakage["applied"] is False
        assert leakage["disabled_by_configuration"] is True

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
                "nawm_auc_raw_median": 1.2,
                "nawm_auc_corrected_median": 1.1,
                "preload_source": "unknown",
                "preload_administered_dose_fraction": None,
                # Advisory key: preload_baseline_delta_r2_advisory.
                "preload_baseline_delta_r2_advisory": 0.12,
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
        assert perf["LeakageCorrection"]["method"] == "bsw"
        assert "Method" not in perf["LeakageCorrection"]
        assert perf["LeakageCorrection"]["nawm_auc_raw_median"] == pytest.approx(1.2)
        assert perf["LeakageCorrection"]["nawm_auc_corrected_median"] == pytest.approx(1.1)

        assert perf["PreloadDetection"]["source"] == "unknown"
        assert perf["PreloadDetection"]["baseline_delta_r2_advisory"] == pytest.approx(0.12, abs=0.001)
        assert "administered_dose_fraction" not in perf["PreloadDetection"]

        assert perf["AIF"]["DetectionLevel"] == "atlas_strict"
        assert "FWHM" not in perf["AIF"]
        assert perf["AIF"]["Source"] == "measured"

        assert perf["QuantificationMode"] == "relative"
        assert "NormalizationReference" not in perf

        assert perf["BaselineFrames"]["n_frames"] == 10
        assert perf["BaselineFrames"]["detection_method"] == "auto_p99"

    def test_unavailable_aif(self) -> None:
        """Unavailable AIF provenance follows detection outcome."""
        sidecar = _make_dsc_sidecar()
        qc = {
            "aif_detection": {"detection_method": "unavailable", "n_aif_voxels_selected": 0},
        }
        result = enrich_provenance_from_qc(sidecar, qc, "DSC")
        assert result["PerfusionProcessing"]["AIF"]["Source"] == "unavailable"

    def test_tikhonov_lambda(self) -> None:
        """Tikhonov sidecars retain selection and residue-integration semantics."""
        sidecar = _make_dsc_sidecar()
        deconvolution = {
            "lambda_mean": 0.15,
            "regularization_selection": "lcurve_upper_extremum",
            "regularization_penalty": "modified_first_difference",
            "penalty_leading_samples": 2,
            "lambda_grid_size": 100,
            "lambda_search_bounds_relative": [0.001, 10.0],
            "lambda_continuous_refinement": False,
            "residue_cbv_integration": "full_length_signed_trapezoid",
            "residue_post_fit_clipping": False,
            "lcurve_selection_counts": {
                "positive_corner": 7, "nonpositive_corner": 2,
                "upper_endpoint": 1, "inactive": 3,
            },
            "residue_diagnostics": {"negative_cbv_voxels": 2},
            "lcurve_corner_fraction": 0.9,
            "lcurve_upper_endpoint_fraction": 0.1,
        }
        qc = {"deconvolution": deconvolution}
        result = enrich_provenance_from_qc(sidecar, qc, "DSC")
        for key, value in deconvolution.items():
            assert result["PerfusionProcessing"]["Deconvolution"][key] == value

    def test_parametric_convergence_and_fallback(self) -> None:
        """Parametric runs propagate solver convergence and fallback rates."""
        sidecar = _make_dsc_sidecar()
        qc = {
            "deconvolution": {
                "solver_convergence_rate": 0.75,
                "fallback_rate": 0.07,
            },
        }
        result = enrich_provenance_from_qc(sidecar, qc, "DSC")
        deconv = result["PerfusionProcessing"]["Deconvolution"]
        assert deconv["solver_convergence_rate"] == pytest.approx(0.75)
        assert deconv["fallback_rate"] == pytest.approx(0.07, abs=1e-6)
        assert "fe_plateau_rate" not in deconv
        assert "convergence_rate" not in deconv

    def test_cpi_provenance_uses_executed_options(self) -> None:
        """Runtime CPI choices replace static construction-time assumptions."""
        sidecar = _make_dsc_sidecar()
        sidecar["PerfusionProcessing"]["Deconvolution"].update({
            "method": "cpi",
            "inference": "vi",
            "fit_domain": "signal",
            "cpi_model": "continuous_cubic_residue",
        })
        control_points = {
            "cpi": {
                "requested": 0,
                "selected": 9,
                "allowed": [5, 6, 7, 8, 9, 10, 11, 12],
                "selection_method": "automatic",
            }
        }
        qc = {
            "deconvolution": {
                "method": "cpi",
                "inference": "svb",
                "fit_domain": "signal",
                "cpi_dispersion": "auto",
                "cpi_infer_delay": True,
                "cpi_control_points": control_points,
                "cpi_leakage_method": "none",
                "cpi_fallback": "osvd",
                "cpi_selection_stage": "ordinary_vi_before_spatial_refinement",
                "cpi_gamma_fraction": 0.25,
                "spatial_sweeps": 3,
            }
        }
        result = enrich_provenance_from_qc(sidecar, qc, "DSC")
        deconv = result["PerfusionProcessing"]["Deconvolution"]
        assert deconv["inference"] == "svb"
        assert deconv["cpi_dispersion"] == "auto"
        assert deconv["cpi_infer_delay"] is True
        assert deconv["cpi_control_points"] == control_points
        assert deconv["cpi_leakage_method"] == "none"
        assert deconv["cpi_fallback"] == "osvd"
        assert deconv["cpi_gamma_fraction"] == pytest.approx(0.25)
        assert deconv["spatial_sweeps"] == 3

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
            "preload": {"preload_source": "unknown"},
        }
        result = enrich_provenance_from_qc(sidecar, dsc_qc, "DCE")
        assert "PerfusionProcessing" in result


def test_partial_dsc_provenance_records_configured_and_executed_methods() -> None:
    sidecar = {"PerfusionProcessing": {"Deconvolution": {"method": "cpi", "fit_domain": "signal"}}}
    result = enrich_provenance_from_qc(sidecar, {
        "deconvolution": {"configured_method": "cpi", "deconvolution_performed": False,
                          "deconvolution_unavailable_reason": "aif_unavailable",
                          "processing_status": "partial"},
        "aif_detection": {"detection_method": "unavailable", "detection_status": "failed"},
    }, "DSC")
    assert result["PerfusionProcessing"]["Deconvolution"] == {
        "configured_method": "cpi", "performed": False,
        "unavailable_reason": "aif_unavailable", "processing_status": "partial",
    }
