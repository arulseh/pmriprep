"""Jinja render tests for pmriprep HTML report templates."""

from pathlib import Path

import pytest
from jinja2 import ChainableUndefined, Environment, FileSystemLoader

from pmriprep.utils.report_utils import _drop_report_nulls

TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "pmriprep" / "templates"


@pytest.fixture
def env():
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=ChainableUndefined,
    )


def _dce_ctx_with_dsc_blocks() -> dict:
    return {
        "qc_metrics": {
            "preprocessing": {},
            "pk_modeling": {
                "dsc_cbv_constraint": {
                    "mode": "skip",
                    "reason": "No DSC data were available for a vp prior.",
                    "skip_reason": "no_dsc_run",
                }
            },
            "dsc_dce_validation": {
                "skipped": True,
                "status": "SKIPPED",
                "reason": "No DSC data were available for a vp prior.",
                "skip_reason": "no_dsc_run",
            },
        },
        "images": {},
        "display_status": {
            "peak_enh": "",
            "baseline_tsnr": "",
            "xval_correlation": "",
            "outliers": "",
        },
    }


def _minimal_parent_ctx(dsc=None, dce=None) -> dict:
    return {
        "subject_id": "sub-test",
        "session_id": "ses-test",
        "pipeline_version": "0.26.6-test",
        "dsc": dsc,
        "dce": dce,
        "anat": None,
        "tumor_seg": None,
        "summary": {"participant": {}, "sessions": [], "qc_summary": {}},
        "about": {},
        "methods_text": "",
        "generated_at": "2026-04-19T00:00:00Z",
    }


def _assert_in_order(rendered: str, *labels: str) -> None:
    """Assert that each visible label follows the preceding label."""
    positions = [rendered.index(label) for label in labels]
    assert positions == sorted(positions)


def test_parent_render_hides_dsc_blocks_when_no_dsc(env):
    """On DCE-only runs, DSC CBV Constraint and DSC-DCE Cross-Validation sections must NOT render."""
    template = env.get_template("report_unified.html")
    ctx = _minimal_parent_ctx(dsc=None, dce=_dce_ctx_with_dsc_blocks())
    rendered = template.render(**ctx)
    assert "<h3>DSC CBV Constraint</h3>" not in rendered
    assert "<h3>DSC-DCE Cross-Validation</h3>" not in rendered


@pytest.mark.parametrize(
    ("template_name", "modality"),
    [
        ("sections/section_dsc.html", "DSC"),
        ("sections/section_dce.html", "DCE"),
    ],
)
def test_registration_description_is_concise(env, template_name, modality):
    """Perfusion registration sections use parallel concise descriptions."""
    template = env.get_template(template_name)
    context = {
        modality.lower(): {
            "qc_metrics": {},
            "images": {"registration": "registration.png"},
            "display_status": {},
        },
    }
    rendered = template.render(**context)
    assert f"Multi-slice overlay showing {modality}-T1w registration." in rendered
    assert "intensity contour" not in rendered


def test_dce_prior_informed_diagnostic_is_not_labeled_validation(env):
    """A DSC-informed vp fit is explicitly reported as non-independent."""
    template = env.get_template("sections/section_dce.html")
    dce = {
        "qc_metrics": {
            "dsc_dce_validation": {
                "status": "PRIOR_INFORMED",
                "prior_informed": True,
                "prior_relative_sd": 0.5,
                "prior_informed_fraction": 0.75,
                "prior_departure": {
                    "z_departure_median": 0.96,
                    "z_departure_p90": 1.33,
                    "frac_prior_dominated": 0.18,
                    "frac_strong_departure": 0.032,
                },
                "dsc_preload_configured": True,
                "metrics": {},
                "interpretation": (
                    "DSC nCBV supplied the vp prior used by retained Bayesian fits; "
                    "the comparison is not independent."
                ),
            },
        },
        "images": {
            "dsc_dce_xval": "prior.png",
            "dsc_dce_region_xval": "prior-region.png",
        },
        "display_status": {},
    }

    rendered = template.render(dce=dce, dsc={"present": True})

    assert "DSC-Informed vp Prior Diagnostics" in rendered
    assert "Prior Informed" in rendered
    assert "PRIOR_INFORMED" not in rendered
    assert "Prior Relative SD" in rendered
    assert "50%" in rendered
    assert "75%" in rendered
    assert "Independence" not in rendered
    assert "Median Absolute Prior Departure" in rendered
    assert "0.96 SD" in rendered
    assert "90th-Percentile Absolute Departure" in rendered
    assert "1.33 SD" in rendered
    assert "&lt; 0.5 Prior SD" in rendered
    assert "18%" in rendered
    assert "&gt; 2 Prior SD" in rendered
    assert "3%" in rendered
    assert '<div class="metric-label">Notes</div>' in rendered
    assert '<div class="metric-note">' in rendered
    assert "including retained fits that did not use the prior" in rendered
    assert "prior-departure summaries above use only unsmoothed retained fits" in rendered
    assert "Preload Config" not in rendered
    assert "DSC-informed vp prior applied" not in rendered
    assert "prior-region.png" in rendered
    _assert_in_order(
        rendered,
        '<div class="metric-label">Status</div>',
        '<div class="metric-label">Prior Relative SD</div>',
        '<div class="metric-label">Retained Fits Using Prior</div>',
        '<div class="metric-label">Median Absolute Prior Departure</div>',
    )
    assert '<div class="metric-label">Interpretation</div>' not in rendered


def test_dce_signal_conversion_qc_renders(env):
    """Conversion fallback and bounded-inversion accounting reaches HTML."""
    template = env.get_template("sections/section_dce.html")
    dce = {
        "qc_metrics": {
            "preprocessing": {},
            "b1_mapping": {"method": "afi"},
            "t1_mapping": {"t1_method": "vfa"},
            "signal_conversion": {
                "t1_invalid_voxels": 12,
                "t1_invalid_fraction": 0.006,
                "b1_missing_voxels": 34,
                "b1_missing_fraction": 0.017,
                "e1_bounded_fraction": 0.00125,
            },
        },
        "images": {"preprocessing_qc": "preproc.png"},
        "display_status": {},
    }
    rendered = template.render(dce=dce)
    assert "Invalid T1 in DCE Space" in rendered
    assert ">12 (0.60%)<" in rendered
    assert "Missing B1 in DCE Space" in rendered
    assert "34 (1.70%)" in rendered
    assert "34 (1.70% brain support)" not in rendered
    assert "SPGR Boundary Hits" in rendered
    assert "0.12%" in rendered
    assert "brain-supported samples" not in rendered
    assert rendered.index("Missing B1 in DCE Space") < rendered.index(
        "T1 Mapping Quality"
    )
    assert rendered.index("Invalid T1 in DCE Space") < rendered.index(
        "DCE Preprocessing QC"
    )


def test_dce_b1_cards_follow_qc_hierarchy(env):
    """B1 status, summaries, support, and processing fields are ordered."""
    template = env.get_template("sections/section_dce.html")
    dce = {
        "qc_metrics": {
            "b1_mapping": {
                "b1_status": "PASS",
                "method": "afi",
                "median_b1": 1.0,
                "mean_b1": 1.01,
                "std_b1": 0.05,
                "pct_correction_gt_10": 12.0,
                "pct_correction_gt_20": 3.0,
                "smoothing_fwhm_mm": 4.0,
                "clip_low": 0.5,
                "clip_high": 1.5,
            },
            "signal_conversion": {
                "b1_missing_voxels": 10,
                "b1_missing_fraction": 0.01,
            },
        },
        "images": {},
        "display_status": {},
    }
    rendered = template.render(dce=dce)
    _assert_in_order(
        rendered,
        "B1 Status",
        "B1 Method",
        "Median B1+",
        "Mean B1+",
        "Std B1+",
        "&gt;10% Correction",
        "&gt;20% Correction",
        "Missing B1 in DCE Space",
        "Smoothing FWHM",
        "Clip Range",
    )


def test_dce_disabled_b1_is_reported_without_figure(env):
    """An explicit configuration skip remains visible as text-only QC."""
    template = env.get_template("sections/section_dce.html")
    dce = {
        "qc_metrics": {
            "b1_mapping": {
                "method": "none",
                "b1_status": "Skipped by configuration",
                "skipped": True,
            },
        },
        "images": {},
        "display_status": {},
    }

    rendered = template.render(dce=dce)

    assert "B1+ Mapping Quality" in rendered
    assert "Skipped by configuration" in rendered
    assert "B1+ Mapping QC" not in rendered


def test_dce_preprocessing_cards_align_with_dsc_order(env):
    """DCE shared acquisition/QC fields use the common DSC/DCE order."""
    template = env.get_template("sections/section_dce.html")
    preprocessing = {
        "motion_status": "PASS",
        "baseline_frames": 5,
        "tr_ms": 3.0,
        "volume_tr_s": 4.0,
        "te_ms": 1.0,
        "n_volumes": 60,
        "voxel_size_mm": [1.0, 1.0, 4.0],
        "n_brain_voxels": 100,
        "peak_enh_pct": 30.0,
        "mean_tsnr": 20.0,
        "mean_cnr": 8.0,
        "baseline_tsnr": 25.0,
    }
    dce = {
        "qc_metrics": {"preprocessing": preprocessing},
        "images": {"preprocessing_qc": "preproc.png"},
        "display_status": {},
    }
    rendered = template.render(dce=dce)
    _assert_in_order(
        rendered,
        "DCE Preprocessing QC",
        "TR (pulse)",
        "Volume TR",
        "TE",
        "Volumes",
        "Voxel Size",
        "Brain Voxels",
        "Peak Enhancement",
        "Mean tSNR",
        "Mean CNR",
        "Baseline Frames",
        "Mean Baseline tSNR",
    )
    assert rendered.index("Baseline Frames") > rendered.index("DCE Preprocessing QC")


def test_dce_aif_cards_follow_qc_hierarchy(env):
    """DCE AIF identity/support precedes peak scaling and shape metrics."""
    template = env.get_template("sections/section_dce.html")
    aif = {
        "aif_status": "PASS",
        "detection_method": "measured",
        "n_aif_voxels_selected": 5,
        "n_aif_voxels_filtered_out": 2,
        "quality_score": 0.8,
        "cnr_mean": 10.0,
        "emitted_peak_mm": 5.0,
        "raw_peak_mm": 4.0,
        "peak_scaling_policy": "one_sided_lower_bound",
        "peak_scaling_applied": True,
        "peak_scaling_factor": 1.25,
        "peak_scaling_trigger": "below_expected_minimum",
        "final_fwhm_seconds": 8.0,
        "washout_mean": 0.3,
        "final_auc": 30.0,
    }
    dce = {
        "qc_metrics": {"aif_detection": aif},
        "images": {},
        "display_status": {},
    }
    rendered = template.render(dce=dce)
    _assert_in_order(
        rendered,
        "AIF Status",
        "Method",
        "AIF Voxels",
        "Quality Score",
        "CNR",
        "Emitted Peak",
        "Raw Peak",
        "Peak Scaling",
        "Peak Scale Factor",
        "FWHM",
        "Washout",
        "AUC",
    )
    assert "Peak Scaling Trigger" not in rendered


def test_dce_vfa_motion_warning_renders(env):
    """Potential inter-VFA motion is visible without implying correction."""
    template = env.get_template("sections/section_dce.html")
    warning = (
        "Potential inter-VFA misalignment detected; VFA images were not "
        "resampled. Review source images and T1-map QC."
    )
    dce = {
        "qc_metrics": {
            "t1_mapping": {
                "t1_method": "nlls",
                "vfa_preprocessing": {
                    "motion_assessment": {
                        "qc_status": "WARN",
                        "correction_applied": False,
                        "warning": warning,
                    },
                },
            },
        },
        "images": {},
        "display_status": {},
    }

    rendered = template.render(dce=dce)
    assert "Inter-VFA Alignment" in rendered
    assert warning in rendered
    assert rendered.index('class="preload-note warning"') < rendered.index(warning)


def test_dce_t1_cards_are_compact_and_ordered(env):
    """T1 acquisition, summaries, and support render in a stable order."""
    template = env.get_template("sections/section_dce.html")
    dce = {
        "qc_metrics": {
            "t1_mapping": {
                "t1_status": "WARN",
                "t1_method": "nlls",
                "n_flip_angles": 3,
                "flip_angles_deg": [5.0, 15.0, 25.0],
                "tr_ms": 20.0,
                "median_t1_ms": 1638.0,
                "fraction_valid": 0.993,
                "t1_support": {
                    "directly_measured_fraction": 0.8937,
                    "locally_interpolated_fraction": 0.0084,
                    "unavailable_voxels": 1314,
                    "unavailable_fraction": 0.0064,
                    "retained_outside_population_reference_voxels": 113008,
                    "retained_outside_population_reference_fraction": 0.5306,
                },
                "t1_correction": {"correction_applied": "none"},
            },
            "signal_conversion": {
                "t1_invalid_voxels": 30194,
                "t1_invalid_fraction": 0.1418,
            },
        },
        "images": {},
        "display_status": {},
    }

    rendered = template.render(dce=dce)
    assert "1,314 (0.64%)" in rendered
    assert "30,194 (14.18%)" in rendered
    assert "53.06%" in rendered
    assert "analysis domain" not in rendered
    assert "brain support" not in rendered
    assert "T1 Source" not in rendered
    labels = (
        "T1 Status",
        "T1 Method",
        "Flip Angles",
        "TR",
        "Median T1",
        "Valid Voxels",
        "Direct T1 Support",
        "Interpolated T1 Support",
        "T1 Unavailable",
        "Outside Population Reference",
        "Invalid T1 in DCE Space",
    )
    positions = [rendered.index(label) for label in labels]
    assert positions == sorted(positions)


def test_dce_no_t1_assumption_renders_without_wm_statistics(env):
    """Configured no-T1 runs use assumed-T1 fields without requiring WM QC."""
    template = env.get_template("sections/section_dce.html")
    dce = {
        "qc_metrics": {
            "t1_mapping": {
                "method": "assumed",
                "t1_status": "Skipped by configuration",
                "skipped": True,
                "assumed_t1_ms": 1200.0,
                "assumed_t1_field_strength_t": 3.0,
                "assumed_t1_policy": "uniform_field_strength_reference",
                "b1_correction": "precomputed",
            },
        },
        "images": {},
        "display_status": {},
    }

    rendered = template.render(dce=dce)

    assert "Skipped by configuration" in rendered
    assert "T1 Value Used" in rendered
    assert "1200 ms" in rendered
    assert "Uniform (1200 ms)" in rendered


def test_dce_skipped_t1_with_incomplete_qc_renders_safely(env):
    """An incomplete skipped-T1 QC dictionary cannot abort report rendering."""
    template = env.get_template("sections/section_dce.html")
    dce = {
        "qc_metrics": {
            "t1_mapping": {
                "t1_status": "Skipped",
                "skipped": True,
            },
        },
        "images": {},
        "display_status": {},
    }

    rendered = template.render(dce=dce)

    assert "T1 Source" in rendered
    assert "Uniform" in rendered


def test_dce_fallback_t1_with_incomplete_qc_renders_safely(env):
    """An incomplete fallback-T1 QC dictionary cannot abort report rendering."""
    template = env.get_template("sections/section_dce.html")
    dce = {
        "qc_metrics": {
            "t1_mapping": {
                "t1_correction": {"correction_applied": "fallback"},
            },
        },
        "images": {},
        "display_status": {},
    }

    rendered = template.render(dce=dce)

    assert "T1 Source" in rendered
    assert "Uniform" in rendered


def test_dce_aif_peak_scaling_qc_renders(env):
    """Raw/emitted peaks plus the resolved scaling policy reach HTML."""
    template = env.get_template("sections/section_dce.html")
    dce = {
        "qc_metrics": {
            "aif_detection": {
                "emitted_peak_mm": 8.0,
                "raw_peak_mm": 4.0,
                "peak_scaling_policy": "one_sided_lower_bound",
                "peak_scaling_applied": True,
                "peak_scaling_factor": 2.0,
                "peak_scaling_trigger": "below_expected_minimum",
            },
        },
        "images": {},
        "display_status": {},
    }
    rendered = template.render(dce=dce)
    assert "Emitted Peak" in rendered
    assert "8.00 mM" in rendered
    assert "Raw Peak" in rendered
    assert "4.00 mM" in rendered
    assert "One-Sided Lower Bound (Applied)" in rendered
    assert "2.00x" in rendered
    assert "Peak Scaling Trigger" not in rendered
    assert "one_sided_lower_bound" not in rendered
    assert "below_expected_minimum" not in rendered


def test_dce_voxel_accounting_renders(env):
    """Named fitting stages and denominators reach the DCE HTML report."""
    template = env.get_template("sections/section_dce.html")
    dce = {
        "qc_metrics": {
            "pk_modeling": {
                "model": "extended_tofts",
                "voxel_accounting": {
                    "brain_mask": {"count": 100},
                    "fitting_eligible": {
                        "count": 80,
                        "fraction_of_brain": 0.8,
                    },
                    "attempted": {"count": 80},
                    "optimizer_converged": {
                        "count": 70,
                        "fraction_of_attempted": 0.875,
                    },
                    "prefit_rejected": {
                        "count": 2,
                        "fraction_of_attempted": 0.025,
                    },
                    "execution_exception": {
                        "count": 1,
                        "fraction_of_attempted": 0.0125,
                    },
                    "fit_unsuccessful": {
                        "count": 3,
                        "fraction_of_attempted": 0.0375,
                    },
                    "retained_nondegenerate": {
                        "count": 65,
                        "fraction_of_attempted": 0.8125,
                    },
                    "quantitative_qc_accepted": {
                        "count": 50,
                        "fraction_of_attempted": 0.625,
                    },
                },
            }
        },
        "images": {},
        "display_status": {},
    }
    rendered = template.render(dce=dce)
    assert "Brain Mask Voxels" in rendered
    assert "Fitting Eligible" in rendered
    assert "Optimizer Converged" in rendered
    assert "Pre-fit Rejected" in rendered
    assert "Execution Exceptions" in rendered
    assert "Unsuccessful Fits" in rendered
    assert "Retained Nondegenerate" in rendered
    assert "Quantitative QC Accepted" in rendered
    assert "70 (87.5%)" in rendered
    assert "80 (80.0%)" in rendered
    assert "of attempted" not in rendered
    assert "of brain" not in rendered


def test_dce_incomplete_voxel_accounting_stage_is_skipped(env):
    """A present but incomplete accounting stage cannot abort rendering."""
    template = env.get_template("sections/section_dce.html")
    dce = {
        "qc_metrics": {
            "pk_modeling": {
                "model": "tofts",
                "voxel_accounting": {"attempted": {}},
            },
        },
        "images": {},
        "display_status": {},
    }

    rendered = template.render(dce=dce)

    assert "Fits Attempted" not in rendered


def test_report_metric_cards_wrap_long_values(env):
    """Shared metric cards wrap identifiers and support secondary details."""
    template = env.get_template("report_unified.html")
    rendered = template.render(**_minimal_parent_ctx())

    assert "overflow-wrap: anywhere" in rendered
    assert ".metric-detail" in rendered


def test_dce_two_cxm_suitability_warning_renders(env):
    """Reason-coded 2CXM screening reaches the DCE HTML without implying failure."""
    template = env.get_template("sections/section_dce.html")
    dce = {
        "qc_metrics": {
            "pk_modeling": {"model": "2cxm"},
            "model_suitability": {
                "2cxm": {
                    "status": "WARN",
                    "reasons": [
                        "temporal_sampling_outside_auto_2cxm_eligibility"
                    ],
                    "metrics": {
                        "volume_tr_seconds": 7.38,
                        "acquisition_duration_seconds": 354.0,
                    },
                }
            },
        },
        "images": {},
        "display_status": {},
    }
    rendered = template.render(dce=dce)
    assert "2CXM Suitability" in rendered
    assert "Temporal sampling is outside the automatic 2CXM selection range" in rendered
    assert "Volume TR 7.38 s" in rendered
    assert "review fit and coverage metrics separately" in rendered


@pytest.mark.parametrize(
    "suitability",
    [
        {"status": None, "metrics": {"volume_tr_seconds": 7.0}},
        {"status": "WARN", "metrics": {"volume_tr_seconds": None}},
        {"status": "WARN"},
    ],
)
def test_dce_incomplete_two_cxm_suitability_renders_safely(env, suitability):
    """Null or missing optional suitability fields cannot abort reporting."""
    template = env.get_template("sections/section_dce.html")
    dce = _drop_report_nulls(
        {
            "qc_metrics": {
                "pk_modeling": {"model": "2cxm"},
                "model_suitability": {"2cxm": suitability},
            },
            "images": {},
            "display_status": {},
        }
    )

    rendered = template.render(dce=dce)

    if suitability.get("status") is None:
        assert "2CXM Suitability" not in rendered
    else:
        assert "2CXM Suitability" in rendered


def test_drop_report_nulls_preserves_valid_falsey_values():
    """Only null mapping entries represent unavailable report fields."""
    value = {
        "missing": None,
        "zero": 0,
        "false": False,
        "empty_mapping": {},
        "empty_list": [],
        "nested": {"missing": None, "value": 1},
        "items": [None, {"missing": None, "value": 2}],
    }

    assert _drop_report_nulls(value) == {
        "zero": 0,
        "false": False,
        "empty_mapping": {},
        "empty_list": [],
        "nested": {"value": 1},
        "items": [None, {"value": 2}],
    }


def test_parent_render_shows_dsc_blocks_when_dsc_present(env):
    """On DCE+DSC runs, both DCE-side DSC blocks must render."""
    template = env.get_template("report_unified.html")
    dsc_ctx = {"qc_metrics": {}, "images": {}, "display_status": {"leaky_voxels": "pass"}}
    ctx = _minimal_parent_ctx(dsc=dsc_ctx, dce=_dce_ctx_with_dsc_blocks())
    rendered = template.render(**ctx)
    assert "<h3>DSC CBV Constraint</h3>" in rendered
    assert "<h3>NAWM-Normalized vp–nCBV Concordance</h3>" in rendered
    assert '<div class="metric" style="grid-column: 1 / -1;">' in rendered
    assert '<div class="metric-label">Reason</div>' in rendered
    assert '<div class="metric-note">No DSC data were available for a vp prior.</div>' in rendered
    assert "No DSC data were available for a vp prior." in rendered


def test_parent_render_tolerates_missing_dsc_qc_sections(env):
    """A missing or unreadable optional QC sidecar must not abort reporting."""
    template = env.get_template("report_unified.html")
    dsc_ctx = {"qc_metrics": {}, "images": {}, "display_status": {}}

    rendered = template.render(**_minimal_parent_ctx(dsc=dsc_ctx, dce=None))

    assert "<h2>DSC Processing</h2>" in rendered


def test_dce_xval_error_hides_raw_exception(env):
    """HTML gives stable guidance while technical failure detail stays in provenance."""
    template = env.get_template("sections/section_dce.html")
    dce = {
        "qc_metrics": {
            "dsc_dce_validation": {
                "skipped": True,
                "status": "error",
                "reason": "/private/work/sub-01 missing input.nii.gz",
            },
        },
        "images": {},
        "display_status": {},
    }

    rendered = template.render(dce=dce, dsc={"qc_metrics": {}})

    assert "DSC–DCE comparison could not be completed" in rendered
    assert '<div class="metric-value">Error</div>' in rendered
    assert "Skipped (informational)" not in rendered
    assert "/private/work" not in rendered


# =============================================================================
# Template AIF caveat
# =============================================================================


def _dsc_ctx_with_aif_level(level: str, **extra) -> dict:
    """Build a minimal DSC context with an AIF detection_method set."""
    qc: dict = {"aif_detection": {"detection_method": level, "quality_score": 0.85}}
    qc.update(extra)
    return {
        "qc_metrics": qc,
        "images": {"aif_curve": ""},
        "display_status": {"leaky_voxels": "pass"},
    }


def test_synthetic_aif_caveat_rendered(env):
    """detection_level == 'synthetic' triggers the caveat alert block."""
    template = env.get_template("sections/section_dsc.html")
    dsc_ctx = _dsc_ctx_with_aif_level("synthetic", synthetic_fallback_reason="no_candidates")
    rendered = template.render(dsc=dsc_ctx)
    assert "Template AIF" in rendered
    assert "no arterial candidates survived automatic detection" in rendered
    assert "no_candidates" not in rendered
    assert "MTT, Tmax" in rendered
    assert "BW K2" in rendered


def test_measured_aif_omits_caveat(env):
    """atlas_strict detection renders no template-AIF caveat."""
    template = env.get_template("sections/section_dsc.html")
    dsc_ctx = _dsc_ctx_with_aif_level("atlas_strict")
    rendered = template.render(dsc=dsc_ctx)
    assert "Template AIF" not in rendered


# =============================================================================
# Deconvolution panel — metric labels
# =============================================================================


def _dsc_ctx_with_deconv(method: str, **deconv_extra) -> dict:
    """Build a DSC context that surfaces the deconvolution metrics panel."""
    deconv = {
        "method": method,
        "cbf_median": 50.0,
        "cbv_median": 1.5,
        "mtt_median": 4.0,
        "tmax_median": 1.5,
        "n_voxels": 1000,
    }
    deconv.update(deconv_extra)
    return {
        "qc_metrics": {
            "deconvolution": deconv,
            "aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.85},
        },
        "images": {"aif_curve": ""},
        "display_status": {
            "leaky_voxels": "pass",
            "rsquared": "pass",
            "fallback": "pass",
        },
    }


def test_deconv_panel_inference_shown_for_cpi(env):
    """CPI runs render an "Inference" metric box (not a "Bayesian" label)."""
    template = env.get_template("sections/section_dsc.html")
    dsc_ctx = _dsc_ctx_with_deconv(
        "cpi", inference="vi", fallback_rate=0.05, fe_plateau_rate=0.0001,
    )
    rendered = template.render(dsc=dsc_ctx)
    assert ">Inference<" in rendered
    assert ">Bayesian<" not in rendered
    assert ">vi<" in rendered


def test_deconv_panel_hides_tmax_for_cpi(env):
    """CPI does not model bolus delay — Tmax stat must be hidden."""
    template = env.get_template("sections/section_dsc.html")
    dsc_ctx = _dsc_ctx_with_deconv(
        "cpi", inference="vi", fallback_rate=0.05, fe_plateau_rate=0.0,
    )
    rendered = template.render(dsc=dsc_ctx)
    assert ">Median Tmax<" not in rendered


def test_deconv_panel_shows_tmax_for_osvd(env):
    """oSVD does model delay — Tmax stat is visible."""
    template = env.get_template("sections/section_dsc.html")
    dsc_ctx = _dsc_ctx_with_deconv("oSVD")
    rendered = template.render(dsc=dsc_ctx)
    assert ">Median Tmax<" in rendered


def test_deconv_panel_fallback_only_for_parametric(env):
    """oSVD Fallback metric box only renders for parametric methods (cpi, vascular)."""
    template = env.get_template("sections/section_dsc.html")

    cpi_ctx = _dsc_ctx_with_deconv(
        "cpi", inference="vi", fallback_rate=0.07, fe_plateau_rate=0.0001,
    )
    cpi_rendered = template.render(dsc=cpi_ctx)
    assert ">oSVD Fallback<" in cpi_rendered
    assert '<div class="metric-value">7.0%</div>' in cpi_rendered

    osvd_ctx = _dsc_ctx_with_deconv("oSVD", fallback_rate=0.0)
    osvd_rendered = template.render(dsc=osvd_ctx)
    assert ">oSVD Fallback<" not in osvd_rendered

    # The "Convergence Rate" label must not appear anywhere.
    assert ">Convergence Rate<" not in cpi_rendered
    assert ">Convergence Rate<" not in osvd_rendered


def test_quality_flag_colour_synthetic_is_poor(env):
    """Synthetic detection_level badges as 'poor' regardless of score."""
    template = env.get_template("sections/section_dsc.html")
    dsc_ctx = _dsc_ctx_with_aif_level("synthetic")
    rendered = template.render(dsc=dsc_ctx)
    # Status badge carries the flag class.
    assert 'class="metric-value poor"' in rendered


# =============================================================================
# Acquisition Advisory section
# =============================================================================


def _summary_ctx(**overrides) -> dict:
    """Build minimal context for section_summary.html."""
    ctx = {
        "overall_recommendation": None,
        "provenance_warnings": {},
        "acquisition_warnings": {},
        "processing_warnings": {},
        "ran_pipelines": {
            "anat": True, "tumor_seg": False, "dsc": True, "dce": False,
        },
        "all_warnings": {},
    }
    ctx.update(overrides)
    return ctx


def test_summary_renders_acquisition_warnings(env):
    """Acquisition advisory section renders when warnings present."""
    template = env.get_template("sections/section_summary.html")
    ctx = _summary_ctx(acquisition_warnings={
        "DSC Processing": ["[DSC_LOW_EFFECTIVE_DOSE] Low dose warning"],
    })
    rendered = template.render(**ctx)
    assert "Acquisition Advisory" in rendered
    assert "DSC_LOW_EFFECTIVE_DOSE" in rendered


def test_summary_hides_acquisition_warnings_when_empty(env):
    """Acquisition advisory section hidden when no warnings."""
    template = env.get_template("sections/section_summary.html")
    ctx = _summary_ctx()
    rendered = template.render(**ctx)
    assert "Acquisition Advisory" not in rendered


def test_summary_renders_processing_warnings(env):
    """Processing advisory section renders when AIF fallback present."""
    template = env.get_template("sections/section_summary.html")
    ctx = _summary_ctx(processing_warnings={
        "DSC Processing": ["AIF fell back to Parker population template"],
    })
    rendered = template.render(**ctx)
    assert "Processing Advisory" in rendered
    assert "Parker population template" in rendered


def test_summary_hides_processing_warnings_when_empty(env):
    """Processing advisory section hidden when no warnings."""
    template = env.get_template("sections/section_summary.html")
    ctx = _summary_ctx()
    rendered = template.render(**ctx)
    assert "Processing Advisory" not in rendered


def _dsc_ctx_with_preload(**preload) -> dict:
    return {
        "qc_metrics": {"preload": preload},
        "images": {},
        "display_status": {"leaky_voxels": "pass"},
    }


def test_unknown_preload_shows_unknown_administered_dose(env):
    """Unknown preload does not imply an administered-dose category."""
    template = env.get_template("sections/section_dsc.html")
    dsc_ctx = _dsc_ctx_with_preload(
        preload_source="unknown",
        preload_administered_dose_fraction=None,
    )
    rendered = template.render(dsc=dsc_ctx)
    assert "Administered Preload" in rendered
    assert '<div class="metric-value">Unknown</div>' in rendered
    assert "Leakage-Correction Adequacy" not in rendered


def test_preload_user_specified_shows_continuous_fraction(env):
    """A supplied preload remains a continuous dose fraction."""
    template = env.get_template("sections/section_dsc.html")
    dsc_ctx = _dsc_ctx_with_preload(
        preload_source="user_specified",
        preload_administered_dose_fraction=0.3,
    )
    rendered = template.render(dsc=dsc_ctx)
    assert "Administered Preload" in rendered
    assert "0.30× standard dose" in rendered
    assert "Leakage-Correction Adequacy" not in rendered


def test_unavailable_k2_reference_has_distinct_warning(env):
    """Unavailable reference reports skipped correction without calibration warning."""
    template = env.get_template("sections/section_dsc.html")
    dsc_ctx = _dsc_ctx_with_preload(
        preload_source="unknown",
        preload_administered_dose_fraction=None,
        k2_signed=True,
        k2_reference_available=False,
        preload_correction_applied=False,
    )

    rendered = template.render(dsc=dsc_ctx)

    assert "K2 reference unavailable; leakage correction was not applied" in rendered


def test_disabled_leakage_correction_has_explicit_note(env):
    """Configuration-driven omission is distinguished from reference failure."""
    template = env.get_template("sections/section_dsc.html")
    dsc_ctx = _dsc_ctx_with_preload(
        preload_source="unknown",
        preload_administered_dose_fraction=None,
        leakage_correction_disabled=True,
        preload_correction_applied=False,
        k2_method="none",
    )

    rendered = template.render(dsc=dsc_ctx)

    assert "Leakage correction was disabled by configuration" in rendered
    assert "baseline-centered concentration series" in rendered
    assert "K2 reference unavailable" not in rendered


def test_dsc_preprocessing_and_aif_cards_follow_common_order(env):
    """DSC shared preprocessing and AIF fields follow the DCE hierarchy."""
    template = env.get_template("sections/section_dsc.html")
    dsc = {
        "qc_metrics": {
            "preprocessing": {
                "tr_ms": 1500.0,
                "te_ms": 30.0,
                "n_volumes": 60,
                "voxel_size_mm": [2.0, 2.0, 4.0],
                "n_brain_voxels": 100,
                "peak_drop_pct": 40.0,
                "mean_tsnr": 20.0,
                "mean_cnr": 8.0,
            },
            "aif_detection": {
                "detection_method": "atlas_strict",
                "quality_score": 0.8,
                "n_aif_voxels_selected": 5,
                "n_aif_voxels_available": 8,
                "n_baseline_frames": 6,
                "arrival_time_seconds": 12.0,
                "aif_metrics": {
                    "peak_conc": 3.0,
                    "ph": 40.0,
                    "fwhm": 6.0,
                    "ttp": 14.0,
                    "psr": 0.8,
                    "brr": 0.9,
                },
            },
        },
        "images": {},
        "display_status": {},
    }
    rendered = template.render(dsc=dsc)
    _assert_in_order(
        rendered,
        "DSC Preprocessing QC",
        "TR",
        "TE",
        "Volumes",
        "Voxel Size",
        "Brain Voxels",
        "Peak Signal Drop",
        "Mean tSNR",
        "Mean CNR",
        "Arterial Input Function",
        "AIF Status",
        "Method",
        "AIF Voxels",
        "Baseline Frames",
        "Arrival Time",
        "Peak Concentration",
        "Peak Height",
        "FWHM",
        "TTP",
        "Peak Signal Ratio",
        "Baseline Return Ratio",
    )


def test_tumor_and_nawm_cards_are_ordered_and_unambiguous(env):
    """Tumor totals lead components and NAWM omits internal atlas provenance."""
    template = env.get_template("sections/section_tumor_seg.html")
    tumor_seg = {
        "qc_metrics": {
            "segmentation": {
                "validation": {
                    "has_tumor": True,
                    "total_tumor_volume_ml": 12.0,
                    "enhancing_volume_ml": 4.0,
                    "non_enhancing_volume_ml": 8.0,
                },
            },
            "nawm": {
                "selected_hemisphere": "right",
                "n_voxels": 1000,
                "volume_ml": 2.0,
                "valid": True,
                "atlas_used": True,
            },
        },
        "images": {},
    }
    rendered = template.render(tumor_seg=tumor_seg)
    _assert_in_order(
        rendered,
        "Tumor Detected",
        "Total Tumor Volume",
        "Enhancing Volume",
        "Non-Enhancing Volume",
        "Selected Hemisphere",
        "NAWM Voxels",
        "NAWM Volume",
        "Minimum Size Met",
    )
    assert "Atlas Used" not in rendered
