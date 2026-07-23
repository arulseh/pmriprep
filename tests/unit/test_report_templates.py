"""Jinja render tests for pmriprep HTML report templates."""

from pathlib import Path

import pytest
from jinja2 import ChainableUndefined, Environment, FileSystemLoader

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
                    "reason": "No DSC data provided",
                    "skip_reason": "no_dsc_run",
                }
            },
            "dsc_dce_validation": {
                "skipped": True,
                "status": "SKIPPED",
                "reason": "No DSC data provided",
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


def test_parent_render_hides_dsc_blocks_when_no_dsc(env):
    """On DCE-only runs, DSC CBV Constraint and DSC-DCE Cross-Validation sections must NOT render."""
    template = env.get_template("report_unified.html")
    ctx = _minimal_parent_ctx(dsc=None, dce=_dce_ctx_with_dsc_blocks())
    rendered = template.render(**ctx)
    assert "<h3>DSC CBV Constraint</h3>" not in rendered
    assert "<h3>DSC-DCE Cross-Validation</h3>" not in rendered


def test_dce_signal_conversion_qc_renders(env):
    """Conversion fallback and bounded-inversion accounting reaches HTML."""
    template = env.get_template("sections/section_dce.html")
    dce = {
        "qc_metrics": {
            "preprocessing": {},
            "signal_conversion": {
                "t1_invalid_voxels": 12,
                "b1_missing_voxels": 34,
                "e1_bounded_fraction": 0.00125,
            },
        },
        "images": {"preprocessing_qc": "preproc.png"},
        "display_status": {},
    }
    rendered = template.render(dce=dce)
    assert "Invalid T1 Voxels" in rendered
    assert ">12<" in rendered
    assert "Missing B1 Voxels" in rendered
    assert ">34<" in rendered
    assert "Bounded SPGR Samples" in rendered
    assert "0.125%" in rendered


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
    assert "one_sided_lower_bound (applied)" in rendered
    assert "2.00x" in rendered
    assert "below_expected_minimum" in rendered


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
    assert "Retained Nondegenerate" in rendered
    assert "Quantitative QC Accepted" in rendered
    assert "87.5% attempted" in rendered


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
    assert "2CXM Suitability Screening" in rendered
    assert "temporal sampling outside auto 2cxm eligibility" in rendered
    assert "volume TR 7.38 s" in rendered
    assert "does not establish biological parameter validity" in rendered


def test_parent_render_shows_dsc_blocks_when_dsc_present(env):
    """On DCE+DSC runs, both DCE-side DSC blocks must render."""
    template = env.get_template("report_unified.html")
    dsc_ctx = {"qc_metrics": {}, "images": {}, "display_status": {"leaky_voxels": "pass"}}
    ctx = _minimal_parent_ctx(dsc=dsc_ctx, dce=_dce_ctx_with_dsc_blocks())
    rendered = template.render(**ctx)
    assert "<h3>DSC CBV Constraint</h3>" in rendered
    assert "<h3>DSC-DCE Cross-Validation</h3>" in rendered


def test_parent_render_tolerates_missing_dsc_qc_sections(env):
    """A missing or unreadable optional QC sidecar must not abort reporting."""
    template = env.get_template("report_unified.html")
    dsc_ctx = {"qc_metrics": {}, "images": {}, "display_status": {}}

    rendered = template.render(**_minimal_parent_ctx(dsc=dsc_ctx, dce=None))

    assert "<h2>DSC Perfusion</h2>" in rendered


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
    assert "no_candidates" in rendered
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
        "DSC Perfusion": ["[DSC_LOW_EFFECTIVE_DOSE] Low dose warning"],
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
        "DSC Perfusion": ["AIF fell back to Parker population template"],
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


def test_preload_auto_shows_k2_adequacy(env):
    """Auto (k2_primary) path labels the box 'K2 Adequacy', not a preload dose."""
    template = env.get_template("sections/section_dsc.html")
    dsc_ctx = _dsc_ctx_with_preload(
        preload_classification="adequate",
        preload_classification_source="k2_primary",
    )
    rendered = template.render(dsc=dsc_ctx)
    assert '<div class="metric-label" title="Reliability of the K2 leakage correction / CBV trustworthiness — not a preload measurement.">K2 Adequacy</div>' in rendered
    assert "Adequate" in rendered
    # No preload-dose framing on the auto path, and the "Preload Source" box removed.
    assert '<div class="metric-label">Preload Status</div>' not in rendered
    assert '<div class="metric-label">Preload Source</div>' not in rendered
    assert '<div class="metric-label">Classification</div>' not in rendered
    assert '<div class="metric-label">Source</div>' not in rendered


def test_preload_user_specified_keeps_preload_status(env):
    """User-specified path keeps the 'Preload Status' label; K2 box reads 'K2 Adequacy'."""
    template = env.get_template("sections/section_dsc.html")
    dsc_ctx = _dsc_ctx_with_preload(
        preload_classification="half_dose",
        preload_classification_source="user_specified",
        preload_k2_classification="adequate",
    )
    rendered = template.render(dsc=dsc_ctx)
    assert '<div class="metric-label">Preload Status</div>' in rendered
    assert "Half Dose" in rendered
    assert "K2 Adequacy" in rendered
    assert "Adequate" in rendered


def test_preload_user_specified_dose_shows_na_when_absent(env):
    """The User-Specified Dose box always renders, reading N/A on auto-detect runs."""
    template = env.get_template("sections/section_dsc.html")
    dsc_ctx = _dsc_ctx_with_preload(
        preload_classification="adequate",
        preload_classification_source="k2_primary",
    )
    rendered = template.render(dsc=dsc_ctx)
    assert "User-Specified Dose" in rendered
    assert '<div class="metric-value">N/A</div>' in rendered


def test_preload_user_specified_dose_shows_value(env):
    """When the user specified a preload dose, the box shows the formatted value."""
    template = env.get_template("sections/section_dsc.html")
    dsc_ctx = _dsc_ctx_with_preload(
        preload_classification="none",
        preload_classification_source="user_specified",
        preload_user_specified_dose=0.0,
    )
    rendered = template.render(dsc=dsc_ctx)
    assert "User-Specified Dose" in rendered
    assert '<div class="metric-value">0.00</div>' in rendered


def test_positive_tail_leakage_warning_renders(env):
    """cbv_positive_tail_fraction > 0.6 → leakage-section over-estimate warning."""
    template = env.get_template("sections/section_dsc.html")
    dsc_ctx = _dsc_ctx_with_preload(
        preload_classification="adequate",
        preload_classification_source="k2_primary",
        cbv_positive_tail_fraction=0.72,
    )
    rendered = template.render(dsc=dsc_ctx)
    assert "over-estimated" in rendered
    assert "72%" in rendered
    # Leakage section must not leak vp/constraint language into this warning.
    assert "vp anchor" not in rendered.split("Preload &amp; Leakage")[-1].split("</div>")[0]


def test_positive_tail_warning_silent_when_low(env):
    """Below threshold → no over-estimate warning."""
    template = env.get_template("sections/section_dsc.html")
    dsc_ctx = _dsc_ctx_with_preload(
        preload_classification="adequate",
        preload_classification_source="k2_primary",
        cbv_positive_tail_fraction=0.30,
    )
    rendered = template.render(dsc=dsc_ctx)
    assert "over-estimated" not in rendered
