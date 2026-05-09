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


def test_parent_render_shows_dsc_blocks_when_dsc_present(env):
    """On DCE+DSC runs, both DCE-side DSC blocks must render."""
    template = env.get_template("report_unified.html")
    dsc_ctx = {"qc_metrics": {}, "images": {}, "display_status": {"leaky_voxels": "pass"}}
    ctx = _minimal_parent_ctx(dsc=dsc_ctx, dce=_dce_ctx_with_dsc_blocks())
    rendered = template.render(**ctx)
    assert "<h3>DSC CBV Constraint</h3>" in rendered
    assert "<h3>DSC-DCE Cross-Validation</h3>" in rendered


# =============================================================================
# Template AIF caveat (v0.29.0)
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
# Deconvolution panel — v0.34.2 metric renames
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
    """CPI runs render an "Inference" metric box (not the legacy "Bayesian" label)."""
    template = env.get_template("sections/section_dsc.html")
    dsc_ctx = _dsc_ctx_with_deconv(
        "cpi", inference="vi", fallback_rate=0.05, fe_plateau_rate=0.0001,
    )
    rendered = template.render(dsc=dsc_ctx)
    assert ">Inference<" in rendered
    assert ">Bayesian<" not in rendered  # old label retired
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

    # The legacy "Convergence Rate" label should be retired everywhere.
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
# Acquisition Advisory section (v0.29.4)
# =============================================================================


def _summary_ctx(**overrides) -> dict:
    """Build minimal context for section_summary.html."""
    ctx = {
        "overall_recommendation": None,
        "provenance_warnings": {},
        "acquisition_warnings": {},
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
