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
    """Build a minimal DSC context with an AIF detection_level set."""
    qc: dict = {"aif": {"detection_level": level, "quality_score": 0.85}}
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


def test_quality_flag_colour_synthetic_is_poor(env):
    """Synthetic detection_level badges as 'poor' regardless of score."""
    template = env.get_template("sections/section_dsc.html")
    dsc_ctx = _dsc_ctx_with_aif_level("synthetic")
    rendered = template.render(dsc=dsc_ctx)
    # Status badge carries the flag class.
    assert 'class="metric-value poor"' in rendered
