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
