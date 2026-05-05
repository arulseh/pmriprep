"""HTML report integration: rendered HTML carries expected sections after PR2.

Section-presence test (NOT byte-equality) — the v0.46.0 aggregator
reads canonical per-image sidecars instead of legacy ``desc-*QC_quality.json``
files. This test wires up a fake session tree, runs
``generate_unified_report``, and confirms key HTML banners (overall
recommendation, warnings, per-pipeline section headings) render.
"""
from __future__ import annotations

import json
from pathlib import Path

from pmriprep.utils.report_utils import generate_unified_report


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def test_unified_report_renders_recommendation_and_warnings(tmp_path: Path) -> None:
    ses = tmp_path / "pmriprep" / "sub-01" / "ses-0001"
    prefix = "sub-01_ses-0001"

    # Anat anchor with two warnings
    _write(
        ses / "anat" / f"{prefix}_space-T1w_desc-preproc_T1w.json",
        {
            "QCWarnings": [
                "Tissue segmentation (WM/GM/CSF probability maps) not available",
                "FLAIR not available",
            ],
        },
    )
    # DSC anchor + minimal resume
    _write(
        ses / "perf" / f"{prefix}_desc-dscPreproc_dsc.json",
        {
            "QCRecommendationAction": "review",
            "QCRecommendationReason": "AIF detection marginal",
        },
    )
    _write(
        ses / "perf" / f"{prefix}_desc-dscResume_pmriprep.json",
        {
            "PMRIPrepInternal": True,
            "aif_qc": {"detection_method": "atlas_relaxed"},
            "preload_qc": {},
            "preprocessing_stats": {"fd_mean": 0.1, "fd_max": 0.4,
                                    "n_volumes": 60, "tr_ms": 1500.0},
        },
    )

    output_html = ses / f"{prefix}.html"
    figures_dir = ses / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    generate_unified_report(
        subject_id="01", session_id="0001",
        output_path=str(output_html), ses_dir=str(ses),
        figures_dir=str(figures_dir),
        ran_pipelines={"anat": True, "tumor_seg": False,
                       "dsc": True, "dce": False},
        command_string="pmriprep ...",
    )

    html = output_html.read_text()
    # Per-pipeline section headings rendered
    assert "Anatomical" in html or "anatomical" in html.lower()
    assert "DSC" in html
    # Recommendation banner reflects DSC review action
    assert "review" in html.lower() or "AIF detection marginal" in html
    # Warnings rendered
    assert "Tissue segmentation" in html
    assert "FLAIR not available" in html
