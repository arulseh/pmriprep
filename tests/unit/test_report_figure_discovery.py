"""Regression tests for entity-preserving QC figure discovery."""

import json
from pathlib import Path

from pmriprep.utils.report_utils import (
    _collect_pipeline_images,
    generate_unified_report,
)


def test_collect_pipeline_images_accepts_source_entities(tmp_path: Path) -> None:
    """Entity-rich perfusion figures remain visible in the HTML report."""
    figure = tmp_path / (
        "sub-test01_ses-visit1_acq-fl3d_ce-GAD_run-1_desc-aifcurve_dce.png"
    )
    figure.touch()

    images = _collect_pipeline_images("test01", "visit1", "dce", tmp_path)

    assert images["aif_curve"] == f"figures/{figure.name}"


def test_collect_pipeline_images_retains_legacy_name_priority(tmp_path: Path) -> None:
    """An exact entity-free filename wins over an additional entity-rich file."""
    legacy = tmp_path / "sub-test01_ses-visit1_desc-motion_dce.png"
    entity_rich = tmp_path / "sub-test01_ses-visit1_run-1_desc-motion_dce.png"
    legacy.touch()
    entity_rich.touch()

    images = _collect_pipeline_images("test01", "visit1", "dce", tmp_path)

    assert images["motion"] == f"figures/{legacy.name}"


def test_collect_pipeline_images_does_not_cross_suffixes(tmp_path: Path) -> None:
    """DCE discovery cannot accidentally collect the corresponding DSC figure."""
    (tmp_path / "sub-01_ses-01_run-1_desc-aifcurve_dsc.png").touch()

    images = _collect_pipeline_images("01", "01", "dce", tmp_path)

    assert "aif_curve" not in images


def test_entity_rich_fit_figure_restores_complete_html_section(
    tmp_path: Path,
) -> None:
    """Figure discovery restores the fit image, parameters, and R-squared."""
    ses_dir = tmp_path / "sub-01" / "ses-01"
    figures_dir = ses_dir / "figures"
    perf_dir = ses_dir / "perf"
    figures_dir.mkdir(parents=True)
    perf_dir.mkdir()
    figure = figures_dir / (
        "sub-01_ses-01_acq-fl3d_run-1_desc-pkfit_dce.png"
    )
    figure.touch()
    anchor = perf_dir / (
        "sub-01_ses-01_acq-fl3d_run-1_desc-dcePreproc_dce.json"
    )
    anchor.write_text(json.dumps({
        "PipelineQC": {
            "pk_modeling": {
                "qc_voxel": "(4, 5, 6)",
                "qc_voxel_ktrans": 0.1234,
                "qc_voxel_ve": 0.3456,
                "qc_voxel_vp": 0.0123,
                "qc_voxel_rsquared": 0.9876,
            },
        },
    }))
    report = ses_dir / "sub-01_ses-01.html"

    generate_unified_report(
        subject_id="01",
        session_id="01",
        output_path=str(report),
        ses_dir=str(ses_dir),
        figures_dir=str(figures_dir),
        ran_pipelines={"anat": False, "tumor_seg": False, "dsc": False, "dce": True},
    )

    html = report.read_text()
    assert "Example Voxel Fit" in html
    assert "(4, 5, 6)" in html
    assert "0.1234 min<sup>-1</sup>" in html
    assert "0.3456" in html
    assert "0.0123" in html
    assert "0.9876" in html
    assert f'img src="figures/{figure.name}"' in html
