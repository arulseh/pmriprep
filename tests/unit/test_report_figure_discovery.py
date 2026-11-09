"""Regression tests for entity-preserving QC figure discovery."""

import json
from pathlib import Path

import pytest

from pmriprep.utils.report_utils import (
    _collect_pipeline_images,
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


@pytest.mark.parametrize("pipeline,writer_name,kept,dropped", [
    ("anat", "AnatFiguresWriter", "brain_mask_filmstrip", "coreg_filmstrip_flair"),
    ("tumor_seg", "TumorSegFiguresWriter", "seg_filmstrip_postcon", "seg_filmstrip_flair"),
    ("dsc", "DSCFiguresWriter", "motion_qc_plot", "dsc_fit_qc_plot"),
    ("dce", "DCEFiguresWriter", "pk_param_maps_qc_plot", "model_selection_qc_plot"),
])
def test_report_figures_follow_latest_writer_outputs(
    tmp_path: Path, pipeline: str, writer_name: str, kept: str, dropped: str,
) -> None:
    """Changed settings hide old optional figures while retaining reusable inputs."""
    from pmriprep.interfaces import qc_writers
    from pmriprep.interfaces.reports import GenerateUnifiedReport
    from pmriprep.utils.report_utils import _load_pipeline_sidecars

    source = tmp_path / "sub-01_ses-01_acq-fast_run-2_dce.nii.gz"
    source.touch()
    png = tmp_path / "current.png"
    png.write_bytes(b"cached figure input")
    writer = getattr(qc_writers, writer_name)(output_dir=str(tmp_path), subject_id="01", session_id="01")
    assert writer._always_run
    if pipeline in {"dsc", "dce"}:
        setattr(writer.inputs, f"source_{pipeline}", str(source))
    setattr(writer.inputs, kept, str(png))
    setattr(writer.inputs, dropped, str(png))
    writer.run(cwd=str(tmp_path))
    ses = tmp_path / "pmriprep" / "sub-01" / "ses-01"
    figures = ses / "figures"
    original = _collect_pipeline_images("01", "01", pipeline, figures)
    assert len(original) == 2

    setattr(writer.inputs, dropped, None)
    writer.run(cwd=str(tmp_path))
    latest = _collect_pipeline_images("01", "01", pipeline, figures)
    assert len(latest) == 1
    assert all((ses / path).read_bytes() == b"cached figure input" for path in original.values())

    if pipeline in {"dsc", "dce"}:
        perf = ses / "perf"
        current_anchor = perf / f"sub-01_ses-01_acq-fast_run-2_desc-{pipeline}Preproc_{pipeline}.json"
        old_anchor = perf / f"sub-01_ses-01_acq-fast_run-1_desc-{pipeline}Preproc_{pipeline}.json"
        old_anchor.write_text(json.dumps({"PipelineQC": {"old_result": True}}))
        qc = {"pk_modeling": {"model": "2cxm", "selection_mode": "fixed"}} if pipeline == "dce" else {
            "deconvolution": {"method": "oSVD"},
        }
        current_anchor.write_text(json.dumps({"PipelineQC": qc}))
        assert _load_pipeline_sidecars(ses, "sub-01_ses-01", pipeline) == qc

    if pipeline == "dce":
        # Use the saved anatomy index alongside the updated DCE results.
        qc_writers.AnatFiguresWriter(output_dir=str(tmp_path), subject_id="01", session_id="01",
                                     brain_mask_filmstrip=str(png)).run(cwd=str(tmp_path))
        anatomy_index = ses / "logs" / "report_anat.json"
        anatomy_before = anatomy_index.read_bytes()
        (ses / "anat").mkdir()
        (ses / "anat" / "sub-01_ses-01_space-T1w_desc-preproc_T1w.json").write_text("{}")
        report = GenerateUnifiedReport(output_dir=str(tmp_path), subject_id="01", session_id="01",
                                       ran_anat=False, ran_dce=True)
        assert report._always_run
        report.run(cwd=str(tmp_path))
        html = (ses / "sub-01_ses-01.html").read_text()
        assert "Two-Compartment Exchange" in html or "2CXM" in html
        assert 'src="figures/sub-01_ses-01_desc-brainmask_anat.png"' in html
        assert all(f'src="{path}"' in html for path in latest.values())
        assert not any(f'src="{path}"' in html for path in set(original.values()) - set(latest.values()))
        assert anatomy_index.read_bytes() == anatomy_before

    setattr(writer.inputs, kept, None)
    writer.run(cwd=str(tmp_path))
    assert _collect_pipeline_images("01", "01", pipeline, figures) == {}
    assert all((ses / path).exists() for path in original.values())


@pytest.mark.parametrize("selection_mode", ["fixed", "auto"])
def test_report_selection_panel_follows_current_anchor(tmp_path: Path, selection_mode: str) -> None:
    """Report-only generation must ignore a leftover auto figure for a fixed fit."""
    from pmriprep.interfaces.reports import GenerateUnifiedReport

    ses = tmp_path / "pmriprep" / "sub-01" / "ses-01"
    (ses / "figures").mkdir(parents=True)
    (ses / "perf").mkdir()
    figure = ses / "figures" / "sub-01_ses-01_desc-modelselection_dce.png"
    figure.touch()
    (ses / "perf" / "sub-01_ses-01_desc-dcePreproc_dce.json").write_text(json.dumps({
        "PipelineQC": {"pk_modeling": {"selection_mode": selection_mode}},
    }))
    GenerateUnifiedReport(subject_id="01", session_id="01", output_dir=str(tmp_path), ran_dce=True).run(
        cwd=str(tmp_path),
    )
    html = (ses / "sub-01_ses-01.html").read_text()
    assert (f'img src="figures/{figure.name}"' in html) == (selection_mode == "auto")
    assert figure.exists(), "Report-only regeneration must not delete derivatives"


def test_entity_rich_fit_figure_restores_complete_html_section(
    tmp_path: Path,
) -> None:
    """Figure discovery restores the fit image, parameters, and R-squared."""
    ses_dir = tmp_path / "pmriprep" / "sub-01" / "ses-01"
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

    from pmriprep.interfaces.reports import GenerateUnifiedReport

    GenerateUnifiedReport(
        subject_id="01",
        session_id="01",
        output_dir=str(tmp_path), ran_dce=True, sloppy=True,
    ).run(cwd=str(tmp_path))

    html = report.read_text()
    assert html.count("Sloppy mode: results are approximate and for testing only.") == 1
    assert "Example Voxel Fit" in html
    assert "(4, 5, 6)" in html
    assert "0.1234 min<sup>-1</sup>" in html
    assert "0.3456" in html
    assert "0.0123" in html
    assert "0.9876" in html
    assert f'img src="figures/{figure.name}"' in html
