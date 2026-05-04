"""Smoke + sidecar-correctness tests for ``init_tumor_seg_datasinks_wf``."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def minimal_bids_with_tumor_inputs(tmp_path: Path) -> tuple[Path, dict]:
    """Build a minimal BIDS root with the four HD-GLIO source modalities."""
    bids = tmp_path / "bids"
    raw = bids / "sub-01" / "ses-0001" / "anat"
    raw.mkdir(parents=True)
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "T", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )
    sources: dict[str, Path] = {}
    for tag, fname in [
        ("source_t1w", "sub-01_ses-0001_T1w.nii.gz"),
        ("source_t1w_postcon", "sub-01_ses-0001_ce-GAD_T1w.nii.gz"),
        ("source_t2w", "sub-01_ses-0001_T2w.nii.gz"),
        ("source_flair", "sub-01_ses-0001_FLAIR.nii.gz"),
    ]:
        path = raw / fname
        nib.save(
            nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)),
            path,
        )
        path.with_suffix("").with_suffix(".json").write_text("{}")
        sources[tag] = path
    return bids, sources


def _make_seg(path: Path, label: int = 2) -> Path:
    """Write a 4x4x4 segmentation NIfTI with a small enhancing region."""
    data = np.zeros((4, 4, 4), dtype=np.int16)
    data[1:3, 1:3, 1:3] = label
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    return path


def _make_mask(path: Path) -> Path:
    nib.save(
        nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4)),
        path,
    )
    return path


def test_tumor_seg_datasinks_smoke(
    tmp_path: Path, minimal_bids_with_tumor_inputs: tuple[Path, dict]
) -> None:
    """All 6 NIfTI outputs + lesionStats + volumes.txt + QC JSON written."""
    from pmriprep.workflows.tumor_seg.datasinks import init_tumor_seg_datasinks_wf

    bids, sources = minimal_bids_with_tumor_inputs
    out_root = tmp_path / "out"

    seg = _make_seg(tmp_path / "seg.nii.gz")
    enhancing = _make_mask(tmp_path / "enhancing.nii.gz")
    nonenhancing = _make_mask(tmp_path / "nonenhancing.nii.gz")
    wholetumor = _make_mask(tmp_path / "wholetumor.nii.gz")
    nawm = _make_mask(tmp_path / "nawm.nii.gz")

    wf = init_tumor_seg_datasinks_wf(
        output_dir=str(out_root),
        bids_dir=str(bids),
        subject_id="01",
        session_id="0001",
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_t1w = str(sources["source_t1w"])
    inp.source_t1w_postcon = str(sources["source_t1w_postcon"])
    inp.source_t2w = str(sources["source_t2w"])
    inp.source_flair = str(sources["source_flair"])
    inp.segmentation = str(seg)
    inp.enhancing_mask = str(enhancing)
    inp.non_enhancing_mask = str(nonenhancing)
    inp.whole_tumor_mask = str(wholetumor)
    inp.nawm_mask = str(nawm)
    inp.qc_metrics = {
        "segmentation": {
            "validation": {
                "total_tumor_volume_ml": 8.0,
                "enhancing_volume_ml": 8.0,
                "non_enhancing_volume_ml": 0.0,
                "total_tumor_voxels": 8,
                "enhancing_voxels": 8,
                "non_enhancing_voxels": 0,
                "voxel_volume_mm3": 1.0,
            }
        }
    }
    wf.run()

    ses = out_root / "pmriprep" / "sub-01" / "ses-0001"
    anat = ses / "anat"
    expected = [
        anat / "sub-01_ses-0001_space-T1w_desc-tumorseg_dseg.nii.gz",
        anat / "sub-01_ses-0001_space-T1w_desc-enhancing_mask.nii.gz",
        anat / "sub-01_ses-0001_space-T1w_desc-nonenhancing_mask.nii.gz",
        anat / "sub-01_ses-0001_space-T1w_desc-wholetumor_mask.nii.gz",
        anat / "sub-01_ses-0001_space-T1w_desc-lesion_mask.nii.gz",
        anat / "sub-01_ses-0001_space-T1w_desc-nawm_mask.nii.gz",
        anat / "sub-01_ses-0001_desc-tumorseg_volumes.txt",
        ses / "sub-01_ses-0001_space-anat_desc-lesionStats.json",
        ses / "sub-01_ses-0001_desc-tumorsegQC_quality.json",
    ]
    for p in expected:
        assert p.exists(), f"Missing: {p.name}"

    dd = out_root / "pmriprep" / "dataset_description.json"
    assert dd.exists()


def test_tumor_seg_sidecar_sources_populated(
    tmp_path: Path, minimal_bids_with_tumor_inputs: tuple[Path, dict]
) -> None:
    """M3 regression: every tumor_seg sidecar must carry Sources (was empty)."""
    from pmriprep.workflows.tumor_seg.datasinks import init_tumor_seg_datasinks_wf

    bids, sources = minimal_bids_with_tumor_inputs
    out_root = tmp_path / "out"
    seg = _make_seg(tmp_path / "seg.nii.gz")
    masks = {
        n: _make_mask(tmp_path / f"{n}.nii.gz")
        for n in ("enhancing", "nonenhancing", "wholetumor", "nawm")
    }

    wf = init_tumor_seg_datasinks_wf(
        output_dir=str(out_root),
        bids_dir=str(bids),
        subject_id="01",
        session_id="0001",
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_t1w = str(sources["source_t1w"])
    inp.source_t1w_postcon = str(sources["source_t1w_postcon"])
    inp.source_t2w = str(sources["source_t2w"])
    inp.source_flair = str(sources["source_flair"])
    inp.segmentation = str(seg)
    inp.enhancing_mask = str(masks["enhancing"])
    inp.non_enhancing_mask = str(masks["nonenhancing"])
    inp.whole_tumor_mask = str(masks["wholetumor"])
    inp.nawm_mask = str(masks["nawm"])
    inp.qc_metrics = {}
    wf.run()

    anat = out_root / "pmriprep" / "sub-01" / "ses-0001" / "anat"
    sidecars = sorted(anat.glob("*.json"))
    assert sidecars, "No sidecars written"

    for sc in sidecars:
        body = json.loads(sc.read_text())
        assert "Sources" in body, f"{sc.name} missing Sources"
        assert isinstance(body["Sources"], list) and body["Sources"], (
            f"{sc.name} has empty Sources"
        )
        assert all(s.startswith("bids::") for s in body["Sources"]), sc.name
        assert body.get("SpatialReference", "").startswith("bids::"), sc.name
        assert body.get("Resolution") == "native", sc.name
        assert isinstance(body.get("VoxelSize"), list), sc.name


def test_tumor_seg_lesion_stats_json_content(
    tmp_path: Path, minimal_bids_with_tumor_inputs: tuple[Path, dict]
) -> None:
    """Lesion stats JSON computes per-region volumes from the segmentation."""
    from pmriprep.workflows.tumor_seg.datasinks import init_tumor_seg_datasinks_wf

    bids, sources = minimal_bids_with_tumor_inputs
    out_root = tmp_path / "out"
    seg = _make_seg(tmp_path / "seg.nii.gz", label=2)  # 8 enhancing voxels
    masks = {
        n: _make_mask(tmp_path / f"{n}.nii.gz")
        for n in ("enhancing", "nonenhancing", "wholetumor", "nawm")
    }

    wf = init_tumor_seg_datasinks_wf(
        output_dir=str(out_root), bids_dir=str(bids),
        subject_id="01", session_id="0001",
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    for tag, p in sources.items():
        setattr(inp, tag, str(p))
    inp.segmentation = str(seg)
    inp.enhancing_mask = str(masks["enhancing"])
    inp.non_enhancing_mask = str(masks["nonenhancing"])
    inp.whole_tumor_mask = str(masks["wholetumor"])
    inp.nawm_mask = str(masks["nawm"])
    inp.qc_metrics = {}
    wf.run()

    lesion_path = (
        out_root / "pmriprep" / "sub-01" / "ses-0001"
        / "sub-01_ses-0001_space-anat_desc-lesionStats.json"
    )
    body = json.loads(lesion_path.read_text())
    assert body["TotalLesionVoxels"] == 8
    assert body["Regions"]["contrast_enhancing_tumor"]["voxel_count"] == 8
    assert body["Regions"]["non_enhancing_tumor"]["voxel_count"] == 0
    assert body["derived_metrics"]["enhancing_fraction"] == 1.0


def test_init_pmriprep_wf_skip_anat_with_tumor_seg(tmp_path: Path) -> None:
    """Regression: ``--skip-anat`` with tumor_seg must not NameError on factory args.

    Catches the class of bug where per-pipeline factory parameters (e.g.
    ``has_t1w_pre``, ``is_postcon_ref``) are computed inside the
    ``if decision.run_anat:`` block but referenced by sibling tumor_seg / DSC
    / DCE blocks. Builds the workflow only — does not execute it.
    """
    import json as _json

    import nibabel as _nib
    import numpy as _np

    from pmriprep.config.tumor_seg import TumorSegConfig
    from pmriprep.config.anat import AnatConfig
    from pmriprep.config.dsc import DSCConfig
    from pmriprep.config.dce import DCEConfig
    from pmriprep.workflows.base import PipelineSkipFlags, init_pmriprep_wf

    # Minimal BIDS dataset with all four HD-GLIO modalities present so
    # decision.run_tumor_seg=True even when --skip-anat=True (which only
    # short-circuits the anat *block*, not the tumor_seg gate).
    bids = tmp_path / "bids"
    raw = bids / "sub-01" / "ses-0001" / "anat"
    raw.mkdir(parents=True)
    (bids / "dataset_description.json").write_text(
        _json.dumps({"Name": "T", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )
    for fname in (
        "sub-01_ses-0001_T1w.nii.gz",
        "sub-01_ses-0001_ce-GAD_T1w.nii.gz",
        "sub-01_ses-0001_T2w.nii.gz",
        "sub-01_ses-0001_FLAIR.nii.gz",
    ):
        _nib.save(
            _nib.Nifti1Image(_np.zeros((4, 4, 4), dtype=_np.float32), _np.eye(4)),
            raw / fname,
        )
        (raw / fname).with_suffix("").with_suffix(".json").write_text(
            _json.dumps({"RepetitionTime": 2.0})
        )

    # Build with skip_anat=True; assert no NameError raised.
    wf = init_pmriprep_wf(
        bids_dir=str(bids),
        output_dir=str(tmp_path / "out"),
        subject_id="01",
        session_id="0001",
        skip_flags=PipelineSkipFlags(skip_anat=True, skip_dsc=True, skip_dce=True),
        anat_config=AnatConfig(),
        tumor_seg_config=TumorSegConfig(),
        dsc_config=DSCConfig(),
        dce_config=DCEConfig(),
        work_dir=str(tmp_path / "work"),
    )
    assert wf is not None
