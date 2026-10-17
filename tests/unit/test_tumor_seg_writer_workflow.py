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
        ("source_t1w", "sub-01_ses-0001_acq-highres_run-1_T1w.nii.gz"),
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


def test_tumor_seg_derivatives_and_sidecars(
    tmp_path: Path, minimal_bids_with_tumor_inputs: tuple[Path, dict]
) -> None:
    """Tumor images carry source provenance, measured volumes and forwarded NAWM QC."""
    from pmriprep.workflows.tumor_seg.datasinks import init_tumor_seg_datasinks_wf

    bids, sources = minimal_bids_with_tumor_inputs
    out_root = tmp_path / "out"
    seg = _make_seg(tmp_path / "seg.nii.gz")
    tumor = nib.load(seg).get_fdata() > 0
    mask_data = {
        "enhancing": tumor,
        "nonenhancing": np.zeros_like(tumor),
        "wholetumor": tumor,
        "nawm": ~tumor,
    }
    masks = {}
    for name, data in mask_data.items():
        masks[name] = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(data.astype(np.uint8), np.eye(4)), masks[name])

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
    inp.qc_metrics = {
        "nawm": {
            "atlas_used": True,
            "compartment_label": "infratentorial",
            "tumor_laterality": "bilateral",
            "selected_hemisphere": "both",
            "laterality_method": "brain_symmetry",
            "nawm_compartment": "supratentorial",
            "f_supra": 0.1,
            "f_cereb": 0.8,
            "f_other": 0.1,
            "atlas_coverage": 0.9,
            "warnings": [
                "infratentorial_tumor_uses_bilateral_supratentorial_nawm"
            ],
            "erosion_voxels_final": 1,
            "erosion_mm_final": 1.25,
            "valid": False,
            "n_voxels": 56,
            "minimum_required_voxels": 100,
            "quantitative_use_exclusion_reason": (
                "insufficient_voxels_after_erosion_relaxation"
            ),
        }
    }
    wf.run()

    anat = out_root / "pmriprep" / "sub-01" / "ses-0001" / "anat"
    expected = [
        anat / "sub-01_ses-0001_acq-highres_run-1_space-T1w_desc-tumorseg_dseg.nii.gz",
        anat / "sub-01_ses-0001_acq-highres_run-1_space-T1w_desc-tumorseg_dseg.tsv",
        anat / "sub-01_ses-0001_acq-highres_run-1_space-T1w_desc-enhancing_mask.nii.gz",
        anat / "sub-01_ses-0001_acq-highres_run-1_space-T1w_desc-nonenhancing_mask.nii.gz",
        anat / "sub-01_ses-0001_acq-highres_run-1_space-T1w_desc-wholetumor_mask.nii.gz",
        anat / "sub-01_ses-0001_acq-highres_run-1_space-T1w_desc-lesion_mask.nii.gz",
        anat / "sub-01_ses-0001_acq-highres_run-1_space-T1w_desc-nawm_mask.nii.gz",
    ]
    for p in expected:
        assert p.exists(), f"Missing: {p.name}"

    dd = out_root / "pmriprep" / "dataset_description.json"
    assert dd.exists()
    sidecars = [
        p.with_suffix("").with_suffix(".json") for p in expected
        if p.name.endswith(".nii.gz")
    ]

    for sc in sidecars:
        body = json.loads(sc.read_text())
        assert "Sources" in body, f"{sc.name} missing Sources"
        assert isinstance(body["Sources"], list) and body["Sources"], (
            f"{sc.name} has empty Sources"
        )
        if "desc-nawm" in sc.name:
            assert body["Sources"] == [
                "bids::sub-01/ses-0001/anat/sub-01_ses-0001_acq-highres_run-1_"
                "space-T1w_desc-wholetumor_mask.nii.gz",
                "bids::sub-01/ses-0001/anat/sub-01_ses-0001_acq-highres_run-1_"
                "space-T1w_label-WM_probseg.nii.gz",
                "bids::sub-01/ses-0001/anat/sub-01_ses-0001_acq-highres_run-1_"
                "space-T1w_desc-brain_mask.nii.gz",
                "bids::sub-01/ses-0001/anat/sub-01_ses-0001_acq-highres_run-1_"
                "space-T1w_desc-preproc_T1w.nii.gz",
            ], sc.name
            assert body["AtlasRestrictionApplied"] is True
            assert body["TumorCompartment"] == "infratentorial"
            assert body["SelectedHemisphere"] == "both"
            assert body["NAWMCompartment"] == "supratentorial"
            assert body["CerebellarOverlapFraction"] == 0.8
            assert body["Warnings"] == [
                "infratentorial_tumor_uses_bilateral_supratentorial_nawm"
            ]
            assert body["WMProbabilityThreshold"] == 0.9
            assert body["TumorExclusionMarginMillimeters"] == 20.0
            assert body["ErosionVoxels"] == 1
            assert body["ErosionMillimeters"] == 1.25
            assert body["ValidForQuantification"] is False
            assert body["VoxelCount"] == 56
            assert body["MinimumRequiredVoxels"] == 100
            assert body["QuantitativeUseExclusionReason"] == (
                "insufficient_voxels_after_erosion_relaxation"
            )
        elif sc.name.endswith("_mask.json"):
            assert body["Sources"] == [
                "bids::sub-01/ses-0001/anat/"
                "sub-01_ses-0001_acq-highres_run-1_space-T1w_"
                "desc-tumorseg_dseg.nii.gz"
            ], sc.name
        else:
            assert all(s.startswith("bids:raw:") for s in body["Sources"]), sc.name
        assert body.get("SpatialReference", "").startswith("bids::"), sc.name
        assert "Resolution" not in body, sc.name
        assert isinstance(body.get("VoxelSize"), list), sc.name

    dseg_json = (
        out_root / "pmriprep" / "sub-01" / "ses-0001" / "anat"
        / "sub-01_ses-0001_acq-highres_run-1_space-T1w_desc-tumorseg_dseg.json"
    )
    body = json.loads(dseg_json.read_text())
    tv = body["TumorVolumetrics"]
    assert tv["TotalLesionVoxels"] == 8
    assert tv["Regions"]["contrast_enhancing_tumor"]["voxel_count"] == 8
    assert tv["Regions"]["non_enhancing_tumor"]["voxel_count"] == 0
    assert tv["EnhancingFraction"] == 1.0
    assert tv["Regions"]["whole_tumor"]["volume_ml"] == pytest.approx(0.008)


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


def test_nawm_sidecar_is_honest_when_supratentorial_atlas_is_unavailable(
    minimal_bids_with_tumor_inputs: tuple[Path, dict],
) -> None:
    """A QC-only fallback must not claim a supratentorial compartment."""
    from pmriprep.utils.sidecar_builders import build_tumor_seg_nawm_sidecar

    bids, sources = minimal_bids_with_tumor_inputs
    sidecar = build_tumor_seg_nawm_sidecar(
        source_paths=[sources["source_t1w"]],
        reference_voxel_source=sources["source_t1w"],
        bids_dir=bids,
        subject="01",
        session="0001",
        qc_metrics={
            "nawm": {
                "atlas_used": False,
                "nawm_compartment": "hemisphere_unrestricted",
                "valid": False,
                "quantitative_use_exclusion_reason": (
                    "supratentorial_atlas_unavailable"
                ),
            }
        },
    )

    assert sidecar["NAWMCompartment"] == "hemisphere_unrestricted"
    assert sidecar["ValidForQuantification"] is False
    assert sidecar["QuantitativeUseExclusionReason"] == (
        "supratentorial_atlas_unavailable"
    )
    assert sidecar["Description"].startswith("Hemisphere-restricted")
