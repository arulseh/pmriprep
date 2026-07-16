"""Sanity tests for Phase 1 foundation pieces of the per-image QC distribution
refactor. These guard the new helpers against regressions during the Phase 2
wiring work, where a silent bug here would surface as a Phase 6 integration
test failure with a much harder root-cause trace.

Coverage targets the critical paths only:
- ``write_tumor_dseg_tsv``: round-trip + label validation + line endings
- ``extract_per_map_lesion_stats``: empty-input cases + happy path
- ``enrich_brain_mask_sidecar`` / ``enrich_preproc_anat_sidecar`` /
  ``enrich_probseg_sidecar``: empty-qc no-op + happy path
- ``build_aif_timeseries_sidecar``: per-modality units (DSC vs DCE) +
  invalid-modality ValueError
- ``build_tumor_seg_dseg_sidecar``: ``Columns`` block present
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib  # type: ignore[import-untyped]
import numpy as np
import pytest

from pmriprep.utils.lesion_stats import (
    extract_per_map_lesion_stats,
    write_tumor_dseg_tsv,
)
from pmriprep.utils.sidecar_builders import (
    build_aif_timeseries_sidecar,
    build_tumor_seg_dseg_sidecar,
    enrich_brain_mask_sidecar,
    enrich_preproc_anat_sidecar,
    enrich_probseg_sidecar,
)


# ---------------------------------------------------------------------------
# write_tumor_dseg_tsv
# ---------------------------------------------------------------------------


def _make_dseg_nifti(tmp_path: Path, labels: dict[int, int]) -> Path:
    """Build a 10x10x10 dseg NIfTI with N voxels per label.

    ``labels`` maps integer label → desired voxel count.
    """
    data = np.zeros((10, 10, 10), dtype=np.int16)
    flat = data.ravel()
    cursor = 0
    for label, count in labels.items():
        if label == 0:
            continue  # background fills the rest
        flat[cursor:cursor + count] = label
        cursor += count
    out = tmp_path / "seg.nii.gz"
    nib.save(nib.Nifti1Image(flat.reshape(data.shape), affine=np.eye(4)), str(out))
    return out


def test_write_tumor_dseg_tsv_round_trip(tmp_path: Path) -> None:
    nii = _make_dseg_nifti(tmp_path, {1: 27, 2: 8})
    out = write_tumor_dseg_tsv(nii, tmp_path / "sub-01_dseg.tsv")

    text = out.read_text()
    lines = text.split("\n")
    # Header per bids-spec-expert PR1: abbreviation column added,
    # redundant volume_ml dropped (consumers can divide volume_mm3/1000).
    assert lines[0] == "index\tname\tabbreviation\tvoxel_count\tvolume_mm3"
    assert "non_enhancing_tumor\tNET\t27" in text
    assert "contrast_enhancing_tumor\tET\t8" in text
    # Background (index 0) MUST NOT appear — BIDS dseg semantics
    assert "\t0\twhole_tumor" not in text
    # Line endings are LF only (Unix), not CRLF
    assert "\r\n" not in text


def test_write_tumor_dseg_tsv_rejects_unexpected_labels(tmp_path: Path) -> None:
    # Label 5 is outside the registered set
    data = np.zeros((4, 4, 4), dtype=np.int16)
    data[0, 0, 0] = 5
    nii = tmp_path / "bad.nii.gz"
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), str(nii))

    with pytest.raises(ValueError, match="unexpected labels"):
        write_tumor_dseg_tsv(nii, tmp_path / "out.tsv")


# ---------------------------------------------------------------------------
# extract_per_map_lesion_stats
# ---------------------------------------------------------------------------


def test_extract_per_map_lesion_stats_none_input() -> None:
    assert extract_per_map_lesion_stats(None, "cbf") is None


def test_extract_per_map_lesion_stats_empty_regions() -> None:
    assert extract_per_map_lesion_stats({"Regions": {}}, "cbf") is None


def test_extract_per_map_lesion_stats_missing_map() -> None:
    aggregate = {
        "Regions": {
            "contrast_enhancing_tumor": {
                "voxel_count": 30,
                "volume_ml": 0.030,
                "perfusion_maps": {"cbv": {"mean": 4.1}},
            },
        },
    }
    assert extract_per_map_lesion_stats(aggregate, "cbf") is None


def test_extract_per_map_lesion_stats_happy_path() -> None:
    aggregate = {
        "Regions": {
            "contrast_enhancing_tumor": {
                "voxel_count": 30,
                "volume_ml": 0.030,
                "perfusion_maps": {"cbf": {"mean": 50.2, "std": 12.1}},
            },
            "non_enhancing_tumor": {
                "voxel_count": 100,
                "volume_ml": 0.100,
                "perfusion_maps": {"cbf": {"mean": 35.0, "std": 18.0}},
            },
        },
    }
    result = extract_per_map_lesion_stats(aggregate, "cbf")
    assert result is not None
    assert set(result["Regions"].keys()) == {"contrast_enhancing_tumor", "non_enhancing_tumor"}
    assert result["Regions"]["contrast_enhancing_tumor"]["stats"] == {"mean": 50.2, "std": 12.1}
    assert result["Regions"]["contrast_enhancing_tumor"]["voxel_count"] == 30


# ---------------------------------------------------------------------------
# enrich_*_sidecar (anat per-image enrichment)
# ---------------------------------------------------------------------------


def test_enrich_brain_mask_sidecar_no_qc_metrics_is_no_op() -> None:
    sc: dict = {"Description": "Brain mask"}
    out = enrich_brain_mask_sidecar(sc, None)
    assert out is sc  # mutate-and-return-self
    # Flat-key convention: no BrainExtraction* keys when qc_metrics is None
    assert all(not k.startswith("BrainExtraction") for k in out)


def test_enrich_brain_mask_sidecar_happy_path() -> None:
    qc = {"brain_extraction": {"tool": "hdbet", "version": "v1.0", "device": "mps"}}
    out = enrich_brain_mask_sidecar({}, qc)
    assert out["BrainExtractionTool"] == "hdbet"
    # Tool version is no longer stamped per-file — it lives in the dataset-level
    # GeneratedBy (BIDS-Derivatives idiom); the custom key was retired.
    assert "BrainExtractionVersion" not in out
    assert out["BrainExtractionDevice"] == "mps"
    # No nested block (regression guard against round-1 convention)
    assert "BrainExtraction" not in out


def test_enrich_preproc_anat_sidecar_happy_path() -> None:
    qc = {"preprocessing": {"denoising": True, "bias_correction": "N4", "acpc_aligned": True}}
    out = enrich_preproc_anat_sidecar({}, qc)
    assert out["PreprocessingDenoising"] is True
    assert out["PreprocessingBiasCorrection"] == "N4"
    assert out["PreprocessingACPCAligned"] is True
    # No nested block
    assert "Preprocessing" not in out


def test_enrich_preproc_anat_sidecar_with_coreg_modality() -> None:
    qc = {
        "preprocessing": {"denoising": False},
        "coregistration": {
            "method": "flirt", "cost_function": "corratio",
            "correlation_ratio": 0.94,
        },
    }
    out = enrich_preproc_anat_sidecar({}, qc, coreg_modality="T2w")
    assert out["CoregistrationT2wMethod"] == "flirt"
    assert out["CoregistrationT2wCostFunction"] == "corratio"
    assert out["CoregistrationT2wCorrelationRatio"] == pytest.approx(0.94)


def test_enrich_preproc_anat_sidecar_t1w_omits_coreg_keys() -> None:
    qc = {
        "preprocessing": {"denoising": True},
        "coregistration": {"method": "flirt"},
    }
    # T1w sidecar passes coreg_modality=None — no Coregistration* keys
    out = enrich_preproc_anat_sidecar({}, qc, coreg_modality=None)
    assert all(not k.startswith("Coregistration") for k in out)


def test_enrich_probseg_sidecar_filters_by_tissue_type() -> None:
    qc = {
        "tissue_segmentation": {
            "WM": {"mean_probability": 0.92, "voxel_count": 100, "volume_ml": 0.1},
            "GM": {"mean_probability": 0.88},
        },
    }
    out_wm = enrich_probseg_sidecar({}, qc, "WM")
    assert out_wm["TissueSegmentationMeanProbability"] == pytest.approx(0.92)
    assert out_wm["TissueSegmentationVoxelCount"] == 100
    assert out_wm["TissueSegmentationVolume"] == pytest.approx(0.1)
    # No nested block
    assert "TissueSegmentationStats" not in out_wm
    # CSF tissue not in qc — no keys written
    out_csf = enrich_probseg_sidecar({}, qc, "CSF")
    assert all(not k.startswith("TissueSegmentation") for k in out_csf)


# ---------------------------------------------------------------------------
# build_aif_timeseries_sidecar (per-modality units verified against AIF writers)
# ---------------------------------------------------------------------------


def test_aif_sidecar_dsc_units_are_delta_r2_star(tmp_path: Path) -> None:
    """DSC AIF curve is ΔR2* in 1/s — verified against interfaces/dsc_aif.py."""
    sc = build_aif_timeseries_sidecar(
        modality="DSC",
        source_paths=[tmp_path / "src.nii.gz"],
        bids_dir=tmp_path,
        subject="01",
        session="01",
        aif_qc=None,
    )
    assert sc["Units"] == "1/s"
    assert sc["Columns"] == ["delta_r2_star"]


def test_aif_sidecar_dce_units_are_plasma_concentration(tmp_path: Path) -> None:
    """DCE AIF curve is plasma [Gd] in mM after /(1-Hct) conversion."""
    sc = build_aif_timeseries_sidecar(
        modality="DCE",
        source_paths=[tmp_path / "src.nii.gz"],
        bids_dir=tmp_path,
        subject="01",
        session="01",
        aif_qc=None,
    )
    assert sc["Units"] == "mM"
    assert sc["Columns"] == ["plasma_concentration_mM"]


def test_aif_sidecar_invalid_modality_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="modality must be"):
        build_aif_timeseries_sidecar(
            modality="ASL",
            source_paths=[tmp_path / "src.nii.gz"],
            bids_dir=tmp_path,
            subject="01",
            session=None,
        )


def test_aif_sidecar_flat_keys_populated(tmp_path: Path) -> None:
    """Flat ``AIF*``-prefixed top-level keys (no nested AIFDetection block)."""
    aif_qc = {
        "detection_method": "manual",
        "n_aif_voxels_selected": 12,
        "aif_metrics": {"peak_conc": 4.2, "fwhm": 6.5},
        "hematocrit": 0.42,
    }
    sc = build_aif_timeseries_sidecar(
        modality="DCE",
        source_paths=[tmp_path / "src.nii.gz"],
        bids_dir=tmp_path,
        subject="01",
        session="01",
        aif_qc=aif_qc,
    )
    assert sc["AIFMethod"] == "manual"
    assert sc["AIFNumberOfVoxelsSelected"] == 12
    assert sc["AIFPeakAmplitude"] == pytest.approx(4.2)
    assert sc["AIFFWHM"] == pytest.approx(6.5)
    assert sc["AIFHematocrit"] == pytest.approx(0.42)
    # Regression: no nested block
    assert "AIFDetection" not in sc


# ---------------------------------------------------------------------------
# build_tumor_seg_dseg_sidecar — Columns block was added to describe TSV
# ---------------------------------------------------------------------------


def test_tumor_seg_dseg_sidecar_describes_tsv_columns(tmp_path: Path) -> None:
    # build_tumor_seg_dseg_sidecar reads voxel sizes from the reference NIfTI
    # Use a segmentation with label values so TumorVolumetrics computes real counts
    seg_data = np.zeros((4, 4, 4), dtype=np.int16)
    seg_data[1:3, 1:3, 1:3] = 2  # 8 enhancing voxels
    ref = tmp_path / "seg.nii.gz"
    nib.save(nib.Nifti1Image(seg_data, affine=np.eye(4)), str(ref))
    sc = build_tumor_seg_dseg_sidecar(
        source_paths=[ref],
        reference_voxel_source=ref,
        bids_dir=tmp_path,
        subject="01",
        session="01",
    )
    assert "Columns" in sc
    # Per bids-spec-expert PR1: abbreviation column added, redundant
    # volume_ml dropped (consumers can divide volume_mm3/1000).
    expected_cols = {"index", "name", "abbreviation", "voxel_count", "volume_mm3"}
    assert set(sc["Columns"].keys()) == expected_cols
    # Per-column metadata follows BIDS Columns convention
    assert "Description" in sc["Columns"]["voxel_count"]
    assert sc["Columns"]["voxel_count"]["Units"] == "1"
    assert "Description" in sc["Columns"]["abbreviation"]

    # TumorVolumetrics block carries per-region volumes
    assert "TumorVolumetrics" in sc
    tv = sc["TumorVolumetrics"]
    assert tv["TotalLesionVoxels"] == 8
    assert tv["Regions"]["contrast_enhancing_tumor"]["voxel_count"] == 8
    assert tv["Regions"]["non_enhancing_tumor"]["voxel_count"] == 0
    assert tv["EnhancingFraction"] == 1.0


def test_tumor_volumetrics_mixed_labels(tmp_path: Path) -> None:
    """Mixed label segmentation produces 0 < EnhancingFraction < 1."""
    seg_data = np.zeros((4, 4, 4), dtype=np.int16)
    seg_data[0, 0, 0] = 1  # 1 non-enhancing
    seg_data[1, 1, 1] = 2  # 1 enhancing
    ref = tmp_path / "seg.nii.gz"
    nib.save(nib.Nifti1Image(seg_data, affine=np.eye(4)), str(ref))
    sc = build_tumor_seg_dseg_sidecar(
        source_paths=[ref], reference_voxel_source=ref,
        bids_dir=tmp_path, subject="01", session="01",
    )
    tv = sc["TumorVolumetrics"]
    assert tv["TotalLesionVoxels"] == 2
    assert tv["Regions"]["contrast_enhancing_tumor"]["voxel_count"] == 1
    assert tv["Regions"]["non_enhancing_tumor"]["voxel_count"] == 1
    assert tv["EnhancingFraction"] == 0.5


def test_tumor_volumetrics_empty_segmentation(tmp_path: Path) -> None:
    """All-zeros segmentation yields EnhancingFraction=None (JSON null)."""
    ref = tmp_path / "seg.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), affine=np.eye(4)), str(ref))
    sc = build_tumor_seg_dseg_sidecar(
        source_paths=[ref], reference_voxel_source=ref,
        bids_dir=tmp_path, subject="01", session="01",
    )
    tv = sc["TumorVolumetrics"]
    assert tv["TotalLesionVoxels"] == 0
    assert tv["EnhancingFraction"] is None
