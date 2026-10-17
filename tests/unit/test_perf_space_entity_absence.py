"""Test the project's derivative space-entity policy.

Native perfusion outputs omit a space entity. Anatomical-reference outputs
use T1w, while structural images resampled into perfusion space use dceref
or dscref. SpatialReference identifies the corresponding reference image.
"""
from __future__ import annotations

import re

import pytest

from pmriprep.utils.sidecar_builders import (
    anat_reference_basename,
    dce_reference_basename,
    dsc_reference_basename,
)


_SPACE_ENTITY = re.compile(r"_space-")


# -----------------------------------------------------------------------------
# Rule 1: perf/ outputs lack space- entity
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "basename",
    [dce_reference_basename(), dsc_reference_basename()],
)
def test_reference_basenames_lack_space_entity(basename: str) -> None:
    """Mean reference image basenames are native perf/ outputs."""
    assert _SPACE_ENTITY.search(basename) is None, (
        f"Reference basename {basename!r} carries space- entity. "
        f"Native perf/ outputs must omit space- per BIDS-fmriprep convention."
    )


def test_dce_copy_perf_map_lacks_space_entity(tmp_path) -> None:
    """End-to-end: _copy_perf_map writes a path without space-."""
    from pmriprep.workflows.dce.datasinks import _copy_perf_map

    src = tmp_path / "in.nii.gz"
    src.write_bytes(b"")  # empty file, just to satisfy shutil.copy2
    out_path = _copy_perf_map(
        in_file=str(src),
        output_dir=str(tmp_path / "out"),
        subject="01",
        session="01",
        desc="dce",
        suffix="Ktrans",
        meta_dict={"Description": "test"},
        datatype="perf",
    )
    assert _SPACE_ENTITY.search(out_path) is None, (
        f"_copy_perf_map (DCE) wrote path with space-: {out_path}"
    )


def test_dsc_copy_perf_map_lacks_space_entity(tmp_path) -> None:
    """End-to-end: _copy_perf_map writes a path without space-."""
    from pmriprep.workflows.dsc.datasinks import _copy_perf_map

    src = tmp_path / "in.nii.gz"
    src.write_bytes(b"")
    out_path = _copy_perf_map(
        in_file=str(src),
        output_dir=str(tmp_path / "out"),
        subject="01",
        session="01",
        desc="dsc",
        suffix="mtt",
        meta_dict={"Description": "test"},
    )
    assert _SPACE_ENTITY.search(out_path) is None, (
        f"_copy_perf_map (DSC) wrote path with space-: {out_path}"
    )


def test_dce_copy_perf_map_anat_mask_has_space_entity(tmp_path) -> None:
    """Anat-side masks include space-dceref with correct entity ordering."""
    from pmriprep.workflows.dce.datasinks import _copy_perf_map

    src = tmp_path / "in.nii.gz"
    src.write_bytes(b"")
    out_path = _copy_perf_map(
        in_file=str(src),
        output_dir=str(tmp_path / "out"),
        subject="01",
        session="01",
        desc="brain",
        suffix="",
        meta_dict={"Description": "test"},
        datatype="anat",
        bids_suffix="mask",
        space="dceref",
    )
    assert "space-dceref" in out_path
    assert out_path.index("space-dceref") < out_path.index("desc-brain")


def test_anat_reference_basename_uses_T1w_space() -> None:
    """Anatomical reference images use the T1w space entity."""
    precon = anat_reference_basename(is_postcon_ref=False)
    postcon = anat_reference_basename(is_postcon_ref=True)
    assert "space-T1w" in precon, (
        f"Precon ref basename should use space-T1w; got {precon!r}"
    )
    assert "space-T1w" in postcon, (
        f"Postcon ref basename should use space-T1w; got {postcon!r}"
    )


def test_anat_spatial_reference_preserves_source_entities() -> None:
    """Anatomical reference URIs identify the derivative that is written."""
    from pmriprep.utils.sidecar_builders import _anat_spatial_reference

    source = "/raw/sub-X/ses-Y/anat/sub-X_ses-Y_acq-mprage_run-2_T1w.nii.gz"
    assert _anat_spatial_reference("X", "Y", source=source) == (
        "bids::sub-X/ses-Y/anat/"
        "sub-X_ses-Y_acq-mprage_run-2_space-T1w_desc-preproc_T1w.nii.gz"
    )


def test_postcon_spatial_reference_adds_forced_contrast_entity() -> None:
    """Metadata-classified postcon references name the forced output entity."""
    from pmriprep.utils.sidecar_builders import _anat_spatial_reference

    source = "/raw/sub-X/ses-Y/anat/sub-X_ses-Y_acq-mprage_run-2_T1w.nii.gz"
    assert _anat_spatial_reference(
        "X", "Y", is_postcon_ref=True, source=source,
    ) == (
        "bids::sub-X/ses-Y/anat/"
        "sub-X_ses-Y_acq-mprage_ce-GAD_run-2_space-T1w_desc-preproc_T1w.nii.gz"
    )
