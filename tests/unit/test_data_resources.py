"""Tests for bundled package resources."""

from pmriprep.data import get_arterial_atlas


def test_default_arterial_atlas_exists() -> None:
    """Resolve the default arterial atlas from the installed resources."""
    atlas = get_arterial_atlas()

    assert atlas.is_file()
    assert atlas.name == "MNI_2mm_MRA_arterial_full_mask_dil.nii.gz"
