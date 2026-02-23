"""PMRIPrep bundled data files.

This module provides access to atlas files and other data bundled with PMRIPrep.
"""

from pathlib import Path

# Base data directory
DATA_DIR = Path(__file__).parent


def get_arterial_atlas(resolution: str = "1mm", variant: str = "dil3") -> Path:
    """Get path to MNI arterial atlas for AIF detection.

    Parameters
    ----------
    resolution : str
        Atlas resolution: '1mm' or '2mm' (default: '1mm')
    variant : str
        Atlas variant suffix, e.g. 'dil3' for 3x dilated (default: 'dil3').
        Use empty string for the non-dilated atlas.

    Returns
    -------
    Path
        Path to the arterial atlas NIfTI file.

    Raises
    ------
    FileNotFoundError
        If the requested atlas file doesn't exist.
    """
    if variant:
        atlas_file = DATA_DIR / "atlases" / "mni_aif" / f"MNI_{resolution}_MRA_arterial_full_mask_{variant}.nii.gz"
    else:
        atlas_file = DATA_DIR / "atlases" / "mni_aif" / f"MNI_{resolution}_MRA_arterial_full_mask.nii.gz"
    if not atlas_file.exists():
        raise FileNotFoundError(f"Arterial atlas not found: {atlas_file}")
    return atlas_file


def get_arterial_atlas_cascade(resolution: str = "2mm") -> list[Path]:
    """Return atlas files from tightest to broadest for cascade search.

    The cascade ordering allows AIF detection to try the most spatially
    specific atlas first (MCA/ACA only), then Circle of Willis, and finally
    the fully dilated MRA mask. This reduces tissue contamination from
    overly broad atlas masks at high spatial resolution.

    Parameters
    ----------
    resolution : str
        Atlas resolution: '1mm' or '2mm' (default: '2mm').

    Returns
    -------
    list[Path]
        Ordered list of existing atlas paths, tightest to broadest.
        May be empty if no atlas files are found.
    """
    # (region_name, dilation_suffix) — ordered tightest to broadest
    variants = [
        ("MCA_ACA", "dil2"),   # Most superior: MCA + ACA, 2× dilated
        ("CoW_mask", "dil2"),  # Circle of Willis, 2× dilated
        ("full_mask", "dil2"), # Full MRA incl. basilar, 2× dilated
    ]
    cascade: list[Path] = []
    for region, dilation in variants:
        suffix = f"_{dilation}" if dilation else ""
        path = (
            DATA_DIR / "atlases" / "mni_aif"
            / f"MNI_{resolution}_MRA_arterial_{region}{suffix}.nii.gz"
        )
        if path.exists():
            cascade.append(path)
    return cascade


def get_lobar_atlas_dir() -> Path:
    """Get path to MNI lobar atlas directory for NAWM region restriction.

    Returns
    -------
    Path
        Path to the lobar atlas directory containing region masks
        (e.g., Supra_L.nii.gz, Cerebellum_R.nii.gz).

    Raises
    ------
    FileNotFoundError
        If the lobar atlas directory doesn't exist.
    """
    atlas_dir = DATA_DIR / "atlases" / "mni_lobar"
    if not atlas_dir.is_dir():
        raise FileNotFoundError(f"Lobar atlas directory not found: {atlas_dir}")
    return atlas_dir
