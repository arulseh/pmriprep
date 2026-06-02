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

    The cascade walks from the supratentorial MCA/ACA territory outward
    to the full MRA arterial mask. CoW is omitted to avoid skull-base
    susceptibility and cavernous-sinus partial-volume contamination.

    Levels (when all files present):
        1. MCA_ACA      — undilated, highest arterial specificity
        2. MCA_ACA_dil  — 1× dilated (~2 mm margin)
        3. full_mask    — full MRA incl. basilar / posterior circulation

    Missing files are silently skipped, so the effective cascade adapts
    to whichever atlas files are installed.

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
        ("MCA_ACA", ""),      # Undilated — tightest, highest arterial specificity
        ("MCA_ACA", "dil"),   # 1× dilated — ~2 mm margin
        ("full_mask", ""),    # Full MRA incl. basilar — broadest atlas
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


def get_mni_brain_template() -> Path:
    """Get path to MNI152NLin6Asym brain-extracted T1w template.

    Uses TemplateFlow (NiPreps standard) to resolve the template path.
    The template is pre-downloaded into ``$TEMPLATEFLOW_HOME`` at container
    build time, so no network access is required at runtime.

    Returns
    -------
    Path
        Path to the skull-stripped 1mm MNI T1w template.

    Raises
    ------
    FileNotFoundError
        If the template cannot be resolved via TemplateFlow.
    """
    try:
        import templateflow.api as tflow  # type: ignore[import-untyped]

        tf_path = tflow.get(
            "MNI152NLin6Asym", resolution=1, desc="brain",
            suffix="T1w", extension=".nii.gz",
        )
        if isinstance(tf_path, Path) and tf_path.exists():
            return tf_path
    except Exception as exc:
        import logging

        logging.getLogger("pmriprep.data").debug(
            "TemplateFlow lookup failed: %s", exc,
        )

    raise FileNotFoundError(
        "MNI152NLin6Asym brain template not found. "
        "Set TEMPLATEFLOW_HOME to a directory containing the template, "
        "or pass a custom template via the mni_template input."
    )


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


def get_boilerplate_bib() -> Path:
    """Get path to the bundled methods-boilerplate BibTeX file.

    Single source of truth for the citation references rendered into the QC
    report Methods section (see ``pmriprep.utils.report_utils``).

    Returns
    -------
    Path
        Path to ``boilerplate.bib``.

    Raises
    ------
    FileNotFoundError
        If the file is missing (e.g. a non-editable install lacking package data).
    """
    bib = DATA_DIR / "boilerplate.bib"
    if not bib.is_file():
        raise FileNotFoundError(f"Boilerplate bibliography not found: {bib}")
    return bib


def get_citation_csl() -> Path:
    """Get path to the bundled numbered (Vancouver-style) citation CSL.

    Returns
    -------
    Path
        Path to ``vancouver.csl``.

    Raises
    ------
    FileNotFoundError
        If the file is missing.
    """
    csl = DATA_DIR / "vancouver.csl"
    if not csl.is_file():
        raise FileNotFoundError(f"Citation CSL not found: {csl}")
    return csl
