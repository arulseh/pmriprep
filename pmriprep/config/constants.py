"""Shared constants for configuration modules."""

from __future__ import annotations

# Named dose presets resolved to float multipliers
# Used by DCE, DSC configs, loader, and CLI
DOSE_PRESETS: dict[str, float] = {"half": 0.5, "full": 1.0, "double": 2.0}

# HD-GLIO tumor segmentation label registry.
# (int_label, snake_case_name, abbreviation, human_label, description)
HDGLIO_TUMOR_SEG_LABELS: tuple[tuple[int, str, str, str, str], ...] = (
    (0, "background", "BG", "Background", "Background (no tumor)"),
    (1, "non_enhancing_tumor", "NET", "Non-enhancing tumor",
     "Non-enhancing tumor (HD-GLIO label 1)"),
    (2, "contrast_enhancing_tumor", "ET", "Enhancing tumor",
     "Contrast-enhancing tumor (HD-GLIO label 2)"),
)

HDGLIO_LABEL_TO_NAME: dict[int, str] = {
    label: name for label, name, _, _, _ in HDGLIO_TUMOR_SEG_LABELS if label > 0
}

HDGLIO_BIDS_LABELS: dict[str, str] = {
    str(label): human for label, _, _, human, _ in HDGLIO_TUMOR_SEG_LABELS
}

HDGLIO_DSEG_ROWS: tuple[tuple[int, str, str, str], ...] = tuple(
    (label, name, abbrev, desc)
    for label, name, abbrev, _, desc in HDGLIO_TUMOR_SEG_LABELS
    if label > 0
)
