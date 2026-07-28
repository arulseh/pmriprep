"""Characterization tests for DSC AIF cascade tier planning."""

import numpy as np

from pmriprep.interfaces.dsc_aif import AIFDetection


def _names(attempts: list[tuple]) -> list[str]:
    """Return tier names from a cascade attempt plan."""
    return [attempt[2] for attempt in attempts]


def test_atlas_plan_precedes_full_atlas_and_brain() -> None:
    """The default plan broadens from atlas to full-atlas and brain tiers."""
    atlas = np.ones((2, 2, 2), dtype=np.uint8)
    full_atlas = np.ones((2, 2, 2), dtype=np.uint8)

    attempts = AIFDetection._build_cascade_attempts(
        "strict", atlas, True, full_atlas, False,
    )

    assert _names(attempts) == [
        "atlas_strict",
        "atlas_relaxed",
        "atlas_permissive",
        "full_atlas_permissive",
        "brain_relaxed",
        "brain_permissive",
    ]
    assert attempts[0][1] is atlas
    assert attempts[3][1] is full_atlas
    assert attempts[0][3] is True


def test_initial_criteria_removes_stricter_tiers() -> None:
    """Starting at relaxed omits strict tiers while retaining broader searches."""
    atlas = np.ones((2, 2, 2), dtype=np.uint8)

    attempts = AIFDetection._build_cascade_attempts(
        "relaxed", atlas, False, None, False,
    )

    assert _names(attempts) == [
        "atlas_relaxed",
        "atlas_permissive",
        "brain_relaxed",
        "brain_permissive",
    ]


def test_strict_roi_constraint_excludes_brain_search() -> None:
    """A strict supplied ROI prevents fallback to unconstrained brain tiers."""
    atlas = np.ones((2, 2, 2), dtype=np.uint8)
    full_atlas = np.ones((2, 2, 2), dtype=np.uint8)

    attempts = AIFDetection._build_cascade_attempts(
        "strict", atlas, False, full_atlas, True,
    )

    assert _names(attempts) == [
        "atlas_strict",
        "atlas_relaxed",
        "atlas_permissive",
        "full_atlas_permissive",
    ]


def test_strict_roi_without_roi_retains_brain_search() -> None:
    """The strict-ROI flag alone does not suppress brain-wide detection."""
    attempts = AIFDetection._build_cascade_attempts(
        "strict", None, False, None, True,
    )

    assert _names(attempts) == ["brain_relaxed", "brain_permissive"]
