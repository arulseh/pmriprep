"""Characterization tests for DSC AIF cascade tier planning."""

import numpy as np
import pytest

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


@pytest.mark.parametrize("filled", [False, True])
def test_supplied_mask_bounds_every_tier(filled: bool) -> None:
    """Even an empty supplied ROI is the boundary for every search tier."""
    atlas = np.full((2, 2, 2), filled, dtype=np.uint8)
    full_atlas = np.ones((2, 2, 2), dtype=np.uint8)

    attempts = AIFDetection._build_cascade_attempts(
        "strict", atlas, False, full_atlas, True,
    )

    assert _names(attempts) == [
        "user_mask_strict",
        "user_mask_relaxed",
        "user_mask_permissive",
    ]
    assert all(roi is atlas and constrained for _, roi, _, constrained in attempts)


def test_no_roi_uses_brain_search() -> None:
    """Without a supplied mask or atlas, detection uses whole-brain tiers."""
    attempts = AIFDetection._build_cascade_attempts(
        "strict", None, False, None, False,
    )

    assert _names(attempts) == ["brain_relaxed", "brain_permissive"]
