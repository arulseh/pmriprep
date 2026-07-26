"""Characterization tests for DSC AIF cascade union state and provenance."""

from types import SimpleNamespace

import numpy as np
import pytest

from pmriprep.interfaces import dsc_aif


def _candidate(
    cost: float,
    coordinate: tuple[int, int, int],
) -> tuple[float, tuple[int, int, int], dict[str, float]]:
    """Build a minimal AIF candidate."""
    return cost, coordinate, {"baseline_return_ratio": 5.0}


def _empty_union() -> dsc_aif._CascadeUnionState:
    """Build an empty mutable cascade union state."""
    return dsc_aif._CascadeUnionState([], set(), {}, {}, {})


def test_union_deduplicates_coordinates_and_preserves_first_metadata() -> None:
    """A broader tier cannot replace a coordinate or metadata already accepted."""
    state = _empty_union()
    shared = _candidate(0.5, (0, 0, 0))
    dsc_aif.AIFDetection._add_candidates_to_union(
        state,
        [shared, _candidate(1.0, (0, 0, 1))],
        "atlas_relaxed",
        3,
        {"search_source": "atlas"},
    )
    dsc_aif.AIFDetection._add_candidates_to_union(
        state,
        [shared, _candidate(1.5, (0, 1, 0)), _candidate(2.0, (1, 0, 0))],
        "brain_relaxed",
        3,
        {"search_source": "brain"},
    )

    assert len(state.candidates) == 3
    assert state.tier_provenance == {"atlas_relaxed": 2, "brain_relaxed": 1}
    assert state.coordinate_origins[(0, 0, 0)] == "atlas_relaxed"
    assert state.filter_metadata == {"search_source": "atlas"}


def test_union_acceptance_rebuilds_trimmed_provenance_in_cascade_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only retained tiers appear, ordered by the configured cascade plan."""
    detector = dsc_aif.AIFDetection()
    atlas_one = _candidate(1.0, (0, 0, 0))
    atlas_two = _candidate(1.1, (0, 0, 1))
    brain = _candidate(1.2, (0, 1, 0))
    discarded = _candidate(1.3, (1, 0, 0))
    state = dsc_aif._CascadeUnionState(
        [discarded, brain, atlas_one, atlas_two],
        {candidate[1] for candidate in (discarded, brain, atlas_one, atlas_two)},
        {"atlas_relaxed": 2, "brain_relaxed": 2},
        {
            discarded[1]: "brain_permissive",
            brain[1]: "brain_relaxed",
            atlas_one[1]: "atlas_relaxed",
            atlas_two[1]: "atlas_relaxed",
        },
        {"search_source": "atlas"},
    )
    accepted_candidates = [brain, atlas_one, atlas_two]
    pool_result = dsc_aif.AIFPoolAcceptanceResult(
        np.zeros(8), np.ones((2, 2, 2), dtype=bool), 1, 2.0, None, 5.0,
    )
    monkeypatch.setattr(
        detector,
        "_try_trimmed_pool_acceptance",
        lambda *args: dsc_aif._TrimmedPoolAcceptanceResult(
            pool_result,
            accepted_candidates,
        ),
    )
    attempts = [
        ("relaxed", None, "atlas_relaxed", True),
        ("relaxed", None, "brain_relaxed", False),
        ("permissive", None, "brain_permissive", False),
    ]
    context = SimpleNamespace(
        concentration=np.zeros((2, 2, 2, 8)),
        brain_mask=np.ones((2, 2, 2)),
        tr=1.5,
    )

    result = detector._try_union_fallback(state, attempts, context, 3)

    assert result is not None
    assert result.detection_level == "atlas_relaxed+brain_relaxed"
    assert result.tier_provenance == {"brain_relaxed": 1, "atlas_relaxed": 2}
    assert "brain_permissive" not in result.tier_provenance
