"""Characterization tests for DSC AIF per-tier candidate preparation."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pmriprep.interfaces.dsc_aif import AIFDetection


def _context() -> SimpleNamespace:
    """Return the subset of detection context consumed by tier preparation."""
    atlas = np.ones((2, 2, 2), dtype=np.uint8)
    return SimpleNamespace(
        concentration=np.zeros((2, 2, 2, 8)),
        brain_mask=np.ones((2, 2, 2), dtype=np.uint8),
        lesion_mask=None,
        signal=None,
        atlas_mask=atlas,
        affine=np.eye(4),
        voxel_sizes=(1.0, 1.0, 1.0),
        lesion_distance=15.0,
        enhancing_mask=None,
        tr=1.5,
    )


def test_tier_candidate_stages_run_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A surviving tier passes through every preparation stage in order."""
    detector = AIFDetection()
    context = _context()
    calls: list[str] = []
    candidates = [
        (0.6, (0, 0, 1), {"baseline_return_ratio": 5.0}),
        (0.5, (0, 0, 0), {"baseline_return_ratio": 5.0}),
    ]

    def prepare(*args: Any) -> np.ndarray:
        calls.append("search")
        return np.ones((2, 2, 2), dtype=bool)

    def find(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        calls.append("find")
        return np.array([[0, 0, 0]]), np.ones((2, 2, 2))

    def score(*args: Any, **kwargs: Any) -> tuple[list[tuple], int]:
        calls.append("score")
        assert kwargs["atlas_mask"] is context.atlas_mask
        return list(candidates), 4

    def spatial(*args: Any, **kwargs: Any) -> tuple[list[tuple], dict]:
        calls.append("spatial")
        return list(candidates), {"distance_filter": "applied"}

    def kinetic(*args: Any, **kwargs: Any) -> tuple[list[tuple], dict]:
        calls.append("kinetic")
        return list(candidates), {"aif_n_after_filters": 1}

    monkeypatch.setattr(detector, "_prepare_search_mask", prepare)
    monkeypatch.setattr(detector, "_find_aif_candidate_voxels", find)
    monkeypatch.setattr(detector, "_score_and_select_candidates", score)
    monkeypatch.setattr(detector, "_apply_post_scoring_filters", spatial)
    monkeypatch.setattr(detector, "_apply_postscore_aif_filters", kinetic)

    result = detector._prepare_tier_candidates(
        context, "relaxed", None, False, 1, 5, 8.0,
    )

    assert calls == ["search", "find", "score", "spatial", "kinetic"]
    assert result.candidates == [candidates[1]]
    assert result.n_available == 4
    assert result.filter_metadata == {"distance_filter": "applied"}
    assert result.postfilter_diagnostics == {"aif_n_after_filters": 1}


def test_empty_search_stops_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tier with no candidate voxels does not invoke downstream stages."""
    detector = AIFDetection()
    context = _context()
    monkeypatch.setattr(
        detector,
        "_prepare_search_mask",
        lambda *args: np.ones((2, 2, 2), dtype=bool),
    )
    monkeypatch.setattr(
        detector,
        "_find_aif_candidate_voxels",
        lambda *args, **kwargs: (np.empty((0, 3), dtype=int), np.zeros((2, 2, 2))),
    )

    def fail_if_called(*args: Any, **kwargs: Any) -> None:
        pytest.fail("candidate scoring should not run for an empty search")

    monkeypatch.setattr(detector, "_score_and_select_candidates", fail_if_called)

    result = detector._prepare_tier_candidates(
        context, "strict", context.atlas_mask, True, 3, 5, 8.0,
    )

    assert result.candidates == []
    assert result.postfilter_diagnostics is None


def test_all_candidates_inside_lesion_margin_exhaust_tier() -> None:
    """A fully excluded spatial pool proceeds to the next cascade tier."""
    detector = AIFDetection()
    lesion_mask = np.zeros((20, 20, 20), dtype=np.uint8)
    lesion_mask[10, 10, 10] = 1
    candidates = [
        (0.1, (10, 10, 11), {}),
        (0.2, (10, 11, 10), {}),
        (0.3, (11, 10, 10), {}),
    ]

    filtered, metadata = detector._apply_post_scoring_filters(
        candidates,
        lesion_mask,
        np.eye(4),
        (1.0, 1.0, 1.0),
        lesion_min_distance_mm=15.0,
    )

    assert filtered == []
    assert metadata["n_distance_excluded"] == 3
    assert metadata["all_candidates_excluded_by_distance"] is True


def test_laterality_accounting_does_not_change_candidate_ranking() -> None:
    """Hemisphere labels are descriptive and cannot alter AIF scores."""
    candidates = [
        (1.0, (8, 0, 0), {}),   # ipsilateral and best-ranked
        (1.5, (-8, 0, 0), {}),  # contralateral
        (2.0, (9, 0, 0), {}),   # ipsilateral
    ]

    original_costs = [candidate[0] for candidate in candidates]
    n_contra, n_ipsi = AIFDetection._count_candidates_by_hemisphere(
        candidates,
        np.eye(4),
        tumor_laterality="right",
    )

    assert [candidate[0] for candidate in candidates] == original_costs
    assert n_contra == 1
    assert n_ipsi == 2
