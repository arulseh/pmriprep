"""Characterization tests for DSC AIF per-tier candidate preparation."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pmriprep.interfaces.dsc_aif import AIFDetection
from pmriprep.utils.aif_laterality import describe_tumor_relative_aif_candidates


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

    def prepare(*args: Any, **kwargs: Any) -> np.ndarray:
        calls.append("search")
        assert kwargs["voxel_sizes"] == context.voxel_sizes
        assert kwargs["apply_posterior_midline_exclusion"] is True
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
        context,
        "relaxed",
        None,
        False,
        None,
        float("nan"),
        float("nan"),
        1,
        5,
        8.0,
    )

    assert calls == ["search", "find", "score", "spatial", "kinetic"]
    assert result.candidates == [candidates[1]]
    assert result.n_available == 4
    assert result.filter_metadata == {
        "distance_filter": "applied",
        "whole_brain_symmetry_method": "scanner_axis_fallback",
    }
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
        lambda *args, **kwargs: np.ones((2, 2, 2), dtype=bool),
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
        context,
        "strict",
        context.atlas_mask,
        True,
        None,
        float("nan"),
        float("nan"),
        3,
        5,
        8.0,
    )

    assert result.candidates == []
    assert result.postfilter_diagnostics is None


@pytest.mark.parametrize("user_mask_supplied", [True, False])
def test_roi_tier_passes_physical_voxel_sizes(
    monkeypatch: pytest.MonkeyPatch,
    user_mask_supplied: bool,
) -> None:
    """Bundled and user-supplied ROIs receive physical voxel sampling."""
    detector = AIFDetection()
    context = _context()
    observed: dict[str, Any] = {}

    def prepare(*args: Any, **kwargs: Any) -> np.ndarray:
        observed.update(kwargs)
        return np.zeros((2, 2, 2), dtype=bool)

    monkeypatch.setattr(detector, "_prepare_search_mask", prepare)
    monkeypatch.setattr(
        detector,
        "_find_aif_candidate_voxels",
        lambda *args, **kwargs: (np.empty((0, 3), dtype=int), np.zeros((2, 2, 2))),
    )

    attempt = detector._build_cascade_attempts(
        "strict", context.atlas_mask, not user_mask_supplied, None, user_mask_supplied,
    )[0]
    detector._prepare_tier_candidates(
        context,
        "strict",
        context.atlas_mask,
        attempt[3],
        None,
        float("nan"),
        float("nan"),
        3,
        5,
        8.0,
    )

    assert observed["voxel_sizes"] == context.voxel_sizes
    assert observed["apply_posterior_midline_exclusion"] is False


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
    lesion_mask = np.zeros((21, 2, 2), dtype=np.uint8)
    lesion_mask[18, 0, 0] = 1
    metadata = describe_tumor_relative_aif_candidates(
        [candidate[1] for candidate in candidates],
        np.eye(4),
        lesion_mask,
    )

    assert [candidate[0] for candidate in candidates] == original_costs
    assert metadata["tumor_laterality"] == "right"
    assert metadata["n_contralateral"] == 1
    assert metadata["n_ipsilateral"] == 2
