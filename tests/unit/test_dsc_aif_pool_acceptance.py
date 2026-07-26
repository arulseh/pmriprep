"""Characterization tests for progressive DSC AIF pool acceptance."""

import logging
from typing import Any

import numpy as np
import pytest

from pmriprep.interfaces.dsc_aif import AIFDetection, AIFPoolAcceptanceResult


def _accepted_result() -> AIFPoolAcceptanceResult:
    """Return a minimal accepted pool result for orchestration tests."""
    return AIFPoolAcceptanceResult(
        curve=np.zeros(8),
        mask=np.ones((2, 2, 2), dtype=bool),
        roi_size=1,
        spr=2.0,
        peak_ratio=None,
        brr=5.0,
    )


def test_progressive_acceptance_removes_worst_brr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected pool is retried after removing its most negative BRR."""
    detector = AIFDetection()
    rejected_coord = (0, 0, 0)
    candidates = [
        (0.1, rejected_coord, {"baseline_return_ratio": -8.0}),
        (1.0, (0, 0, 1), {"baseline_return_ratio": 4.0}),
        (2.0, (0, 1, 0), {"baseline_return_ratio": 6.0}),
        (3.0, (1, 0, 0), {"baseline_return_ratio": 8.0}),
    ]
    attempted_coords: list[list[tuple[int, int, int]]] = []

    def try_pool(pool: list[tuple], *args: Any) -> AIFPoolAcceptanceResult | None:
        attempted_coords.append([candidate[1] for candidate in pool])
        return _accepted_result() if len(pool) == 3 else None

    monkeypatch.setattr(detector, "_try_aif_pool_acceptance", try_pool)
    result = detector._try_trimmed_pool_acceptance(
        candidates, np.zeros((2, 2, 2, 8)), np.ones((2, 2, 2)), 1.5, 3,
    )

    assert result is not None
    assert [len(pool) for pool in attempted_coords] == [4, 3]
    assert rejected_coord not in [candidate[1] for candidate in result.candidates]


def test_missing_brr_removes_highest_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When BRR is unavailable, a rejected pool removes its highest cost."""
    detector = AIFDetection()
    rejected_coord = (1, 0, 0)
    candidates = [
        (0.1, (0, 0, 0), {}),
        (1.0, (0, 0, 1), {}),
        (2.0, (0, 1, 0), {}),
        (3.0, rejected_coord, {}),
    ]

    def try_pool(pool: list[tuple], *args: Any) -> AIFPoolAcceptanceResult | None:
        return _accepted_result() if len(pool) == 3 else None

    monkeypatch.setattr(detector, "_try_aif_pool_acceptance", try_pool)
    result = detector._try_trimmed_pool_acceptance(
        candidates, np.zeros((2, 2, 2, 8)), np.ones((2, 2, 2)), 1.5, 3,
    )

    assert result is not None
    assert rejected_coord not in [candidate[1] for candidate in result.candidates]


def test_pool_below_minimum_is_not_evaluated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pool below the acceptance floor cannot invoke pool evaluation."""
    detector = AIFDetection()

    def fail_if_called(*args: Any) -> None:
        pytest.fail("pool evaluation should not run below the pool minimum")

    monkeypatch.setattr(detector, "_try_aif_pool_acceptance", fail_if_called)
    result = detector._try_trimmed_pool_acceptance(
        [(0.1, (0, 0, 0), {}), (0.2, (0, 0, 1), {})],
        np.zeros((2, 2, 2, 8)),
        np.ones((2, 2, 2)),
        1.5,
        3,
    )

    assert result is None


def test_single_best_uses_lowest_cost_and_retries_larger_roi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback selects by cost and tries the 3-voxel ROI after ROI 1."""
    detector = AIFDetection()
    best_coord = (0, 0, 1)
    candidates = [
        (2.0, (0, 0, 0), {}),
        (0.5, best_coord, {}),
        (1.0, (0, 1, 0), {}),
    ]
    attempted_sizes: list[int] = []
    attempted_coords: list[tuple[int, int, int]] = []

    def compute(
        pool: list[tuple],
        concentration: np.ndarray,
        brain_mask: np.ndarray,
        tr: float,
        *,
        roi_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        attempted_sizes.append(roi_size)
        attempted_coords.append(pool[0][1])
        return np.full(8, roi_size, dtype=float), np.ones((2, 2, 2), dtype=bool)

    monkeypatch.setattr(detector, "_compute_roi_averaged_aif", compute)
    monkeypatch.setattr(
        detector,
        "_aif_global_max_in_fp",
        lambda curve, baseline, tr: bool(curve[0] == 3),
    )
    monkeypatch.setattr(detector, "_compute_aif_brr", lambda *args: 5.0)

    result = detector._try_single_best_fallback(
        candidates,
        np.zeros((2, 2, 2, 8)),
        np.ones((2, 2, 2)),
        1.5,
        3,
        logging.getLogger(__name__),
    )

    assert result is not None
    assert result.candidate[1] == best_coord
    assert result.roi_size == 3
    assert attempted_sizes == [1, 3]
    assert attempted_coords == [best_coord, best_coord]


def test_single_best_rejects_all_rois_below_brr_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No single-candidate fallback is accepted below the BRR floor."""
    detector = AIFDetection()
    attempted_sizes: list[int] = []

    def compute(*args: Any, roi_size: int) -> tuple[np.ndarray, np.ndarray]:
        attempted_sizes.append(roi_size)
        return np.ones(8), np.ones((2, 2, 2), dtype=bool)

    monkeypatch.setattr(detector, "_compute_roi_averaged_aif", compute)
    monkeypatch.setattr(detector, "_aif_global_max_in_fp", lambda *args: True)
    monkeypatch.setattr(detector, "_compute_aif_brr", lambda *args: -4.0)

    result = detector._try_single_best_fallback(
        [(0.5, (0, 0, 0), {})],
        np.zeros((2, 2, 2, 8)),
        np.ones((2, 2, 2)),
        1.5,
        3,
        logging.getLogger(__name__),
    )

    assert result is None
    assert attempted_sizes == [1, 3]
