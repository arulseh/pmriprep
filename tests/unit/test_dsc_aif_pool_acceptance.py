"""Characterization tests for progressive DSC AIF pool acceptance."""

from typing import Any

import numpy as np
import pytest

from pmriprep.interfaces.dsc_aif import AIFDetection, DualRoiResult


def _accepted_result() -> DualRoiResult:
    """Return a minimal accepted dual-ROI result for orchestration tests."""
    return DualRoiResult(
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

    def try_pool(pool: list[tuple], *args: Any) -> DualRoiResult | None:
        attempted_coords.append([candidate[1] for candidate in pool])
        return _accepted_result() if len(pool) == 3 else None

    monkeypatch.setattr(detector, "_try_aif_dual_roi", try_pool)
    result = detector._try_trimmed_dual_roi(
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

    def try_pool(pool: list[tuple], *args: Any) -> DualRoiResult | None:
        return _accepted_result() if len(pool) == 3 else None

    monkeypatch.setattr(detector, "_try_aif_dual_roi", try_pool)
    result = detector._try_trimmed_dual_roi(
        candidates, np.zeros((2, 2, 2, 8)), np.ones((2, 2, 2)), 1.5, 3,
    )

    assert result is not None
    assert rejected_coord not in [candidate[1] for candidate in result.candidates]


def test_pool_below_minimum_is_not_evaluated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pool below the acceptance floor cannot invoke dual-ROI evaluation."""
    detector = AIFDetection()

    def fail_if_called(*args: Any) -> None:
        pytest.fail("dual-ROI evaluation should not run below the pool minimum")

    monkeypatch.setattr(detector, "_try_aif_dual_roi", fail_if_called)
    result = detector._try_trimmed_dual_roi(
        [(0.1, (0, 0, 0), {}), (0.2, (0, 0, 1), {})],
        np.zeros((2, 2, 2, 8)),
        np.ones((2, 2, 2)),
        1.5,
        3,
    )

    assert result is None
