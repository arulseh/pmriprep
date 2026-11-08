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


@pytest.mark.parametrize("spr, peak_ratio, brr, reason", [
    (0.0, 1.0, 5.0, None),
    (0.0, 1.0, float("nan"), None),
    (101.0, 1.0, 5.0, "secondary_peak"),
    (0.0, None, 5.0, "peak_ratio"),
])
def test_single_best_uses_lowest_cost_and_shared_curve_gate(
    monkeypatch: pytest.MonkeyPatch, spr: float, peak_ratio: float | None,
    brr: float, reason: str | None,
) -> None:
    """The best voxel must pass the pool curve gate, including its NaN-BRR policy."""
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
        AIFDetection,
        "_aif_global_max_in_fp",
        staticmethod(lambda curve, baseline, tr: True),
    )
    monkeypatch.setattr(AIFDetection, "_compute_aif_brr", lambda *args: brr)
    monkeypatch.setattr(AIFDetection, "_compute_aif_spr", lambda *args: (spr, 4, 1.0))
    monkeypatch.setattr("pmriprep.interfaces.dsc_aif._compute_aif_peak_ratio", lambda *args, **kw: peak_ratio)

    result, rejected = detector._try_single_best_fallback(
        candidates,
        np.zeros((2, 2, 2, 8)),
        np.ones((2, 2, 2)),
        1.5,
        3,
        logging.getLogger(__name__),
    )

    if reason is None:
        assert result is not None and rejected is None
        assert result.candidate[1] == best_coord
        assert result.roi_size == 1
    else:
        assert result is None and rejected is not None
        assert rejected.qc["reason"] == reason
        assert rejected.qc["candidate_coords"] == [list(best_coord)]
        assert rejected.qc["roi_size"] == 1
    assert attempted_sizes == [1]
    assert attempted_coords == [best_coord]


def test_single_best_rejects_below_brr_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No single-candidate fallback is accepted below the BRR floor."""
    detector = AIFDetection()
    attempted_sizes: list[int] = []

    def compute(*args: Any, roi_size: int) -> tuple[np.ndarray, np.ndarray]:
        attempted_sizes.append(roi_size)
        return np.ones(8), np.ones((2, 2, 2), dtype=bool)

    monkeypatch.setattr(detector, "_compute_roi_averaged_aif", compute)
    monkeypatch.setattr(AIFDetection, "_aif_global_max_in_fp", staticmethod(lambda *args: True))
    monkeypatch.setattr(AIFDetection, "_compute_aif_brr", lambda *args: -4.0)
    monkeypatch.setattr(AIFDetection, "_compute_aif_spr", lambda *args: (0.0, 4, 1.0))
    monkeypatch.setattr("pmriprep.interfaces.dsc_aif._compute_aif_peak_ratio", lambda *args, **kw: 1.0)

    result, rejected = detector._try_single_best_fallback(
        [(0.5, (0, 0, 0), {})],
        np.zeros((2, 2, 2, 8)),
        np.ones((2, 2, 2)),
        1.5,
        3,
        logging.getLogger(__name__),
    )

    assert result is None
    assert attempted_sizes == [1]

    assert rejected is not None and rejected.qc["reason"] == "negative_tail"
