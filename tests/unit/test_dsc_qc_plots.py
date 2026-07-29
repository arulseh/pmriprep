"""Tests for ``pmriprep.utils.dsc_qc_plots`` selectors and helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pmriprep.utils.dsc_qc_plots import (
    _select_dsc_qc_voxel,
    generate_dsc_preprocessing_qc_plot,
)


@pytest.fixture
def cbf_volume() -> np.ndarray:
    """Synthetic 3-D CBF map with a clear median in the brain interior.

    Z-axis is deliberately ≥7 slices so 3-iteration erosion (used by
    the selector) leaves a non-empty interior in z.
    """
    cbf = np.zeros((20, 20, 11), dtype=np.float64)
    # Brain interior fills most of the volume with a graded CBF
    cbf[2:18, 2:18, 1:10] = 1.0
    cbf[5:15, 5:15, 3:8] = 2.0
    cbf[8:12, 8:12, 4:7] = 3.0
    return cbf


@pytest.fixture
def brain_mask(cbf_volume: np.ndarray) -> np.ndarray:
    return cbf_volume > 0


@pytest.fixture
def enhancing_mask(cbf_volume: np.ndarray) -> np.ndarray:
    """Small enhancing region with cbf>0."""
    enh = np.zeros_like(cbf_volume)
    enh[9:11, 9:11, 5:6] = 1.0
    return enh


def _binary_erosion(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    from scipy.ndimage import binary_erosion as _erosion
    return _erosion(mask, iterations=iterations)


class TestSelectDSCQCVoxelTiers:
    """Tier ordering: enhancing → eroded brain → centre."""

    def test_returns_coord_and_tier_tuple(
        self,
        cbf_volume: np.ndarray,
        brain_mask: np.ndarray,
    ) -> None:
        result = _select_dsc_qc_voxel(cbf_volume, brain_mask, _binary_erosion)
        assert isinstance(result, tuple) and len(result) == 2
        coord, tier = result
        assert isinstance(coord, tuple) and len(coord) == 3
        assert tier in {"enhancing", "eroded_brain", "center"}

    def test_enhancing_tier_wins_when_available(
        self,
        cbf_volume: np.ndarray,
        brain_mask: np.ndarray,
        enhancing_mask: np.ndarray,
    ) -> None:
        """When the enhancing mask has cbf>0 voxels, tier 1 must fire and
        return ``"enhancing"`` — even though tier 2 (eroded brain) also
        has many candidates.
        """
        coord, tier = _select_dsc_qc_voxel(
            cbf_volume, brain_mask, _binary_erosion,
            enhancing_mask=enhancing_mask,
        )
        assert tier == "enhancing"
        assert enhancing_mask[coord] > 0.5

    def test_eroded_brain_tier_when_no_enhancing_mask(
        self,
        cbf_volume: np.ndarray,
        brain_mask: np.ndarray,
    ) -> None:
        coord, tier = _select_dsc_qc_voxel(
            cbf_volume, brain_mask, _binary_erosion,
        )
        assert tier == "eroded_brain"
        assert brain_mask[coord]

    def test_eroded_brain_tier_when_enhancing_mask_empty(
        self,
        cbf_volume: np.ndarray,
        brain_mask: np.ndarray,
    ) -> None:
        empty_enh = np.zeros_like(cbf_volume)
        coord, tier = _select_dsc_qc_voxel(
            cbf_volume, brain_mask, _binary_erosion,
            enhancing_mask=empty_enh,
        )
        assert tier == "eroded_brain"

    def test_center_fallback_when_cbf_all_zero(
        self,
        brain_mask: np.ndarray,
    ) -> None:
        """If no voxel has cbf>0 (degenerate), the centre fallback fires."""
        cbf_zero = np.zeros_like(brain_mask, dtype=np.float64)
        coord, tier = _select_dsc_qc_voxel(
            cbf_zero, brain_mask, _binary_erosion,
        )
        assert tier == "center"
        assert brain_mask[coord]


def test_preprocessing_qc_uses_baseline_tsnr_and_signal_drop_cnr(tmp_path) -> None:
    """Displayed acquisition metrics match the stored-map signal conventions."""
    curve = np.array([100.0, 102.0, 98.0, 75.0, 85.0])
    signal = np.broadcast_to(curve, (3, 3, 3, curve.size)).copy()
    mask = np.ones((3, 3, 3), dtype=bool)

    _, stats = generate_dsc_preprocessing_qc_plot(
        signal,
        mask,
        baseline_frames=3,
        volume_tr_s=1.5,
        output_path=str(tmp_path / "dsc_qc.png"),
    )

    assert stats["mean_tsnr"] == pytest.approx(50.0)
    assert stats["mean_cnr"] == pytest.approx(12.5)
    assert stats["fraction_cnr_gt_4"] == 1.0
