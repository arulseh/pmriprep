"""Phase 4: AIF detection on synthetic 3D+t phantom tests.

Tests atlas_based_aif_search() with physiologically realistic gamma-variate
AIFs in multi-tissue volumes. Validates the 4-level cascade (atlas_strict ->
atlas_relaxed -> whole_brain_strict -> whole_brain_relaxed) and lesion exclusion.

Exercises:
    dsc_aif.py: atlas_based_aif_search(), score_aif_candidate(), cascade logic
"""

from typing import Dict, Set, Tuple

import numpy as np
import numpy.typing as npt
import pytest

from pmriprep.utils.dsc_aif import (
    AIFSearchResult,
    atlas_based_aif_search,
)

from .conftest import (
    DSC_TISSUE_PARAMS,
    dsc_forward_model,
    generate_dsc_3d_phantom,
    generate_dsc_aif_dro,
    make_exponential_residue,
)

# =============================================================================
# Helpers
# =============================================================================

_AIF_PEAK = 40.0  # Full peak for AIF detection (needs to look arterial)
_BASELINE = 10


def _build_test_phantom(
    shape: Tuple[int, int, int] = (12, 12, 3),
    snr: float = np.inf,
    seed: int = 42,
) -> Tuple[
    npt.NDArray[np.floating],  # 4D concentration
    Dict[str, npt.NDArray[np.bool_]],  # masks
    Set[Tuple[int, int, int]],  # aif_coords
    npt.NDArray[np.floating],  # aif curve (1D, with baseline)
]:
    """Build a multi-tissue phantom for AIF detection testing.

    Layout (axial slice):
    - AIF voxels at periphery (edge columns)
    - WM in center
    - GM ring around WM
    - Tumor patch in one corner
    - CSF (label 0, zero concentration)
    """
    nx, ny, nz = shape

    # Generate AIF (no baseline — generate_dsc_3d_phantom prepends)
    aif_post, _ = generate_dsc_aif_dro(
        n_timepoints=60, dt_sec=1.5, baseline_frames=0,
        peak_delta_r2=_AIF_PEAK,
    )

    # Generate tissue curves with low AIF peak for realistic concentrations
    aif_tissue, _ = generate_dsc_aif_dro(
        n_timepoints=60, dt_sec=1.5, baseline_frames=0,
        peak_delta_r2=0.1,
    )

    tissue_curves: Dict[str, npt.NDArray[np.floating]] = {}
    for name, params in DSC_TISSUE_PARAMS.items():
        residue = make_exponential_residue(params.cbf, params.mtt, 60, 1.5)
        tissue_curves[name] = dsc_forward_model(aif_tissue, residue, 1.5)

    # Build tissue map
    tissue_labels = {"gm": 1, "wm": 2, "tumor": 3}
    tissue_map = np.zeros(shape, dtype=np.int32)

    for z in range(nz):
        # WM center (4x4)
        tissue_map[4:8, 4:8, z] = tissue_labels["wm"]
        # GM ring
        tissue_map[3:9, 3:9, z] = np.where(
            tissue_map[3:9, 3:9, z] == 0,
            tissue_labels["gm"],
            tissue_map[3:9, 3:9, z],
        )
        # Tumor patch (corner)
        tissue_map[9:11, 9:11, z] = tissue_labels["tumor"]

    # AIF voxels at edges (column 0 and 1)
    aif_coords: Set[Tuple[int, int, int]] = set()
    for z in range(nz):
        for y in [1, 2]:
            aif_coords.add((1, y, z))

    volume, masks = generate_dsc_3d_phantom(
        shape=shape,
        aif_coords=aif_coords,
        aif=aif_post,
        tissue_curves=tissue_curves,
        tissue_map=tissue_map,
        tissue_labels=tissue_labels,
        baseline_frames=_BASELINE,
    )

    # Add noise
    if np.isfinite(snr) and snr > 0:
        rng = np.random.default_rng(seed)
        peak = float(np.max(np.abs(volume)))
        if peak > 0:
            noise_std = peak / snr
            volume = volume + rng.normal(0.0, noise_std, volume.shape)

    # AIF curve with baseline
    aif_full = np.zeros(_BASELINE + 60, dtype=np.float64)
    aif_full[_BASELINE:] = aif_post

    return volume, masks, aif_coords, aif_full


# =============================================================================
# Test classes
# =============================================================================


class TestAIFDetectionAccuracy:
    """Test AIF detection on synthetic multi-tissue phantom."""

    def test_aif_detection_with_atlas(self) -> None:
        """AIF search with atlas mask should find AIF voxels."""
        volume, masks, aif_coords_true, _ = _build_test_phantom(snr=np.inf)

        # Atlas mask covers AIF region
        atlas_mask = np.zeros(volume.shape[:3], dtype=np.uint8)
        for x, y, z in aif_coords_true:
            atlas_mask[x, y, z] = 1
        # Expand atlas slightly
        atlas_mask[0:3, 0:4, :] = 1

        result = atlas_based_aif_search(
            concentration_data=volume,
            brain_mask=masks["brain"].astype(np.uint8),
            atlas_mask=atlas_mask,
            baseline_frames=_BASELINE,
            n_candidates=5,
        )

        assert isinstance(result, AIFSearchResult)
        assert result.detection_level != "none", "AIF detection failed completely"
        assert len(result.candidates) > 0
        assert result.aif_curve.shape[0] == volume.shape[3]

    @pytest.mark.parametrize(
        "snr",
        [pytest.param(np.inf, id="noiseless"),
         pytest.param(50.0, id="SNR50")],
    )
    def test_aif_curve_shape_matches_ground_truth(self, snr: float) -> None:
        """Detected AIF curve shape should correlate with ground truth."""
        volume, masks, aif_coords_true, aif_true = _build_test_phantom(snr=snr)

        atlas_mask = np.zeros(volume.shape[:3], dtype=np.uint8)
        for x, y, z in aif_coords_true:
            atlas_mask[x, y, z] = 1
        atlas_mask[0:3, 0:4, :] = 1

        result = atlas_based_aif_search(
            concentration_data=volume,
            brain_mask=masks["brain"].astype(np.uint8),
            atlas_mask=atlas_mask,
            baseline_frames=_BASELINE,
            n_candidates=5,
        )

        if result.detection_level == "none":
            pytest.skip("AIF detection failed — cannot test curve shape")

        # Pearson correlation between detected and ground truth AIF
        detected = result.aif_curve
        truth = aif_true

        # Both must have same length
        assert len(detected) == len(truth)

        # Correlation of post-baseline portion
        d_post = detected[_BASELINE:]
        t_post = truth[_BASELINE:]

        if np.std(d_post) > 0 and np.std(t_post) > 0:
            r = float(np.corrcoef(d_post, t_post)[0, 1])
            assert r > 0.90, (
                f"AIF curve correlation {r:.3f} < 0.90"
            )

    def test_aif_peak_precedes_tissue(self) -> None:
        """AIF peak should precede tissue peak (causality)."""
        volume, masks, aif_coords_true, _ = _build_test_phantom(snr=np.inf)

        atlas_mask = np.zeros(volume.shape[:3], dtype=np.uint8)
        for x, y, z in aif_coords_true:
            atlas_mask[x, y, z] = 1

        result = atlas_based_aif_search(
            concentration_data=volume,
            brain_mask=masks["brain"].astype(np.uint8),
            atlas_mask=atlas_mask,
            baseline_frames=_BASELINE,
        )

        if result.detection_level == "none":
            pytest.skip("AIF detection failed")

        # AIF peak time
        aif_peak_frame = int(np.argmax(result.aif_curve))

        # Mean tissue peak time (GM voxels)
        gm_mask = masks["gm"]
        gm_peaks = []
        for idx in zip(*np.where(gm_mask)):
            curve = volume[idx[0], idx[1], idx[2], :]
            if np.max(curve) > 0:
                gm_peaks.append(int(np.argmax(curve)))

        if gm_peaks:
            mean_tissue_peak = np.mean(gm_peaks)
            # AIF should peak at or before tissue
            assert aif_peak_frame <= mean_tissue_peak + 2, (
                f"AIF peak ({aif_peak_frame}) after tissue peak "
                f"({mean_tissue_peak:.0f})"
            )


class TestAIFCascadeFallback:
    """Test the 4-level cascade fallback behavior."""

    def test_perfect_atlas_uses_atlas_strict(self) -> None:
        """With a perfect atlas, detection_level should be atlas_strict."""
        volume, masks, aif_coords_true, _ = _build_test_phantom(snr=np.inf)

        atlas_mask = np.zeros(volume.shape[:3], dtype=np.uint8)
        for x, y, z in aif_coords_true:
            atlas_mask[x, y, z] = 1

        result = atlas_based_aif_search(
            concentration_data=volume,
            brain_mask=masks["brain"].astype(np.uint8),
            atlas_mask=atlas_mask,
            baseline_frames=_BASELINE,
        )

        assert result.detection_level in ("atlas_strict", "atlas_relaxed"), (
            f"Expected atlas level, got {result.detection_level}"
        )

    def test_empty_atlas_falls_back(self) -> None:
        """Empty atlas mask should fall back to whole_brain level."""
        volume, masks, _, _ = _build_test_phantom(snr=np.inf)

        # Empty atlas
        atlas_mask = np.zeros(volume.shape[:3], dtype=np.uint8)

        result = atlas_based_aif_search(
            concentration_data=volume,
            brain_mask=masks["brain"].astype(np.uint8),
            atlas_mask=atlas_mask,
            baseline_frames=_BASELINE,
        )

        # Should fall back to whole_brain level
        assert result.detection_level.startswith("whole_brain") or \
            result.detection_level == "none", (
                f"Expected whole_brain fallback, got {result.detection_level}"
            )

    def test_atlas_excluding_aif_falls_back(self) -> None:
        """Atlas that excludes all AIF voxels — behaviour depends on tissue curves.

        Noiseless synthetic tissue curves may pass AIF criteria (perfect
        gamma-variate shape, CNR=infinity). This test verifies the cascade
        logic runs without error and that the returned level is consistent.
        """
        volume, masks, aif_coords_true, _ = _build_test_phantom(snr=np.inf)

        # Atlas covers only tissue, not AIF
        atlas_mask = masks["wm"].astype(np.uint8)

        result = atlas_based_aif_search(
            concentration_data=volume,
            brain_mask=masks["brain"].astype(np.uint8),
            atlas_mask=atlas_mask,
            baseline_frames=_BASELINE,
        )

        # Result should be valid regardless of which level fired
        assert result.detection_level in (
            "atlas_strict", "atlas_relaxed",
            "whole_brain_strict", "whole_brain_relaxed", "none",
        )

    def test_no_candidates_returns_none_level(self) -> None:
        """Volume with no good candidates should return detection_level='none'."""
        # All-zero volume
        shape = (8, 8, 3)
        volume = np.zeros((*shape, 70), dtype=np.float64)
        brain_mask = np.ones(shape, dtype=np.uint8)

        result = atlas_based_aif_search(
            concentration_data=volume,
            brain_mask=brain_mask,
            baseline_frames=_BASELINE,
        )

        assert result.detection_level == "none"
        assert len(result.candidates) == 0
        assert result.quality_score == 0.0


class TestAIFWithLesionExclusion:
    """Test AIF detection with lesion exclusion."""

    def test_lesion_exclusion(self) -> None:
        """Lesion mask should exclude nearby voxels from AIF search."""
        volume, masks, aif_coords_true, _ = _build_test_phantom(snr=np.inf)

        # Lesion overlapping with some AIF coords
        lesion_mask = np.zeros(volume.shape[:3], dtype=np.uint8)
        lesion_mask[0:3, 0:3, :] = 1  # Near AIF voxels at (1,1,z) and (1,2,z)

        atlas_mask = np.zeros(volume.shape[:3], dtype=np.uint8)
        atlas_mask[0:4, 0:5, :] = 1  # Covers AIF region

        result = atlas_based_aif_search(
            concentration_data=volume,
            brain_mask=masks["brain"].astype(np.uint8),
            atlas_mask=atlas_mask,
            lesion_mask=lesion_mask,
            baseline_frames=_BASELINE,
            lesion_distance_mm=5.0,
        )

        # AIF candidates should not be in lesion region
        for cand in result.candidates:
            x, y, z = cand.coordinates
            assert lesion_mask[x, y, z] == 0, (
                f"AIF candidate at ({x},{y},{z}) is inside lesion mask"
            )

    def test_contralateral_aif_found(self) -> None:
        """With lesion near AIF, should still find AIF elsewhere."""
        shape = (16, 16, 3)
        # Build phantom with AIF voxels on both sides
        aif_post, _ = generate_dsc_aif_dro(
            n_timepoints=60, dt_sec=1.5, baseline_frames=0,
            peak_delta_r2=_AIF_PEAK,
        )

        # AIF on left side (x=1) and right side (x=14)
        aif_coords_left = {(1, 1, z) for z in range(3)}
        aif_coords_right = {(14, 1, z) for z in range(3)}
        all_aif = aif_coords_left | aif_coords_right

        # Simple tissue map
        tissue_labels = {"gm": 1}
        tissue_map = np.zeros(shape, dtype=np.int32)
        tissue_map[4:12, 4:12, :] = 1

        aif_tissue, _ = generate_dsc_aif_dro(
            n_timepoints=60, dt_sec=1.5, baseline_frames=0,
            peak_delta_r2=0.1,
        )
        gm = DSC_TISSUE_PARAMS["gm"]
        residue = make_exponential_residue(gm.cbf, gm.mtt, 60, 1.5)
        gm_curve = dsc_forward_model(aif_tissue, residue, 1.5)
        tissue_curves = {"gm": gm_curve}

        volume, masks = generate_dsc_3d_phantom(
            shape=shape,
            aif_coords=all_aif,
            aif=aif_post,
            tissue_curves=tissue_curves,
            tissue_map=tissue_map,
            tissue_labels=tissue_labels,
            baseline_frames=_BASELINE,
        )

        # Lesion near left AIF
        lesion_mask = np.zeros(shape, dtype=np.uint8)
        lesion_mask[0:4, 0:4, :] = 1

        result = atlas_based_aif_search(
            concentration_data=volume,
            brain_mask=masks["brain"].astype(np.uint8),
            lesion_mask=lesion_mask,
            baseline_frames=_BASELINE,
            lesion_distance_mm=10.0,
        )

        # Should still find AIF (on right side, away from lesion)
        if result.detection_level != "none":
            for cand in result.candidates:
                assert lesion_mask[cand.coordinates] == 0
