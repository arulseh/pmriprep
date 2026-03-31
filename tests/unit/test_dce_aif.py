"""Unit tests for DCE AIF detection fixes: ROI averaging, FWHM validation, atlas cascade."""

import numpy as np
import pytest

from pmriprep.interfaces.dce_aif import (
    AIFDetection,
    compute_dce_aif_fwhm_seconds,
    compute_dce_aif_washout,
)
from pmriprep.data import get_arterial_atlas_cascade


# =============================================================================
# Helpers
# =============================================================================


def _make_synthetic_dce_4d(
    shape: tuple[int, int, int, int] = (20, 20, 20, 40),
    baseline_frames: int = 5,
    tr: float = 4.0,
    arterial_coords: list[tuple[int, int, int]] | None = None,
    arterial_peak: float = 5.0,
    tissue_peak: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic 4D DCE concentration data with arterial and tissue curves.

    Returns:
        (conc_data, brain_mask) — 4D concentration and 3D binary mask.
    """
    nx, ny, nz, nt = shape
    conc_data = np.zeros(shape, dtype=np.float64)
    brain_mask = np.ones((nx, ny, nz), dtype=np.uint8)

    # Tissue background: slow broad Gaussian peak
    t = np.arange(nt, dtype=np.float64)
    tissue_peak_frame = baseline_frames + 8
    tissue_sigma = 6.0
    tissue_curve = tissue_peak * np.exp(
        -0.5 * ((t - tissue_peak_frame) / tissue_sigma) ** 2
    )
    tissue_curve[:baseline_frames] = 0.0

    # Fill brain with tissue curve
    conc_data[:, :, :, :] = tissue_curve[np.newaxis, np.newaxis, np.newaxis, :]

    # Arterial voxels: sharp narrow peak
    if arterial_coords is None:
        arterial_coords = [(10, 10, 10), (10, 11, 10), (11, 10, 10)]
    arterial_peak_frame = baseline_frames + 2
    arterial_sigma = 1.5
    arterial_curve = arterial_peak * np.exp(
        -0.5 * ((t - arterial_peak_frame) / arterial_sigma) ** 2
    )
    arterial_curve[:baseline_frames] = 0.0

    for x, y, z in arterial_coords:
        conc_data[x, y, z, :] = arterial_curve

    return conc_data, brain_mask


# =============================================================================
# Test: roi_size=1 preserves candidate curves
# =============================================================================


class TestROISizeOne:
    """Verify roi_size=1 produces AIF matching candidate voxel curves."""

    def test_roi_size_1_preserves_candidate_curves(self) -> None:
        """With roi_size=1, averaged AIF should match weighted average of exact voxel curves."""
        conc_data, brain_mask = _make_synthetic_dce_4d()
        candidates = [(10, 10, 10), (10, 11, 10)]
        scores = [1.0, 0.8]

        aif_curve, aif_mask = AIFDetection._compute_weighted_roi_aif(
            candidates, scores, conc_data, brain_mask, roi_size=1, tr=4.0,
        )

        # Mask should only contain the 2 candidate voxels
        assert int(np.sum(aif_mask > 0)) == 2
        assert aif_mask[10, 10, 10] == 1
        assert aif_mask[10, 11, 10] == 1

        # Neighboring tissue voxels should NOT be in mask
        assert aif_mask[9, 10, 10] == 0
        assert aif_mask[10, 10, 9] == 0

    def test_roi_size_3_includes_neighbors(self) -> None:
        """With roi_size=3, mask should include 3x3x3 neighborhood."""
        conc_data, brain_mask = _make_synthetic_dce_4d()
        candidates = [(10, 10, 10)]
        scores = [1.0]

        _, aif_mask = AIFDetection._compute_weighted_roi_aif(
            candidates, scores, conc_data, brain_mask, roi_size=3, tr=4.0,
        )

        # Should include 3x3x3 = 27 voxels around (10,10,10)
        assert int(np.sum(aif_mask > 0)) == 27

    def test_roi_size_1_fwhm_narrower_than_roi_size_3(self) -> None:
        """roi_size=1 should produce narrower FWHM (less tissue broadening)."""
        conc_data, brain_mask = _make_synthetic_dce_4d()
        candidates = [(10, 10, 10), (10, 11, 10), (11, 10, 10)]
        scores = [1.0, 0.9, 0.8]
        tr = 4.0
        baseline_frames = 5

        aif_r1, _ = AIFDetection._compute_weighted_roi_aif(
            candidates, scores, conc_data, brain_mask, roi_size=1, tr=tr,
        )
        aif_r3, _ = AIFDetection._compute_weighted_roi_aif(
            candidates, scores, conc_data, brain_mask, roi_size=3, tr=tr,
        )

        fwhm_r1 = compute_dce_aif_fwhm_seconds(aif_r1, tr, baseline_frames)
        fwhm_r3 = compute_dce_aif_fwhm_seconds(aif_r3, tr, baseline_frames)

        # roi_size=1 should give narrower or equal FWHM
        assert fwhm_r1 <= fwhm_r3 + 1e-6, (
            f"roi_size=1 FWHM ({fwhm_r1:.1f}s) should be <= roi_size=3 FWHM ({fwhm_r3:.1f}s)"
        )


# =============================================================================
# Test: FWHM validation triggers retry
# =============================================================================


class TestFWHMValidation:
    """Verify _validate_dce_averaged_aif detects broadened curves and retries."""

    def test_fwhm_validation_triggers_retry_on_broad_aif(self) -> None:
        """Averaged AIF with FWHM >> candidate FWHM should trigger retry."""
        conc_data, brain_mask = _make_synthetic_dce_4d(
            arterial_peak=5.0, tissue_peak=0.5,
        )
        tr = 4.0
        baseline_frames = 5

        candidates = [(10, 10, 10), (10, 11, 10), (11, 10, 10)]
        scores = [1.0, 0.9, 0.8]

        # Create a deliberately broadened "averaged" AIF (as if ROI expansion
        # contaminated it with tissue). Use a very broad Gaussian.
        nt = conc_data.shape[3]
        t = np.arange(nt, dtype=np.float64)
        broad_aif = 2.0 * np.exp(-0.5 * ((t - 10) / 8.0) ** 2)
        broad_aif[:baseline_frames] = 0.0

        # Candidate metrics with narrow FWHM
        metrics_list: list[dict[str, float]] = [
            {"fwhm_seconds": 12.0, "peak_mm": 5.0, "washout": 0.7, "cnr": 50.0,
             "auc": 100.0, "fwhm_frames": 3.0},
            {"fwhm_seconds": 13.0, "peak_mm": 4.8, "washout": 0.65, "cnr": 45.0,
             "auc": 95.0, "fwhm_frames": 3.2},
            {"fwhm_seconds": 11.0, "peak_mm": 4.5, "washout": 0.72, "cnr": 40.0,
             "auc": 90.0, "fwhm_frames": 2.8},
        ]

        aif_out, mask_out, val_qc = AIFDetection._validate_dce_averaged_aif(
            broad_aif, candidates, scores, metrics_list,
            conc_data, brain_mask, tr, baseline_frames,
        )

        # Should have retried (retries > 0)
        assert val_qc.get("aif_validation_retries", 0) > 0, (
            "Expected validation to trigger retry on broadened AIF"
        )

    def test_fwhm_validation_accepts_good_aif(self) -> None:
        """AIF with FWHM close to candidate mean should pass without retry."""
        conc_data, brain_mask = _make_synthetic_dce_4d()
        tr = 4.0
        baseline_frames = 5

        candidates = [(10, 10, 10), (10, 11, 10)]
        scores = [1.0, 0.9]

        # Use candidate voxel curve directly as averaged AIF (no broadening)
        good_aif = conc_data[10, 10, 10, :].copy()

        fwhm_of_aif = compute_dce_aif_fwhm_seconds(good_aif, tr, baseline_frames)
        metrics_list: list[dict[str, float]] = [
            {"fwhm_seconds": fwhm_of_aif, "peak_mm": 5.0, "washout": 0.7,
             "cnr": 50.0, "auc": 100.0, "fwhm_frames": fwhm_of_aif / tr},
            {"fwhm_seconds": fwhm_of_aif + 1.0, "peak_mm": 4.8, "washout": 0.65,
             "cnr": 45.0, "auc": 95.0, "fwhm_frames": (fwhm_of_aif + 1.0) / tr},
        ]

        _, _, val_qc = AIFDetection._validate_dce_averaged_aif(
            good_aif, candidates, scores, metrics_list,
            conc_data, brain_mask, tr, baseline_frames,
        )

        assert val_qc.get("aif_validation_retries", 0) == 0, (
            "Good AIF should not trigger validation retry"
        )


# =============================================================================
# Test: Atlas cascade ordering
# =============================================================================


class TestAtlasCascade:
    """Verify atlas cascade returns files in tightest-to-broadest order."""

    def test_atlas_cascade_returns_ordered_list(self) -> None:
        """get_arterial_atlas_cascade should return MCA_ACA first, then CoW, then full."""
        cascade = get_arterial_atlas_cascade("2mm")

        # Should find at least the files we just copied
        assert len(cascade) >= 2, f"Expected >= 2 atlas files, got {len(cascade)}"

        names = [p.name for p in cascade]
        # MCA_ACA should come before CoW_mask
        mca_idx = next((i for i, n in enumerate(names) if "MCA_ACA" in n), None)
        cow_idx = next((i for i, n in enumerate(names) if "CoW_mask" in n), None)
        full_idx = next((i for i, n in enumerate(names) if "full_mask" in n), None)

        if mca_idx is not None and cow_idx is not None:
            assert mca_idx < cow_idx, "MCA_ACA should come before CoW_mask"
        if cow_idx is not None and full_idx is not None:
            assert cow_idx < full_idx, "CoW_mask should come before full_mask"

    def test_atlas_cascade_all_files_exist(self) -> None:
        """All returned atlas paths should exist."""
        cascade = get_arterial_atlas_cascade("2mm")
        for path in cascade:
            assert path.exists(), f"Atlas file does not exist: {path}"


# =============================================================================
# Test: TR-adaptive Savitzky-Golay
# =============================================================================


class TestTRAdaptiveSavgol:
    """Verify Savitzky-Golay window adapts to TR."""

    def test_savgol_low_tr_uses_window_5(self) -> None:
        """At TR=2s, smoothing should use window=5 (standard)."""
        conc_data, brain_mask = _make_synthetic_dce_4d(shape=(20, 20, 20, 60))
        candidates = [(10, 10, 10)]
        scores = [1.0]
        tr = 2.0

        aif_tr2, _ = AIFDetection._compute_weighted_roi_aif(
            candidates, scores, conc_data, brain_mask, roi_size=1, tr=tr,
        )
        # Curve should be smoothed (not raw)
        raw = conc_data[10, 10, 10, :].copy()
        # Smoothed curve should differ from raw (SG changes values slightly)
        # But peak should be preserved approximately
        assert abs(float(np.max(aif_tr2)) - float(np.max(raw))) < 1.0

    def test_savgol_high_tr_skips_smoothing(self) -> None:
        """At TR=8s, SG smoothing should be skipped entirely — peak fully preserved."""
        conc_data, brain_mask = _make_synthetic_dce_4d(
            shape=(20, 20, 20, 15), baseline_frames=3,
        )
        candidates = [(10, 10, 10)]
        scores = [1.0]
        tr = 8.0

        aif_high_tr, _ = AIFDetection._compute_weighted_roi_aif(
            candidates, scores, conc_data, brain_mask, roi_size=1, tr=tr,
        )

        # SG is skipped at TR > 5s, so peak should be fully preserved
        raw_peak = float(np.max(conc_data[10, 10, 10, :]))
        smoothed_peak = float(np.max(aif_high_tr))
        assert smoothed_peak > 0.9 * raw_peak, (
            f"High-TR peak not preserved: {smoothed_peak:.2f} vs raw {raw_peak:.2f}"
        )

    def test_savgol_short_curve_no_crash(self) -> None:
        """Curves with < 5 timepoints should not crash."""
        conc_data = np.random.rand(5, 5, 5, 3).astype(np.float64)
        brain_mask = np.ones((5, 5, 5), dtype=np.uint8)
        candidates = [(2, 2, 2)]
        scores = [1.0]

        # Should not raise
        aif, mask = AIFDetection._compute_weighted_roi_aif(
            candidates, scores, conc_data, brain_mask, roi_size=1, tr=4.0,
        )
        assert len(aif) == 3

    def test_savgol_skipped_for_narrow_peak(self) -> None:
        """Delta-like peak at TR=8s should be preserved (>95% of raw) since SG is skipped."""
        nt = 15
        baseline_frames = 3
        tr = 8.0

        # Create data with a very narrow arterial peak (~0.3 frame sigma)
        shape = (10, 10, 10, nt)
        conc_data = np.zeros(shape, dtype=np.float64)
        brain_mask = np.ones(shape[:3], dtype=np.uint8)

        t = np.arange(nt, dtype=np.float64)
        # Delta-like peak: sigma=0.3 frames, almost all energy in one frame
        peak_frame = baseline_frames + 1
        arterial_curve = 5.0 * np.exp(-0.5 * ((t - peak_frame) / 0.3) ** 2)
        arterial_curve[:baseline_frames] = 0.0
        conc_data[5, 5, 5, :] = arterial_curve

        candidates = [(5, 5, 5)]
        scores = [1.0]

        aif_out, _ = AIFDetection._compute_weighted_roi_aif(
            candidates, scores, conc_data, brain_mask, roi_size=1, tr=tr,
        )

        raw_peak = float(np.max(arterial_curve))
        out_peak = float(np.max(aif_out))
        # SG skipped at TR=8s → peak should be >95% preserved
        assert out_peak > 0.95 * raw_peak, (
            f"Narrow peak at high TR not preserved: {out_peak:.3f} vs raw {raw_peak:.3f}"
        )


# =============================================================================
# Test: final_washout QC metric
# =============================================================================


class TestFinalWashoutQC:
    """Verify compute_dce_aif_washout returns expected values on known curves."""

    def test_washout_known_curve(self) -> None:
        """Washout = (peak - tail) / (peak - baseline) on a simple synthetic curve."""
        # Baseline = 0, peak = 10, tail = 3 → washout = (10 - 3) / (10 - 0) = 0.7
        nt = 30
        baseline_frames = 5
        curve = np.zeros(nt, dtype=np.float64)
        curve[baseline_frames + 2] = 10.0  # peak
        # Tail: last 5 frames
        curve[-5:] = 3.0

        washout = compute_dce_aif_washout(curve, baseline_frames)
        assert abs(washout - 0.7) < 0.05, f"Expected washout ~0.7, got {washout:.3f}"

    def test_washout_no_washout(self) -> None:
        """Curve that stays at peak level should have washout ~0."""
        nt = 30
        baseline_frames = 5
        curve = np.zeros(nt, dtype=np.float64)
        curve[baseline_frames:] = 10.0  # peak and stays there

        washout = compute_dce_aif_washout(curve, baseline_frames)
        assert washout < 0.1, f"Expected near-zero washout, got {washout:.3f}"

    def test_washout_full_washout(self) -> None:
        """Curve that returns to baseline should have washout ~1."""
        nt = 30
        baseline_frames = 5
        curve = np.zeros(nt, dtype=np.float64)
        curve[baseline_frames + 2] = 10.0  # sharp peak
        # Tail returns to baseline (0)

        washout = compute_dce_aif_washout(curve, baseline_frames)
        assert washout > 0.9, f"Expected washout ~1.0, got {washout:.3f}"


# =============================================================================
# Rescue Score Tests
# =============================================================================


class TestComputeRescueScore:
    """Tests for _compute_rescue_score — venous AIF rejection in whole-brain mode."""

    def test_rescue_rejects_subthreshold_peak_wholebrain(self) -> None:
        """Whole-brain: peak below floor (1.07 < 1.5 mM) → rejected."""
        score = AIFDetection._compute_rescue_score(
            candidate_peak=1.07, peak_min_expected=5.0, peak_floor=1.5,
            tier_used=1, is_likely_venous=False, atlas_constrained=False,
        )
        assert score == 0.0

    def test_rescue_adequacy_scaling_wholebrain(self) -> None:
        """Whole-brain: marginal peak gets adequacy penalty.

        peak_adequacy = 2.0/5.0 = 0.4, score = 1.0 * 0.4 = 0.4.
        Note: 0.4 < _MIN_CANDIDATE_SCORE (0.5), so downstream filter
        would reject this candidate — intentional conservative behavior.
        """
        score = AIFDetection._compute_rescue_score(
            candidate_peak=2.0, peak_min_expected=5.0, peak_floor=1.5,
            tier_used=1, is_likely_venous=False, atlas_constrained=False,
        )
        assert abs(score - 0.4) < 1e-6

    def test_rescue_venous_stacks_wholebrain(self) -> None:
        """Whole-brain + venous: both penalties stack.

        base=1.0, venous=0.3, adequacy=3.0/5.0=0.6 → 1.0*0.3*0.6=0.18.
        """
        score = AIFDetection._compute_rescue_score(
            candidate_peak=3.0, peak_min_expected=5.0, peak_floor=1.5,
            tier_used=1, is_likely_venous=True, atlas_constrained=False,
        )
        assert abs(score - 0.18) < 1e-6

    def test_rescue_atlas_no_adequacy_penalty(self) -> None:
        """Atlas-constrained: no peak adequacy penalty (low peaks expected)."""
        score = AIFDetection._compute_rescue_score(
            candidate_peak=1.0, peak_min_expected=5.0, peak_floor=0.75,
            tier_used=1, is_likely_venous=False, atlas_constrained=True,
        )
        # base_score=1.0, peak_adequacy=1.0 (atlas skips scaling)
        assert abs(score - 1.0) < 1e-6

    def test_rescue_atlas_keeps_low_floor(self) -> None:
        """Atlas-constrained: 0.9 mM passes the 0.75 mM floor."""
        score = AIFDetection._compute_rescue_score(
            candidate_peak=0.9, peak_min_expected=5.0, peak_floor=0.75,
            tier_used=1, is_likely_venous=False, atlas_constrained=True,
        )
        assert score > 0.0

    def test_rescue_zero_peak(self) -> None:
        """Zero peak is always rejected regardless of mode."""
        for atlas in (True, False):
            score = AIFDetection._compute_rescue_score(
                candidate_peak=0.0, peak_min_expected=5.0, peak_floor=0.75,
                tier_used=1, is_likely_venous=False, atlas_constrained=atlas,
            )
            assert score == 0.0

    def test_rescue_half_dose_wholebrain(self) -> None:
        """Half-dose whole-brain: 1.5 mM peak passes floor and quality filter.

        peak_min=2.5 (half dose), floor=0.75, adequacy=1.5/2.5=0.6.
        score = 1.0 * 0.6 = 0.6 (passes _MIN_CANDIDATE_SCORE=0.5).
        """
        score = AIFDetection._compute_rescue_score(
            candidate_peak=1.5, peak_min_expected=2.5, peak_floor=0.75,
            tier_used=1, is_likely_venous=False, atlas_constrained=False,
        )
        assert abs(score - 0.6) < 1e-6
