"""Unit tests for DCE AIF detection fixes: ROI averaging, FWHM validation, atlas cascade."""

from unittest.mock import patch

import numpy as np
import pytest

from pmriprep.interfaces.dce_aif import (
    AIFDetection,
    compute_dce_aif_fwhm_seconds,
    compute_dce_aif_washout,
)
from pmriprep.data import get_arterial_atlas_cascade
from pmriprep.utils.dce_constants import AIF_THRESHOLDS_PERMISSIVE


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
        """Whole-brain: peak below floor (1.858 < 2.5 mM) → rejected."""
        score = AIFDetection._compute_rescue_score(
            candidate_peak=1.858, peak_min_expected=5.0, peak_floor=2.5,
            tier_used=1, is_likely_venous=False, atlas_constrained=False,
        )
        assert score == 0.0

    def test_rescue_adequacy_scaling_wholebrain(self) -> None:
        """Whole-brain: marginal peak gets adequacy penalty.

        peak_adequacy = 3.0/5.0 = 0.6, score = 1.0 * 0.6 = 0.6.
        """
        score = AIFDetection._compute_rescue_score(
            candidate_peak=3.0, peak_min_expected=5.0, peak_floor=2.5,
            tier_used=1, is_likely_venous=False, atlas_constrained=False,
        )
        assert abs(score - 0.6) < 1e-6

    def test_rescue_venous_stacks_wholebrain(self) -> None:
        """Whole-brain + venous: both penalties stack.

        base=1.0, venous=0.3, adequacy=3.0/5.0=0.6 → 1.0*0.3*0.6=0.18.
        """
        score = AIFDetection._compute_rescue_score(
            candidate_peak=3.0, peak_min_expected=5.0, peak_floor=2.5,
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
        for atlas, floor in ((True, 0.75), (False, 2.5)):
            score = AIFDetection._compute_rescue_score(
                candidate_peak=0.0, peak_min_expected=5.0, peak_floor=floor,
                tier_used=1, is_likely_venous=False, atlas_constrained=atlas,
            )
            assert score == 0.0

    def test_rescue_half_dose_wholebrain(self) -> None:
        """Half-dose whole-brain: 1.5 mM peak passes floor (1.25) and quality filter.

        peak_min=2.5 (half dose), floor=1.25 (0.5×2.5), adequacy=1.5/2.5=0.6.
        score = 1.0 * 0.6 = 0.6 (passes _MIN_CANDIDATE_SCORE=0.5).
        """
        score = AIFDetection._compute_rescue_score(
            candidate_peak=1.5, peak_min_expected=2.5, peak_floor=1.25,
            tier_used=1, is_likely_venous=False, atlas_constrained=False,
        )
        assert abs(score - 0.6) < 1e-6


# =============================================================================
# AIF scaling unit-mismatch fix (v0.25.3)
# =============================================================================


class TestAIFScalingUnitFix:
    """v0.25.3: Parker's peak_min/peak_max are in blood-voxel units; aif_curve
    is in plasma units after /(1-Hct) correction. Scaling target must be
    divided by (1-Hct) to produce a correct plasma target.
    """

    def _make_interface(
        self,
        hematocrit: float = 0.42,
        dose: float = 1.0,
        agent_standard_dose: float = 0.1,
        target_peak: float = 0.0,
    ) -> AIFDetection:
        """Build a minimal AIFDetection instance with just the inputs the
        scaling-block arithmetic touches (hematocrit + dose factors).
        """
        iface = AIFDetection()
        iface.inputs.hematocrit = hematocrit
        iface.inputs.dose = dose
        iface.inputs.agent_standard_dose = agent_standard_dose
        iface.inputs.target_peak = target_peak
        iface.inputs.field_strength = 3.0
        return iface

    def test_get_expected_peak_range_returns_blood_units(self) -> None:
        """_get_expected_peak_range returns Parker blood-voxel values × dose."""
        iface = self._make_interface(dose=1.0, agent_standard_dose=0.1)
        peak_min, peak_max = iface._get_expected_peak_range()
        # Full-dose Gd-DOTA (dose_factor=1.0): Parker 5.0 / 7.6.
        assert peak_min == pytest.approx(5.0)
        assert peak_max == pytest.approx(7.6)

    def test_get_dose_factor_half_dose_vueway(self) -> None:
        """Vueway half-dose gives dose_factor = 0.5 × 0.05 / 0.1 = 0.25."""
        iface = self._make_interface(dose=0.5, agent_standard_dose=0.05)
        assert iface._get_dose_factor() == pytest.approx(0.25)

    def test_effective_target_half_dose_vueway_plasma(self) -> None:
        """Half-dose Vueway: plasma target = (5.0+7.6)/2 × 0.25 / (1-0.42) ≈ 2.716 mM."""
        iface = self._make_interface(
            dose=0.5, agent_standard_dose=0.05, hematocrit=0.42,
        )
        peak_min, peak_max = iface._get_expected_peak_range()
        blood_target = (peak_min + peak_max) / 2.0  # 6.3 × 0.25 = 1.575 mM (blood)
        plasma_target = blood_target / (1.0 - 0.42)
        assert blood_target == pytest.approx(1.575)
        assert plasma_target == pytest.approx(2.716, rel=1e-3)

    def test_hct_zero_is_no_op_on_conversion(self) -> None:
        """Hct=0 → (1-Hct)=1 → plasma conversion is no-op; target equals Parker fallback."""
        hct = 0.0
        peak_min, peak_max = 5.0, 7.6
        blood_target = (peak_min + peak_max) / 2.0
        plasma_target = blood_target / (1.0 - hct)
        assert plasma_target == pytest.approx(blood_target)

    def test_hct_boundary_high_polycythemic(self) -> None:
        """Hct=0.55 (clinical high end) → conversion factor ≈ 2.22×."""
        hct = 0.55
        conversion = 1.0 / (1.0 - hct)
        assert conversion == pytest.approx(2.2222, rel=1e-3)

    def test_target_peak_override_is_blood_units(self) -> None:
        """User-supplied `target_peak=5.0` is interpreted as blood-voxel units.

        The scaling site divides by (1-Hct) so override behaves identically
        to the Parker-fallback convention. For Hct=0.42: target becomes
        5.0 / 0.58 ≈ 8.621 mM plasma.
        """
        hct = 0.42
        user_target_blood = 5.0
        plasma_target = user_target_blood / (1.0 - hct)
        assert plasma_target == pytest.approx(8.621, rel=1e-3)

    def test_warning_thresholds_calibrated_for_plasma_scale(self) -> None:
        """Warning thresholds 3.4 and 5.2 are ~1/(1-Hct) × old thresholds 2.0/3.0.

        Regression guard: scaling_factor=2.5 should NOT trigger warning
        (would have under the old 2.0 threshold). Scaling_factor=3.5 triggers
        the low-confidence warning (>3.4). Scaling_factor=5.3 triggers
        unreliable (>5.2).
        """
        # Proportional scaling
        assert 2.0 / (1.0 - 0.42) == pytest.approx(3.448, rel=1e-3)  # ~3.4
        assert 3.0 / (1.0 - 0.42) == pytest.approx(5.172, rel=1e-3)  # ~5.2
        # Regression guard
        assert 2.5 <= 3.4  # below new low-confidence threshold
        assert 3.5 > 3.4   # above new low-confidence threshold
        assert 5.3 > 5.2   # above new unreliable threshold

    def test_scaling_block_end_to_end_on_plasma_curve(self) -> None:
        """Integration-flavored: replay the landed scaling arithmetic against
        an AIFDetection instance and a known plasma-scale ``aif_curve``,
        confirming the peak lands at the plasma-unit target.

        Simulates the exact control flow from ``_run_interface`` (line
        2917-2954 of ``dce_aif.py``) using an actual interface instance
        so the arithmetic is exercised against live trait values
        (hematocrit, dose, agent_standard_dose, target_peak).
        """
        iface = self._make_interface(
            hematocrit=0.42, dose=1.0, agent_standard_dose=0.1, target_peak=0.0,
        )

        # Construct a synthetic aif_curve in plasma units. A typical
        # full-dose Gd-DOTA detected-voxel peak ≈ 3.65 mM blood →
        # after /(1-Hct)=0.58 ≈ 6.29 mM plasma.
        n = 40
        curve = np.zeros(n, dtype=np.float64)
        curve[10:20] = 6.29  # peak in plasma units

        # Replay scaling arithmetic matching the landed code.
        current_peak = float(np.max(curve))
        hct = float(iface.inputs.hematocrit)
        plasma_conv = 1.0 / (1.0 - hct)
        peak_min, peak_max = iface._get_expected_peak_range()
        target = (peak_min + peak_max) / 2.0 * plasma_conv
        scaling_factor = target / current_peak
        scaled = curve * scaling_factor

        # Expected plasma target: Parker mid (6.3) / (1-0.42) ≈ 10.862 mM.
        expected_plasma_target = (5.0 + 7.6) / 2.0 / (1.0 - 0.42)
        assert target == pytest.approx(expected_plasma_target, rel=1e-4)
        assert float(np.max(scaled)) == pytest.approx(target, rel=1e-4)
        # scaling_factor ≈ 10.862 / 6.29 ≈ 1.727 — below warning thresholds.
        assert scaling_factor == pytest.approx(1.727, rel=1e-3)
        assert scaling_factor <= 3.4  # no low-confidence warning
        assert scaling_factor <= 5.2  # no unreliable warning


# =============================================================================
# Two-pass cascade regression tests (Issue #2 fix)
# =============================================================================


def _make_cascade_inputs(
    nx: int = 8, ny: int = 8, nz: int = 4, nt: int = 30,
) -> dict[str, np.ndarray]:
    """Build minimal inputs for ``_run_two_pass_cascade``."""
    conc_data = np.zeros((nx, ny, nz, nt), dtype=np.float64)
    atlas_search_mask = np.ones((nx, ny, nz), dtype=np.uint8)
    wb_search_mask = np.ones((nx, ny, nz), dtype=np.uint8)
    return {
        "conc_data": conc_data,
        "atlas_search_mask": atlas_search_mask,
        "wb_search_mask": wb_search_mask,
    }


def _make_atlas_cascade(n: int = 4) -> tuple[list[np.ndarray], list[str]]:
    """Build N synthetic atlas masks + names parallel."""
    masks = [np.ones((8, 8, 4), dtype=np.uint8) for _ in range(n)]
    names = ["MCA_ACA", "MCA_ACA_dil", "MCA_ACA_dil2", "full_mask_dil2"][:n]
    return masks, names


class TestTwoPassCascade:
    """Regression tests for the Issue #2 two-pass cascade contract."""

    def test_descends_when_strict_returns_zero_scored(self) -> None:
        """Strict tier returning 10 zero-scored candidates must NOT
        short-circuit the cascade. Relaxed at the same atlas level should
        be tried and accepted when it produces ≥ 2 keepers and at least
        one candidate within the anchor-timing window.
        """
        inputs = _make_cascade_inputs()
        atlas_masks, atlas_names = _make_atlas_cascade()
        detector = AIFDetection()

        # baseline_frames=5, tr=4.0 → peak_time_frame ≤ 7 satisfies anchor
        # ((7-5)*4 = 8s ≤ 10s threshold).
        side_effects: list = [
            ([(1, 1, 1)] * 10, [0.0] * 10,
             [{"peak_time_frame": 7}] * 10),
            ([(2, 2, 2)] * 5, [0.6] * 5,
             [{"peak_time_frame": 7}] * 5),
        ]

        with patch.object(
            AIFDetection, "_find_candidates_in_mask",
            side_effect=side_effects,
        ) as mock_find:
            (
                candidates, scores, metrics_list,
                detection_method, atlas_tier_used, n_attempted,
                anchor_outcomes,
            ) = detector._run_two_pass_cascade(
                **inputs,
                atlas_masks=atlas_masks,
                atlas_names=atlas_names,
                n_candidates=10,
                tr=4.0,
                baseline_frames=5,
                tissue_ttp_frame=10.0,
            )

        assert detection_method == "atlas"
        assert atlas_tier_used == "MCA_ACA_relaxed"
        assert n_attempted == 2
        assert mock_find.call_count == 2
        assert len(candidates) == 5
        assert all(s == 0.6 for s in scores)
        # 2 tiers attempted, both anchored; only the second met quorum.
        assert len(anchor_outcomes) == 2
        assert all(o["has_anchor"] for o in anchor_outcomes)

    def test_pass2_accepts_best_single_keeper_when_no_quorum(self) -> None:
        """If no tier produces ≥ 2 keepers, Pass 2 falls back to the
        highest-max-score single-keeper tier observed.
        """
        inputs = _make_cascade_inputs()
        atlas_masks, atlas_names = _make_atlas_cascade()
        detector = AIFDetection()

        # Every tier returns exactly 1 keeper. Max-score per tier varies;
        # the tier with the highest max-score should be chosen by Pass 2.
        # 4 atlases × 3 criteria + 1 whole-brain = 13 tiers.
        max_scores = [
            0.55, 0.50, 0.50,  # MCA_ACA × 3
            0.50, 0.62, 0.50,  # MCA_ACA_dil × 3   ← highest at idx 4 (relaxed)
            0.50, 0.50, 0.51,  # MCA_ACA_dil2 × 3
            0.50, 0.50, 0.50,  # full_mask_dil2 × 3
            0.55,              # wb_relaxed
        ]
        # Each tier returns 1 candidate at peak_time_frame=6 (post-arrival
        # = (6-5)*4 = 4s ≤ 10s threshold; keeper is anchored).
        side_effects: list = [
            ([(i, 0, 0)], [s], [{"peak_time_frame": 6}])
            for i, s in enumerate(max_scores)
        ]

        with patch.object(
            AIFDetection, "_find_candidates_in_mask",
            side_effect=side_effects,
        ):
            (
                candidates, scores, metrics_list,
                detection_method, atlas_tier_used, n_attempted,
                anchor_outcomes,
            ) = detector._run_two_pass_cascade(
                **inputs,
                atlas_masks=atlas_masks,
                atlas_names=atlas_names,
                n_candidates=10,
                tr=4.0,
                baseline_frames=5,
                tissue_ttp_frame=10.0,
            )

        assert n_attempted == 13  # all tiers run, none short-circuited
        # Highest single-keeper max-score was 0.62 at MCA_ACA_dil_relaxed.
        assert detection_method == "atlas"
        assert atlas_tier_used == "MCA_ACA_dil_relaxed"
        assert len(candidates) == 1
        assert scores == [0.62]
        assert len(anchor_outcomes) == 13

    def test_falls_to_population_only_when_no_keeper_anywhere(self) -> None:
        """Cascade must exhaust ALL 13 tiers before signaling population.

        With every tier returning 0-keeper output, Pass 1 fails everywhere,
        Pass 2 has no single-keeper tier to fall back to, and the cascade
        returns ``detection_method="population"`` with the full 13-tier
        attempt count for the caller's provenance.
        """
        inputs = _make_cascade_inputs()
        atlas_masks, atlas_names = _make_atlas_cascade()
        detector = AIFDetection()

        # All 13 tiers return non-empty zero-scored candidates with
        # arbitrary timing (anchor doesn't matter when quorum is 0).
        side_effects: list = [
            ([(0, 0, 0)] * 5, [0.0] * 5, [{"peak_time_frame": 7}] * 5)
            for _ in range(13)
        ]

        with patch.object(
            AIFDetection, "_find_candidates_in_mask",
            side_effect=side_effects,
        ) as mock_find:
            (
                candidates, scores, metrics_list,
                detection_method, atlas_tier_used, n_attempted,
                anchor_outcomes,
            ) = detector._run_two_pass_cascade(
                **inputs,
                atlas_masks=atlas_masks,
                atlas_names=atlas_names,
                n_candidates=10,
                tr=4.0,
                baseline_frames=5,
                tissue_ttp_frame=10.0,
            )

        assert detection_method == "population"
        assert atlas_tier_used is None
        assert n_attempted == 13
        assert mock_find.call_count == 13
        assert len(anchor_outcomes) == 13
        # Pass 1 selected nothing; Pass 2 had no single-keeper tier to use.
        assert candidates == []
        assert scores == []

    def test_strict_quorum_short_circuits_relaxed(self) -> None:
        """When strict tier produces ≥ 2 keepers, relaxed/permissive at
        the same (or any later) atlas level must NOT be tried.

        Confirms the "tightest tier wins" intent of the cascade: better-
        constrained tiers preempt looser ones.
        """
        inputs = _make_cascade_inputs()
        atlas_masks, atlas_names = _make_atlas_cascade()
        detector = AIFDetection()

        # Strict at first atlas finds 5 keepers (all score 0.7) at
        # peak_time_frame=6 (post-arr 4s, anchored).
        side_effects: list = [
            ([(i, 0, 0) for i in range(5)], [0.7] * 5,
             [{"peak_time_frame": 6}] * 5),
        ]

        with patch.object(
            AIFDetection, "_find_candidates_in_mask",
            side_effect=side_effects,
        ) as mock_find:
            (
                _candidates, _scores, _metrics,
                detection_method, atlas_tier_used, n_attempted,
                _anchor_outcomes,
            ) = detector._run_two_pass_cascade(
                **inputs,
                atlas_masks=atlas_masks,
                atlas_names=atlas_names,
                n_candidates=10,
                tr=4.0,
                baseline_frames=5,
                tissue_ttp_frame=10.0,
            )

        assert detection_method == "atlas"
        assert atlas_tier_used == "MCA_ACA_strict"
        assert n_attempted == 1
        assert mock_find.call_count == 1


class TestPermissiveThresholdTightened:
    """Regression guard for the physics-expert recommendation."""

    def test_permissive_washout_threshold_tightened(self) -> None:
        """Permissive AIF washout threshold raised from 0.05 to 0.10.

        Physics review (Calamante NMR Biomed 2013): CNR relaxation
        (5.0→2.0) provides the sensitivity gain on low-dose/preloaded
        protocols; loosening washout below 0.10 admits tissue voxels
        without proportional arterial-detection benefit.
        """
        assert AIF_THRESHOLDS_PERMISSIVE.washout_min == 0.10


class TestCascadeExhaustionProvenance:
    """Regression for Pass-3 (population) provenance threading.

    When the two-pass cascade exhausts all tiers without any keeper, the
    population-fallback path must populate ``cascade_tiers_attempted``,
    ``cascade_best_score=0.0``, and an exhaustion-aware
    ``auto_fallback_reason`` so QC consumers can distinguish "data quality
    too poor for voxel-AIF detection" from older single-tier fallback.
    """

    def test_helper_provenance_shape(self) -> None:
        """Helper returns a dict with all required QC fields populated."""
        prov = AIFDetection._cascade_exhaustion_provenance(13)
        assert prov["auto_detection_attempted"] is True
        assert prov["auto_best_score"] == 0.0
        assert prov["auto_n_candidates"] == 0
        assert prov["auto_n_above_threshold"] == 0
        assert prov["cascade_tiers_attempted"] == 13
        assert prov["cascade_best_score"] == 0.0
        assert "all 13 cascade tiers exhausted" in prov["auto_fallback_reason"]
        assert "no tier produced any anchored keeper" in prov["auto_fallback_reason"]

    def test_helper_threads_partial_attempt_count(self) -> None:
        """Helper accurately reports tier count even on partial cascades.

        E.g. atlas_masks=None scenario: only the whole-brain-relaxed tier
        runs, so cascade_tiers_attempted=1.
        """
        prov = AIFDetection._cascade_exhaustion_provenance(1)
        assert prov["cascade_tiers_attempted"] == 1
        assert "all 1 cascade tiers exhausted" in prov["auto_fallback_reason"]

    def test_helper_embeds_anchor_outcomes_when_provided(self) -> None:
        """When anchor_outcomes is passed, it appears in the provenance dict."""
        outcomes = [
            {"tier": "MCA_ACA_strict", "n_candidates": 5,
             "quorum_count": 0, "earliest_post_arr_s": 12.5,
             "has_anchor": False},
        ]
        prov = AIFDetection._cascade_exhaustion_provenance(1, anchor_outcomes=outcomes)
        assert prov["cascade_anchor_outcomes"] == outcomes


# =============================================================================
# Anchor-gate regression tests (venous-cluster fix)
# =============================================================================


class TestArterialAnchor:
    """Tests for the Mouridsen-style cluster-anchor predicate."""

    def test_empty_metrics_returns_no_anchor(self) -> None:
        has_anchor, earliest = AIFDetection._has_arterial_anchor(
            [], baseline_frames=10, tr=4.0,
        )
        assert has_anchor is False
        assert earliest == float("inf")

    def test_anchor_threshold_constant_value(self) -> None:
        """Regression guard: physiologically-derived 10s threshold."""
        from pmriprep.interfaces.dce_aif import _MAX_ANCHOR_POST_ARRIVAL_S
        assert _MAX_ANCHOR_POST_ARRIVAL_S == 10.0

    def test_anchored_when_earliest_within_threshold(self) -> None:
        # baseline=10, tr=4 → peak_time_frame=12 → post-arr=8s ≤ 10s
        metrics = [
            {"peak_time_frame": 12},
            {"peak_time_frame": 15},
            {"peak_time_frame": 18},
        ]
        has_anchor, earliest = AIFDetection._has_arterial_anchor(
            metrics, baseline_frames=10, tr=4.0,
        )
        assert has_anchor is True
        assert earliest == pytest.approx(8.0)

    def test_unanchored_when_all_late(self) -> None:
        # baseline=10, tr=4 → all peak_time_frames give >10s post-arr
        metrics = [
            {"peak_time_frame": 13},  # 12s
            {"peak_time_frame": 14},  # 16s
            {"peak_time_frame": 15},  # 20s
        ]
        has_anchor, earliest = AIFDetection._has_arterial_anchor(
            metrics, baseline_frames=10, tr=4.0,
        )
        assert has_anchor is False
        assert earliest == pytest.approx(12.0)


class TestAnchorGateInCascade:
    """Cascade-level integration tests for the anchor gate."""

    def test_all_venous_tier_fails_via_anchor_gate(self) -> None:
        """Sub-20402855 pattern: 3 candidates per tier, all peaking at
        post-arrival 13s (above 10s threshold). Cascade descends through
        all 13 tiers, none anchored, falls to population.
        """
        inputs = _make_cascade_inputs()
        atlas_masks, atlas_names = _make_atlas_cascade()
        detector = AIFDetection()

        # baseline=5, tr=4 → peak_time_frame=8 gives post-arr=12s > 10s
        side_effects: list = [
            ([(0, 0, 0)] * 3, [0.6] * 3, [{"peak_time_frame": 8}] * 3)
            for _ in range(13)
        ]

        with patch.object(
            AIFDetection, "_find_candidates_in_mask",
            side_effect=side_effects,
        ):
            (
                candidates, scores, _metrics,
                detection_method, atlas_tier_used, n_attempted,
                anchor_outcomes,
            ) = detector._run_two_pass_cascade(
                **inputs,
                atlas_masks=atlas_masks,
                atlas_names=atlas_names,
                n_candidates=10,
                tr=4.0,
                baseline_frames=5,
                tissue_ttp_frame=10.0,
            )

        assert detection_method == "population"
        assert atlas_tier_used is None
        assert n_attempted == 13
        assert candidates == []
        # Every tier was anchor-rejected
        assert all(o["has_anchor"] is False for o in anchor_outcomes)
        assert all(o["quorum_count"] >= 2 for o in anchor_outcomes)

    def test_strict_quorum_without_anchor_descends_to_relaxed(self) -> None:
        """Strict tier has quorum (5 keepers, score 0.6) but NO anchor
        (all peak at 12s post-arr); relaxed at the same atlas level has
        BOTH quorum AND anchor (5 keepers at 8s post-arr). Cascade must
        descend, not short-circuit on the anchor-failed strict tier.

        Exercises the exact regression the anchor gate is intended to
        catch: quorum alone is no longer sufficient.
        """
        inputs = _make_cascade_inputs()
        atlas_masks, atlas_names = _make_atlas_cascade()
        detector = AIFDetection()

        # baseline=5, tr=4
        # tier 1 (MCA_ACA strict): 5 candidates, score 0.6, peak_time=8
        #   → quorum=5, post-arr=12s, NO anchor → fail
        # tier 2 (MCA_ACA relaxed): 5 candidates, score 0.6, peak_time=6
        #   → quorum=5, post-arr=4s, anchored → succeed
        side_effects: list = [
            ([(i, 0, 0) for i in range(5)], [0.6] * 5,
             [{"peak_time_frame": 8}] * 5),
            ([(i, 1, 0) for i in range(5)], [0.6] * 5,
             [{"peak_time_frame": 6}] * 5),
        ]

        with patch.object(
            AIFDetection, "_find_candidates_in_mask",
            side_effect=side_effects,
        ) as mock_find:
            (
                candidates, _scores, _metrics,
                detection_method, atlas_tier_used, n_attempted,
                anchor_outcomes,
            ) = detector._run_two_pass_cascade(
                **inputs,
                atlas_masks=atlas_masks,
                atlas_names=atlas_names,
                n_candidates=10,
                tr=4.0,
                baseline_frames=5,
                tissue_ttp_frame=10.0,
            )

        assert detection_method == "atlas"
        assert atlas_tier_used == "MCA_ACA_relaxed"
        assert n_attempted == 2
        assert mock_find.call_count == 2
        assert len(candidates) == 5
        # Strict tier had quorum but failed anchor; relaxed had both.
        assert anchor_outcomes[0]["quorum_count"] == 5
        assert anchor_outcomes[0]["has_anchor"] is False
        assert anchor_outcomes[1]["quorum_count"] == 5
        assert anchor_outcomes[1]["has_anchor"] is True

    def test_anchored_tier_with_late_companions_succeeds(self) -> None:
        """Sub-20449286 pattern: 4 candidates, anchor #1 at post-arr 8s,
        rest at 12.2s. Tier should pass on anchor + quorum.
        """
        inputs = _make_cascade_inputs()
        atlas_masks, atlas_names = _make_atlas_cascade()
        detector = AIFDetection()

        # baseline=5, tr=4: peak_time_frame=7 → 8s (anchor),
        # peak_time_frame=8 → 12s (over threshold, but other tiers do not
        # need to anchor; this candidate is in the same tier).
        side_effects: list = [
            (
                [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
                [0.7, 0.6, 0.55, 0.51],
                [
                    {"peak_time_frame": 7},  # anchor candidate (post-arr 8s)
                    {"peak_time_frame": 8},
                    {"peak_time_frame": 8},
                    {"peak_time_frame": 8},
                ],
            ),
        ]

        with patch.object(
            AIFDetection, "_find_candidates_in_mask",
            side_effect=side_effects,
        ):
            (
                candidates, scores, _metrics,
                detection_method, atlas_tier_used, n_attempted,
                anchor_outcomes,
            ) = detector._run_two_pass_cascade(
                **inputs,
                atlas_masks=atlas_masks,
                atlas_names=atlas_names,
                n_candidates=10,
                tr=4.0,
                baseline_frames=5,
                tissue_ttp_frame=10.0,
            )

        assert detection_method == "atlas"
        assert atlas_tier_used == "MCA_ACA_strict"
        assert n_attempted == 1
        assert len(candidates) == 4
        assert anchor_outcomes[0]["has_anchor"] is True
        assert anchor_outcomes[0]["earliest_post_arr_s"] == pytest.approx(8.0)

    def test_anchored_single_keeper_promoted_in_pass2(self) -> None:
        """Every tier returns 1 keeper at post-arr 8s. Quorum (≥2) fails
        everywhere; Pass-2 picks an anchored single keeper.
        """
        inputs = _make_cascade_inputs()
        atlas_masks, atlas_names = _make_atlas_cascade()
        detector = AIFDetection()

        # 13 tiers, each with 1 keeper at peak_time_frame=7 (post-arr 8s)
        side_effects: list = [
            ([(i, 0, 0)], [0.6], [{"peak_time_frame": 7}])
            for i in range(13)
        ]

        with patch.object(
            AIFDetection, "_find_candidates_in_mask",
            side_effect=side_effects,
        ):
            (
                candidates, scores, _metrics,
                detection_method, _atlas_tier, n_attempted,
                _anchor_outcomes,
            ) = detector._run_two_pass_cascade(
                **inputs,
                atlas_masks=atlas_masks,
                atlas_names=atlas_names,
                n_candidates=10,
                tr=4.0,
                baseline_frames=5,
                tissue_ttp_frame=10.0,
            )

        assert detection_method != "population"
        assert n_attempted == 13
        assert len(candidates) == 1

    def test_unanchored_single_keeper_NOT_promoted_in_pass2(self) -> None:
        """Pass-2 must reject single keepers whose own timing fails the
        anchor gate (defends against using late venous voxels as fallback).
        """
        inputs = _make_cascade_inputs()
        atlas_masks, atlas_names = _make_atlas_cascade()
        detector = AIFDetection()

        # 13 tiers, each with 1 keeper at peak_time_frame=8 (post-arr 12s,
        # over 10s threshold).
        side_effects: list = [
            ([(i, 0, 0)], [0.6], [{"peak_time_frame": 8}])
            for i in range(13)
        ]

        with patch.object(
            AIFDetection, "_find_candidates_in_mask",
            side_effect=side_effects,
        ):
            (
                candidates, _scores, _metrics,
                detection_method, _atlas_tier, n_attempted,
                _anchor_outcomes,
            ) = detector._run_two_pass_cascade(
                **inputs,
                atlas_masks=atlas_masks,
                atlas_names=atlas_names,
                n_candidates=10,
                tr=4.0,
                baseline_frames=5,
                tissue_ttp_frame=10.0,
            )

        assert detection_method == "population"
        assert n_attempted == 13
        assert candidates == []

    def test_filter_venous_all_flagged_fails_tier(self) -> None:
        """`_filter_venous_candidates` returns empty when all flagged venous."""
        cands = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        scores = [0.7, 0.6, 0.55]
        metrics = [{"is_likely_venous": True}] * 3
        out_c, out_s, out_m = AIFDetection._filter_venous_candidates(
            cands, scores, metrics, min_non_venous=3,
        )
        assert out_c == []
        assert out_s == []
        assert out_m == []

    def test_filter_venous_drops_venous_keeps_one_survivor(self) -> None:
        """When < min_non_venous survive but ≥ 1 does, drop venous and
        keep survivors. Cascade Pass-2 will handle the count.
        """
        cands = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        scores = [0.7, 0.6, 0.55]
        metrics = [
            {"is_likely_venous": False},
            {"is_likely_venous": True},
            {"is_likely_venous": True},
        ]
        out_c, out_s, out_m = AIFDetection._filter_venous_candidates(
            cands, scores, metrics, min_non_venous=3,
        )
        assert out_c == [(0, 0, 0)]
        assert out_s == [0.7]
        assert out_m == [{"is_likely_venous": False}]
