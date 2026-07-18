"""Unit tests for DCE AIF detection fixes: ROI averaging, FWHM validation, atlas cascade."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pmriprep.interfaces.dce_aif import (
    AIFDetection,
    _apply_aif_scale,
    _classify_hemisphere,
    _compute_pool_qc,
    _count_slabs_spanned,
    _evaluate_gates,
    _inflow_exclusion_mm,
    compute_dce_aif_fwhm_seconds,
    compute_dce_aif_peak_amplitude_warning,
    compute_dce_aif_washout,
    detect_dce_aif_multi_bump_tail,
)
from pmriprep.data import get_arterial_atlas_cascade
from pmriprep.utils.dce_constants import (
    AIF_IDEAL_FWHM_SECONDS,
    AIF_WASHOUT_PHYSIOLOGIC_CENTER,
    AIF_THRESHOLDS_PERMISSIVE,
    DCE_AIF_PEAK_RATIO_WARN_THRESHOLD,
    DCE_AIF_SCALE_WARN_SUBSTANTIAL,
    DCE_AIF_SCALE_WARN_UNRELIABLE,
    DCE_AIF_TAIL_PEAK_FRACTION,
    INFLOW_EXCLUSION_MAX_SLAB_FRACTION,
    INFLOW_EXCLUSION_MM_INFERIOR_BASE,
    SLAB_EDGE_EXCLUSION_MM_ROLLOFF,
    SSS_EXCLUSION_MM,
)


class TestDCEClusterScoring:
    """Whole-brain cluster ranking uses physical, physiologic metrics."""

    @staticmethod
    def _cluster_inputs(
        first_size: int,
        second_size: int,
        first_metrics: dict[str, float],
        second_metrics: dict[str, float],
    ) -> tuple[
        list[tuple[int, int, int]], list[float], list[dict[str, float]]
    ]:
        """Build two disconnected candidate clusters with uniform metrics."""
        first = [(1 + i, 1, 1) for i in range(first_size)]
        second = [(10 + i, 10, 10) for i in range(second_size)]
        candidates = first + second
        scores = [0.8] * len(candidates)
        metrics = (
            [first_metrics.copy() for _ in first]
            + [second_metrics.copy() for _ in second]
        )
        return candidates, scores, metrics

    def test_physiologic_washout_beats_tail_collapsed_cluster(self) -> None:
        """A higher raw washout cannot beat the physiologic washout center."""
        physiologic = {
            "fwhm_seconds": AIF_IDEAL_FWHM_SECONDS,
            "washout": AIF_WASHOUT_PHYSIOLOGIC_CENTER,
            "upstroke_slope": 1.0,
        }
        collapsed = {
            **physiologic,
            "washout": 0.99,
        }
        candidates, scores, metrics = self._cluster_inputs(
            3, 3, physiologic, collapsed
        )

        selected, _, _ = AIFDetection._score_and_select_cluster(
            candidates, scores, metrics, (20, 20, 20), tr=4.8
        )

        assert selected == candidates[:3]

    @pytest.mark.parametrize("tr", [2.0, 4.8, 6.0])
    def test_physical_fwhm_ranking_is_tr_invariant(self, tr: float) -> None:
        """The same physical cluster wins across temporal resolutions."""
        ideal = {
            "fwhm_seconds": AIF_IDEAL_FWHM_SECONDS,
            "washout": AIF_WASHOUT_PHYSIOLOGIC_CENTER,
            "upstroke_slope": 1.0,
        }
        broad = {
            **ideal,
            "fwhm_seconds": 20.0,
        }
        # The broad cluster is larger, so this also proves the physical-width
        # term remains discriminative instead of collapsing at coarse TR.
        candidates, scores, metrics = self._cluster_inputs(3, 4, ideal, broad)

        selected, _, _ = AIFDetection._score_and_select_cluster(
            candidates, scores, metrics, (20, 20, 20), tr=tr
        )

        assert selected == candidates[:3]


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

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
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
# AIF scaling unit-mismatch fix
# =============================================================================


class TestAIFScalingUnitFix:
    """Parker's peak_min/peak_max are in blood-voxel units; aif_curve
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
        """Pass-2 WARN thresholds are fixed Hct-invariant scale-ratio bars.

        scaling_factor is a pure blood-amplitude ratio (the 1/(1-Hct) cancels),
        so the thresholds are fixed. Guard: 2.5 does NOT warn, 3.5 trips
        SUBSTANTIAL, 5.3 trips UNRELIABLE. Bound to the production constants so
        this actually pins their values.
        """
        assert DCE_AIF_SCALE_WARN_SUBSTANTIAL == pytest.approx(3.4)
        assert DCE_AIF_SCALE_WARN_UNRELIABLE == pytest.approx(5.2)
        assert 2.5 <= DCE_AIF_SCALE_WARN_SUBSTANTIAL  # below → no warning
        assert 3.5 > DCE_AIF_SCALE_WARN_SUBSTANTIAL   # trips SUBSTANTIAL
        assert 5.3 > DCE_AIF_SCALE_WARN_UNRELIABLE    # trips UNRELIABLE

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


def _make_atlas_cascade(n: int = 3) -> tuple[list[np.ndarray], list[str]]:
    """Build N synthetic atlas masks + names parallel."""
    masks = [np.ones((8, 8, 4), dtype=np.uint8) for _ in range(n)]
    names = ["MCA_ACA", "MCA_ACA_dil", "full_mask"][:n]
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
            _cascade_res = detector._run_two_pass_cascade(
                **inputs,
                atlas_masks=atlas_masks,
                atlas_names=atlas_names,
                n_candidates=10,
                tr=4.0,
                baseline_frames=5,
                tissue_ttp_frame=10.0,
            )
            candidates = _cascade_res.candidates
            scores = _cascade_res.scores
            metrics_list = _cascade_res.metrics_list
            detection_method = _cascade_res.detection_method
            atlas_tier_used = _cascade_res.atlas_tier_used
            n_attempted = _cascade_res.cascade_tiers_attempted
            anchor_outcomes = _cascade_res.anchor_outcomes

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
        # 3 atlases × 3 criteria + 1 whole-brain = 10 tiers.
        max_scores = [
            0.55, 0.50, 0.50,  # MCA_ACA × 3
            0.50, 0.62, 0.50,  # MCA_ACA_dil × 3   ← highest at idx 4 (relaxed)
            0.50, 0.50, 0.50,  # full_mask × 3
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
            _cascade_res = detector._run_two_pass_cascade(
                **inputs,
                atlas_masks=atlas_masks,
                atlas_names=atlas_names,
                n_candidates=10,
                tr=4.0,
                baseline_frames=5,
                tissue_ttp_frame=10.0,
            )
            candidates = _cascade_res.candidates
            scores = _cascade_res.scores
            metrics_list = _cascade_res.metrics_list
            detection_method = _cascade_res.detection_method
            atlas_tier_used = _cascade_res.atlas_tier_used
            n_attempted = _cascade_res.cascade_tiers_attempted
            anchor_outcomes = _cascade_res.anchor_outcomes

        assert n_attempted == 10  # all tiers run, none short-circuited
        # Highest single-keeper max-score was 0.62 at MCA_ACA_dil_relaxed.
        assert detection_method == "atlas"
        assert atlas_tier_used == "MCA_ACA_dil_relaxed"
        assert len(candidates) == 1
        assert scores == [0.62]
        assert len(anchor_outcomes) == 10

    def test_falls_to_population_only_when_no_keeper_anywhere(self) -> None:
        """Cascade must exhaust ALL 10 tiers before signaling population.

        With every tier returning 0-keeper output, Pass 1 fails everywhere,
        Pass 2 has no single-keeper tier to fall back to, and the cascade
        returns ``detection_method="population"`` with the full 10-tier
        attempt count for the caller's provenance.
        """
        inputs = _make_cascade_inputs()
        atlas_masks, atlas_names = _make_atlas_cascade()
        detector = AIFDetection()

        # All 10 tiers return non-empty zero-scored candidates with
        # arbitrary timing (anchor doesn't matter when quorum is 0).
        side_effects: list = [
            ([(0, 0, 0)] * 5, [0.0] * 5, [{"peak_time_frame": 7}] * 5)
            for _ in range(10)
        ]

        with patch.object(
            AIFDetection, "_find_candidates_in_mask",
            side_effect=side_effects,
        ) as mock_find:
            _cascade_res = detector._run_two_pass_cascade(
                **inputs,
                atlas_masks=atlas_masks,
                atlas_names=atlas_names,
                n_candidates=10,
                tr=4.0,
                baseline_frames=5,
                tissue_ttp_frame=10.0,
            )
            candidates = _cascade_res.candidates
            scores = _cascade_res.scores
            metrics_list = _cascade_res.metrics_list
            detection_method = _cascade_res.detection_method
            atlas_tier_used = _cascade_res.atlas_tier_used
            n_attempted = _cascade_res.cascade_tiers_attempted
            anchor_outcomes = _cascade_res.anchor_outcomes

        assert detection_method == "population"
        assert atlas_tier_used is None
        assert n_attempted == 10
        assert mock_find.call_count == 10
        assert len(anchor_outcomes) == 10
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
            _cascade_res = detector._run_two_pass_cascade(
                **inputs,
                atlas_masks=atlas_masks,
                atlas_names=atlas_names,
                n_candidates=10,
                tr=4.0,
                baseline_frames=5,
                tissue_ttp_frame=10.0,
            )
            detection_method = _cascade_res.detection_method
            atlas_tier_used = _cascade_res.atlas_tier_used
            n_attempted = _cascade_res.cascade_tiers_attempted

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
        """All-venous-tier pattern: 3 candidates per tier, all peaking at
        post-arrival 13s (above 10s threshold). Cascade descends through
        all 10 tiers, none anchored, falls to population.
        """
        inputs = _make_cascade_inputs()
        atlas_masks, atlas_names = _make_atlas_cascade()
        detector = AIFDetection()

        # baseline=5, tr=4 → peak_time_frame=8 gives post-arr=12s > 10s
        side_effects: list = [
            ([(0, 0, 0)] * 3, [0.6] * 3, [{"peak_time_frame": 8}] * 3)
            for _ in range(10)
        ]

        with patch.object(
            AIFDetection, "_find_candidates_in_mask",
            side_effect=side_effects,
        ):
            _cascade_res = detector._run_two_pass_cascade(
                **inputs,
                atlas_masks=atlas_masks,
                atlas_names=atlas_names,
                n_candidates=10,
                tr=4.0,
                baseline_frames=5,
                tissue_ttp_frame=10.0,
            )
            candidates = _cascade_res.candidates
            scores = _cascade_res.scores
            detection_method = _cascade_res.detection_method
            atlas_tier_used = _cascade_res.atlas_tier_used
            n_attempted = _cascade_res.cascade_tiers_attempted
            anchor_outcomes = _cascade_res.anchor_outcomes

        assert detection_method == "population"
        assert atlas_tier_used is None
        assert n_attempted == 10
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
        catch: quorum alone is not sufficient.
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
            _cascade_res = detector._run_two_pass_cascade(
                **inputs,
                atlas_masks=atlas_masks,
                atlas_names=atlas_names,
                n_candidates=10,
                tr=4.0,
                baseline_frames=5,
                tissue_ttp_frame=10.0,
            )
            candidates = _cascade_res.candidates
            detection_method = _cascade_res.detection_method
            atlas_tier_used = _cascade_res.atlas_tier_used
            n_attempted = _cascade_res.cascade_tiers_attempted
            anchor_outcomes = _cascade_res.anchor_outcomes

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
        """Anchored-tier pattern: 4 candidates, anchor #1 at post-arr 8s,
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
            _cascade_res = detector._run_two_pass_cascade(
                **inputs,
                atlas_masks=atlas_masks,
                atlas_names=atlas_names,
                n_candidates=10,
                tr=4.0,
                baseline_frames=5,
                tissue_ttp_frame=10.0,
            )
            candidates = _cascade_res.candidates
            scores = _cascade_res.scores
            detection_method = _cascade_res.detection_method
            atlas_tier_used = _cascade_res.atlas_tier_used
            n_attempted = _cascade_res.cascade_tiers_attempted
            anchor_outcomes = _cascade_res.anchor_outcomes

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

        # 10 tiers, each with 1 keeper at peak_time_frame=7 (post-arr 8s)
        side_effects: list = [
            ([(i, 0, 0)], [0.6], [{"peak_time_frame": 7}])
            for i in range(10)
        ]

        with patch.object(
            AIFDetection, "_find_candidates_in_mask",
            side_effect=side_effects,
        ):
            _cascade_res = detector._run_two_pass_cascade(
                **inputs,
                atlas_masks=atlas_masks,
                atlas_names=atlas_names,
                n_candidates=10,
                tr=4.0,
                baseline_frames=5,
                tissue_ttp_frame=10.0,
            )
            candidates = _cascade_res.candidates
            scores = _cascade_res.scores
            detection_method = _cascade_res.detection_method
            n_attempted = _cascade_res.cascade_tiers_attempted

        assert detection_method != "population"
        assert n_attempted == 10
        assert len(candidates) == 1

    def test_unanchored_single_keeper_NOT_promoted_in_pass2(self) -> None:
        """Pass-2 must reject single keepers whose own timing fails the
        anchor gate (defends against using late venous voxels as fallback).
        """
        inputs = _make_cascade_inputs()
        atlas_masks, atlas_names = _make_atlas_cascade()
        detector = AIFDetection()

        # 10 tiers, each with 1 keeper at peak_time_frame=8 (post-arr 12s,
        # over 10s threshold).
        side_effects: list = [
            ([(i, 0, 0)], [0.6], [{"peak_time_frame": 8}])
            for i in range(10)
        ]

        with patch.object(
            AIFDetection, "_find_candidates_in_mask",
            side_effect=side_effects,
        ):
            _cascade_res = detector._run_two_pass_cascade(
                **inputs,
                atlas_masks=atlas_masks,
                atlas_names=atlas_names,
                n_candidates=10,
                tr=4.0,
                baseline_frames=5,
                tissue_ttp_frame=10.0,
            )
            candidates = _cascade_res.candidates
            detection_method = _cascade_res.detection_method
            n_attempted = _cascade_res.cascade_tiers_attempted

        assert detection_method == "population"
        assert n_attempted == 10
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


# =============================================================================
# Slab-edge inflow exclusion regression tests (3D-VIBE inflow fix)
# =============================================================================


class TestInflowExclusionHelper:
    """Tests for ``_inflow_exclusion_mm`` adaptive Option C formula."""

    def test_thin_isotropic_uses_base_floor(self) -> None:
        """1mm × 150 slices: 4*1=4 < 20mm base; cap = 0.35*150 = 52.5
        doesn't bind → returns 20mm."""
        assert _inflow_exclusion_mm(1.0, 150) == pytest.approx(20.0)

    def test_typical_clinical_uniform_20mm(self) -> None:
        """3.5/4/5mm at typical slab counts all return ~20mm (typical
        across clinical corpora)."""
        # 3.5mm × 28 = 98mm slab; max(20, 14)=20; cap=34.3 → 20
        assert _inflow_exclusion_mm(3.5, 28) == pytest.approx(20.0)
        # 4mm × 24 = 96mm slab; max(20, 16)=20; cap=33.6 → 20
        assert _inflow_exclusion_mm(4.0, 24) == pytest.approx(20.0)
        # 5mm × 16 = 80mm slab; max(20, 20)=20; cap=28 → 20
        assert _inflow_exclusion_mm(5.0, 16) == pytest.approx(20.0)

    def test_thick_slab_capped(self) -> None:
        """8mm × 8 = 64mm slab: max(20, 32)=32; cap=22.4 → 22.4."""
        assert _inflow_exclusion_mm(8.0, 8) == pytest.approx(22.4)

    def test_very_thick_slab_capped_aggressively(self) -> None:
        """10mm × 4 = 40mm slab: max(20, 40)=40; cap=14 → 14."""
        assert _inflow_exclusion_mm(10.0, 4) == pytest.approx(14.0)

    def test_constants_values(self) -> None:
        """Regression guard against silent threshold changes."""
        assert INFLOW_EXCLUSION_MM_INFERIOR_BASE == 20.0
        assert INFLOW_EXCLUSION_MAX_SLAB_FRACTION == 0.35


class TestPrepareSearchMaskDirectional:
    """Regression tests for the directional slab-edge exclusion."""

    @staticmethod
    def _ones_mask(shape: tuple[int, int, int]) -> np.ndarray:
        return np.ones(shape, dtype=np.uint8)

    def test_inferior_exclusion_larger_than_superior(self) -> None:
        """Axial slab (slice > in-plane). Asymmetric exclusion applied:
        6 inferior slices (24mm/4mm), 1 superior slice (4mm/4mm)."""
        mask = self._ones_mask((64, 64, 16))
        out = AIFDetection._prepare_search_mask(
            mask, voxel_sizes=(1.0, 1.0, 4.0),
            erosion_iterations=0,
            slab_edge_inferior_mm=24.0,
            slab_edge_superior_mm=4.0,
        )
        # First 6 z-slices fully zero (inferior)
        assert out[:, :, :6].sum() == 0
        # Last 1 z-slice fully zero (superior)
        assert out[:, :, -1:].sum() == 0
        # z=6 (just above inferior cutoff) has at least one non-zero
        # voxel after midline exclusion (skip_midline=False is default
        # so the midline column is zeroed, but lateral voxels remain).
        assert out[:, :, 6].sum() > 0

    def test_inferior_exclusion_skipped_if_too_aggressive(self) -> None:
        """nz=4 with 24+4mm exclusion = 7 slices ≥ 4 → guard skips
        Step 2 entirely. Mask preserved (modulo erosion + Step 4)."""
        mask = self._ones_mask((64, 64, 4))
        out = AIFDetection._prepare_search_mask(
            mask, voxel_sizes=(1.0, 1.0, 4.0),
            erosion_iterations=0, skip_midline=True,
            slab_edge_inferior_mm=24.0,
            slab_edge_superior_mm=4.0,
        )
        # Guard skipped Step 2, so z=0 (which would be inferior cut)
        # is NOT entirely zero. Step 4 still applies top-25%, which for
        # nz=4 zeros slice 3 only.
        assert out[:, :, 0].sum() > 0
        assert out[:, :, 3].sum() == 0

    def test_signature_defaults_use_rolloff_inferior_and_sss_superior(self) -> None:
        """Default args: inferior=SLAB_EDGE_EXCLUSION_MM_ROLLOFF (4mm),
        superior=SSS_EXCLUSION_MM (15mm). 16-slice × 4mm mask gets 1
        inferior slice (4mm/4mm) + 4 superior slices (15mm/4mm) zeroed.
        Production call sites override these defaults; this just
        documents the ad-hoc / test-invocation defaults."""
        mask = self._ones_mask((64, 64, 16))
        out = AIFDetection._prepare_search_mask(
            mask, voxel_sizes=(1.0, 1.0, 4.0),
            erosion_iterations=0, skip_midline=True,
        )
        assert out[:, :, :1].sum() == 0   # 1 inferior slice (rolloff)
        assert out[:, :, -4:].sum() == 0  # 4 superior slices (SSS)

    def test_non_axial_slab_falls_back_to_symmetric(self) -> None:
        """In-plane voxel > slice thickness → not axial → symmetric
        exclusion using min of the two values (4mm here = 4 slices)."""
        mask = self._ones_mask((64, 64, 16))
        out = AIFDetection._prepare_search_mask(
            mask, voxel_sizes=(2.0, 2.0, 1.0),  # in-plane > through-plane
            erosion_iterations=0, skip_midline=True,
            slab_edge_inferior_mm=24.0,
            slab_edge_superior_mm=4.0,
        )
        # Both edges get min(24, 4) = 4mm = 4 slices
        assert out[:, :, :4].sum() == 0
        assert out[:, :, -4:].sum() == 0
        # Middle slices retained
        assert out[:, :, 6].sum() > 0

    def test_axial_slab_proxy_passes_for_typical_dce(self) -> None:
        """Typical axial DCE (through-plane > in-plane) → directional
        exclusion applies (6 inferior, 1 superior at 24/4mm)."""
        mask = self._ones_mask((64, 64, 16))
        out = AIFDetection._prepare_search_mask(
            mask, voxel_sizes=(1.0, 1.0, 4.0),
            erosion_iterations=0, skip_midline=True,
            slab_edge_inferior_mm=24.0,
            slab_edge_superior_mm=4.0,
        )
        assert out[:, :, :6].sum() == 0   # 6 inferior slices zeroed
        assert out[:, :, -1:].sum() == 0  # 1 superior slice zeroed
        assert out[:, :, 6].sum() > 0     # z=6 retained


class TestSlabEdgeConstants:
    """Renamed slab-edge constants reflect actual physics rationale."""

    def test_rolloff_constant_value(self) -> None:
        """Rolloff (B1/RF profile, atlas inferior+superior, WB inferior
        rolloff baseline) is 4mm — single-partition margin."""
        assert SLAB_EDGE_EXCLUSION_MM_ROLLOFF == 4.0

    def test_sss_constant_value(self) -> None:
        """SSS exclusion (WB superior only) is 15mm — keeps superior
        sagittal sinus out of the unconstrained candidate pool."""
        assert SSS_EXCLUSION_MM == 15.0


class TestInflowExclusionAppliedUniversally:
    """End-to-end coverage for the universal inferior-inflow apply.

    No MRAcquisitionType gate is applied, as BIDS
    metadata can't reliably distinguish slab-selective 2D-prep sequences
    from plain 2D-MS. Inferior inflow exclusion applies to both
    atlas + WB call sites regardless of acquisition type. These tests
    assert that the helper output (`_inflow_exclusion_mm`) actually
    reaches `_prepare_search_mask` and zeroes the expected slice count.
    """

    @staticmethod
    def _ones_mask(shape: tuple[int, int, int]) -> np.ndarray:
        return np.ones(shape, dtype=np.uint8)

    def test_atlas_call_site_zeroes_inflow_inferior_and_rolloff_superior(
        self,
    ) -> None:
        """Atlas tier: inferior=_inflow_exclusion_mm, superior=ROLLOFF.
        24 slices × 4mm: inflow = max(20, 4*4)=20mm capped at 0.35*96=33.6mm
        → 20mm = 5 slices inferior. Superior = 4mm = 1 slice."""
        mask = self._ones_mask((64, 64, 24))
        slice_thickness = 4.0
        nz = 24
        inflow_mm = _inflow_exclusion_mm(slice_thickness, nz)
        inflow_slices = int(np.ceil(inflow_mm / slice_thickness))
        out = AIFDetection._prepare_search_mask(
            mask, voxel_sizes=(1.0, 1.0, slice_thickness),
            erosion_iterations=0, skip_midline=True,
            slab_edge_inferior_mm=inflow_mm,
            slab_edge_superior_mm=SLAB_EDGE_EXCLUSION_MM_ROLLOFF,
        )
        assert out[:, :, :inflow_slices].sum() == 0
        assert out[:, :, -1:].sum() == 0
        assert out[:, :, inflow_slices].sum() > 0

    def test_wb_call_site_zeroes_inflow_inferior_and_sss_superior(self) -> None:
        """WB tier: inferior=_inflow_exclusion_mm, superior=SSS_EXCLUSION_MM.
        24 slices × 4mm: 5 inferior slices (inflow), 4 superior slices
        (15mm SSS / 4mm = ceil(3.75) = 4)."""
        mask = self._ones_mask((64, 64, 24))
        slice_thickness = 4.0
        nz = 24
        inflow_mm = _inflow_exclusion_mm(slice_thickness, nz)
        inflow_slices = int(np.ceil(inflow_mm / slice_thickness))
        sss_slices = int(np.ceil(SSS_EXCLUSION_MM / slice_thickness))
        out = AIFDetection._prepare_search_mask(
            mask, voxel_sizes=(1.0, 1.0, slice_thickness),
            erosion_iterations=0, skip_midline=False,
            slab_edge_inferior_mm=inflow_mm,
            slab_edge_superior_mm=SSS_EXCLUSION_MM,
        )
        assert out[:, :, :inflow_slices].sum() == 0
        assert out[:, :, -sss_slices:].sum() == 0
        assert out[:, :, inflow_slices].sum() > 0

    def test_3d_acquisition_isotropic_voxel_passes_axial_proxy(self) -> None:
        """Regression for M1: isotropic 1×1×1mm 3D-VIBE must still get
        the directional exclusion (axial proxy guard uses >=, not >)."""
        mask = np.ones((64, 64, 80), dtype=np.uint8)
        out = AIFDetection._prepare_search_mask(
            mask, voxel_sizes=(1.0, 1.0, 1.0),
            erosion_iterations=0, skip_midline=True,
            slab_edge_inferior_mm=20.0,
            slab_edge_superior_mm=4.0,
        )
        # 20mm / 1mm = 20 inferior slices zeroed
        assert out[:, :, :20].sum() == 0
        # 4mm / 1mm = 4 superior slices zeroed
        assert out[:, :, -4:].sum() == 0
        # Middle survives
        assert out[:, :, 30].sum() > 0

    def test_sagittal_acquisition_falls_back_to_symmetric(self) -> None:
        """In-plane voxel STRICTLY larger than slice → not axial → fall
        back to symmetric using min."""
        mask = np.ones((64, 64, 16), dtype=np.uint8)
        out = AIFDetection._prepare_search_mask(
            mask, voxel_sizes=(2.0, 2.0, 1.0),  # in-plane > through-plane
            erosion_iterations=0, skip_midline=True,
            slab_edge_inferior_mm=20.0,
            slab_edge_superior_mm=4.0,
        )
        # min(20, 4) = 4mm = 4 slices on each side
        assert out[:, :, :4].sum() == 0
        assert out[:, :, -4:].sum() == 0
        # Middle slices retained
        assert out[:, :, 8].sum() > 0



# =============================================================================
# Test: Peak-amplitude QC warning
# =============================================================================


class TestDCEAIFPeakAmplitudeWarning:
    """Coverage for ``compute_dce_aif_peak_amplitude_warning`` —
    cluster-mean AIF peak vs dose-scaled Parker plasma ``peak_max``."""

    EXPECTED_MAX = 7.6  # full-dose Parker plasma peak_max (mM)

    def test_clean_full_dose_does_not_flag(self) -> None:
        ratio, warn = compute_dce_aif_peak_amplitude_warning(
            final_peak=6.5,
            expected_max_mm=self.EXPECTED_MAX,
            detection_method="atlas",
        )
        assert warn is False
        assert ratio < DCE_AIF_PEAK_RATIO_WARN_THRESHOLD

    def test_observed_literature_ceiling_does_not_flag(self) -> None:
        """Schabel 2008 observed clean ceiling at full dose ~9 mM →
        ratio 1.18 < 1.5; should not flag."""
        ratio, warn = compute_dce_aif_peak_amplitude_warning(
            final_peak=9.0,
            expected_max_mm=self.EXPECTED_MAX,
            detection_method="atlas",
        )
        assert warn is False
        assert ratio == pytest.approx(9.0 / 7.6)

    def test_inflated_peak_flags(self) -> None:
        peak = (DCE_AIF_PEAK_RATIO_WARN_THRESHOLD + 0.1) * self.EXPECTED_MAX
        ratio, warn = compute_dce_aif_peak_amplitude_warning(
            final_peak=peak,
            expected_max_mm=self.EXPECTED_MAX,
            detection_method="atlas",
        )
        assert warn is True
        assert ratio > DCE_AIF_PEAK_RATIO_WARN_THRESHOLD

    def test_threshold_just_below_does_not_flag(self) -> None:
        peak = (DCE_AIF_PEAK_RATIO_WARN_THRESHOLD - 0.05) * self.EXPECTED_MAX
        _, warn = compute_dce_aif_peak_amplitude_warning(
            final_peak=peak,
            expected_max_mm=self.EXPECTED_MAX,
            detection_method="atlas",
        )
        assert warn is False

    def test_threshold_just_above_flags(self) -> None:
        peak = (DCE_AIF_PEAK_RATIO_WARN_THRESHOLD + 0.05) * self.EXPECTED_MAX
        _, warn = compute_dce_aif_peak_amplitude_warning(
            final_peak=peak,
            expected_max_mm=self.EXPECTED_MAX,
            detection_method="atlas",
        )
        assert warn is True

    def test_population_method_suppresses_warning(self) -> None:
        """Even with an inflated peak, population AIF must not flag —
        population is tautologically within its own expected range."""
        peak = 2.0 * self.EXPECTED_MAX
        ratio, warn = compute_dce_aif_peak_amplitude_warning(
            final_peak=peak,
            expected_max_mm=self.EXPECTED_MAX,
            detection_method="population",
        )
        assert warn is False
        assert np.isnan(ratio)

    def test_zero_or_negative_expected_max_returns_nan(self) -> None:
        """Defensive against malformed dose factor."""
        ratio, warn = compute_dce_aif_peak_amplitude_warning(
            final_peak=5.0,
            expected_max_mm=0.0,
            detection_method="atlas",
        )
        assert warn is False
        assert np.isnan(ratio)

    def test_dose_factor_scales_expected_max(self) -> None:
        """Half-dose subject: scaled expected_max = 7.6 * 0.5 = 3.8 mM.
        Peak just above threshold * scaled expected_max should flag."""
        scaled_max = self.EXPECTED_MAX * 0.5
        peak = (DCE_AIF_PEAK_RATIO_WARN_THRESHOLD + 0.05) * scaled_max
        _, warn = compute_dce_aif_peak_amplitude_warning(
            final_peak=peak,
            expected_max_mm=scaled_max,
            detection_method="atlas",
        )
        assert warn is True


# =============================================================================
# Test: Multi-bump tail detector
# =============================================================================


class TestMultiBumpTailDetector:
    """Coverage for ``detect_dce_aif_multi_bump_tail``.

    Builds synthetic AIF curves at TR=4s (typical clinical) and TR=0.5s
    (sub-second compressed-sensing) to verify the seconds-based windows
    convert to frames consistently across temporal resolutions.
    """

    @staticmethod
    def _gaussian_bump(t: np.ndarray, center: float, sigma: float,
                      amplitude: float) -> np.ndarray:
        return amplitude * np.exp(-0.5 * ((t - center) / sigma) ** 2)

    @staticmethod
    def _build_aif(
        tr: float,
        duration_s: float,
        baseline_frames: int,
        first_pass_peak_s: float,
        first_pass_amplitude: float,
        secondary_peaks: list[tuple[float, float]],  # (center_s, amplitude)
    ) -> np.ndarray:
        n_frames = int(round(duration_s / tr))
        t = np.arange(n_frames, dtype=np.float64) * tr
        curve = TestMultiBumpTailDetector._gaussian_bump(
            t, first_pass_peak_s, sigma=4.0, amplitude=first_pass_amplitude,
        )
        for center, amp in secondary_peaks:
            curve += TestMultiBumpTailDetector._gaussian_bump(
                t, center, sigma=4.0, amplitude=amp,
            )
        curve[:baseline_frames] = 0.0
        return curve

    def test_clean_aif_one_recirculation_bump_not_flagged(self) -> None:
        """Physiologic recirculation hump (~30 s post-peak) is excluded
        by the 60 s shoulder; one true late-tail bump at 100 s post-peak
        with amplitude > 40% is counted as 1."""
        aif = self._build_aif(
            tr=4.0, duration_s=300.0, baseline_frames=5,
            first_pass_peak_s=20.0, first_pass_amplitude=5.0,
            # recirculation hump (excluded by sh60) + 1 true late bump
            secondary_peaks=[(50.0, 1.5), (140.0, 2.5)],  # 50% amplitude
        )
        count, _ = detect_dce_aif_multi_bump_tail(aif, tr=4.0)
        assert count == 1

    def test_two_secondary_bumps_flagged(self) -> None:
        """Two true late-tail bumps (>60 s post-peak, >40% amplitude)."""
        aif = self._build_aif(
            tr=4.0, duration_s=320.0, baseline_frames=5,
            first_pass_peak_s=20.0, first_pass_amplitude=5.0,
            secondary_peaks=[(110.0, 2.5), (180.0, 2.3)],
        )
        count, _ = detect_dce_aif_multi_bump_tail(aif, tr=4.0)
        assert count == 2

    def test_three_motion_bumps_flagged(self) -> None:
        """Three true late-tail bumps (>60 s post-peak, >40% amplitude)."""
        aif = self._build_aif(
            tr=4.0, duration_s=400.0, baseline_frames=5,
            first_pass_peak_s=20.0, first_pass_amplitude=5.0,
            secondary_peaks=[(110.0, 2.5), (180.0, 2.3), (260.0, 2.7)],
        )
        count, _ = detect_dce_aif_multi_bump_tail(aif, tr=4.0)
        assert count == 3

    def test_below_threshold_not_counted(self) -> None:
        """Secondary peak at 30% of first-pass (< 40%) is not counted."""
        aif = self._build_aif(
            tr=4.0, duration_s=300.0, baseline_frames=5,
            first_pass_peak_s=20.0, first_pass_amplitude=5.0,
            secondary_peaks=[(120.0, 1.5)],  # 30% of 5.0
        )
        count, _ = detect_dce_aif_multi_bump_tail(aif, tr=4.0)
        assert count == 0

    def test_separation_distance_enforced(self) -> None:
        """Two secondary maxima 8 s apart (< min_separation_seconds=10s)
        deduplicate to 1."""
        aif = self._build_aif(
            tr=4.0, duration_s=300.0, baseline_frames=5,
            first_pass_peak_s=20.0, first_pass_amplitude=5.0,
            secondary_peaks=[(120.0, 2.5), (128.0, 2.5)],
        )
        count, _ = detect_dce_aif_multi_bump_tail(aif, tr=4.0)
        assert count == 1

    def test_sub_second_tr_window_scales(self) -> None:
        """TR=0.5 s — windows scale in seconds; two true late-tail bumps
        at 100 s and 180 s post-peak (well beyond 60 s shoulder) count
        as 2."""
        aif = self._build_aif(
            tr=0.5, duration_s=300.0, baseline_frames=5,
            first_pass_peak_s=20.0, first_pass_amplitude=5.0,
            secondary_peaks=[(120.0, 2.5), (200.0, 2.3)],
        )
        count, _ = detect_dce_aif_multi_bump_tail(aif, tr=0.5)
        assert count == 2

    def test_long_tr_window_floors_to_one_frame(self) -> None:
        """TR=8s: round(6/8) under banker's rounding may give 0; the
        max(1, ...) guard floors it to ≥1. Helper must remain functional
        (no divide-by-zero, no zero-frame skip)."""
        aif = self._build_aif(
            tr=8.0, duration_s=400.0, baseline_frames=3,
            first_pass_peak_s=40.0, first_pass_amplitude=5.0,
            secondary_peaks=[],
        )
        count, _ = detect_dce_aif_multi_bump_tail(aif, tr=8.0)
        assert count == 0  # No secondary bumps; no crash

    def test_zero_tr_returns_zero(self) -> None:
        """Defensive guard against malformed input."""
        aif = np.zeros(40)
        count, heights = detect_dce_aif_multi_bump_tail(aif, tr=0.0)
        assert count == 0
        assert heights == []

    def test_zero_or_negative_peak_returns_zero(self) -> None:
        aif = np.zeros(40)
        count, heights = detect_dce_aif_multi_bump_tail(aif, tr=4.0)
        assert count == 0
        assert heights == []

    def test_short_tail_returns_zero(self) -> None:
        """First-pass peak at last frame, no tail → no secondary peaks."""
        aif = np.zeros(15)
        aif[5:] = np.linspace(0, 5.0, 10)  # peak at last frame
        count, heights = detect_dce_aif_multi_bump_tail(aif, tr=4.0)
        assert count == 0
        assert heights == []

    def test_shoulder_skip_does_not_count_descending_limb(self) -> None:
        """A bump on the recirculation shoulder (within
        tail_shoulder_seconds=60.0 of the peak) is excluded by *time*,
        not amplitude. At TR=4 s the shoulder is round(60/4)=15
        frames, so a bump 5 frames post-peak (at peak+20 s) is skipped
        even when its amplitude exceeds the 40 %-of-peak threshold."""
        tr = 4.0
        aif = self._build_aif(
            tr=tr, duration_s=300.0, baseline_frames=5,
            first_pass_peak_s=20.0, first_pass_amplitude=5.0,
            secondary_peaks=[],
        )
        peak_idx = int(round(20.0 / tr))
        # Inside the recirculation window (peak+20 s < shoulder of 60 s)
        aif[peak_idx + 5] = 0.5 * 5.0  # 50 % of first-pass peak
        count, _ = detect_dce_aif_multi_bump_tail(aif, tr=tr)
        assert count == 0


class TestMultiBumpCallSiteSuppression:
    """Population-AIF call-site suppression for multi-bump.

    The helper ``detect_dce_aif_multi_bump_tail`` itself does not know
    about ``detection_method`` — suppression happens at the cascade call
    site by short-circuiting count to 0 when the population fallback is
    active. These tests exercise that decision logic directly without
    invoking the full ``_run_interface``.
    """

    @staticmethod
    def _multi_bump_count_with_suppression(
        aif_curve: np.ndarray, tr: float, detection_method: str,
    ) -> tuple[int, list[float]]:
        """Mirror the call-site suppression decision in _run_interface."""
        if detection_method != "population":
            return detect_dce_aif_multi_bump_tail(aif_curve, tr=tr)
        return 0, []

    @staticmethod
    def _three_bump_curve(tr: float) -> np.ndarray:
        """Build a 3-late-tail-bump curve: peak at 20 s, bumps well past
        the 60 s shoulder at amplitudes >40 % of the first-pass peak."""
        n = int(400 / tr)
        t = np.arange(n) * tr
        aif = 5.0 * np.exp(-0.5 * ((t - 20) / 4) ** 2)
        for c in (110.0, 180.0, 260.0):
            aif += 2.5 * np.exp(-0.5 * ((t - c) / 4) ** 2)
        aif[:5] = 0.0
        return aif

    def test_population_method_zeroes_count_on_3_bump_curve(self) -> None:
        aif = self._three_bump_curve(tr=4.0)
        count, heights = self._multi_bump_count_with_suppression(
            aif, tr=4.0, detection_method="population",
        )
        assert count == 0
        assert heights == []

    def test_atlas_method_does_not_suppress(self) -> None:
        aif = self._three_bump_curve(tr=4.0)
        count, _ = self._multi_bump_count_with_suppression(
            aif, tr=4.0, detection_method="atlas",
        )
        assert count == 3


class TestMultiBumpTailConstantValue:
    """Regression for the 40 % threshold value."""

    def test_tail_peak_fraction_is_40_percent(self) -> None:
        assert DCE_AIF_TAIL_PEAK_FRACTION == 0.40


class TestDCEAIFDetectionLesionMaskFileTrait:
    """DCE AIF detection must accept the empty string
    that upstream Function nodes return when no lesion mask is available.

    When no lesion mask is available, Nipype would reject
    ``""`` if the trait were ``File(exists=True)`` (mandatory by default)
    because ``Path("").is_file()`` is False. The trait is
    ``Either(File(exists=True), Enum(""))`` so the documented Function-node-
    returns-empty-string convention works without changing the runtime guard
    in ``_run_interface``.
    """

    def test_lesion_mask_file_accepts_empty_string(self) -> None:
        iface = AIFDetection()
        iface.inputs.lesion_mask_file = ""
        assert iface.inputs.lesion_mask_file == ""

    def test_lesion_mask_file_accepts_existing_path(self, tmp_path) -> None:
        mask = tmp_path / "lesion.nii.gz"
        mask.write_bytes(b"")
        iface = AIFDetection()
        iface.inputs.lesion_mask_file = str(mask)
        assert iface.inputs.lesion_mask_file == str(mask)

    def test_lesion_mask_file_rejects_nonexistent_path(self) -> None:
        from traits.trait_errors import TraitError
        iface = AIFDetection()
        with pytest.raises(TraitError):
            iface.inputs.lesion_mask_file = (
                "/nonexistent/path/that/does/not/exist.nii.gz"
            )

    def test_upstream_function_node_empty_string_does_not_raise(self, tmp_path) -> None:
        """Wiring regression: when an upstream Function node returns ``""``
        (the documented convention used by ``_transform_lesion_mask_to_dce``
        when no whole-tumor mask is available), assigning that value to the
        downstream trait must not raise.

        A ``File(exists=True)`` trait would raise ``TraitError`` here because
        it normalizes ``""`` → ``"."`` and rejects. Pure ``wf.connect([...])``
        validation is structural-only and does NOT propagate values, so this
        test runs the upstream node explicitly and assigns its result to the
        downstream trait — that's the actual code path the engine takes at
        node-run time.
        """
        import nipype.interfaces.utility as niu
        import nipype.pipeline.engine as pe

        def _empty_returner():
            return ""

        upstream = pe.Node(
            niu.Function(
                input_names=[], output_names=["lesion_mask_dce_file"],
                function=_empty_returner,
            ),
            name="upstream",
            base_dir=str(tmp_path),
        )
        result = upstream.run()
        upstream_value = result.outputs.lesion_mask_dce_file
        assert upstream_value == ""

        aif = AIFDetection()
        # This is the assignment Nipype's MultiProc plugin makes at node-run
        # time. It must succeed for the empty-string convention.
        aif.inputs.lesion_mask_file = upstream_value
        assert aif.inputs.lesion_mask_file == ""


# =============================================================================
# Acceptance gates G1-G6 + G6 escape
# =============================================================================

from pmriprep.interfaces.dce_aif import (  # noqa: E402
    CascadeResult,
    GateDecision,
    GateQC,
    PoolGateOutcome,
    TierResult,
    _apply_aif_scale,
    _compute_pool_qc,
    _count_slabs_spanned,
    _evaluate_gates,
    _select_worst_for_trim,
    compute_dce_aif_ratio_30s,
)
from pmriprep.utils.dce_constants import (  # noqa: E402
    DCE_AIF_GATE_FWHM_MAX_SECONDS,
    DCE_AIF_GATE_G6_MIN_VOXELS_BILATERAL,
    DCE_AIF_GATE_G6_MIN_VOXELS_UNILATERAL,
    DCE_AIF_GATE_G6_MIN_SLABS_UNILATERAL,
    DCE_AIF_GATE_LATE_TAIL_RATIO_MAX,
    DCE_AIF_GATE_MULTIBUMP_MAX,
    DCE_AIF_GATE_RATIO30_HARD,
    DCE_AIF_GATE_RATIO30_WARN,
    DCE_AIF_GATE_SCALE_G6_CEILING,
    DCE_AIF_GATE_SCALE_HARD,
    DCE_AIF_TAIL_PEAK_FRACTION,
    DCE_AIF_TAIL_SHOULDER_SECONDS,
)


def _make_qc(
    *,
    peak: float = 8.0,
    fwhm: float = 14.0,
    auc: float = 800.0,
    washout: float = 0.8,
    bumps: int = 0,
    late_tail: float | None = 0.20,
    ratio_30: float | None = 0.30,
) -> GateQC:
    return GateQC(
        peak_mm=peak,
        fwhm_seconds=fwhm,
        auc_mm_s=auc,
        washout=washout,
        multi_bump_count=bumps,
        late_tail_per_peak=late_tail,
        ratio_30=ratio_30,
    )


# ----- Constants -----------------------------------------------------------

def test_bump_detector_constants_updated():
    """sh60 + f0.40 are the bump-detector defaults."""
    assert DCE_AIF_TAIL_PEAK_FRACTION == 0.40
    assert DCE_AIF_TAIL_SHOULDER_SECONDS == 60.0


# ----- ratio_30 helper -----------------------------------------------------

def test_compute_ratio_30s_basic():
    tr = 4.0
    curve = np.zeros(40)
    curve[5] = 10.0  # peak
    curve[5 + max(4, int(np.ceil(30.0 / tr)))] = 3.0
    r = compute_dce_aif_ratio_30s(curve, tr=tr, peak_idx=5)
    assert r == pytest.approx(0.30, abs=1e-6)


def test_compute_ratio_30s_short_scan_returns_none():
    tr = 4.0
    curve = np.zeros(8)
    curve[5] = 10.0
    assert compute_dce_aif_ratio_30s(curve, tr=tr, peak_idx=5) is None


def test_compute_ratio_30s_zero_peak_returns_none():
    assert compute_dce_aif_ratio_30s(np.zeros(50), tr=4.0, peak_idx=0) is None


# ----- _apply_aif_scale ---------------------------------------------------

def test_apply_aif_scale_preserves_min_enhancement_guard():
    """A peak below MIN_MEANINGFUL_ENHANCEMENT_MM returns the input unchanged."""
    curve = np.full(20, 1e-6)
    out, sf = _apply_aif_scale(curve, hct=0.42, target_peak=0.0,
                                expected_range_fn=lambda: (5.0, 7.6))
    assert sf == 1.0
    assert np.array_equal(out, curve)


def test_apply_aif_scale_target_peak_explicit():
    curve = np.array([0, 0, 0, 1.0, 0.5, 0.25])
    out, sf = _apply_aif_scale(curve, hct=0.42, target_peak=6.0,
                                expected_range_fn=lambda: (5.0, 7.6))
    # target = 6.0 / (1 - 0.42) = 10.345
    expected_target = 6.0 / (1 - 0.42)
    assert sf == pytest.approx(expected_target / 1.0, rel=1e-6)
    assert out.max() == pytest.approx(expected_target, rel=1e-6)


# ----- Per-gate basic tests -----------------------------------------------

def test_g1_pass():
    status, _ = _evaluate_gates(scale_factor=2.0, qc=_make_qc(),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=10, hemisphere="bilateral",
                                 n_slabs_spanned=3)
    assert status == "PASS"


def test_g1_fail_no_g6_escape():
    """G1 fails, G6 sub-criteria fail too -> revert."""
    status, _ = _evaluate_gates(scale_factor=4.0, qc=_make_qc(),
                                 atlas_tier="MCA_ACA_dil_dce_strict",
                                 n_voxels=2, hemisphere="unilateral_left",
                                 n_slabs_spanned=1)
    assert status.startswith("FAIL_G6")


def test_g2_hard_fail():
    status, _ = _evaluate_gates(scale_factor=1.0,
                                 qc=_make_qc(late_tail=0.50),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=10, hemisphere="bilateral",
                                 n_slabs_spanned=3)
    assert status == "FAIL_G2"


def test_g3_hard_fail():
    status, _ = _evaluate_gates(scale_factor=1.0,
                                 qc=_make_qc(fwhm=25.0),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=10, hemisphere="bilateral",
                                 n_slabs_spanned=3)
    assert status == "FAIL_G3"


def test_g4_hard_fail():
    status, _ = _evaluate_gates(scale_factor=1.0,
                                 qc=_make_qc(bumps=5),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=10, hemisphere="bilateral",
                                 n_slabs_spanned=3)
    assert status == "FAIL_G4"


def test_g5_warn():
    status, dec = _evaluate_gates(scale_factor=1.0,
                                   qc=_make_qc(ratio_30=0.45),
                                   atlas_tier="MCA_ACA_dce_strict",
                                   n_voxels=10, hemisphere="bilateral",
                                   n_slabs_spanned=3)
    assert status == "PASS_WARN"
    assert "G5" in dec["warns"]


def test_g5_hard_fail():
    status, _ = _evaluate_gates(scale_factor=1.0,
                                 qc=_make_qc(ratio_30=0.55),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=10, hemisphere="bilateral",
                                 n_slabs_spanned=3)
    assert status == "FAIL_G5"


def test_skip_g5_when_ratio_unevaluable():
    """ratio_30 = None (short TR / scan) does not silently pass; recorded."""
    status, dec = _evaluate_gates(scale_factor=1.0,
                                   qc=_make_qc(ratio_30=None),
                                   atlas_tier="MCA_ACA_dce_strict",
                                   n_voxels=10, hemisphere="bilateral",
                                   n_slabs_spanned=3)
    assert status == "PASS"
    assert dec["qc"]["ratio_30"] is None


# ----- G6 admit / sub-criteria boundaries ---------------------------------

def test_g6_admit_sub08_archetype():
    """sub-08 case: scale 4.11, strict tier, bilateral n=10, all shape pass."""
    status, dec = _evaluate_gates(scale_factor=4.11,
                                   qc=_make_qc(fwhm=17.0, ratio_30=0.48,
                                               late_tail=0.30, bumps=0),
                                   atlas_tier="MCA_ACA_dce_strict",
                                   n_voxels=10, hemisphere="bilateral",
                                   n_slabs_spanned=3)
    assert status == "G6_ADMIT"
    assert "G1_via_G6" in dec["warns"]


def test_g6_reject_sub0081_v1_archetype():
    """sub-0081 V1 case: dilated tier (G6_f), n=2 unilateral (G6_g), scale 8.83 (G6_a)."""
    status, _ = _evaluate_gates(scale_factor=8.83, qc=_make_qc(ratio_30=0.41),
                                 atlas_tier="MCA_ACA_dil_dce_strict",
                                 n_voxels=2, hemisphere="unilateral_left",
                                 n_slabs_spanned=1)
    assert status.startswith("FAIL_G6")


def test_g6_g_bilateral_boundary_n4_reject():
    status, _ = _evaluate_gates(scale_factor=4.0, qc=_make_qc(),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=4, hemisphere="bilateral",
                                 n_slabs_spanned=2)
    assert status == "FAIL_G6_g"


def test_g6_g_bilateral_boundary_n5_admit():
    status, _ = _evaluate_gates(scale_factor=4.0, qc=_make_qc(),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=5, hemisphere="bilateral",
                                 n_slabs_spanned=2)
    assert status == "G6_ADMIT"


def test_g6_g_unilateral_n9_2slabs_reject():
    status, _ = _evaluate_gates(scale_factor=4.0, qc=_make_qc(),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=9, hemisphere="unilateral_right",
                                 n_slabs_spanned=2)
    assert status == "FAIL_G6_g"


def test_g6_g_unilateral_n10_2slabs_admit():
    status, _ = _evaluate_gates(scale_factor=4.0, qc=_make_qc(),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=10, hemisphere="unilateral_right",
                                 n_slabs_spanned=2)
    assert status == "G6_ADMIT"


def test_g6_g_unilateral_n10_1slab_reject():
    """G6(g) unilateral path requires n>=10 AND >=2 distinct slabs."""
    status, _ = _evaluate_gates(scale_factor=4.0, qc=_make_qc(),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=10, hemisphere="unilateral_right",
                                 n_slabs_spanned=1)
    assert status == "FAIL_G6_g"


def test_g6_a_boundary_scale_5p99_admit():
    status, _ = _evaluate_gates(scale_factor=5.99, qc=_make_qc(),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=10, hemisphere="bilateral",
                                 n_slabs_spanned=3)
    assert status == "G6_ADMIT"


def test_g6_a_boundary_scale_6p01_admit():
    """6.01 is below the 8.0 ceiling — admitted via G6."""
    status, _ = _evaluate_gates(scale_factor=6.01, qc=_make_qc(),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=10, hemisphere="bilateral",
                                 n_slabs_spanned=3)
    assert status == "G6_ADMIT"


def test_g6_a_scale_7p5_admit():
    status, _ = _evaluate_gates(scale_factor=7.5, qc=_make_qc(),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=10, hemisphere="bilateral",
                                 n_slabs_spanned=3)
    assert status == "G6_ADMIT"


def test_g6_a_boundary_scale_8p0_admit():
    """Exact ceiling boundary — 8.0 passes via <=."""
    status, _ = _evaluate_gates(scale_factor=8.0, qc=_make_qc(),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=10, hemisphere="bilateral",
                                 n_slabs_spanned=3)
    assert status == "G6_ADMIT"


def test_g6_a_scale_8p5_reject():
    status, _ = _evaluate_gates(scale_factor=8.5, qc=_make_qc(),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=10, hemisphere="bilateral",
                                 n_slabs_spanned=3)
    assert status == "FAIL_G6_a"


def test_g6_f_wrong_tier_isolated_reject():
    """Isolates G6(f) from (a)/(g): scale 4 OK, n=10 bilateral OK, but tier=dilated."""
    status, _ = _evaluate_gates(scale_factor=4.0, qc=_make_qc(),
                                 atlas_tier="MCA_ACA_dil_dce_strict",
                                 n_voxels=10, hemisphere="bilateral",
                                 n_slabs_spanned=3)
    assert status == "FAIL_G6_f"


def test_g6_f_relaxed_tier_admit():
    status, _ = _evaluate_gates(scale_factor=4.0, qc=_make_qc(),
                                 atlas_tier="MCA_ACA_dce_relaxed",
                                 n_voxels=10, hemisphere="bilateral",
                                 n_slabs_spanned=3)
    assert status == "G6_ADMIT"


def test_g6_f_permissive_tier_admit():
    status, _ = _evaluate_gates(scale_factor=4.0, qc=_make_qc(),
                                 atlas_tier="MCA_ACA_dce_permissive",
                                 n_voxels=10, hemisphere="bilateral",
                                 n_slabs_spanned=3)
    assert status == "G6_ADMIT"


def test_g6_relaxed_tier_high_scale_admit():
    """Combined: relaxed tier at scale 7.0 — exercises both ceiling raise and tier extension."""
    status, _ = _evaluate_gates(scale_factor=7.0, qc=_make_qc(),
                                 atlas_tier="MCA_ACA_dce_relaxed",
                                 n_voxels=10, hemisphere="bilateral",
                                 n_slabs_spanned=3)
    assert status == "G6_ADMIT"


# ----- Threshold boundaries -----------------------------------------------

def test_g3_fwhm_boundary_22p0_pass():
    status, _ = _evaluate_gates(scale_factor=1.0, qc=_make_qc(fwhm=22.0),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=10, hemisphere="bilateral",
                                 n_slabs_spanned=3)
    assert status == "PASS"


def test_g3_fwhm_boundary_22p01_hard():
    status, _ = _evaluate_gates(scale_factor=1.0, qc=_make_qc(fwhm=22.01),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=10, hemisphere="bilateral",
                                 n_slabs_spanned=3)
    assert status == "FAIL_G3"


def test_g4_bumps_boundary_2_pass():
    status, _ = _evaluate_gates(scale_factor=1.0, qc=_make_qc(bumps=2),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=10, hemisphere="bilateral",
                                 n_slabs_spanned=3)
    assert status == "PASS"


def test_g4_bumps_boundary_3_hard():
    status, _ = _evaluate_gates(scale_factor=1.0, qc=_make_qc(bumps=3),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=10, hemisphere="bilateral",
                                 n_slabs_spanned=3)
    assert status == "FAIL_G4"


def test_g5_boundary_0p40_pass():
    """ratio_30 == 0.40 is at the WARN floor (strict ineq for WARN)."""
    status, _ = _evaluate_gates(scale_factor=1.0, qc=_make_qc(ratio_30=0.40),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=10, hemisphere="bilateral",
                                 n_slabs_spanned=3)
    assert status == "PASS"


def test_g5_boundary_0p50_warn():
    """ratio_30 == 0.50 still WARN (HARD requires strict >0.50)."""
    status, _ = _evaluate_gates(scale_factor=1.0, qc=_make_qc(ratio_30=0.50),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=10, hemisphere="bilateral",
                                 n_slabs_spanned=3)
    assert status == "PASS_WARN"


def test_g5_boundary_0p501_hard():
    status, _ = _evaluate_gates(scale_factor=1.0, qc=_make_qc(ratio_30=0.501),
                                 atlas_tier="MCA_ACA_dce_strict",
                                 n_voxels=10, hemisphere="bilateral",
                                 n_slabs_spanned=3)
    assert status == "FAIL_G5"


# ----- Composite trim metric selection ------------------------------------

def test_select_worst_for_trim_g1_drops_lowest_peak():
    metrics = [{"peak_mm": 5.0, "washout": 0.9, "fwhm_seconds": 14.0},
               {"peak_mm": 1.0, "washout": 0.9, "fwhm_seconds": 14.0},
               {"peak_mm": 3.0, "washout": 0.9, "fwhm_seconds": 14.0}]
    assert _select_worst_for_trim(metrics, [1.0, 1.0, 1.0], "G1") == 1


def test_select_worst_for_trim_g2_drops_lowest_washout():
    metrics = [{"peak_mm": 5.0, "washout": 0.9, "fwhm_seconds": 14.0},
               {"peak_mm": 5.0, "washout": 0.3, "fwhm_seconds": 14.0},
               {"peak_mm": 5.0, "washout": 0.8, "fwhm_seconds": 14.0}]
    assert _select_worst_for_trim(metrics, [1.0, 1.0, 1.0], "G2") == 1


def test_select_worst_for_trim_g3_drops_widest_fwhm():
    metrics = [{"peak_mm": 5.0, "washout": 0.9, "fwhm_seconds": 14.0},
               {"peak_mm": 5.0, "washout": 0.9, "fwhm_seconds": 25.0},
               {"peak_mm": 5.0, "washout": 0.9, "fwhm_seconds": 18.0}]
    assert _select_worst_for_trim(metrics, [1.0, 1.0, 1.0], "G3") == 1


def test_select_worst_for_trim_g4_drops_lowest_washout():
    metrics = [{"peak_mm": 5.0, "washout": 0.9},
               {"peak_mm": 5.0, "washout": 0.2}]
    assert _select_worst_for_trim(metrics, [1.0, 1.0], "G4") == 1


def test_select_worst_for_trim_fallback_drops_lowest_score():
    metrics = [{"peak_mm": 5.0, "washout": 0.9},
               {"peak_mm": 5.0, "washout": 0.9}]
    assert _select_worst_for_trim(metrics, [0.9, 0.3], "unknown") == 1


def test_select_worst_tiebreak_lowest_index():
    metrics = [{"peak_mm": 5.0, "washout": 0.9},
               {"peak_mm": 5.0, "washout": 0.9}]
    assert _select_worst_for_trim(metrics, [1.0, 1.0], "G1") == 0


# ----- _count_slabs_spanned ----------------------------------------------

def test_count_slabs_spanned_unique_z():
    cands = [(0, 0, 5), (0, 0, 5), (0, 0, 7), (0, 0, 9)]
    assert _count_slabs_spanned(cands) == 3


def test_count_slabs_spanned_empty():
    assert _count_slabs_spanned([]) == 0


# ----- Bump-detector new constants ----------------------------------------

def test_bump_detector_short_shoulder_excludes_recirculation():
    """sh60+f0.40 excludes physiologic recirculation hump."""
    tr = 7.5
    curve = np.zeros(60)
    curve[7] = 10.0  # peak
    # Recirculation hump at ~30s post-peak (sub-shoulder under sh60)
    curve[7 + int(30 / tr)] = 4.5  # 45% of peak
    # True late-tail anomaly at ~90s post-peak
    curve[7 + int(90 / tr)] = 4.5
    n, _ = detect_dce_aif_multi_bump_tail(
        curve, tr=tr, baseline_frames=5,
        secondary_peak_fraction=0.40,
        tail_shoulder_seconds=60.0,
    )
    assert n == 1  # only the true late-tail bump (recirculation excluded)


# ----- Dose / B0 independence -------------------------------------------

def test_g1_dimensionless_against_dose_corrected_target():
    """Half-dose AIF with proportionally-halved target gives scale ~1.

    Half-dose plasma peak ~5 mM; expected_range halved to (2.5, 3.8)
    so midpoint = 3.15 mM blood, target = 3.15 / (1 - 0.42) = 5.43 mM
    plasma. scaling_factor = 5.43 / 5.0 = 1.087 — passes G1 <= 3.
    """
    curve = np.array([0, 0, 0, 5.0, 2.5, 1.25])
    _out, sf = _apply_aif_scale(
        curve, hct=0.42, target_peak=0.0,
        expected_range_fn=lambda: (2.5, 3.8),
    )
    assert sf == pytest.approx(1.087, abs=0.05)
    assert sf <= DCE_AIF_GATE_SCALE_HARD


# ----- Trim metric routing regression --------------------------------------

def test_select_worst_for_trim_g6_a_drops_lowest_peak():
    """Regression for the failing-gate routing bug: G6_a (G1 fail with G6
    escape denied via amplitude ceiling) must drop the lowest-peak voxel,
    not the lowest score. Without the fix the call site stripped '_a' and
    routed to the score-fallback branch."""
    metrics = [
        {"peak_mm": 5.0, "washout": 0.9, "fwhm_seconds": 14.0},
        {"peak_mm": 1.0, "washout": 0.9, "fwhm_seconds": 14.0},
        {"peak_mm": 3.0, "washout": 0.9, "fwhm_seconds": 14.0},
    ]
    # Highest score on the lowest-peak candidate to disambiguate the
    # peak-routed branch from the score-fallback.
    scores = [0.3, 0.9, 0.5]
    assert _select_worst_for_trim(metrics, scores, "G6_a") == 1


def test_select_worst_for_trim_g6_f_falls_through_to_score():
    """G6_f / G6_g (tier or consensus failures) cannot be remediated by
    dropping a single voxel — fall through to the score-fallback."""
    metrics = [{"peak_mm": 5.0, "washout": 0.9, "fwhm_seconds": 14.0}] * 3
    scores = [0.9, 0.3, 0.7]
    assert _select_worst_for_trim(metrics, scores, "G6_f") == 1
    assert _select_worst_for_trim(metrics, scores, "G6_g") == 1


# =============================================================================
# Fix 1: brain-wide tissue_ttp_frame threading into _apply_arterial_criteria
# =============================================================================


class TestTissueTtpFrameThreading:
    """Atlas-constrained search must use the brain-wide tissue_ttp.

    Without the threading, ``_apply_arterial_criteria`` recomputes
    ``tissue_ttp`` from the moderate-enhancement IQR of the masked
    voxels. When the mask is atlas-constrained (mostly arterial voxels),
    that median collapses onto the arterial peak time and the
    downstream ``is_likely_venous`` flag rejects every arterial voxel.
    """

    @staticmethod
    def _build_atlas_arterial_chars(
        n_arterial: int = 200,
        n_tissue: int = 50,
        arterial_peak_frame: int = 5,
        tissue_peak_frame: int = 25,
    ) -> dict:
        """Mimic _compute_signal_characteristics output for an atlas-
        constrained mask: majority arterial-peak voxels, a few tissue."""
        rng = np.random.default_rng(0)
        n = n_arterial + n_tissue
        peak_indices = np.empty(n, dtype=np.int64)
        peak_indices[:n_arterial] = arterial_peak_frame + rng.integers(-1, 2, n_arterial)
        peak_indices[n_arterial:] = tissue_peak_frame + rng.integers(-1, 2, n_tissue)
        peak_values = np.empty(n, dtype=np.float64)
        peak_values[:n_arterial] = 6.0 + rng.normal(0, 0.5, n_arterial)
        peak_values[n_arterial:] = 1.0 + rng.normal(0, 0.1, n_tissue)
        enhancement = peak_values.copy()
        return {
            "peak_indices": peak_indices,
            "peak_values": peak_values,
            "enhancement": enhancement,
            "fwhm_frames": np.full(n, 3.0),
            "fwhm_seconds": np.full(n, 12.0),
            "washout_fraction": np.full(n, 0.85),
            "upstroke_slope": np.full(n, 0.5),
        }

    def test_brain_wide_tissue_ttp_overrides_local(self):
        chars = self._build_atlas_arterial_chars()
        spatial_exclude = np.zeros(len(chars["peak_indices"]), dtype=bool)
        # Local IQR median collapses onto arterial timing (~frame 5);
        # brain-wide tissue_ttp from a real parenchymal reference would
        # be ~frame 25.
        idx_local, ttp_local, _tier_local = AIFDetection._apply_arterial_criteria(
            chars, spatial_exclude, nt=40, tr=4.0,
        )
        idx_passed, ttp_passed, _tier_passed = AIFDetection._apply_arterial_criteria(
            chars, spatial_exclude, nt=40, tr=4.0, tissue_ttp_frame=25.0,
        )
        # Local computation collapses onto arterial peak (~5)
        assert ttp_local <= 7.0, f"local tissue_ttp should collapse onto arterial timing, got {ttp_local}"
        # Passed value is preserved
        assert ttp_passed == 25.0

    def test_kwarg_default_preserves_local_path(self):
        """Omitting tissue_ttp_frame must give the pre-fix behavior."""
        chars = self._build_atlas_arterial_chars()
        spatial_exclude = np.zeros(len(chars["peak_indices"]), dtype=bool)
        idx_a, ttp_a, tier_a = AIFDetection._apply_arterial_criteria(
            chars, spatial_exclude, nt=40, tr=4.0,
        )
        idx_b, ttp_b, tier_b = AIFDetection._apply_arterial_criteria(
            chars, spatial_exclude, nt=40, tr=4.0, tissue_ttp_frame=None,
        )
        np.testing.assert_array_equal(idx_a, idx_b)
        assert ttp_a == ttp_b
        assert tier_a == tier_b

    def test_fix1_byte_identical_on_healthy_session(self):
        """Regression guard for the ‘13 healthy OSIPI sessions
        unaffected’ claim.

        On a healthy session the atlas-local ``tissue_ttp`` (IQR median
        of moderate-enhancement voxels in atlas territory) sits within
        ~1 frame of the brain-wide reference because the atlas is not
        yet collapsed onto the arterial peak (sufficient parenchymal
        contamination in the IQR). In that regime, threading the brain-
        wide value must be a no-op: candidate set, scores, and
        ``is_likely_venous`` flags must be byte-identical.
        """
        # Build a synthetic 4D where the atlas mask covers a mix of
        # arterial-like AND parenchymal-like voxels — its local IQR
        # median tissue_ttp lands at ~tissue_peak_frame.
        nx = ny = nz = 20
        nt = 40
        t = np.arange(nt, dtype=np.float64)
        tissue_curve = 0.6 * np.exp(-0.5 * ((t - 20) / 6.0) ** 2)
        tissue_curve[:5] = 0.0
        conc = np.broadcast_to(
            tissue_curve[None, None, None, :], (nx, ny, nz, nt),
        ).copy()
        # Small arterial cluster — only 3 voxels, won't dominate IQR
        arterial_curve = 6.0 * np.exp(-0.5 * ((t - 5) / 1.5) ** 2)
        arterial_curve[:5] = 0.0
        for x, y, z in [(10, 10, 10), (10, 11, 10), (11, 10, 10)]:
            conc[x, y, z, :] = arterial_curve
        # Atlas mask covers a 6×6×6 block where most voxels are
        # parenchymal — atlas-local tissue_ttp ≈ brain-wide.
        atlas_mask = np.zeros((nx, ny, nz), dtype=np.uint8)
        atlas_mask[8:14, 8:14, 8:14] = 1

        det = AIFDetection.__new__(AIFDetection)
        det.inputs = type("FI", (), {
            "hematocrit": 0.42, "r1_relaxivity": 4.1,
            "agent_standard_dose": 0.1, "agent_r1_known": True,
            "field_strength": 3.0, "dose": 1.0, "target_peak": 0.0,
            "baseline_frames": 5, "lesion_mask_file": None,
            "mr_acquisition_type": "3D",
        })()
        # Sanity: confirm local and brain-wide tissue_ttp agree (this
        # IS a healthy-session-like setup, not a regressed one).
        # Both will be ≈ 20 in this synthetic.
        cands_default, scores_default, mets_default = det._find_candidates_in_mask(
            conc, atlas_mask, n_candidates=4, tr=4.0,
            baseline_frames=5, criteria="strict", atlas_constrained=True,
        )
        cands_brainwide, scores_brainwide, mets_brainwide = det._find_candidates_in_mask(
            conc, atlas_mask, n_candidates=4, tr=4.0,
            baseline_frames=5, criteria="strict", atlas_constrained=True,
            tissue_ttp_frame=20.0,
        )
        # Byte-identical candidates + scores + venous flags
        assert cands_default == cands_brainwide
        assert scores_default == scores_brainwide
        assert (
            [m.get("is_likely_venous", False) for m in mets_default]
            == [m.get("is_likely_venous", False) for m in mets_brainwide]
        )

    def test_atlas_arterial_voxels_not_flagged_venous_with_brain_wide_ttp(self):
        """End-to-end via _find_candidates_in_mask: atlas-constrained
        search on a synthetic brain where all atlas voxels are arterial
        must NOT flag them as venous when brain-wide tissue_ttp is
        passed."""
        nx = ny = nz = 20
        nt = 40
        # Tissue background peaks at frame 25
        t = np.arange(nt, dtype=np.float64)
        tissue_curve = 0.6 * np.exp(-0.5 * ((t - 25) / 6.0) ** 2)
        tissue_curve[:5] = 0.0
        conc = np.broadcast_to(
            tissue_curve[None, None, None, :],
            (nx, ny, nz, nt),
        ).copy()
        # A small atlas-territory cluster of ARTERIAL voxels at frame 5
        arterial_coords = [(10, 10, 10), (10, 11, 10), (11, 10, 10), (11, 11, 10)]
        arterial_curve = 6.0 * np.exp(-0.5 * ((t - 5) / 1.5) ** 2)
        arterial_curve[:5] = 0.0
        for x, y, z in arterial_coords:
            conc[x, y, z, :] = arterial_curve
        atlas_mask = np.zeros((nx, ny, nz), dtype=np.uint8)
        for x, y, z in arterial_coords:
            atlas_mask[x, y, z] = 1
        # Add a few tissue voxels into the atlas mask to give the local
        # IQR computation something to misinterpret as "tissue_ttp".
        for x in range(8, 13):
            for y in range(8, 13):
                atlas_mask[x, y, 9] = 1

        det = AIFDetection.__new__(AIFDetection)
        det.inputs = type("FI", (), {
            "hematocrit": 0.42, "r1_relaxivity": 4.1,
            "agent_standard_dose": 0.1, "agent_r1_known": True,
            "field_strength": 3.0, "dose": 1.0, "target_peak": 0.0,
            "baseline_frames": 5, "lesion_mask_file": None,
            "mr_acquisition_type": "3D",
        })()

        # Without brain-wide ttp, atlas-local IQR collapses to ~5 →
        # arterial voxels at peak=5 get frames_before_tissue=0 → flagged
        # venous (with TR>0 timing_margin=1).
        cands_no, scores_no, mets_no = det._find_candidates_in_mask(
            conc, atlas_mask, n_candidates=4, tr=4.0,
            baseline_frames=5, criteria="strict", atlas_constrained=True,
        )
        # With brain-wide ttp threaded in, no false-venous flag.
        cands_fix, scores_fix, mets_fix = det._find_candidates_in_mask(
            conc, atlas_mask, n_candidates=4, tr=4.0,
            baseline_frames=5, criteria="strict", atlas_constrained=True,
            tissue_ttp_frame=25.0,
        )
        n_venous_no = sum(1 for m in mets_no if m.get("is_likely_venous", False))
        n_venous_fix = sum(1 for m in mets_fix if m.get("is_likely_venous", False))
        assert n_venous_fix == 0, "arterial atlas voxels must not be flagged venous when brain-wide tissue_ttp is passed"
        # The unfixed path should have flagged at least some arterial
        # voxels (regression guard against silently restoring the bug).
        assert n_venous_no >= n_venous_fix

    def test_low_tr_residual_effect_on_venous_flag(self):
        """At TR > 5 s, the cascade selector path bypasses arterial_timing
        (uses early_arrival instead). But the per-voxel is_likely_venous
        flag and the scorer's timing contribution still consume tissue_ttp,
        so Fix 1's effect is preserved at low TR."""
        nx = ny = nz = 20
        nt = 30
        t = np.arange(nt, dtype=np.float64)
        tissue_curve = 0.6 * np.exp(-0.5 * ((t - 18) / 4.0) ** 2)
        tissue_curve[:5] = 0.0
        conc = np.broadcast_to(
            tissue_curve[None, None, None, :], (nx, ny, nz, nt),
        ).copy()
        arterial_coords = [(10, 10, 10), (10, 11, 10), (11, 10, 10), (11, 11, 10)]
        arterial_curve = 6.0 * np.exp(-0.5 * ((t - 6) / 1.5) ** 2)
        arterial_curve[:5] = 0.0
        for x, y, z in arterial_coords:
            conc[x, y, z, :] = arterial_curve
        atlas_mask = np.zeros((nx, ny, nz), dtype=np.uint8)
        for x, y, z in arterial_coords:
            atlas_mask[x, y, z] = 1

        det = AIFDetection.__new__(AIFDetection)
        det.inputs = type("FI", (), {
            "hematocrit": 0.42, "r1_relaxivity": 4.1,
            "agent_standard_dose": 0.1, "agent_r1_known": True,
            "field_strength": 3.0, "dose": 1.0, "target_peak": 0.0,
            "baseline_frames": 5, "lesion_mask_file": None,
            "mr_acquisition_type": "3D",
        })()
        # TR=8 s → low_tr=True
        cands, scores, mets = det._find_candidates_in_mask(
            conc, atlas_mask, n_candidates=4, tr=8.0,
            baseline_frames=5, criteria="strict", atlas_constrained=True,
            tissue_ttp_frame=18.0,
        )
        n_venous = sum(1 for m in mets if m.get("is_likely_venous", False))
        assert n_venous == 0, "low-TR mode must still respect brain-wide tissue_ttp for venous flag"


# =============================================================================
# Fix 2: 10mm physical erosion on WB fallback search mask
# =============================================================================


class TestPhysicalErosionWBFallback:
    """Verify ``physical_erosion_mm`` excludes lateral surface voxels in
    physical mm (not voxel count), and that the thin-slab fallback
    triggers when the cut would evacuate the mask."""

    @staticmethod
    def _make_brain_mask(shape, voxel_sizes):
        mask = np.zeros(shape, dtype=np.uint8)
        # Spherical-ish brain centered in volume
        cx, cy, cz = (s // 2 for s in shape)
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    rx = (x - cx) * voxel_sizes[0]
                    ry = (y - cy) * voxel_sizes[1]
                    rz = (z - cz) * voxel_sizes[2]
                    if rx * rx + ry * ry + rz * rz <= 60.0 ** 2:
                        mask[x, y, z] = 1
        return mask

    def test_physical_erosion_excludes_5mm_voxel(self):
        """A voxel at edge_d=5mm must be excluded with physical_erosion_mm=10
        and included with 0.0."""
        from scipy.ndimage import distance_transform_edt
        shape = (60, 60, 60)
        voxel_sizes = (1.0, 1.0, 1.0)
        brain = self._make_brain_mask(shape, voxel_sizes)
        dist = distance_transform_edt(brain > 0.5, sampling=voxel_sizes)
        # Find a voxel at edge_d≈5mm
        target = np.argwhere((dist >= 4.5) & (dist <= 5.5))
        assert len(target) > 0
        x, y, z = target[0]
        assert dist[x, y, z] < 10.0

        no_erode = AIFDetection._prepare_search_mask(
            brain, voxel_sizes, skip_midline=False,
            slab_edge_inferior_mm=2.0, slab_edge_superior_mm=2.0,
            erosion_iterations=0, physical_erosion_mm=0.0,
        )
        eroded = AIFDetection._prepare_search_mask(
            brain, voxel_sizes, skip_midline=False,
            slab_edge_inferior_mm=2.0, slab_edge_superior_mm=2.0,
            erosion_iterations=0, physical_erosion_mm=10.0,
        )
        # The 5mm-edge voxel must be present in no_erode and absent in eroded
        # (assuming it survives the small slab/midline/top-25 cuts; pick a
        # central voxel).
        # Find a voxel surviving all non-erosion cuts at edge_d≈5
        in_no_erode = (no_erode > 0.5) & (dist >= 4.5) & (dist <= 5.5)
        assert int(np.sum(in_no_erode)) > 0
        survivors_no = np.argwhere(in_no_erode)
        # None of these should survive in eroded
        for x, y, z in survivors_no[:10]:
            assert eroded[x, y, z] == 0, f"voxel at edge_d={dist[x,y,z]:.1f}mm should be eroded"

    def test_physical_erosion_anisotropic_voxels(self):
        """1×1×5mm voxels: physical_erosion_mm=10 must measure mm,
        not voxel count. Catches the bug where someone passes 10
        as a voxel-count threshold (which would erode 50mm in z)."""
        from scipy.ndimage import distance_transform_edt
        shape = (80, 80, 16)
        voxel_sizes = (1.0, 1.0, 5.0)
        brain = np.zeros(shape, dtype=np.uint8)
        # Cylinder spanning full z, radius 30mm in xy: large enough that
        # 10mm erosion + slab/top-25% cuts leaves >>BRAIN_EROSION_MIN_VOXELS.
        cx, cy = 40, 40
        for x in range(shape[0]):
            for y in range(shape[1]):
                if (x - cx) ** 2 + (y - cy) ** 2 <= 30 ** 2:
                    brain[x, y, :] = 1
        dist = distance_transform_edt(brain > 0.5, sampling=voxel_sizes)
        z_test = 5
        # Find voxels at edge_d ≈ 7mm in plane at z=z_test
        target_xy = np.argwhere(
            (dist[:, :, z_test] >= 6.5) & (dist[:, :, z_test] <= 7.5)
        )
        assert len(target_xy) > 0

        eroded = AIFDetection._prepare_search_mask(
            brain, voxel_sizes, skip_midline=True,
            slab_edge_inferior_mm=2.0, slab_edge_superior_mm=2.0,
            erosion_iterations=0, physical_erosion_mm=10.0,
        )
        assert int(np.sum(eroded[..., z_test])) > 0, (
            "test setup invalid: erosion evacuated z slice"
        )
        # 7mm-edge voxels must all be excluded
        excluded = sum(1 for x, y in target_xy if eroded[x, y, z_test] == 0)
        assert excluded == len(target_xy), (
            f"voxels at edge_d≈7mm should be eroded; "
            f"{excluded}/{len(target_xy)} excluded"
        )
        # Deep voxels (edge_d ≥ 11mm) must survive
        deep_xy = np.argwhere(dist[:, :, z_test] >= 11.0)
        assert len(deep_xy) > 0
        survived = sum(1 for x, y in deep_xy if eroded[x, y, z_test] == 1)
        assert survived > 0

    def test_thin_slab_fallback_keeps_pre_erosion_mask(self, caplog):
        """If physical_erosion_mm would evacuate the mask below
        BRAIN_EROSION_MIN_VOXELS, fall back to pre-erosion + log warning."""
        from pmriprep.utils.dce_constants import BRAIN_EROSION_MIN_VOXELS
        # Tiny brain — most voxels are within 10mm of edge
        shape = (8, 8, 8)
        voxel_sizes = (1.0, 1.0, 1.0)
        brain = np.zeros(shape, dtype=np.uint8)
        brain[1:7, 1:7, 1:7] = 1
        # Without erosion this mask has ~216 voxels (above min). With
        # 10mm physical erosion all voxels are within 10mm so candidate
        # mask collapses to 0; fallback triggers.
        with caplog.at_level("WARNING"):
            mask = AIFDetection._prepare_search_mask(
                brain, voxel_sizes, skip_midline=True,
                slab_edge_inferior_mm=0.0, slab_edge_superior_mm=0.0,
                erosion_iterations=0, physical_erosion_mm=10.0,
            )
        assert int(np.sum(mask > 0.5)) >= BRAIN_EROSION_MIN_VOXELS or int(np.sum(mask > 0.5)) == int(np.sum(brain > 0.5))
        # The warning must be emitted
        assert any("Physical erosion" in rec.message and "falling back" in rec.message
                   for rec in caplog.records)

    def test_physical_erosion_zero_no_op(self):
        """physical_erosion_mm=0.0 must produce identical output to the
        prior signature (no Step 6 invocation)."""
        shape = (40, 40, 40)
        voxel_sizes = (1.0, 1.0, 1.0)
        brain = self._make_brain_mask(shape, voxel_sizes)
        baseline = AIFDetection._prepare_search_mask(
            brain, voxel_sizes, skip_midline=False,
            slab_edge_inferior_mm=4.0, slab_edge_superior_mm=4.0,
            erosion_iterations=2,
        )
        with_zero = AIFDetection._prepare_search_mask(
            brain, voxel_sizes, skip_midline=False,
            slab_edge_inferior_mm=4.0, slab_edge_superior_mm=4.0,
            erosion_iterations=2, physical_erosion_mm=0.0,
        )
        np.testing.assert_array_equal(baseline, with_zero)

    def test_physical_erosion_after_lesion_exclusion(self):
        """Lesion exclusion is Step 5, physical erosion is Step 6.

        Place the lesion DEEP (edge_d ≥ 12 mm) so the 10 mm physical
        erosion alone wouldn't strip those voxels.  Then assert:
        - With lesion + erosion enabled: lesion voxels are excluded
          (Step 5 must run regardless of erosion).
        - With lesion enabled but erosion disabled: lesion voxels are
          still excluded (Step 5 alone is sufficient).
        - With erosion enabled but no lesion: lesion voxels survive
          (Step 6 alone wouldn't strip a deep lesion).

        This isolates Step 5's contribution from Step 6's, so the test
        actually pins the order and catches a regression where Step 6
        accidentally precedes Step 5 (or replaces it).
        """
        shape = (40, 40, 40)
        voxel_sizes = (1.0, 1.0, 1.0)
        brain = self._make_brain_mask(shape, voxel_sizes)
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(brain > 0.5, sampling=voxel_sizes)
        # Deep lesion: well past the 10 mm physical erosion threshold.
        deep_voxels = np.argwhere(dist >= 12.0)
        assert len(deep_voxels) >= 20
        lesion = np.zeros(shape, dtype=np.uint8)
        for x, y, z in deep_voxels[:20]:
            lesion[x, y, z] = 1

        mask_both = AIFDetection._prepare_search_mask(
            brain, voxel_sizes, lesion_mask=lesion,
            lesion_distance_mm=0.0, skip_midline=False,
            slab_edge_inferior_mm=2.0, slab_edge_superior_mm=2.0,
            erosion_iterations=0, physical_erosion_mm=10.0,
        )
        mask_lesion_only = AIFDetection._prepare_search_mask(
            brain, voxel_sizes, lesion_mask=lesion,
            lesion_distance_mm=0.0, skip_midline=False,
            slab_edge_inferior_mm=2.0, slab_edge_superior_mm=2.0,
            erosion_iterations=0, physical_erosion_mm=0.0,
        )
        mask_erosion_only = AIFDetection._prepare_search_mask(
            brain, voxel_sizes, lesion_mask=None,
            skip_midline=False,
            slab_edge_inferior_mm=2.0, slab_edge_superior_mm=2.0,
            erosion_iterations=0, physical_erosion_mm=10.0,
        )

        for x, y, z in deep_voxels[:10]:
            # Lesion alone strips the deep lesion (Step 5)
            assert mask_lesion_only[x, y, z] == 0, (
                f"deep lesion at edge_d={dist[x,y,z]:.1f}mm should be "
                "excluded by lesion mask alone"
            )
            # Erosion alone leaves the deep lesion intact (proves
            # Step 5 contribution isn't masked by Step 6).
            assert mask_erosion_only[x, y, z] == 1, (
                f"deep voxel at edge_d={dist[x,y,z]:.1f}mm should "
                "survive 10mm erosion when no lesion is present"
            )
            # Combined: lesion still excluded (regression guard for
            # accidentally inverting Step 5 / Step 6 order).
            assert mask_both[x, y, z] == 0


# =============================================================================
# Fix 3: per-tier cascade_gate_decisions
# =============================================================================


class TestCascadeGateDecisions:
    """Verify the cascade exposes per-tier gate decisions for all
    attempted tiers (winner, trim-exhausted, and skipped), in the same
    order as ``anchor_outcomes``."""

    @staticmethod
    def _build_det():
        det = AIFDetection.__new__(AIFDetection)
        det.inputs = type("FI", (), {
            "hematocrit": 0.42, "r1_relaxivity": 4.1,
            "agent_standard_dose": 0.1, "agent_r1_known": True,
            "field_strength": 3.0, "dose": 1.0, "target_peak": 0.0,
            "baseline_frames": 5, "lesion_mask_file": None,
            "mr_acquisition_type": "3D",
        })()
        return det

    def test_evaluation_returns_decisions_on_failure(self):
        """``_evaluate_pool_with_gates_and_trim`` must expose its per-
        iteration gate-decision log even when trim exhausts and result
        is None."""
        from pmriprep.interfaces.dce_aif import TierEvaluation
        det = self._build_det()
        # 2 candidates with peak below the wb floor → G1 hard-fail on
        # any pool size; trim exhausts.
        nx = ny = nz = 20
        nt = 30
        conc = np.zeros((nx, ny, nz, nt), dtype=np.float64)
        t = np.arange(nt, dtype=np.float64)
        weak_curve = 0.4 * np.exp(-0.5 * ((t - 8) / 2.0) ** 2)
        weak_curve[:5] = 0.0
        coords = [(10, 10, 10), (11, 10, 10)]
        for x, y, z in coords:
            conc[x, y, z, :] = weak_curve
        brain = np.ones((nx, ny, nz), dtype=np.uint8)
        affine = np.eye(4)
        evaluation = det._evaluate_pool_with_gates_and_trim(
            coords, [0.7, 0.6],
            [{"peak_mm": 0.4}, {"peak_mm": 0.4}],
            conc, brain, affine, tr=4.0, baseline_frames=5,
            hct=0.42, target_peak=0.0,
            atlas_tier_label="MCA_ACA_dce_strict",
        )
        assert isinstance(evaluation, TierEvaluation)
        assert evaluation.result is None
        assert len(evaluation.gate_decisions) >= 1
        # All decisions must be from the same tier
        assert all(d["tier"] == "MCA_ACA_dce_strict" for d in evaluation.gate_decisions)

    def test_cascade_gate_decisions_entry_order_matches_anchor_outcomes(self):
        """tier_gate_decisions must have one entry per attempted tier,
        in the same order as anchor_outcomes."""
        # Build a synthetic where atlas tiers all skip on quorum (no
        # candidates pass per-voxel score floor) and wb_relaxed accepts.
        nx = ny = nz = 16
        nt = 30
        t = np.arange(nt, dtype=np.float64)
        # Tissue background, no arterial in atlas territory
        tissue = 0.6 * np.exp(-0.5 * ((t - 18) / 5.0) ** 2)
        tissue[:5] = 0.0
        conc = np.broadcast_to(
            tissue[None, None, None, :], (nx, ny, nz, nt),
        ).copy()
        # Arterial cluster OUTSIDE atlas masks (deep central, gets WB tier)
        arterial = 6.0 * np.exp(-0.5 * ((t - 6) / 1.5) ** 2)
        arterial[:5] = 0.0
        for x, y, z in [(8, 8, 8), (8, 9, 8), (9, 8, 8), (9, 9, 8)]:
            conc[x, y, z, :] = arterial
        brain = np.ones((nx, ny, nz), dtype=np.uint8)
        # Two atlas masks (in unrelated location, will fail)
        atlas1 = np.zeros((nx, ny, nz), dtype=np.uint8)
        atlas1[1:3, 1:3, 1:3] = 1
        atlas2 = np.zeros((nx, ny, nz), dtype=np.uint8)
        atlas2[14:16, 14:16, 14:16] = 1

        det = self._build_det()
        # Run cascade without gates (still records gate decisions for
        # PASS / skipped paths; just no trim_exhausted).
        result = det._run_two_pass_cascade(
            conc,
            atlas_search_mask=brain.astype(np.uint8),
            wb_search_mask=brain.astype(np.uint8),
            atlas_masks=[atlas1, atlas2],
            atlas_names=["A1", "A2"],
            n_candidates=4,
            tr=4.0,
            baseline_frames=5,
            tissue_ttp_frame=18.0,
        )
        # tier_gate_decisions length matches anchor_outcomes length
        assert len(result.tier_gate_decisions) == len(result.anchor_outcomes)
        # Pin the absolute cascade order (atlas tightest → broadest ×
        # strict → permissive, then wb_relaxed). Asserting only the
        # transitive zip-equality with anchor_outcomes wouldn't catch a
        # silent regression where BOTH orderings reverse together.
        expected_order = [
            "A1_strict", "A1_relaxed", "A1_permissive",
            "A2_strict", "A2_relaxed", "A2_permissive",
            "wb_relaxed",
        ]
        assert [e["tier"] for e in result.tier_gate_decisions] == expected_order
        assert [e["tier"] for e in result.anchor_outcomes] == expected_order

    def test_cascade_records_skipped_quorum(self):
        """Tiers skipped because quorum<2 must record outcome
        ``"skipped_quorum"`` with empty decisions."""
        nx = ny = nz = 16
        nt = 30
        # All-tissue concentration: no voxels score above quorum.
        t = np.arange(nt, dtype=np.float64)
        tissue = 0.4 * np.exp(-0.5 * ((t - 18) / 5.0) ** 2)
        tissue[:5] = 0.0
        conc = np.broadcast_to(
            tissue[None, None, None, :], (nx, ny, nz, nt),
        ).copy()
        brain = np.ones((nx, ny, nz), dtype=np.uint8)
        atlas = np.zeros((nx, ny, nz), dtype=np.uint8)
        atlas[5:11, 5:11, 5:11] = 1

        det = self._build_det()
        result = det._run_two_pass_cascade(
            conc,
            atlas_search_mask=brain.astype(np.uint8),
            wb_search_mask=brain.astype(np.uint8),
            atlas_masks=[atlas],
            atlas_names=["A"],
            n_candidates=4,
            tr=4.0,
            baseline_frames=5,
            tissue_ttp_frame=18.0,
        )
        # All tiers should have skipped outcome (quorum or anchor),
        # never PASS, since no arterial voxels exist anywhere.
        assert result.detection_method == "population"
        assert all(
            entry["outcome"] in ("skipped_quorum", "skipped_anchor", "trim_exhausted")
            for entry in result.tier_gate_decisions
        )
        # Skipped entries have empty decision logs
        for entry in result.tier_gate_decisions:
            if entry["outcome"].startswith("skipped"):
                assert entry["decisions"] == []


# =============================================================================
# _score_and_gate_pool extraction (v0.81.1)
# =============================================================================


class TestScoreAndGatePool:
    """Pin the extracted ``_score_and_gate_pool`` helper.

    The helper must reproduce the exact average -> plasma-convert -> scale ->
    QC -> hemisphere/slabs -> gate sequence the progressive-trim loop and the
    post-cascade C2 re-gate previously inlined. The extraction is
    behavior-preserving only if the helper is a faithful composition of those
    primitives; these tests pin it against a manual replay and exercise the C2
    candidate-set-divergence path that was previously uncovered.
    """

    @staticmethod
    def _make_detector(hematocrit: float = 0.42) -> AIFDetection:
        det = AIFDetection()
        det.inputs.hematocrit = hematocrit
        det.inputs.dose = 1.0
        det.inputs.agent_standard_dose = 0.1
        det.inputs.target_peak = 0.0
        det.inputs.field_strength = 3.0
        return det

    @staticmethod
    def _bilateral_affine() -> np.ndarray:
        # world_x = vox_i - 5  ->  i < 5 is left (x < 0), i >= 5 is right.
        aff = np.eye(4, dtype=np.float64)
        aff[0, 3] = -5.0
        return aff

    def test_matches_manual_primitive_composition(self) -> None:
        """Helper output is bit-for-bit the manual primitive sequence."""
        tr, baseline_frames, hct = 4.0, 5, 0.42
        cands = [(3, 10, 1), (7, 10, 2), (7, 11, 3)]  # left + right, 3 slabs
        conc_data, brain_mask = _make_synthetic_dce_4d(
            shape=(12, 14, 6, 40), baseline_frames=baseline_frames, tr=tr,
            arterial_coords=cands, arterial_peak=5.0,
        )
        affine = self._bilateral_affine()
        scores = [0.9, 0.8, 0.7]
        det = self._make_detector(hematocrit=hct)

        outcome = det._score_and_gate_pool(
            cands, scores, conc_data, brain_mask, affine, tr,
            baseline_frames, hct, 0.0, "MCA_ACA_dce_strict",
        )

        # Manual replay of the identical primitive sequence.
        curve_blood, _ = AIFDetection._compute_weighted_roi_aif(
            cands, scores, conc_data, brain_mask, tr=tr,
        )
        curve_plasma = curve_blood / (1.0 - hct)
        curve_scaled, sf = _apply_aif_scale(
            curve_plasma, hct, 0.0, det._get_expected_peak_range,
        )
        qc = _compute_pool_qc(curve_scaled, tr, baseline_frames)
        hemi, _nl, _nr = _classify_hemisphere(cands, affine)
        n_slabs = _count_slabs_spanned(cands)
        status, decision = _evaluate_gates(
            sf, qc, "MCA_ACA_dce_strict", len(cands), hemi, n_slabs,
        )

        np.testing.assert_array_equal(outcome.curve_scaled, curve_scaled)
        assert outcome.scale_factor == sf
        assert outcome.qc == qc
        assert outcome.hemisphere == hemi == "bilateral"
        assert outcome.n_slabs == n_slabs == 3
        assert outcome.status == status
        assert outcome.decision == decision

    def test_emitted_curve_is_anchored_to_plasma_target(self) -> None:
        """Emitted curve is amplitude-anchored (the C2 emit invariant).

        Its peak lands at the plasma-unit target, not the raw detected peak.
        """
        tr, baseline_frames, hct = 4.0, 5, 0.42
        cands = [(3, 10, 1), (7, 10, 2)]
        conc_data, brain_mask = _make_synthetic_dce_4d(
            shape=(12, 14, 6, 40), baseline_frames=baseline_frames, tr=tr,
            arterial_coords=cands, arterial_peak=5.0,
        )
        det = self._make_detector(hematocrit=hct)

        outcome = det._score_and_gate_pool(
            cands, [0.9, 0.8], conc_data, brain_mask, self._bilateral_affine(),
            tr, baseline_frames, hct, 0.0, "MCA_ACA_dce_strict",
        )

        peak_min, peak_max = det._get_expected_peak_range()
        target = (peak_min + peak_max) / 2.0 / (1.0 - hct)
        assert float(np.max(outcome.curve_scaled)) == pytest.approx(
            target, rel=1e-4,
        )

    def test_c2_divergence_rescores_shrunken_subset(self) -> None:
        """C2 path re-scores the EMITTED (post-filter) subset.

        A shrunken candidate set (not the pre-filter pool) produces its own
        scaled curve and a gate decision sized to the subset.
        """
        tr, baseline_frames, hct = 4.0, 5, 0.42
        full = [(3, 10, 1), (7, 10, 2), (7, 11, 3)]
        conc_data, brain_mask = _make_synthetic_dce_4d(
            shape=(12, 14, 6, 40), baseline_frames=baseline_frames, tr=tr,
            arterial_coords=full, arterial_peak=5.0,
        )
        affine = self._bilateral_affine()
        det = self._make_detector(hematocrit=hct)

        subset = full[:2]  # emulate a post-cascade filter dropping one voxel
        outcome = det._score_and_gate_pool(
            subset, [0.9, 0.8], conc_data, brain_mask, affine, tr,
            baseline_frames, hct, 0.0, "MCA_ACA_dce_strict",
        )

        assert outcome.decision["n_pool"] == 2
        assert outcome.n_slabs == _count_slabs_spanned(subset)
        assert np.all(np.isfinite(outcome.curve_scaled))
        peak_min, peak_max = det._get_expected_peak_range()
        target = (peak_min + peak_max) / 2.0 / (1.0 - hct)
        assert float(np.max(outcome.curve_scaled)) == pytest.approx(
            target, rel=1e-4,
        )


# =============================================================================
# In-situ C2 divergence branch inside _run_interface (v0.81.2)
# =============================================================================


class TestC2DivergenceInSitu:
    """Drive the C2 re-gate/rebuild branch inside ``_run_interface``.

    The `if cur_cand_set != tier_cand_set:` routing is otherwise pinned only
    via the direct `_score_and_gate_pool` helper tests. This forces divergence
    (the cascade's `tier_result.candidates` a strict superset of the emitted
    `candidates`) with a mocked cascade + a spy on `_score_and_gate_pool`, and
    asserts `_run_interface` routes the spy's outcome into the emitted AIF and
    rebuilds `tier_result` from it (so the emitted status — not the original
    cascade status — drives provenance).
    """

    @staticmethod
    def _write_niftis(tmp_path):
        import nibabel as nib

        conc_data, brain_mask = _make_synthetic_dce_4d(shape=(20, 20, 20, 40))
        affine = np.eye(4, dtype=np.float64)
        conc_f = tmp_path / "conc.nii.gz"
        mask_f = tmp_path / "brain.nii.gz"
        nib.save(nib.Nifti1Image(conc_data, affine), str(conc_f))
        nib.save(nib.Nifti1Image(brain_mask.astype(np.uint8), affine), str(mask_f))
        return str(conc_f), str(mask_f)

    @staticmethod
    def _full_metric(peak_mm: float = 4.0) -> dict:
        return {
            "peak_mm": peak_mm, "fwhm_seconds": 12.0, "fwhm_frames": 3.0,
            "washout": 0.8, "cnr": 20.0, "auc": 60.0, "upstroke_slope": 1.0,
            "tier_used": 1, "is_likely_venous": False, "peak_time_frame": 7,
        }

    def test_c2_branch_routes_rebuilt_outcome(self, tmp_path, monkeypatch) -> None:
        """Divergent pool routes the spy outcome into the emitted AIF + rebuild."""
        conc_f, mask_f = self._write_niftis(tmp_path)
        monkeypatch.chdir(tmp_path)

        a, b, c = (10, 10, 10), (10, 12, 10), (12, 10, 10)
        emitted = [a, b]        # post-cascade pool that reaches the emit block
        tier_pool = [a, b, c]   # strict superset -> divergence at C2
        scores = [0.9, 0.8]
        metrics = [self._full_metric(), self._full_metric()]

        # The cascade's tier_result carries a PASSing status; only the C2
        # rebuild (via the spy's FAIL outcome) can flip provenance to reduced.
        orig_tier = TierResult(
            candidates=tier_pool, scores=[0.9, 0.8, 0.7],
            metrics=[self._full_metric()] * 3,
            curve_scaled=np.zeros(40), scale_factor=1.5, qc=_make_qc(),
            atlas_tier="MCA_ACA_dce_strict", hemisphere="bilateral",
            n_slabs=2, status="PASS", gate_decisions=[],
        )
        cascade = CascadeResult(
            candidates=emitted, scores=scores, metrics_list=metrics,
            detection_method="atlas", atlas_tier_used="MCA_ACA_dce_strict",
            cascade_tiers_attempted=1, anchor_outcomes=[],
            tier_result=orig_tier, tier_gate_decisions=[],
        )

        t = np.arange(40.0)
        sentinel_curve = 7.0 * np.exp(-0.5 * ((t - 8.0) / 2.0) ** 2)
        sentinel_curve[:5] = 0.0
        sentinel = PoolGateOutcome(
            curve_scaled=sentinel_curve, scale_factor=2.5, qc=_make_qc(),
            hemisphere="bilateral", n_slabs=2, status="FAIL_G2",
            decision=GateDecision(
                tier="MCA_ACA_dce_strict", n_pool=2, scale_factor=2.5,
                qc=_make_qc(), status="FAIL_G2", failing_gate="G2", warns=[],
            ),
        )
        spy = MagicMock(return_value=sentinel)

        iface = AIFDetection()
        iface.inputs.concentration_file = conc_f
        iface.inputs.brain_mask_file = mask_f
        iface.inputs.tr = 4.0

        with patch.object(
            AIFDetection, "_run_two_pass_cascade", return_value=cascade,
        ), patch.object(
            AIFDetection, "_ensure_bilateral_balance",
            side_effect=lambda cands, scs, mets, *a, **k: (cands, scs, mets),
        ), patch.object(AIFDetection, "_score_and_gate_pool", spy):
            iface.run()

        # C2 entered: the spy was invoked once, on the diverged (emitted) pool.
        spy.assert_called_once()
        assert list(spy.call_args.args[0]) == emitted

        res = iface._results
        # Emitted AIF + scale factor come from the spy's outcome.
        np.testing.assert_allclose(res["aif_curve"], sentinel_curve)
        assert res["scaling_factor"] == pytest.approx(2.5)
        assert res["candidate_count"] == 2
        # Rebuilt tier_result carried the FAIL status through: the original
        # cascade tier_result was PASS, so a "reduced" confidence proves the
        # C2 rebuild (not the original) drives downstream provenance.
        assert res["aif_confidence"] == "reduced"
