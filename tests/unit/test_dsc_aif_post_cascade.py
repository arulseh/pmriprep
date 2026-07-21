"""Characterization tests for DSC AIF post-cascade finalization."""

import logging

import numpy as np
import pytest

from pmriprep.interfaces import dsc_aif


def _context(*, disable_fallback: bool = False) -> dsc_aif._DSCAIFDetectionContext:
    concentration = np.zeros((2, 2, 2, 6), dtype=float)
    return dsc_aif._DSCAIFDetectionContext(
        concentration=concentration,
        affine=np.eye(4),
        brain_mask=np.ones(concentration.shape[:3], dtype=bool),
        atlas_mask=None,
        atlas_is_mra=False,
        full_atlas_mask=None,
        lesion_mask=None,
        enhancing_mask=None,
        signal=None,
        aif_method="auto",
        initial_criteria="strict",
        n_candidates=10,
        baseline_frames=2,
        gamma_fit_mode="auto",
        disable_fallback=disable_fallback,
        lesion_distance=15.0,
        n_timepoints=6,
        field_strength=3.0,
        synthetic_effective_dose=1.0,
        voxel_sizes=(1.0, 1.0, 1.0),
        tr=1.5,
    )


def _cascade(*, accepted: bool = True) -> dsc_aif._CascadeDetectionResult:
    curve = np.array([0.0, 0.0, 1.0, 4.0, 2.0, 0.5])
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    candidates = [(0.1, (0, 0, 0), {"snr": 10.0})] if accepted else []
    return dsc_aif._CascadeDetectionResult(
        accepted_aif=(curve, mask, 1) if accepted else None,
        candidates=candidates,
        n_available=len(candidates),
        detection_level="atlas_strict" if accepted else "none",
        filter_metadata={},
        initial_level="atlas_strict",
        tiers_attempted=["atlas_strict"],
        tier_provenance={"atlas_strict": 1} if accepted else {},
        postfilter_diagnostics={"atlas_strict": {"aif_n_after_filters": 1}},
    )


def _detected_result(*, baseline_frames: int = 2) -> dsc_aif._FinalizedDetectedAIF:
    curve = np.array([0.0, 0.0, 1.0, 4.0, 2.0, 0.5])
    mask = np.zeros((2, 2, 2), dtype=np.uint8)
    mask[0, 0, 0] = 1
    return dsc_aif._FinalizedDetectedAIF(
        curve=curve,
        mask=mask,
        quality_score=0.8,
        qc_metrics={
            "aif_quality": "detected",
            "aif_voxel_coord": [0, 0, 0],
            "cascade_downgraded": False,
        },
        baseline_frames=baseline_frames,
    )


def test_gamma_r2_floor_replaces_detected_state_with_synthetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected sub-floor gamma fit clears spatial state and uses synthetic AIF."""
    detector = dsc_aif.AIFDetection()
    synthetic = np.array([0.0, 0.0, 2.0, 8.0, 3.0, 1.0])
    monkeypatch.setattr(
        detector,
        "_finalize_detected_aif",
        lambda *args: _detected_result(baseline_frames=4),
    )

    def reject_gamma(
        *args: object,
    ) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
        curve = np.asarray(args[0])
        qc_metrics = args[-1]
        assert isinstance(qc_metrics, dict)
        qc_metrics["gamma_fit_r_squared"] = -0.25
        return False, curve * 0.9, curve, curve * 0.9

    monkeypatch.setattr(detector, "_apply_gamma_fit_if_appropriate", reject_gamma)
    monkeypatch.setattr(
        detector,
        "_prepare_cached_synthetic_fallback",
        lambda *args: (
            synthetic,
            {"synthetic_peak": 8.0, "synthetic_aif_arrival_frame": 3},
        ),
    )

    state = detector._finalize_post_cascade(
        _context(), _cascade(), 2, logging.getLogger(__name__)
    )

    np.testing.assert_array_equal(state.curve, synthetic)
    np.testing.assert_array_equal(state.measured_curve, synthetic)
    assert not state.mask.any()
    assert state.gamma_curve is None
    assert state.raw_curve is None
    assert state.detection_level == "synthetic"
    assert state.quality_score == 0.3
    assert state.qc_metrics["synthetic_fallback_reason"] == "gamma_r2_floor"
    assert state.qc_metrics["gamma_r2_at_fallback"] == -0.25
    assert state.qc_metrics["bootstrap_baseline_frames"] == 2
    assert state.qc_metrics["baseline_frames_used"] == 2
    assert state.qc_metrics["n_baseline_frames"] == 2
    assert state.qc_metrics["candidate_baseline_frames"] == 4
    assert state.qc_metrics["arrival_time_seconds"] == 4.5
    assert state.qc_metrics["aif_metrics"]["arrival"] == 4.5
    assert state.qc_metrics["rejected_measured_aif"]["baseline_frames"] == 4
    assert state.qc_metrics["cbv_auc_denominator_is_synthetic"] is True
    assert "aif_voxel_coord" not in state.qc_metrics
    assert "cascade_postfilter_diags" in state.qc_metrics


def test_disabled_fallback_retains_sub_floor_detected_aif(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabling fallback retains the measured curve and exposes degraded QC."""
    detector = dsc_aif.AIFDetection()
    monkeypatch.setattr(
        detector,
        "_finalize_detected_aif",
        lambda *args: _detected_result(),
    )

    def reject_gamma(*args: object) -> tuple[bool, None, np.ndarray, np.ndarray]:
        curve = np.asarray(args[0])
        qc_metrics = args[-1]
        assert isinstance(qc_metrics, dict)
        qc_metrics["gamma_fit_r_squared"] = -0.25
        return False, None, curve, curve * 0.9

    monkeypatch.setattr(detector, "_apply_gamma_fit_if_appropriate", reject_gamma)

    state = detector._finalize_post_cascade(
        _context(disable_fallback=True),
        _cascade(),
        2,
        logging.getLogger(__name__),
    )

    assert state.detection_level == "atlas_strict"
    assert state.quality_score == 0.8
    assert state.mask.sum() == 1
    assert state.qc_metrics["aif_gamma_fit_below_floor_retained"] is True
    assert state.qc_metrics["gamma_r2_retained"] == -0.25
    assert state.qc_metrics["aif_quality"] == "detected"


def test_cascade_exhaustion_builds_complete_synthetic_qc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-candidate fallback stamps acquisition and postfilter QC in one state."""
    detector = dsc_aif.AIFDetection()
    synthetic = np.array([0.0, 0.0, 2.0, 8.0, 3.0, 1.0])
    monkeypatch.setattr(
        detector,
        "_prepare_cached_synthetic_fallback",
        lambda *args: (synthetic, {"synthetic_peak": 8.0}),
    )

    state = detector._finalize_post_cascade(
        _context(), _cascade(accepted=False), 3, logging.getLogger(__name__)
    )

    assert state.detection_level == "synthetic"
    assert state.candidate_count == 0
    assert state.baseline_frames == 3
    assert state.qc_metrics["synthetic_fallback_reason"] == "no_candidates"
    assert state.qc_metrics["gamma_fit_rejected_reason"] == "synthetic_aif"
    assert state.qc_metrics["aif_baseline_frames"] == 3
    assert state.qc_metrics["bootstrap_baseline_frames"] == 2
    assert state.qc_metrics["baseline_frames_used"] == 3
    assert state.qc_metrics["cascade_postfilter_diags"] == (
        _cascade(accepted=False).postfilter_diagnostics
    )
