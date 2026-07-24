"""Characterization tests for detected DSC AIF finalization."""

from types import SimpleNamespace

import numpy as np
import pytest

from pmriprep.interfaces import dsc_aif


@pytest.mark.parametrize(
    ("preliminary", "bootstrap", "refined", "confidence", "expected", "source"),
    [
        (5, 5, 12, "medium", 12, "selected_aif_fallback_recovery"),
        (14, 5, 7, "high", 7, "selected_aif_high_confidence"),
        (14, 5, 19, "medium", 14, "whole_brain_percentile"),
    ],
)
def test_selected_aif_refinement_preserves_confidence_policy(
    monkeypatch: pytest.MonkeyPatch,
    preliminary: int,
    bootstrap: int,
    refined: int,
    confidence: str,
    expected: int,
    source: str,
) -> None:
    """Selected arterial timing refines only under the established safeguards."""
    monkeypatch.setattr(
        "pmriprep.utils.baseline_detection.detect_baseline_ensemble",
        lambda *args, **kwargs: (
            refined,
            {"confidence": confidence, "method": "change_point"},
        ),
    )

    result, diagnostics = dsc_aif.AIFDetection._refine_selected_aif_baseline(
        np.r_[np.zeros(10), np.ones(20)],
        preliminary_baseline=preliminary,
        bootstrap_baseline=bootstrap,
        tr=1.5,
    )

    assert result == expected
    assert diagnostics["baseline_refinement_source"] == source
    assert diagnostics["baseline_frames_used"] == expected


def test_detected_aif_finalization_builds_mask_and_cascade_qc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cascade result is converted into mask, quality, and provenance QC."""
    detector = dsc_aif.AIFDetection()
    curve = np.array([0.0, 0.0, 0.0, 4.0, 10.0, 5.0, 2.0, 1.0])
    accepted_mask = np.zeros((3, 3, 3), dtype=bool)
    accepted_mask[1, 1, 1] = True
    candidates = [
        (1.0, (1, 1, 1), {"peak_time_frame": 4, "tail_fraction_late": 0.1}),
        (3.0, (1, 1, 2), {"peak_time_frame": 6, "tail_fraction_late": 0.2}),
    ]
    cascade = dsc_aif._CascadeDetectionResult(
        accepted_aif=(curve, accepted_mask, 1),
        candidates=candidates,
        n_available=4,
        detection_level="atlas_relaxed+brain_relaxed",
        filter_metadata={"distance_filter": "applied"},
        initial_level="atlas_strict",
        tiers_attempted=["atlas_strict", "atlas_relaxed", "brain_relaxed"],
        tier_provenance={"atlas_relaxed": 1, "brain_relaxed": 1},
        postfilter_diagnostics={
            "atlas_relaxed": {
                "aif_n_after_filters": 1,
                "aif_n_filtered_peakspread": 1,
                "aif_n_filtered_tail": 0,
            },
            "brain_relaxed": {
                "aif_n_after_filters": 1,
                "aif_n_filtered_peakspread": 0,
                "aif_n_filtered_tail": 2,
            },
        },
    )
    context = SimpleNamespace(
        concentration=np.zeros((3, 3, 3, 8)),
        brain_mask=np.ones((3, 3, 3)),
        affine=np.eye(4),
        tr=1.5,
        n_candidates=10,
        baseline_frames=3,
    )
    monkeypatch.setattr(
        detector,
        "_finalize_aif_metrics",
        lambda *args, **kwargs: (curve, {"n_aif_voxels_available": 4}),
    )
    monkeypatch.setattr(detector, "_compute_aif_spr", lambda *args: (3.0, 4, 10.0))

    monkeypatch.setattr(
        detector,
        "_refine_selected_aif_baseline",
        lambda *args, **kwargs: (3, {"baseline_refinement_source": "test"}),
    )
    result = detector._finalize_detected_aif(cascade, context, 3)

    assert result.mask[1, 1, 1] == 1
    assert result.mask[1, 1, 2] == 1
    assert int(result.mask.sum()) == 2
    assert result.quality_score == pytest.approx(1.0 / 3.0)
    assert result.qc_metrics["cascade_downgraded"] is True
    assert result.qc_metrics["aif_tier_provenance"] == cascade.tier_provenance
    assert result.qc_metrics["aif_n_after_filters"] == 2
    assert result.qc_metrics["aif_n_filtered_peakspread"] == 1
    assert result.qc_metrics["aif_n_filtered_tail"] == 2
    assert result.qc_metrics["aif_peak_time_spread_frames"] == 2.0
    assert result.qc_metrics["aif_tail_max_fraction"] == 0.2
