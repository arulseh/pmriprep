"""Characterization tests for detected DSC AIF finalization."""

from types import SimpleNamespace

import numpy as np
import pytest

from pmriprep.interfaces import dsc_aif


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
    )
    monkeypatch.setattr(
        detector,
        "_finalize_aif_metrics",
        lambda *args, **kwargs: (curve, {"n_aif_voxels_available": 4}),
    )
    monkeypatch.setattr(detector, "_compute_aif_spr", lambda *args: (3.0, 4, 10.0))

    result = detector._finalize_detected_aif(cascade, context, 3, 3)

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
