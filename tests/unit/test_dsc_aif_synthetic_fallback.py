"""Characterization tests for cached DSC synthetic-AIF fallback preparation."""

import numpy as np
import pytest

from pmriprep.interfaces.dsc_aif import AIFDetection


def _detector_with_cache() -> AIFDetection:
    """Return a detector with the synthetic cache populated."""
    detector = AIFDetection()
    detector._cached_synth_aif = np.array([0.0, 10.0, 2.0])
    detector._cached_synth_metrics = {"synthetic_aif_target_peak": 10.0}
    return detector


def test_tissue_peak_adds_relative_synthetic_qc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A positive tissue peak produces both tissue and peak-ratio metrics."""
    detector = _detector_with_cache()
    monkeypatch.setattr(
        detector,
        "_compute_tissue_max_for_synthetic",
        lambda concentration, brain_mask: 4.0,
    )

    curve, metrics = detector._prepare_cached_synthetic_fallback(
        np.zeros((2, 2, 2, 3)),
        np.ones((2, 2, 2)),
    )

    assert metrics["synthetic_aif_tissue_p95_delta_r2star"] == 4.0
    assert metrics["synthetic_aif_peak_to_tissue_ratio"] == 2.5
    assert curve is not detector._cached_synth_aif
    assert metrics is not detector._cached_synth_metrics


def test_zero_tissue_peak_is_recorded_without_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero tissue peak remains visible in QC without dividing by zero."""
    detector = _detector_with_cache()
    monkeypatch.setattr(
        detector,
        "_compute_tissue_max_for_synthetic",
        lambda concentration, brain_mask: 0.0,
    )

    _, metrics = detector._prepare_cached_synthetic_fallback(
        np.zeros((2, 2, 2, 3)),
        np.ones((2, 2, 2)),
    )

    assert metrics["synthetic_aif_tissue_p95_delta_r2star"] == 0.0
    assert "synthetic_aif_peak_to_tissue_ratio" not in metrics


def test_missing_tissue_peak_leaves_cached_metrics_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unavailable tissue signal does not add tissue-relative QC fields."""
    detector = _detector_with_cache()
    monkeypatch.setattr(
        detector,
        "_compute_tissue_max_for_synthetic",
        lambda concentration, brain_mask: None,
    )

    _, metrics = detector._prepare_cached_synthetic_fallback(
        np.zeros((2, 2, 2, 3)),
        np.ones((2, 2, 2)),
    )

    assert metrics == {"synthetic_aif_target_peak": 10.0}
