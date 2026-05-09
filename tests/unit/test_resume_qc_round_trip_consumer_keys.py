"""Round-trip parity oracle: consumer-key set survives resume sidecar.

Code-reviewer round 2 requirement. Defines the set of keys actually read
by downstream consumers (``dsc_cbv_constraint.py``, ``dce_compute.py``)
from the producer-shape ``aif_qc`` / ``preload_qc`` /
``preprocessing_stats`` dicts. Asserts the round-trip
in-process-dict → resume sidecar → resume helper recovers each consumer
key verbatim.

This guards against silent field omissions in the resume sidecar — the
verbatim-companion design only works if the dict round-trip is truly
verbatim. Future filtering ("PII protection", schema enforcement, etc.)
must add to / not remove from this oracle.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pmriprep.utils.sidecar_builders import build_dsc_resume_sidecar
from pmriprep.workflows.base import _find_previous_dsc_outputs
from pmriprep.workflows.dsc.datasinks import _write_resume_sidecar


# Keys read by downstream consumers from the producer dicts. Sources:
# - dsc_cbv_constraint.py:_normalize_aif_qc_keys (translates
#   aif_metrics.psr → aif_peak_height, aif_metrics.brr →
#   baseline_return_ratio, gamma_fit_auto_metrics.peak_ratio →
#   gamma_peak_ratio)
# - dsc_cbv_constraint.py:assess_dsc_cbv_quality
# - dce_compute.py:1567-1620 PK setup
# - utils/dsc_aif_provenance.py
# - utils/qc_standards.py:recommend_dsc_qc
#
# Detection-time AIF metrics live under the nested ``aif_metrics`` key;
# the round-trip oracle compares the nested dict by value.
_AIF_QC_CONSUMER_KEYS = {
    "detection_method",
    "n_aif_voxels_selected",
    "aif_metrics",                       # nested: psr, brr, peak_conc, ttp, fwhm, ...
    "cbv_auc_denominator_is_synthetic",
    "synthetic",
    "gamma_fit_applied",
    "gamma_fit_auto_metrics",            # contains peak_ratio sub-key
    "hematocrit",
}

_PRELOAD_QC_CONSUMER_KEYS = {
    "preload_classification",
    "k2_method",
    "k2_mean", "k2_std", "k2_max", "k2_median",
    "percent_leaky_voxels", "percent_severe_leakage",
    "percent_above_clinical_max",
    "n_brain_voxels",
    "auc_mean",
    "kep_used",
}

_PREPROC_STATS_CONSUMER_KEYS = {
    "fraction_cnr_gt_4",
    "mean_cnr",
    "fd_mean", "fd_max",
    "percent_outliers",
    "motion_status",
}


@pytest.fixture
def representative_in_process_dicts():
    """Producer-shape dicts populated with every consumer key."""
    aif_qc = {
        "detection_method": "atlas_strict",
        "n_aif_voxels_selected": 6,
        "aif_metrics": {
            "peak_conc": 12.5,
            "fwhm": 6.2,
            "ttp": 18.0,
            "psr": 78.0,
            "brr": 92.0,
        },
        "cbv_auc_denominator_is_synthetic": False,
        "synthetic": False,
        "gamma_fit_applied": True,
        "gamma_fit_auto_metrics": {"peak_ratio": 1.05, "shape_score": 0.9},
        "hematocrit": 0.42,
        # Bonus producer keys (not consumed but should still round-trip)
        "quality_score": 0.81,
    }
    preload_qc = {
        "preload_classification": "none",
        "k2_method": "bsw",
        "k2_mean": 0.011, "k2_std": 0.005,
        "k2_max": 0.087, "k2_median": 0.012,
        "percent_leaky_voxels": 4.7,
        "percent_severe_leakage": 0.3,
        "percent_above_clinical_max": 0.0,
        "n_brain_voxels": 142_000,
        "auc_mean": 1.23,
        "kep_used": None,
    }
    preprocessing_stats = {
        "fraction_cnr_gt_4": 0.81,
        "mean_cnr": 5.4,
        "fd_mean": 0.18, "fd_max": 0.62,
        "percent_outliers": 1.5,
        "motion_status": "PASS",
        # Bonus
        "tr_ms": 1500.0, "n_volumes": 60,
    }
    return aif_qc, preload_qc, preprocessing_stats


def test_consumer_keys_round_trip(tmp_path, representative_in_process_dicts):
    aif_qc, preload_qc, preprocessing_stats = representative_in_process_dicts
    meta = build_dsc_resume_sidecar(
        aif_qc=aif_qc, preload_qc=preload_qc,
        preprocessing_stats=preprocessing_stats,
    )
    _write_resume_sidecar(
        output_dir=str(tmp_path), subject="X", session="Y", meta_dict=meta,
    )
    recovered = _find_previous_dsc_outputs(
        output_dir=tmp_path, subject_id="X", session_id="Y",
    )

    # Every consumer key in the in-process dict MUST appear in the
    # recovered dict with the same value. Future filtering of the
    # resume sidecar would silently break consumers; this test catches it.
    rec_aif = recovered["aif_qc"]
    assert rec_aif is not None
    for key in _AIF_QC_CONSUMER_KEYS:
        assert key in rec_aif, f"aif_qc consumer key '{key}' lost in round-trip"
        assert rec_aif[key] == aif_qc[key], (
            f"aif_qc[{key!r}] changed: in={aif_qc[key]!r} out={rec_aif[key]!r}"
        )

    rec_preload = recovered["preload_qc"]
    assert rec_preload is not None
    for key in _PRELOAD_QC_CONSUMER_KEYS:
        assert key in rec_preload, f"preload_qc consumer key '{key}' lost"
        assert rec_preload[key] == preload_qc[key]

    rec_preproc = recovered["preprocessing_stats"]
    assert rec_preproc is not None
    for key in _PREPROC_STATS_CONSUMER_KEYS:
        assert key in rec_preproc, f"preprocessing_stats consumer key '{key}' lost"
        assert rec_preproc[key] == preprocessing_stats[key]


def test_normalizer_handles_round_tripped_aif_qc(tmp_path, representative_in_process_dicts):
    """End-to-end: round-tripped dict feeds dsc_cbv_constraint normalizer cleanly."""
    aif_qc, preload_qc, preprocessing_stats = representative_in_process_dicts
    meta = build_dsc_resume_sidecar(
        aif_qc=aif_qc, preload_qc=preload_qc,
        preprocessing_stats=preprocessing_stats,
    )
    _write_resume_sidecar(
        output_dir=str(tmp_path), subject="X", session="Y", meta_dict=meta,
    )
    recovered = _find_previous_dsc_outputs(
        output_dir=tmp_path, subject_id="X", session_id="Y",
    )

    from pmriprep.utils.dsc_cbv_constraint import _normalize_aif_qc_keys

    normalized = _normalize_aif_qc_keys(dict(recovered["aif_qc"]))
    # Translator creates the post-normalization keys from producer keys
    assert "aif_peak_height" in normalized
    assert normalized["aif_peak_height"] == pytest.approx(0.78)
    assert "baseline_return_ratio" in normalized
    assert normalized["baseline_return_ratio"] == pytest.approx(0.92)
    assert "gamma_peak_ratio" in normalized
    assert normalized["gamma_peak_ratio"] == pytest.approx(1.05)
