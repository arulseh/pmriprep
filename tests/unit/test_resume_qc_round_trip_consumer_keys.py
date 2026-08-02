"""Round-trip parity oracle: consumer-key set survives work/ checkpoint.

Defines the set of keys read by downstream consumers from the
producer-shape dicts. Asserts the round-trip via the work/ checkpoint
recovers each consumer key verbatim.
"""
from __future__ import annotations


import pytest

from pmriprep.workflows.base import _find_previous_dsc_outputs
from pmriprep.workflows.dsc.datasinks import _write_dsc_resume_checkpoint

_AIF_QC_CONSUMER_KEYS = {
    "detection_method",
    "n_aif_voxels_selected",
    "aif_metrics",
    "cbv_auc_denominator_is_synthetic",
    "synthetic",
    "gamma_fit_applied",
    "gamma_fit_auto_metrics",
    "hematocrit",
}

_PRELOAD_QC_CONSUMER_KEYS = {
    "preload_source",
    "preload_administered_dose_fraction",
    "k2_method",
    "k2_mean", "k2_std", "k2_min", "k2_max", "k2_median", "k2_p05", "k2_p95",
    "percent_negative_k2", "percent_k2_undefined",
    "n_brain_voxels", "n_k2_valid",
    "auc_mean",
    "kep_used",
    "percent_negative_k2",
    "k2_signed",
    "k2_reference_available",
    "k2_reference_total_voxels",
    "k2_reference_selected_voxels",
    "k2_reference_selected_fraction",
    "k2_reference_baseline_frames",
    "k2_reference_tail_frames",
    "k2_reference_selection",
    "k2_reference_selection_domain",
    "k2_reference_peak",
    "k2_reference_failure_reason",
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
        "quality_score": 0.81,
    }
    preload_qc = {
        "preload_source": "user_specified",
        "preload_administered_dose_fraction": 0.0,
        "k2_method": "bsw",
        "k2_mean": 0.011, "k2_std": 0.005,
        "k2_min": -0.021, "k2_max": 0.087, "k2_median": 0.012,
        "k2_p05": -0.004, "k2_p95": 0.033,
        "n_brain_voxels": 142_000,
        "n_k2_valid": 141_000,
        "auc_mean": 1.23,
        "kep_used": None,
        "percent_negative_k2": 41.0,
        "percent_k2_undefined": 1000 / 1420,
        "k2_signed": True,
        "k2_reference_available": True,
        "k2_reference_total_voxels": 1000,
        "k2_reference_selected_voxels": 600,
        "k2_reference_selected_fraction": 0.6,
        "k2_reference_baseline_frames": 20,
        "k2_reference_tail_frames": 10,
        "k2_reference_selection": "raw_signal_final10_mean_1_baseline_sample_sd",
        "k2_reference_selection_domain": "raw_signal",
        "k2_reference_peak": 4.0,
        "k2_reference_failure_reason": "",
    }
    preprocessing_stats = {
        "fraction_cnr_gt_4": 0.81,
        "mean_cnr": 5.4,
        "fd_mean": 0.18, "fd_max": 0.62,
        "percent_outliers": 1.5,
        "motion_status": "PASS",
        "tr_ms": 1500.0, "n_volumes": 60,
    }
    return aif_qc, preload_qc, preprocessing_stats


def test_consumer_keys_round_trip(tmp_path, representative_in_process_dicts):
    aif_qc, preload_qc, preprocessing_stats = representative_in_process_dicts
    _write_dsc_resume_checkpoint(
        work_dir=str(tmp_path),
        subject_id="X", session_id="Y",
        aif_qc=aif_qc, preload_qc=preload_qc,
        preprocessing_stats=preprocessing_stats,
    )
    recovered = _find_previous_dsc_outputs(
        output_dir=tmp_path, subject_id="X", session_id="Y",
        work_dir=tmp_path,
    )

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
    aif_qc, preload_qc, preprocessing_stats = representative_in_process_dicts
    _write_dsc_resume_checkpoint(
        work_dir=str(tmp_path),
        subject_id="X", session_id="Y",
        aif_qc=aif_qc, preload_qc=preload_qc,
        preprocessing_stats=preprocessing_stats,
    )
    recovered = _find_previous_dsc_outputs(
        output_dir=tmp_path, subject_id="X", session_id="Y",
        work_dir=tmp_path,
    )

    from pmriprep.utils.dsc_cbv_constraint import _normalize_aif_qc_keys

    normalized = _normalize_aif_qc_keys(dict(recovered["aif_qc"]))
    assert "aif_peak_height" in normalized
    assert normalized["aif_peak_height"] == pytest.approx(0.78)
    assert "baseline_return_ratio" in normalized
    assert normalized["baseline_return_ratio"] == pytest.approx(0.92)
    assert "gamma_peak_ratio" in normalized
    assert normalized["gamma_peak_ratio"] == pytest.approx(1.05)
