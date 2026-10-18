"""DSC AIF interface orchestration, diagnostic curves and output contracts."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pmriprep.interfaces import dsc_aif
import nibabel as nib
from nipype.interfaces.base import isdefined


def _cascade(*, accepted: bool = True) -> dsc_aif._CascadeDetectionResult:
    curve = np.array([0.0, 0.0, 1.0, 4.0, 2.0, 0.5])
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    candidates = [(0.1, (0, 0, 0), {"snr": 10.0})] if accepted else []
    return dsc_aif._CascadeDetectionResult(
        accepted_aif=(curve, mask, 1) if accepted else None,
        candidates=candidates,
        n_available=len(candidates),
        detection_level="atlas_strict" if accepted else "unavailable",
        filter_metadata={},
        initial_level="atlas_strict",
        tiers_attempted=["atlas_strict"],
        tier_provenance={"atlas_strict": 1} if accepted else {},
        postfilter_diagnostics={"atlas_strict": {"aif_n_after_filters": 1}},
        spatial_filter_diagnostics={},
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


@pytest.mark.parametrize("outcome", ["accepted", "accepted_fit_failed", "rejected", "no_candidates"])
def test_run_interface_preserves_analysis_and_diagnostic_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    """The public runner exports usable input separately from diagnostic artifacts."""
    concentration = tmp_path / "concentration.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 6)), np.eye(4)), concentration)
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), mask)
    detector = dsc_aif.AIFDetection(
        concentration_file=str(concentration), brain_mask_file=str(mask),
        baseline_frames=3, tr=1.5,
    )
    accepted = outcome.startswith("accepted")
    cascade = _cascade(accepted=accepted)
    measured = _detected_result()
    diagnostic = measured.curve * 0.5
    if outcome == "rejected":
        cascade = cascade._replace(rejected=dsc_aif._RejectedAIF(
            measured.curve, measured.mask,
            {"reason": "quality_checks_failed", "candidate_coords": [[0, 0, 0]],
             "candidate_scores": [0.1]},
        ))
    def fixed_cascade(
        context: dsc_aif._DSCAIFDetectionContext, baseline_frames: int, logger: Any,
    ) -> dsc_aif._CascadeDetectionResult:
        assert baseline_frames == 2
        return cascade

    def diagnostic_fit(
        curve: np.ndarray, baseline_frames: int, tr: float, qc: dict[str, Any],
    ) -> np.ndarray | None:
        assert baseline_frames == 2 and tr == 1.5
        np.testing.assert_array_equal(curve, measured.curve)
        qc["gamma_fit_converged"] = outcome != "accepted_fit_failed"
        if outcome == "accepted_fit_failed":
            qc["gamma_fit_failure_reason"] = "optimizer_not_converged"
            return None
        qc["gamma_fit_r_squared"] = -0.25
        return diagnostic

    monkeypatch.setattr(detector, "_prepare_detection_baseline", lambda *args: dsc_aif._TissueBaseline(2, {"confidence": "high"}))
    monkeypatch.setattr(detector, "_run_candidate_cascade", fixed_cascade)
    monkeypatch.setattr(detector, "_finalize_detected_aif", lambda *args: measured)
    monkeypatch.setattr(detector, "_fit_gamma_for_qc", diagnostic_fit)
    outputs = detector.run(cwd=str(tmp_path)).outputs
    qc = outputs.qc_metrics
    json.dumps(qc, allow_nan=False)
    assert outputs.aif_baseline_frames == 2
    assert qc["bootstrap_baseline_frames"] == 3
    assert qc["tissue_baseline_confidence"] == "high"
    assert qc["cascade_tiers_attempted"] == ["atlas_strict"]
    assert qc["cbv_auc_denominator_available"] is accepted
    assert outputs.detection_level == ("atlas_strict" if accepted else "unavailable")
    if accepted:
        np.testing.assert_array_equal(np.loadtxt(outputs.out_aif_curve), measured.curve)
        np.testing.assert_array_equal(nib.load(outputs.out_aif_mask).get_fdata(), measured.mask)
        assert qc["detection_status"] == "accepted"
        assert outputs.candidate_count == 1
    else:
        assert not isdefined(outputs.out_aif_curve)
        assert not isdefined(outputs.out_aif_mask)
        assert qc["detection_status"] == "failed"
        assert outputs.candidate_count == qc["n_aif_voxels_selected"] == 0
        assert qc["failure_reason"] == ("quality_checks_failed" if outcome == "rejected" else "no_candidates")
    if outcome == "rejected":
        np.testing.assert_array_equal(np.loadtxt(outputs.out_rejected_aif_curve), measured.curve)
        np.testing.assert_array_equal(nib.load(outputs.out_rejected_aif_mask).get_fdata(), measured.mask)
        assert qc["rejected_candidate"]["candidate_coords"] == [[0, 0, 0]]
    else:
        assert not isdefined(outputs.out_rejected_aif_curve)
    if outcome in {"accepted", "rejected"}:
        np.testing.assert_array_equal(np.loadtxt(outputs.out_gamma_aif_curve), diagnostic)
        fit_qc = qc["rejected_candidate"] if outcome == "rejected" else qc
        assert fit_qc["gamma_fit_r_squared"] == -0.25
    else:
        assert not isdefined(outputs.out_gamma_aif_curve)
        if outcome == "accepted_fit_failed":
            assert qc["gamma_fit_failure_reason"] == "optimizer_not_converged"


def test_gamma_diagnostic_preserves_measured_curve_and_reports_window() -> None:
    from pmriprep.utils.dsc_aif import fit_gamma_variate
    time = np.arange(80, dtype=float)
    first = np.maximum(time - 10, 0)**3 * np.exp(-np.maximum(time - 10, 0) / 2)
    second = 0.2 * np.maximum(time - 32, 0)**3 * np.exp(-np.maximum(time - 32, 0) / 3)
    curve = first + second
    original = curve.copy()
    qc = {}
    gamma = dsc_aif.AIFDetection._fit_gamma_for_qc(curve, 8, 1.0, qc)
    np.testing.assert_array_equal(curve, original)
    assert gamma is not None
    assert qc["gamma_fit_converged"]
    start, end = qc["gamma_fit_diagnostics"]["first_pass_window"]
    expected_fraction = np.trapezoid(curve[start:end]) / np.trapezoid(curve[start:])
    assert qc["gamma_fit_diagnostics"]["first_pass_area_fraction"] == pytest.approx(expected_fraction)
    assert expected_fraction < 0.8
    reference = fit_gamma_variate(curve, baseline_frames=8, remove_recirculation=True, preserve_auc=True, tr=1.0)
    np.testing.assert_allclose(gamma, reference.fitted_curve)
