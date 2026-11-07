"""Supplied search regions constrain arterial detection and its fallback pools."""

import json
import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest
from nipype.interfaces.base import isdefined

from pmriprep.interfaces import dce_aif, dsc_aif
from pmriprep.utils.aif_provenance import build_aif_provenance_block


def _inputs(tmp_path: Path, modality: str, region: np.ndarray) -> dict[str, object]:
    """Write concentration with an arterial pulse everywhere outside the ROI."""
    time = np.arange(40, dtype=float)
    curve = 5 * np.exp(-0.5 * ((time - 8) / 1.0) ** 2)
    concentration = np.broadcast_to(curve, (*region.shape, 40)).copy()
    concentration[region > 0.5] = 0
    paths = {}
    for name, array in (
        ("concentration", concentration), ("brain_mask", np.ones_like(region)),
        ("aif_mask", region),
    ):
        path = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(array.astype(np.float32), np.diag([2., 2., 2., 1.])), path)
        paths[f"{name}_file"] = str(path)
    return {**paths, "tr": 1.5 if modality == "DSC" else 4.0, "baseline_frames": 5}


@pytest.mark.parametrize("modality", ["DSC", "DCE"])
@pytest.mark.parametrize("empty", [False, True])
def test_exhausted_user_mask_uses_nonfatal_fallback(
    tmp_path: Path, modality: str, empty: bool,
) -> None:
    """An empty or signal-free ROI cannot escape to arterial voxels elsewhere."""
    region = np.zeros((16, 16, 16))
    if not empty:
        region[6:10, 6:10, 6:10] = 1
    cls = dsc_aif.AIFDetection if modality == "DSC" else dce_aif.AIFDetection
    detector = cls(**_inputs(tmp_path, modality, region))
    outputs = detector.run(cwd=str(tmp_path)).outputs
    qc = outputs.qc_metrics
    assert qc["detection_status"] == "failed"
    assert qc["search_region"] == "user_mask"
    assert qc["failure_reason"] == "user_mask_exhausted"
    assert qc["failure_detail"] == ("empty_user_mask" if empty else "no_candidates")
    assert outputs.candidate_count == 0
    block = build_aif_provenance_block(qc, modality=modality)
    assert block["SearchRegion"] == "user_mask"
    assert block["FailureReason"] == "user_mask_exhausted"
    json.dumps(block, allow_nan=False)
    if modality == "DSC":
        assert outputs.detection_level == "unavailable"
        assert not isdefined(outputs.out_aif_curve)
        assert not qc["cbv_auc_denominator_available"]
        tiers = qc["cascade_tiers_attempted"]
    else:
        assert outputs.detection_method == "population"
        assert np.max(np.loadtxt(outputs.out_aif_curve)) > 0
        assert qc["auto_fallback_reason"] == "user_mask_exhausted"
        tiers = [item["tier"] for item in qc["cascade_gate_decisions"]]
    assert tiers == [f"user_mask_{criterion}" for criterion in ("strict", "relaxed", "permissive")]


@pytest.mark.parametrize("route", ["pool", "union", "single"])
def test_dsc_user_mask_bounds_cascade_and_neighborhood_averaging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str,
) -> None:
    """Tier, union and expanded single-voxel averages use only supplied voxels."""
    region = np.zeros((16, 16, 16))
    coords = [(8, 5, 7), (8, 6, 7), (8, 7, 7)]
    for coord in coords:
        region[coord] = 1
    detector = dsc_aif.AIFDetection(**_inputs(tmp_path, "DSC", region))
    context = detector._load_detection_context(logging.getLogger(__name__))
    time = np.arange(40)
    pulse = 5 * np.exp(-0.5 * ((time - 10) / 2) ** 2)
    signal = np.full_like(context.concentration, np.nan)
    for coord in coords:
        context.concentration[coord] = pulse
        signal[coord] = 1000 * np.exp(-0.04 * pulse)
    context = context._replace(signal=signal)
    # Candidate scoring is controlled so each acceptance route can be isolated;
    # search-mask preparation, ROI averaging and cascade accumulation run normally.
    def score(voxels: np.ndarray, *args: Any, **kwargs: Any) -> tuple[list[tuple], int]:
        available = {tuple(v) for v in voxels}
        assert available == set(coords)
        criteria = args[6]
        keep = coords if route == "pool" else [coords[("strict", "relaxed", "permissive").index(criteria)]]
        return [(0.1 + i / 10, c, {}) for i, c in enumerate(keep)], len(keep)

    monkeypatch.setattr(detector, "_score_and_select_candidates", score)
    monkeypatch.setattr(detector, "_apply_postscore_aif_filters", lambda candidates, **kw: (candidates, {}))
    monkeypatch.setattr(detector, "_aif_passes_quality", lambda *a: dsc_aif.AifQualityResult(route != "single", 0., 1., 0.))
    if route == "single":
        # Exercise the expanded neighborhood after the single voxel fails its gate.
        outcomes = iter([False, True])
        monkeypatch.setattr(detector, "_aif_global_max_in_fp", lambda *a: next(outcomes))
        monkeypatch.setattr(detector, "_compute_aif_brr", lambda *a: 0.)
    result = detector._run_candidate_cascade(context, 5, logging.getLogger(__name__))
    assert result.accepted_aif is not None
    curve, used, roi_size = result.accepted_aif
    assert np.any(used) and np.all(region[used] > 0.5)
    assert all(region[c[1]] > 0.5 for c in result.candidates)
    expected_curve, _ = detector._compute_roi_averaged_aif(
        result.candidates, context.concentration, region, context.tr, roi_size=roi_size,
    )
    np.testing.assert_allclose(curve, expected_curve)
    assert len(result.candidates) == (1 if route == "single" else 3)
    assert roi_size == (3 if route == "single" else 1)
    if route == "union":
        assert result.detection_level == "user_mask_strict+user_mask_relaxed+user_mask_permissive"
    final = detector._finalize_post_cascade(
        context, result, dsc_aif._TissueBaseline(5, {"confidence": "high"}), logging.getLogger(__name__),
    )
    block = build_aif_provenance_block(final.qc_metrics, modality="DSC")
    assert block["Source"] == "measured" and block["SearchRegion"] == "user_mask"
    assert np.isfinite(final.qc_metrics["aif_metrics"]["peak_height_percent"])


@pytest.mark.parametrize("count,rejected", [(1, False), (2, False), (2, True)])
def test_dce_user_mask_bounds_pool_and_singleton(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int, rejected: bool,
) -> None:
    """Both cascade passes take their candidates from the supplied region."""
    region = np.zeros((16, 16, 16))
    region[7, 7:7 + count, 7] = 1
    detector = dce_aif.AIFDetection(**_inputs(tmp_path, "DCE", region))
    concentration = nib.load(detector.inputs.concentration_file).get_fdata()
    pulse = 5 * np.exp(-0.5 * (np.arange(40) - 7) ** 2)
    if rejected:
        pulse[15:] = 4.5
    concentration[region > 0.5] = pulse
    nib.save(nib.Nifti1Image(concentration, np.diag([2., 2., 2., 1.])), detector.inputs.concentration_file)
    def find(concentration: np.ndarray, mask: np.ndarray, *args: Any, **kwargs: Any) -> tuple[list, list, list]:
        candidates = [tuple(v) for v in np.argwhere(mask > 0.5)]
        assert candidates and all(region[c] > 0.5 for c in candidates)
        metrics = {"peak_time_frame": 7, "peak_mm": 5., "fwhm_seconds": 9.4,
                   "washout": 1., "cnr": 30., "auc": 50.}
        return candidates, [0.8] * len(candidates), [metrics] * len(candidates)
    monkeypatch.setattr(detector, "_find_candidates_in_mask", find)
    outputs = detector.run(cwd=str(tmp_path)).outputs
    qc = outputs.qc_metrics
    if rejected:
        assert outputs.detection_method == "population"
        assert qc["detection_status"] == "failed"
        assert qc["failure_reason"] == "user_mask_exhausted"
        assert qc["failure_detail"] == "quality_checks_failed"
        assert isdefined(outputs.out_rejected_aif_mask)
        used = nib.load(outputs.out_rejected_aif_mask).get_fdata() > 0.5
        assert np.any(used) and np.all(region[used] > 0.5)
        return
    assert outputs.candidate_count == count
    used = nib.load(outputs.out_aif_mask).get_fdata() > 0.5
    assert np.count_nonzero(used) == count and np.all(region[used] > 0.5)
    assert all(region[tuple(c)] > 0.5 for c in qc["candidate_coords"])
    assert qc["atlas_tier_used"] == "user_mask_strict"
    assert len(qc["cascade_gate_decisions"]) == (3 if count == 1 else 1)
    block = build_aif_provenance_block(qc, modality="DCE")
    assert block["Source"] == "measured" and block["SearchRegion"] == "user_mask"


@pytest.mark.parametrize("name", ["my_roi.nii.gz", "MNI_2mm_MRA_arterial_MCA_ACA_dce.nii.gz"])
def test_dce_user_filename_cannot_enable_internal_atlas_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    """An internal-atlas-like filename cannot grant G6 eligibility to a user ROI."""
    region = np.ones((4, 4, 4))
    path = tmp_path / name
    nib.save(nib.Nifti1Image(region, np.eye(4)), path)
    detector = dce_aif.AIFDetection(aif_mask_file=str(path))
    _, names, _, _ = detector._load_atlas_and_lesion_masks()
    assert names == ["user_mask"]
    observed = []
    evaluate = dce_aif._evaluate_gates
    def gate(*args: Any, **kwargs: Any) -> tuple[dce_aif.GateStatus, dce_aif.GateDecision]:
        observed.append(kwargs["g6_atlas_eligible"])
        return evaluate(*args, **kwargs)
    monkeypatch.setattr(dce_aif, "_evaluate_gates", gate)
    # The same otherwise eligible pool is evaluated with supplied and internal inputs.
    time = np.arange(40)
    pulse = np.exp(-0.5 * ((time - 7) / .7) ** 2)
    concentration = np.broadcast_to(pulse, (4, 4, 4, 40))
    coords = [(i, 0, k) for i in range(4) for k in (1, 2)] + [(0, 1, 1), (0, 1, 2)]
    args = (coords, [0.8] * len(coords), concentration, region, 4., 5, .42, 0., "MCA_ACA_dce_strict")
    supplied = detector._score_and_gate_pool(*args)
    internal = dce_aif.AIFDetection()._score_and_gate_pool(*args)
    assert observed == [False, True]
    assert supplied.status == "FAIL_G6_f"
    assert internal.status == "G6_ADMIT"
