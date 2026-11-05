"""Routed slice fitting, time serialization and scheduling estimates."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pmriprep.interfaces.dce_compute import ComputeDCEMaps
from pmriprep.models.dce.conditional_covariance import covariance, jacobian
from pmriprep.models.dce.delayed import DelayedPchip
from pmriprep.models.dce.profile_fitting import tofts_predict
from pmriprep.utils.mem import estimate_dce_mem_gb


@pytest.mark.parametrize("true_delay", [0.0, 2.0, 12.0])
def test_routed_slices_preserve_background_missing_values_and_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, true_delay: float,
) -> None:
    import weakref
    from typing import Any
    from pmriprep.models.dce.local_runner import LocalDelayRunner, SliceFits

    original = LocalDelayRunner.fit_slice
    previous: list[weakref.ReferenceType[Any]] = []

    def check_release(self: LocalDelayRunner, *args: Any, **kwargs: Any) -> SliceFits:
        assert all(reference() is None for reference in previous)
        result = original(self, *args, **kwargs)
        previous[:] = [weakref.ref(profile) for profile in result.profiles.values()]
        return result

    monkeypatch.setattr(LocalDelayRunner, "fit_slice", check_release)
    times = np.arange(40) * 7.4 / 60
    aif = 4 * np.exp(-((times - 0.6) / 0.13) ** 2)
    context = DelayedPchip(aif, times, true_delay)
    curve = tofts_predict(context, np.array([0.2, 0.3])) + 0.04 * context.cp
    concentration = np.tile(curve, (3, 3, 2, 1))
    concentration[0, 0, 0] = 0
    mask = np.ones((3, 3, 2), dtype=np.uint8)
    mask[2, 2, 1] = 0
    nib.save(nib.Nifti1Image(concentration, np.eye(4)), tmp_path / "conc.nii.gz")
    nib.save(nib.Nifti1Image(mask, np.eye(4)), tmp_path / "mask.nii.gz")
    np.savetxt(tmp_path / "aif.txt", aif)
    output = ComputeDCEMaps(
        concentration_file=str(tmp_path / "conc.nii.gz"),
        brain_mask_file=str(tmp_path / "mask.nii.gz"), aif_file=str(tmp_path / "aif.txt"),
        volume_tr=7.4, baseline_frames=3, model="auto", n_procs=2,
        delay_correction=True, sloppy=True, save_fitted_curves=True,
    ).run(cwd=str(tmp_path)).outputs
    values = nib.load(output.out_ktrans).get_fdata()
    assert values[2, 2, 1] == 0 and np.isnan(values[0, 0, 0])
    accepted = mask.astype(bool)
    accepted[0, 0, 0] = False
    np.testing.assert_allclose(values[accepted], 0.2, rtol=1e-4)
    np.testing.assert_array_equal(nib.load(output.out_model_selection).get_fdata()[accepted], 3)
    qc = output.qc_metrics
    assert qc["voxel_accounting"]["optimizer_converged"]["count"] == np.count_nonzero(accepted)
    assert qc["delay_correction"]["counts"]["selected"] == np.count_nonzero(accepted)
    assert qc["delay_correction"]["approximate_test_mode"] is True
    assert qc["local_fitting_execution"]["peak_slice_profile_bytes"] > 0
    assert qc["local_fitting_execution"]["n_procs"] == 2
    assert qc["two_cxm_auto_sampling_eligible"] is False
    assert qc["qc_acceptance_policy"]["fit"]["uncertainty_used_for_acceptance"] is True
    delay = nib.load(output.out_delay).get_fdata()
    np.testing.assert_array_equal(delay[accepted], true_delay)
    counts = qc["delay_correction"]["counts"]
    for name in ("selected", "reference"):
        assert counts[name + "_at_lower_bound"] == (accepted.sum() if true_delay == 0 else 0)
        assert counts[name + "_at_upper_bound"] == (accepted.sum() if true_delay == 12 else 0)
    assert np.isnan(delay[~accepted]).all()
    assert qc["timing_diagnostics"]["sensitivity_refits_performed"] is False

    assert qc["timing_diagnostics"]["tau_e"]["defined"] == accepted.sum()
    assert qc["timing_diagnostics"]["tau_e"]["below_frame"] == 0
    assert sum(qc["delay_correction"]["support_histogram"].values()) == accepted.sum()


def test_rounded_uniform_time_file_can_reach_fitting(tmp_path: Path) -> None:
    time = np.round(np.arange(30) * 2 / 60, 8)
    uniform = np.arange(30) * time[1]
    aif = 3 * np.exp(-((uniform - 0.3) / 0.1) ** 2)
    context = DelayedPchip(aif, uniform)
    curve = 0.1 * context.convolution(0) + 0.04 * context.cp
    nib.save(nib.Nifti1Image(curve[None, None, None, :], np.eye(4)), tmp_path / "conc.nii.gz")
    nib.save(nib.Nifti1Image(np.ones((1, 1, 1)), np.eye(4)), tmp_path / "mask.nii.gz")
    np.savetxt(tmp_path / "aif.txt", aif)
    np.savetxt(tmp_path / "times.txt", time)
    output = ComputeDCEMaps(
        concentration_file=str(tmp_path / "conc.nii.gz"),
        brain_mask_file=str(tmp_path / "mask.nii.gz"), aif_file=str(tmp_path / "aif.txt"),
        time_points_file=str(tmp_path / "times.txt"), volume_tr=2,
        baseline_frames=3, model="patlak", delay_correction=False,
    ).run(cwd=str(tmp_path)).outputs
    assert nib.load(output.out_ktrans).get_fdata().item() == pytest.approx(0.1, rel=1e-5)


def test_conditional_covariance_matches_linear_theory_and_marks_rank_loss() -> None:
    x = np.arange(10) / 10
    design = np.column_stack([np.ones(10), x])
    derivative, schemes = jacobian(np.array([0.2, 0.3]), np.ones(2), (), lambda p: design @ p)
    np.testing.assert_allclose(derivative, design, atol=1e-10)
    cov, _, reason = covariance(derivative, 0.08)
    assert reason == "ok" and schemes == ["central", "central"]
    np.testing.assert_allclose(cov, 0.08 / 8 * np.linalg.inv(design.T @ design), rtol=1e-9)
    cov, _, reason = covariance(np.ones((10, 2)), 0.08)
    assert reason == "rank_deficient" and np.isnan(cov).all()
    derivative, schemes = jacobian(np.array([0.0, 1.0]), np.ones(2), (), lambda p: design @ p)
    assert schemes == ["forward", "backward"]
    np.testing.assert_allclose(derivative, design, atol=1e-8)


def test_reported_quality_and_nlls_identification_both_gate_binary_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing NLLS evidence fails QC without erasing VI or fallback estimates."""
    from copy import deepcopy
    from typing import Any
    from pmriprep.models.dce import identification, local_refinement
    from pmriprep.models.dce.delay_profiles import DelayProfile
    from pmriprep.models.dce.fixed_inference import FixedDelayModel, Refinement

    times = np.arange(40) * 4.76 / 60
    aif = 4 * np.exp(-((times - .5) / .12)**2)
    context = DelayedPchip(aif, times)
    curve = .1 * context.convolution(0) + .04 * context.cp
    shape = (4, 1, 1)
    nib.save(nib.Nifti1Image(np.tile(curve, (*shape, 1)), np.eye(4)), tmp_path / "conc.nii.gz")
    nib.save(nib.Nifti1Image(np.ones(shape), np.eye(4)), tmp_path / "mask.nii.gz")
    np.savetxt(tmp_path / "aif.txt", aif)
    original = identification.profile_excursion

    def excursion(profile: DelayProfile, member: int, index: int) -> np.ndarray:
        result = original(profile, member, index)
        return result if member != 0 else np.full_like(result, np.nan)

    calls = []

    def refine(model: FixedDelayModel, **kwargs: Any) -> Refinement:
        fit = deepcopy(model.fit(model.tissue, model.input_aif, model.input_times))
        calls.append(fit.parameters.copy())
        # Distinguish reported-fit quality from good identification at its seed.
        fit.r_squared = -1 if len(calls) == 2 else fit.r_squared
        fallback = len(calls) == 3
        return Refinement(fit, "vi", "nlls" if fallback else "vi", model.name, 0, fallback,
                          "test_numerical_fallback" if fallback else None)

    monkeypatch.setattr(identification, "profile_excursion", excursion)
    monkeypatch.setattr(local_refinement, "refine_vi", refine)
    output = ComputeDCEMaps(
        concentration_file=str(tmp_path / "conc.nii.gz"), brain_mask_file=str(tmp_path / "mask.nii.gz"),
        aif_file=str(tmp_path / "aif.txt"), volume_tr=4.76, baseline_frames=3,
        model="patlak", inference="vi", n_procs=1, delay_correction=False,
    ).run(cwd=str(tmp_path)).outputs
    np.testing.assert_array_equal(nib.load(output.out_qc_mask).get_fdata().ravel(), [0, 0, 1, 1])
    np.testing.assert_array_equal(nib.load(output.out_retained_nondegenerate_mask).get_fdata(), 1)
    np.testing.assert_array_equal(nib.load(output.out_delay).get_fdata(), 0)
    np.testing.assert_allclose(nib.load(output.out_ktrans).get_fdata(), .1, rtol=1e-6)
    qc = output.qc_metrics
    evidence = qc["identification_qc"]["by_model"]["patlak"]
    assert evidence["evaluated"] == 4 and evidence["primary_pass"] == 3
    assert evidence["parameters"]["Ktrans"]["failures"] == {"output_unavailable": 1}
    assert qc["inference_fallback_reasons"] == {"test_numerical_fallback": 1}
    assert qc["qc_acceptance_policy"]["identification"]["basis"] == "selected_nlls"
    assert qc["timing_diagnostics"]["tau_e"]["applicable_retained"] == 0
    assert qc["timing_diagnostics"]["sensitivity_refits_performed"] is False
    for name in ("selected", "reference"):
        for edge in ("lower", "upper"):
            assert qc["delay_correction"]["counts"][name + "_at_" + edge + "_bound"] == 0


def test_memory_reservation_accounts_for_grid_families_and_workers(tmp_path: Path) -> None:
    path = tmp_path / "dynamic.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((10, 12, 2, 20), np.float32), np.eye(4)), path)
    full = estimate_dce_mem_gb(path, "auto", 4, True, False)
    coarse = estimate_dce_mem_gb(path, "auto", 4, True, True)
    fixed = estimate_dce_mem_gb(path, "auto", 4, False, False)
    assert full > coarse > fixed
    assert fixed > estimate_dce_mem_gb(path, "auto", 1, False, False)
    assert full > estimate_dce_mem_gb(path, "patlak", 4, True, False)
    # 176 bytes per three-family point, with two 128-member blocks per worker.
    assert full - coarse == pytest.approx((120 + 2 * 128 * 4) * 108 * 176 / 1e9)
