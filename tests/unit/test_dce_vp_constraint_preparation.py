"""Characterization tests for DSC-derived DCE vp-constraint preparation."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import nibabel as nib
import numpy as np
import pytest

from pmriprep.interfaces.dce_compute import (
    ComputeDCEMaps,
    _DCEFitPreparation,
    _DCEVPConstraint,
)
from pmriprep.utils.dsc_cbv_constraint import ConstraintMode


def _prepared(
    *, populated: bool = False, inference: str = "map",
) -> _DCEFitPreparation:
    """Build auto-model preparation state with an optional populated domain."""
    shape = (5, 5, 5)
    concentration = np.zeros((*shape, 4), dtype=float)
    fitting_mask = np.zeros(shape, dtype=bool)
    if populated:
        concentration[..., 3] = 0.1
        fitting_mask[:] = True
    return _DCEFitPreparation(
        concentration=concentration,
        affine=np.eye(4),
        header=None,
        mask=np.ones(shape, dtype=float),
        aif=np.array([0.0, 2.0, 1.0, 0.5]),
        time_points=np.array([0.0, 1 / 30, 2 / 30, 0.1]),
        n_timepoints=4,
        selection_tr_seconds=2.0,
        model_name="patlak",
        is_auto=True,
        candidate_names=["patlak"],
        baseline_frames=1,
        tsnr_threshold=0.0,
        tsnr_map=np.ones(shape, dtype=float),
        peak_enhancement=None,
        fitting_mask=fitting_mask,
        inference=inference,
        mcmc_samples=500,
        save_fitted_curves=False,
    )


def _write_mask(path: Path) -> str:
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5)), np.eye(4)), path)
    return str(path)


def test_dce_only_registration_qc_sentinel_is_trait_safe(tmp_path: Path) -> None:
    """Inactive DSC state must remain assignable to the strict Dict trait."""
    from pmriprep.config.dce import DCEConfig
    from pmriprep.workflows.dce.base import init_dce_preproc_wf

    bids = tmp_path / "bids"
    perf = bids / "sub-0001" / "ses-0001" / "perf"
    perf.mkdir(parents=True)
    (perf / "sub-0001_ses-0001_dce.nii.gz").touch()
    (bids / "dataset_description.json").write_text(
        '{"Name": "T", "BIDSVersion": "1.9.0"}'
    )
    workflow = init_dce_preproc_wf(
        bids_dir=bids,
        output_dir=bids / "derivatives",
        subject_id="0001",
        session_id="0001",
        config=DCEConfig(),
        has_tissue_seg=True,
    )
    inputnode = workflow.get_node("inputnode")
    pk_modeling = next(
        node for node in workflow._get_all_nodes() if node.name == "pk_modeling"
    )

    assert inputnode.inputs.dsc_registration_qc == {}
    pk_modeling.inputs.dsc_registration_qc = inputnode.inputs.dsc_registration_qc
    assert pk_modeling.inputs.dsc_registration_qc == {}


@pytest.mark.parametrize("is_auto", [False, True])
def test_prepare_vp_constraint_distinguishes_absent_dsc_run(
    tmp_path: Path, is_auto: bool,
) -> None:
    """With no DSC inputs, the typed state retains explicit no-run provenance."""
    interface = ComputeDCEMaps(brain_mask_file=_write_mask(tmp_path / "mask.nii.gz"))

    prepared = _prepared()._replace(is_auto=is_auto)
    result = interface._prepare_vp_constraint(prepared)

    assert result.vp_map is None
    assert result.mode == ConstraintMode.SKIP
    assert result.vp_uncertainty == 0.2
    assert result.qc["mode"] == "skip"
    assert result.qc["skip_reason"] == "no_dsc_run"
    assert result.model_name == "patlak"
    assert result.candidate_names == ["patlak"]


@pytest.mark.parametrize("is_auto", [False, True])
def test_prepare_vp_constraint_rejects_nlls_prior_before_loading_dsc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    is_auto: bool,
) -> None:
    """Explicit and auto-selected Patlak NLLS omit the DSC prior."""
    monkeypatch.setattr(
        "pmriprep.utils.dsc_cbv_constraint.prepare_vp_constraint_from_data",
        lambda **kwargs: pytest.fail("NLLS must not prepare a DSC prior"),
    )
    interface = ComputeDCEMaps(
        brain_mask_file=_write_mask(tmp_path / "mask.nii.gz"),
        dsc_ncbv_file="ncbv.nii.gz",
        dsc_aif_qc={"detection_method": "atlas"},
        constraint_mode="auto",
    )

    result = interface._prepare_vp_constraint(
        _prepared(populated=True, inference="nlls")._replace(is_auto=is_auto)
    )

    assert result.vp_map is None
    assert result.mode == ConstraintMode.SKIP
    assert result.vp_uncertainty == 0.0
    assert result.qc["skip_reason"] == "nlls_constraint_unsupported"
    assert result.candidate_names == ["patlak"]


@pytest.mark.parametrize(
    ("model", "mode", "reason", "message"),
    [
        ("tofts", "auto", "model_without_vp", "no vp parameter"),
        ("patlak", "skip", "disabled_by_configuration", "disabled by configuration"),
    ],
)
def test_prepare_vp_constraint_reports_model_and_configuration_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    model: str, mode: str, reason: str, message: str,
) -> None:
    """Present DSC inputs are not described as missing when the prior is disabled."""
    monkeypatch.setattr(
        "pmriprep.utils.dsc_cbv_constraint.prepare_vp_constraint_from_data",
        lambda **kwargs: pytest.fail("An inapplicable prior must not load DSC data"),
    )
    interface = ComputeDCEMaps(
        brain_mask_file=_write_mask(tmp_path / "mask.nii.gz"),
        dsc_ncbv_file="ncbv.nii.gz", constraint_mode=mode,
    )
    prepared = _prepared(inference="vi")._replace(
        model_name=model, is_auto=False, candidate_names=[],
    )

    result = interface._prepare_vp_constraint(prepared)

    assert result.mode == ConstraintMode.SKIP
    assert result.vp_map is None
    assert result.vp_uncertainty == 0.0
    assert result.qc["skip_reason"] == reason
    assert message in result.qc["reason"]


@pytest.mark.parametrize("model", ["patlak", "auto"])
def test_patlak_vi_maps_use_qualified_dsc_prior(tmp_path: Path, model: str) -> None:
    """Explicit and auto-selected Patlak use the same qualified spatial prior."""
    time = np.arange(166, dtype=float) / 30.0
    u = np.maximum(time - 0.5, 0.0)
    aif = (u / 0.12) ** 3 * np.exp(-u / 0.12)
    aif *= 5.0 / np.max(aif)
    integral = np.r_[0.0, np.cumsum((aif[:-1] + aif[1:]) * np.diff(time) / 2)]
    tissue = 0.02 * integral + 0.08 * aif
    tissue += np.random.default_rng(7).normal(0, 0.003, time.size)
    mask = np.ones((1, 1, 1), dtype=np.uint8)
    for name, values in (
        ("concentration", tissue.reshape(1, 1, 1, -1)),
        ("mask", mask), ("ncbv", np.ones(mask.shape)),
    ):
        nib.save(nib.Nifti1Image(values, np.eye(4)), tmp_path / f"{name}.nii.gz")
    np.savetxt(tmp_path / "aif.txt", aif)
    interface = ComputeDCEMaps(
        concentration_file=str(tmp_path / "concentration.nii.gz"),
        brain_mask_file=str(tmp_path / "mask.nii.gz"),
        aif_file=str(tmp_path / "aif.txt"),
        dsc_ncbv_file=str(tmp_path / "ncbv.nii.gz"),
        model=model, inference="vi", volume_tr=2.0, baseline_frames=15,
        aif_bat_alignment="off", n_procs=1,
        dsc_aif_qc={
            "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995, "mean_cnr": 10.0},
            "aif_peak_height": 0.4, "baseline_return_ratio": 0.15,
            "detection_method": "atlas_strict",
        },
        dsc_preload_qc={"k2_reference_available": True},
        dsc_registration_qc={"qc_status": "PASS"},
        dce_registration_qc={"qc_status": "PASS"},
    )

    outputs = interface.run(cwd=str(tmp_path)).outputs

    assert outputs.constraint_qc["mode"] == "soft"
    assert outputs.constraint_qc["vp_median"] == pytest.approx(0.01)
    assert outputs.qc_metrics["n_constrained"] == 1
    assert outputs.qc_metrics["inference_fallback"]["n_nlls_fallback"] == 0
    assert np.isfinite(nib.load(outputs.out_vp).get_fdata()).all()
    if model == "auto":
        assert nib.load(outputs.out_model_selection).get_fdata().item() == 1


def test_prepare_vp_constraint_reports_missing_nawm_before_nlls_policy(
    tmp_path: Path,
) -> None:
    """A missing nCBV anchor is the primary NLLS skip reason."""
    interface = ComputeDCEMaps(
        brain_mask_file=_write_mask(tmp_path / "mask.nii.gz"),
        dsc_aif_qc={"detection_method": "atlas"},
        constraint_mode="auto",
    )

    result = interface._prepare_vp_constraint(
        _prepared(populated=True, inference="nlls")
    )

    assert result.vp_map is None
    assert result.mode == ConstraintMode.SKIP
    assert result.qc["skip_reason"] == "no_nawm_mask"
    assert result.qc["reason"] == (
        "No NAWM-normalized DSC CBV map was available for a vp prior."
    )


def test_prepare_vp_constraint_preserves_soft_prior_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid SOFT prior preserves its width and promotes auto candidates."""
    prepared = _prepared(populated=True)
    vp_map = np.full(prepared.mask.shape, 0.1, dtype=float)
    captured: dict[str, Any] = {}

    def fake_prepare(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            vp_map=vp_map,
            config=SimpleNamespace(mode=ConstraintMode.SOFT, vp_uncertainty=0.25),
            qc_report={"mode": "soft", "reason": "quality_passed"},
        )

    monkeypatch.setattr(
        "pmriprep.utils.dsc_cbv_constraint.prepare_vp_constraint_from_data",
        fake_prepare,
    )
    interface = ComputeDCEMaps(
        brain_mask_file=_write_mask(tmp_path / "mask.nii.gz"),
        dsc_ncbv_file="ncbv.nii.gz",
        dsc_aif_qc={"detection_method": "atlas"},
        dsc_preprocessing_stats={"fraction_cnr_gt_4": 0.8, "mean_cnr": 6.0},
        dsc_preload_qc={"k2_reference_available": True, "k2_r2": 0.9},
        dsc_registration_qc={"qc_status": "WARN"},
        dce_registration_qc={"qc_status": "PASS"},
        constraint_mode="auto",
        dsc_dose=0.5,
        dsc_agent_standard_dose=0.1,
        volume_tr=2.0,
    )

    result = interface._prepare_vp_constraint(prepared)

    assert result.vp_map is vp_map
    assert result.mode == ConstraintMode.SOFT
    assert result.vp_uncertainty == 0.25
    assert result.model_name == "extended_tofts"
    assert result.candidate_names == ["patlak", "extended_tofts"]
    assert captured["dose"] == 0.5
    assert captured["agent_standard_dose"] == 0.1
    assert captured["dsc_registration_qc"] == {"qc_status": "WARN"}
    assert captured["dce_registration_qc"] == {"qc_status": "PASS"}
    assert captured["dsc_qc_data"]["cnr_whole_brain"] == {
        "fraction_cnr_gt_4": 0.8,
        "mean_cnr": 6.0,
    }


def test_prepare_vp_constraint_clears_soft_state_after_diagnostic_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-preparation failure cannot retain a prior reported as skipped."""
    prepared = _prepared(populated=True)

    def fake_prepare(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            vp_map=np.full((4, 4, 4), 0.1, dtype=float),
            config=SimpleNamespace(mode=ConstraintMode.SOFT, vp_uncertainty=0.25),
            qc_report={"mode": "soft", "reason": "quality_passed"},
        )

    monkeypatch.setattr(
        "pmriprep.utils.dsc_cbv_constraint.prepare_vp_constraint_from_data",
        fake_prepare,
    )
    interface = ComputeDCEMaps(
        brain_mask_file=_write_mask(tmp_path / "mask.nii.gz"),
        dsc_ncbv_file="ncbv.nii.gz",
        dsc_aif_qc={"detection_method": "atlas"},
        constraint_mode="auto",
    )

    result = interface._prepare_vp_constraint(prepared)

    assert result.vp_map is None
    assert result.mode == ConstraintMode.SKIP
    assert result.vp_uncertainty == 0.0
    assert result.qc["mode"] == "skip"
    assert result.qc["skip_reason"] == "constraint_preparation_failed"
    assert result.model_name == "patlak"
    assert result.candidate_names == ["patlak"]


def test_prepare_fit_dispatch_preserves_voxel_order_and_prior_admission() -> None:
    """Dispatch state keeps NumPy mask order and admits only finite positive vp."""
    prepared = _prepared(populated=True)
    vp_map = np.full(prepared.mask.shape, 0.1, dtype=float)
    vp_map[0, 0, 0] = np.nan
    vp_map[0, 0, 1] = -0.1
    constraint = _DCEVPConstraint(
        vp_map=vp_map,
        mode=ConstraintMode.SOFT,
        vp_uncertainty=0.25,
        qc={"mode": "soft"},
        model_name="extended_tofts",
        candidate_names=["patlak", "extended_tofts"],
    )
    motion_weights = np.array([1.0, 0.5, 1.0, 1.0])
    interface = ComputeDCEMaps(n_procs=0, sloppy=True, random_seed=17)

    dispatch = interface._prepare_fit_dispatch(
        prepared, prepared.aif, constraint, motion_weights,
    )

    assert len(dispatch.voxel_data) == 125
    assert dispatch.voxel_data[0].x == 0
    assert dispatch.voxel_data[0].y == 0
    assert dispatch.voxel_data[0].z == 0
    assert dispatch.voxel_data[0].voxel_vp is None
    assert dispatch.voxel_data[1].voxel_vp is None
    assert dispatch.voxel_data[2].voxel_vp == 0.1
    assert dispatch.n_procs == 1
    assert dispatch.sloppy is True
    assert dispatch.random_seed == 17
    assert dispatch.init_args[0] is prepared.aif
    assert dispatch.init_args[9] is motion_weights
    assert dispatch.init_args[-1] == ["patlak", "extended_tofts"]
