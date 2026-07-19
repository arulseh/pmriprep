"""Characterization tests for DSC-derived DCE vp-constraint preparation."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import nibabel as nib
import numpy as np
import pytest

from pmriprep.interfaces.dce_compute import ComputeDCEMaps, _DCEFitPreparation
from pmriprep.utils.dsc_cbv_constraint import ConstraintMode


def _prepared(*, populated: bool = False) -> _DCEFitPreparation:
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
        inference="nlls",
        mcmc_samples=500,
        save_fitted_curves=False,
    )


def _write_mask(path: Path) -> str:
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5)), np.eye(4)), path)
    return str(path)


def test_prepare_vp_constraint_distinguishes_absent_dsc_run(tmp_path: Path) -> None:
    """With no DSC inputs, the typed state retains explicit no-run provenance."""
    interface = ComputeDCEMaps(brain_mask_file=_write_mask(tmp_path / "mask.nii.gz"))

    result = interface._prepare_vp_constraint(_prepared(), _prepared().aif)

    assert result.vp_map is None
    assert result.mode == ConstraintMode.SKIP
    assert result.vp_uncertainty == 0.2
    assert result.qc["mode"] == "skip"
    assert result.qc["skip_reason"] == "no_dsc_run"
    assert result.model_name == "patlak"
    assert result.candidate_names == ["patlak"]


def test_prepare_vp_constraint_preserves_soft_prior_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid SOFT prior emits overshoot/lag QC and promotes auto candidates."""
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
        dsc_preload_qc={"preload_classification": "adequate", "k2_r2": 0.9},
        constraint_mode="auto",
        dsc_dose=0.5,
        dsc_agent_standard_dose=0.1,
        volume_tr=2.0,
    )

    result = interface._prepare_vp_constraint(prepared, prepared.aif)

    assert result.vp_map is vp_map
    assert result.mode == ConstraintMode.SOFT
    assert result.vp_uncertainty == 0.25
    assert result.model_name == "extended_tofts"
    assert result.candidate_names == ["patlak", "extended_tofts"]
    assert result.qc["aif_tissue_overshoot_fraction"] == 1.0
    assert result.qc["peak_alignment_lag_s"] == 4.0
    assert result.qc["volume_tr_s"] == 2.0
    assert "acquisition_limited_note" in result.qc
    assert captured["dose"] == 0.5
    assert captured["agent_standard_dose"] == 0.1
    assert captured["preload_classification"] == "adequate"
    assert captured["aif_qc_data"]["cnr_whole_brain"] == {
        "fraction_cnr_gt_4": 0.8,
        "mean_cnr": 6.0,
    }


def test_prepare_vp_constraint_marks_synthetic_aif_diagnostics_inapplicable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic AIF provenance emits stable not-applicable overshoot fields."""
    prepared = _prepared(populated=True)

    def fake_prepare(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            vp_map=None,
            config=SimpleNamespace(mode=ConstraintMode.SKIP, vp_uncertainty=0.2),
            qc_report={"mode": "skip", "reason": "synthetic_aif"},
        )

    monkeypatch.setattr(
        "pmriprep.utils.dsc_cbv_constraint.prepare_vp_constraint_from_data",
        fake_prepare,
    )
    interface = ComputeDCEMaps(
        brain_mask_file=_write_mask(tmp_path / "mask.nii.gz"),
        dsc_aif_qc={
            "detection_method": "synthetic",
            "cbv_auc_denominator_is_synthetic": True,
        },
        constraint_mode="auto",
    )

    result = interface._prepare_vp_constraint(prepared, prepared.aif)

    assert result.mode == ConstraintMode.SKIP
    assert result.qc["aif_tissue_overshoot_fraction"] is None
    assert result.qc["overshoot_applicable"] is False
    assert result.qc["overshoot_reason"] == "synthetic_aif"
    assert result.candidate_names == ["patlak"]
