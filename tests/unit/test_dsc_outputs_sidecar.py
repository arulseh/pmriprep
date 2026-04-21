"""Integration tests for DSC sidecar AIF-provenance fields (v0.28.3).

Exercises the full ``_run_interface`` write path of ``DSCDerivativeWriter``
with a controlled ``qc_metrics`` dict and asserts the resulting JSON
sidecars carry the expected ``AifSource`` and ``CrossSubjectComparable``
values across the six writer sites (raw CBV variants, nCBV/nCBF variants,
smoothed hemodynamic variants).

Catches regressions where one writer site drifts — e.g. someone setting
``CrossSubjectComparable=False`` on a smoothed nCBV sidecar by mistake.
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pmriprep.interfaces.dsc_outputs import DSCDerivativeWriter


def _write_dummy_nifti(path: Path) -> str:
    """Write a 4×4×4 zero-filled NIfTI and return its path as str."""
    data = np.zeros((4, 4, 4), dtype=np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, path)  # type: ignore[attr-defined]
    return str(path)


@pytest.fixture
def minimal_bids_dir(tmp_path):
    """Minimal BIDS dataset root for `_build_base_sidecar` to consume."""
    bids = tmp_path / "bids"
    bids.mkdir()
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "test", "BIDSVersion": "1.8.0"})
    )
    return bids


@pytest.fixture
def minimal_inputs(tmp_path, minimal_bids_dir):
    """Dict of trait values that satisfies DSCDerivativeWriter mandatories."""
    return {
        "output_dir": str(tmp_path / "derivatives"),
        "subject_id": "test01",
        "session_id": "01",
        "bids_dir": str(minimal_bids_dir),
        "mtt_map": _write_dummy_nifti(tmp_path / "mtt.nii.gz"),
        "tmax_map": _write_dummy_nifti(tmp_path / "tmax.nii.gz"),
        "psr_map": _write_dummy_nifti(tmp_path / "psr.nii.gz"),
        "ph_map": _write_dummy_nifti(tmp_path / "ph.nii.gz"),
        "rel_cbf_map": _write_dummy_nifti(tmp_path / "rcbf.nii.gz"),
        "rel_cbv_map": _write_dummy_nifti(tmp_path / "rcbv.nii.gz"),
        "rel_cbv_auc_map": _write_dummy_nifti(tmp_path / "rcbv_auc.nii.gz"),
        "ncbv_map": _write_dummy_nifti(tmp_path / "ncbv.nii.gz"),
        "ncbf_map": _write_dummy_nifti(tmp_path / "ncbf.nii.gz"),
    }


def _load_sidecar(derivatives_root: Path, suffix: str) -> dict:
    """Load the one sidecar matching the given BIDS desc suffix."""
    matches = list(derivatives_root.rglob(f"*{suffix}*.json"))
    assert len(matches) == 1, (
        f"Expected exactly one sidecar matching {suffix!r}; "
        f"found {len(matches)}: {matches}"
    )
    with open(matches[0]) as f:
        return json.load(f)


def _run_writer(inputs: dict, qc_metrics: dict) -> Path:
    """Run DSCDerivativeWriter and return the derivatives root path."""
    writer = DSCDerivativeWriter()
    for k, v in inputs.items():
        setattr(writer.inputs, k, v)
    writer.inputs.qc_metrics = qc_metrics
    writer.run()
    return Path(inputs["output_dir"])


class TestMeasuredAifSidecars:
    """AIF detection at atlas_relaxed → measured AifSource everywhere."""

    def test_raw_cbv_auc(self, minimal_inputs) -> None:
        qc = {"aif": {"detection_level": "atlas_relaxed"}}
        derivatives = _run_writer(minimal_inputs, qc)
        sidecar = _load_sidecar(derivatives, "dscRelAUC_cbv")
        assert sidecar["AifSource"] == "measured"
        assert sidecar["CrossSubjectComparable"] is True

    def test_raw_rel_cbv_deconv(self, minimal_inputs) -> None:
        qc = {"aif": {"detection_level": "atlas_strict"}}
        derivatives = _run_writer(minimal_inputs, qc)
        # Raw deconv CBV uses dscRel prefix — distinct from dscRelAUC
        matches = list(derivatives.rglob("*desc-dscRel_cbv*.json"))
        assert len(matches) == 1
        sidecar = json.loads(matches[0].read_text())
        assert sidecar["AifSource"] == "measured"
        assert sidecar["CrossSubjectComparable"] is True

    def test_ncbv(self, minimal_inputs) -> None:
        qc = {"aif": {"detection_level": "brain_relaxed"}}
        derivatives = _run_writer(minimal_inputs, qc)
        sidecar = _load_sidecar(derivatives, "dscNawmNorm_cbv")
        assert sidecar["AifSource"] == "measured"
        assert sidecar["CrossSubjectComparable"] is True

    def test_ncbf(self, minimal_inputs) -> None:
        qc = {"aif": {"detection_level": "atlas_relaxed"}}
        derivatives = _run_writer(minimal_inputs, qc)
        sidecar = _load_sidecar(derivatives, "dscNawmNorm_cbf")
        assert sidecar["AifSource"] == "measured"
        assert sidecar["CrossSubjectComparable"] is True


class TestSyntheticAifSidecars:
    """AIF cascade exhausted → synthetic AifSource; raw CBV gets
    CrossSubjectComparable=false, nCBV variants stay True."""

    def test_raw_cbv_auc_flagged_false(self, minimal_inputs) -> None:
        qc = {"aif": {"detection_level": "synthetic"}}
        derivatives = _run_writer(minimal_inputs, qc)
        sidecar = _load_sidecar(derivatives, "dscRelAUC_cbv")
        assert sidecar["AifSource"] == "synthetic"
        assert sidecar["CrossSubjectComparable"] is False

    def test_raw_rel_cbv_deconv_flagged_false(self, minimal_inputs) -> None:
        qc = {"aif": {"detection_level": "synthetic"}}
        derivatives = _run_writer(minimal_inputs, qc)
        matches = list(derivatives.rglob("*desc-dscRel_cbv*.json"))
        assert len(matches) == 1
        sidecar = json.loads(matches[0].read_text())
        assert sidecar["AifSource"] == "synthetic"
        assert sidecar["CrossSubjectComparable"] is False

    def test_ncbv_stays_comparable(self, minimal_inputs) -> None:
        """nCBV is a NAWM-normalised ratio → AIF amplitude cancels →
        still CrossSubjectComparable=True even on synthetic-AIF subjects."""
        qc = {"aif": {"detection_level": "synthetic"}}
        derivatives = _run_writer(minimal_inputs, qc)
        sidecar = _load_sidecar(derivatives, "dscNawmNorm_cbv")
        assert sidecar["AifSource"] == "synthetic"
        assert sidecar["CrossSubjectComparable"] is True

    def test_ncbf_stays_comparable(self, minimal_inputs) -> None:
        qc = {"aif": {"detection_level": "synthetic"}}
        derivatives = _run_writer(minimal_inputs, qc)
        sidecar = _load_sidecar(derivatives, "dscNawmNorm_cbf")
        assert sidecar["AifSource"] == "synthetic"
        assert sidecar["CrossSubjectComparable"] is True

    def test_legacy_flag_also_triggers(self, minimal_inputs) -> None:
        """Backwards-compat: legacy cbv_auc_denominator_is_synthetic flag
        still drives the synthetic branch even when detection_level is a
        legitimate cascade tier."""
        qc = {"aif": {
            "detection_level": "atlas_relaxed",
            "cbv_auc_denominator_is_synthetic": True,
        }}
        derivatives = _run_writer(minimal_inputs, qc)
        sidecar = _load_sidecar(derivatives, "dscRelAUC_cbv")
        assert sidecar["AifSource"] == "synthetic"
        assert sidecar["CrossSubjectComparable"] is False


class TestSmoothedSidecars:
    """Smoothed variants inherit the same AIF provenance logic as their
    non-smoothed parents."""

    def test_smoothed_cbv_synthetic(self, minimal_inputs, tmp_path) -> None:
        # Populate a smooth_hemodynamic dict with a rel_cbv filepath so
        # the smoothed-variant writer path fires with fwhm > 0.
        minimal_inputs["smooth_fwhm"] = 3.0
        minimal_inputs["smooth_hemodynamic"] = {
            "rel_cbv": _write_dummy_nifti(tmp_path / "rcbv_sm.nii.gz"),
            "ncbv": _write_dummy_nifti(tmp_path / "ncbv_sm.nii.gz"),
        }
        qc = {"aif": {"detection_level": "synthetic"}}
        derivatives = _run_writer(minimal_inputs, qc)
        raw = _load_sidecar(derivatives, "dscRelAUCSm3_cbv")
        assert raw["AifSource"] == "synthetic"
        assert raw["CrossSubjectComparable"] is False
        ncbv_sm = _load_sidecar(derivatives, "dscNawmNormSm3_cbv")
        assert ncbv_sm["AifSource"] == "synthetic"
        assert ncbv_sm["CrossSubjectComparable"] is True

    def test_smoothed_cbv_measured(self, minimal_inputs, tmp_path) -> None:
        minimal_inputs["smooth_fwhm"] = 3.0
        minimal_inputs["smooth_hemodynamic"] = {
            "rel_cbv": _write_dummy_nifti(tmp_path / "rcbv_sm.nii.gz"),
            "ncbv": _write_dummy_nifti(tmp_path / "ncbv_sm.nii.gz"),
        }
        qc = {"aif": {"detection_level": "atlas_strict"}}
        derivatives = _run_writer(minimal_inputs, qc)
        raw = _load_sidecar(derivatives, "dscRelAUCSm3_cbv")
        assert raw["AifSource"] == "measured"
        assert raw["CrossSubjectComparable"] is True


class TestMissingAifQc:
    """Defensive behaviour when qc_metrics is absent or malformed."""

    def test_missing_qc_defaults_to_measured(self, minimal_inputs) -> None:
        """If no qc_metrics provided, default to measured + comparable
        (so ancient pipelines or test shims don't break)."""
        writer = DSCDerivativeWriter()
        for k, v in minimal_inputs.items():
            setattr(writer.inputs, k, v)
        # Deliberately do not set qc_metrics
        writer.run()
        derivatives = Path(minimal_inputs["output_dir"])
        sidecar = _load_sidecar(derivatives, "dscRelAUC_cbv")
        assert sidecar["AifSource"] == "measured"
        assert sidecar["CrossSubjectComparable"] is True
