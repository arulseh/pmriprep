"""Smoke + sidecar-correctness tests for ``init_dce_datasinks_wf``."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def _write_nii(path: Path) -> str:
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), path)
    return str(path)


@pytest.fixture
def minimal_bids(tmp_path: Path) -> tuple[Path, str]:
    bids = tmp_path / "bids"
    perf = bids / "sub-01" / "ses-0001" / "perf"
    perf.mkdir(parents=True)
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "T", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )
    src = perf / "sub-01_ses-0001_dce.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), src)
    return bids, str(src)


def test_dce_factory_smoke(tmp_path: Path, minimal_bids: tuple[Path, str]) -> None:
    """Mandatory PK output (Ktrans) writes correctly with M3 sidecar fields."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    out = tmp_path / "out"

    wf = init_dce_datasinks_wf(
        output_dir=str(out), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        pk_model="auto",
        present_inputs=set(),  # only mandatory ktrans_map
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_dce = src
    inp.ktrans_map = _write_nii(tmp_path / "ktrans.nii.gz")
    inp.qc_metrics = {}
    wf.run()

    perf_out = out / "pmriprep" / "sub-01" / "ses-0001" / "perf"
    ktrans = perf_out / "sub-01_ses-0001_desc-dce_Ktrans.nii.gz"
    assert ktrans.exists(), f"Missing: {ktrans}"
    sidecar = json.loads(
        (perf_out / "sub-01_ses-0001_desc-dce_Ktrans.json").read_text()
    )
    assert sidecar["Description"].startswith("Volume transfer constant")
    assert sidecar["Units"] == "1/min"
    assert sidecar["SpatialReference"].startswith("bids::pmriprep/")
    assert sidecar["Resolution"] == "native"
    assert isinstance(sidecar["VoxelSize"], list)
    assert sidecar["KtransConvention"] == "plasma_Tofts1999"


def test_dce_factory_with_bayesian_uncertainty(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """Bayesian-inference run emits uncertainty/CV/CI95 sinks per PK param."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    out = tmp_path / "out"

    present = {
        "ve_map", "vp_map",
        "ktrans_uncertainty", "ve_uncertainty", "vp_uncertainty",
        "ktrans_cv", "ve_cv", "vp_cv",
        "ktrans_ci95_lower", "ktrans_ci95_upper",
    }
    wf = init_dce_datasinks_wf(
        output_dir=str(out), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        pk_model="extended_tofts",
        present_inputs=present,
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_dce = src
    inp.ktrans_map = _write_nii(tmp_path / "k.nii.gz")
    for attr in present:
        setattr(inp, attr, _write_nii(tmp_path / f"{attr}.nii.gz"))
    inp.qc_metrics = {}
    wf.run()

    perf_out = out / "pmriprep" / "sub-01" / "ses-0001" / "perf"
    expected = [
        "sub-01_ses-0001_desc-dce_Ktrans.nii.gz",
        "sub-01_ses-0001_desc-dce_ve.nii.gz",
        "sub-01_ses-0001_desc-dce_vp.nii.gz",
        "sub-01_ses-0001_desc-dceUncertainty_Ktrans.nii.gz",
        "sub-01_ses-0001_desc-dceCV_Ktrans.nii.gz",
        "sub-01_ses-0001_desc-dceCI95lower_Ktrans.nii.gz",
        "sub-01_ses-0001_desc-dceCI95upper_Ktrans.nii.gz",
    ]
    for fname in expected:
        assert (perf_out / fname).exists(), f"Missing: {fname}"


def test_dce_factory_model_free_no_model_entity(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """Model-free maps (IAUC60, peakEnhancement, etc.) drop the model entity."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    out = tmp_path / "out"

    present = {"iauc60_map", "peak_enhancement_map", "bat_map"}
    wf = init_dce_datasinks_wf(
        output_dir=str(out), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        pk_model="auto",
        present_inputs=present,
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_dce = src
    inp.ktrans_map = _write_nii(tmp_path / "k.nii.gz")
    for attr in present:
        setattr(inp, attr, _write_nii(tmp_path / f"{attr}.nii.gz"))
    inp.qc_metrics = {}
    wf.run()

    perf_out = out / "pmriprep" / "sub-01" / "ses-0001" / "perf"
    # No model entity in filename
    assert (perf_out / "sub-01_ses-0001_desc-dce_IAUC60.nii.gz").exists()
    assert (perf_out / "sub-01_ses-0001_desc-dce_peakEnhancement.nii.gz").exists()
    bat_sidecar = json.loads(
        (perf_out / "sub-01_ses-0001_desc-dce_BAT.json").read_text()
    )
    assert bat_sidecar["DetectionMethod"] == "threshold_then_gradient_fallback"


def test_patlak_only_no_ve_kep(tmp_path: Path, minimal_bids: tuple[Path, str]) -> None:
    """Patlak runs do not populate ve/kep — factory must not crash on Undefined."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    out = tmp_path / "out"

    # Patlak only emits Ktrans + vp; ve/kep are NOT in present_inputs.
    present = {"vp_map"}
    wf = init_dce_datasinks_wf(
        output_dir=str(out), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        pk_model="patlak",
        present_inputs=present,
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_dce = src
    inp.ktrans_map = _write_nii(tmp_path / "k.nii.gz")
    inp.vp_map = _write_nii(tmp_path / "vp.nii.gz")
    inp.qc_metrics = {}
    wf.run()  # Must not crash

    perf_out = out / "pmriprep" / "sub-01" / "ses-0001" / "perf"
    assert (perf_out / "sub-01_ses-0001_desc-dce_Ktrans.nii.gz").exists()
    assert (perf_out / "sub-01_ses-0001_desc-dce_vp.nii.gz").exists()
    # No ve/kep files
    assert not list(perf_out.glob("*_ve.nii.gz"))
    assert not list(perf_out.glob("*_kep.nii.gz"))


def test_2cxm_bayesian_fp_ps_uncertainty(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """2CXM Bayesian inference emits Fp/PS uncertainty maps."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    out = tmp_path / "out"

    present = {
        "ve_map", "kep_map", "vp_map",
        "fp_uncertainty", "ps_uncertainty",
    }
    wf = init_dce_datasinks_wf(
        output_dir=str(out), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        pk_model="2cxm",
        present_inputs=present,
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_dce = src
    inp.ktrans_map = _write_nii(tmp_path / "k.nii.gz")
    for attr in present:
        setattr(inp, attr, _write_nii(tmp_path / f"{attr}.nii.gz"))
    inp.qc_metrics = {}
    wf.run()

    perf_out = out / "pmriprep" / "sub-01" / "ses-0001" / "perf"
    assert (perf_out / "sub-01_ses-0001_desc-dceUncertainty_Fp.nii.gz").exists()
    assert (perf_out / "sub-01_ses-0001_desc-dceUncertainty_PS.nii.gz").exists()


def test_init_pmriprep_wf_dce_patlak_doesnt_crash(tmp_path: Path) -> None:
    """Build init_pmriprep_wf with --dce-model patlak and verify no NameError /
    crash on factory construction.
    """
    import json as _json

    from pmriprep.config.tumor_seg import TumorSegConfig
    from pmriprep.config.anat import AnatConfig
    from pmriprep.config.dsc import DSCConfig
    from pmriprep.config.dce import DCEConfig
    from pmriprep.workflows.base import PipelineSkipFlags, init_pmriprep_wf

    bids = tmp_path / "bids"
    raw = bids / "sub-01" / "ses-0001"
    (raw / "anat").mkdir(parents=True)
    (raw / "perf").mkdir(parents=True)
    (bids / "dataset_description.json").write_text(
        _json.dumps({"Name": "T", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )
    nib.save(
        nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)),
        raw / "anat" / "sub-01_ses-0001_T1w.nii.gz",
    )
    (raw / "anat" / "sub-01_ses-0001_T1w.json").write_text(
        _json.dumps({"RepetitionTime": 2.0})
    )
    nib.save(
        nib.Nifti1Image(np.zeros((4, 4, 4, 5), dtype=np.float32), np.eye(4)),
        raw / "perf" / "sub-01_ses-0001_dce.nii.gz",
    )
    (raw / "perf" / "sub-01_ses-0001_dce.json").write_text(
        _json.dumps({
            "RepetitionTime": 4.0,
            "FlipAngle": 12.0,
            "MagneticFieldStrength": 3.0,
            "EchoTime": 0.0014,
        })
    )

    dce_cfg = DCEConfig()
    dce_cfg.model = "patlak"
    dce_cfg.inference = "nlls"
    dce_cfg.no_t1 = True
    dce_cfg.b1_method = "none"

    wf = init_pmriprep_wf(
        bids_dir=str(bids),
        output_dir=str(tmp_path / "out"),
        subject_id="01",
        session_id="0001",
        skip_flags=PipelineSkipFlags(skip_anat=True, skip_dsc=True, skip_dce=False),
        anat_config=AnatConfig(),
        tumor_seg_config=TumorSegConfig(),
        dsc_config=DSCConfig(),
        dce_config=dce_cfg,
        work_dir=str(tmp_path / "work"),
    )
    assert wf is not None


def test_dce_factory_warnings_in_anchor_sidecar(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """``QCWarnings`` lands on ``desc-dcePreproc_dce.json`` from has_* flags."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    out = tmp_path / "out"

    # Include dce_moco so the anchor preproc sidecar is emitted; everything
    # else absent so all six presence warnings fire.
    wf = init_dce_datasinks_wf(
        output_dir=str(out), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        pk_model="tofts",
        present_inputs={"dce_moco"},
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_dce = src
    inp.ktrans_map = _write_nii(tmp_path / "k.nii.gz")
    inp.dce_moco = _write_nii(tmp_path / "moco.nii.gz")
    inp.qc_metrics = {"ok": True}
    wf.run()

    anchor = (
        out / "pmriprep" / "sub-01" / "ses-0001" / "perf"
        / "sub-01_ses-0001_desc-dcePreproc_dce.json"
    )
    assert anchor.exists(), f"Anchor sidecar missing: {anchor}"
    sidecar = json.loads(anchor.read_text())
    warnings = sidecar.get("QCWarnings", [])
    assert any("B1+" in w for w in warnings)
    assert any("T1 relaxation" in w for w in warnings)
    assert any("Brain mask" in w for w in warnings)
    assert any("registration transform" in w for w in warnings)

    # Legacy session-root QC JSON must not exist
    legacy = (
        out / "pmriprep" / "sub-01" / "ses-0001"
        / "sub-01_ses-0001_desc-dceQC_quality.json"
    )
    assert not legacy.exists()
