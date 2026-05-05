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


# ---------------------------------------------------------------------------
# Dual-T1w-underlay support (v0.47.0): a session configured with both
# t1w_precon and t1w_postcon must produce TWO anat-side cross-modality
# outputs differing only by the BIDS ce-GAD entity.
# ---------------------------------------------------------------------------


def _build_dce_underlay_wf(
    bids: Path, src: str, out: Path, tmp_path: Path,
    present_inputs: set, work_subdir: str,
    *, t1w_in_dce: str | None = None,
    t1w_postcon_in_dce: str | None = None,
):
    """Helper: build the DCE datasinks workflow with the given underlays."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    wf = init_dce_datasinks_wf(
        output_dir=str(out), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        pk_model="tofts",
        present_inputs=present_inputs,
    )
    wf.base_dir = str(tmp_path / work_subdir)
    inp = wf.inputs.inputnode
    inp.source_dce = src
    inp.ktrans_map = _write_nii(tmp_path / f"{work_subdir}-k.nii.gz")
    inp.qc_metrics = {}
    if t1w_in_dce is not None:
        inp.t1w_in_dce = t1w_in_dce
    if t1w_postcon_in_dce is not None:
        inp.t1w_postcon_in_dce = t1w_postcon_in_dce
    wf.run()


def test_dce_t1w_precon_underlay_lands_in_anat(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """Precon-only underlay: lands in anat/ with no ce- entity."""
    bids, src = minimal_bids
    out = tmp_path / "out"
    underlay = _write_nii(tmp_path / "t1w_in_dce.nii.gz")

    _build_dce_underlay_wf(
        bids, src, out, tmp_path,
        present_inputs={"t1w_in_dce"},
        work_subdir="work-precon",
        t1w_in_dce=underlay,
    )

    anat_dir = out / "pmriprep" / "sub-01" / "ses-0001" / "anat"
    expected = anat_dir / "sub-01_ses-0001_space-dce_T1w.nii.gz"
    assert expected.exists(), (
        f"precon underlay missing at {expected}; "
        f"anat/ contains: {sorted(p.name for p in anat_dir.iterdir())}"
    )
    assert not (anat_dir / "sub-01_ses-0001_ce-GAD_space-dce_T1w.nii.gz").exists()
    sidecar = json.loads(expected.with_suffix("").with_suffix(".json").read_text())
    assert "Pre-contrast" in sidecar["Description"]


def test_dce_t1w_postcon_underlay_lands_in_anat(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """Postcon-only underlay: lands in anat/ with ce-GAD entity."""
    bids, src = minimal_bids
    out = tmp_path / "out"
    underlay = _write_nii(tmp_path / "t1w_postcon_in_dce.nii.gz")

    _build_dce_underlay_wf(
        bids, src, out, tmp_path,
        present_inputs={"t1w_postcon_in_dce"},
        work_subdir="work-postcon",
        t1w_postcon_in_dce=underlay,
    )

    anat_dir = out / "pmriprep" / "sub-01" / "ses-0001" / "anat"
    expected = anat_dir / "sub-01_ses-0001_ce-GAD_space-dce_T1w.nii.gz"
    assert expected.exists(), (
        f"postcon underlay missing at {expected}; "
        f"anat/ contains: {sorted(p.name for p in anat_dir.iterdir())}"
    )
    assert not (anat_dir / "sub-01_ses-0001_space-dce_T1w.nii.gz").exists()
    sidecar = json.loads(expected.with_suffix("").with_suffix(".json").read_text())
    assert "Post-contrast" in sidecar["Description"]


def test_dce_both_t1w_underlays(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """Both underlays configured: two anat/ outputs differing by ce- entity."""
    bids, src = minimal_bids
    out = tmp_path / "out"
    pre = _write_nii(tmp_path / "t1w_in_dce.nii.gz")
    post = _write_nii(tmp_path / "t1w_postcon_in_dce.nii.gz")

    _build_dce_underlay_wf(
        bids, src, out, tmp_path,
        present_inputs={"t1w_in_dce", "t1w_postcon_in_dce"},
        work_subdir="work-both",
        t1w_in_dce=pre,
        t1w_postcon_in_dce=post,
    )

    anat_dir = out / "pmriprep" / "sub-01" / "ses-0001" / "anat"
    precon = anat_dir / "sub-01_ses-0001_space-dce_T1w.nii.gz"
    postcon = anat_dir / "sub-01_ses-0001_ce-GAD_space-dce_T1w.nii.gz"
    assert precon.exists() and postcon.exists(), (
        f"both underlays expected; anat/ contains: "
        f"{sorted(p.name for p in anat_dir.iterdir())}"
    )
    pre_meta = json.loads(precon.with_suffix("").with_suffix(".json").read_text())
    post_meta = json.loads(postcon.with_suffix("").with_suffix(".json").read_text())
    assert "Pre-contrast" in pre_meta["Description"]
    assert "Post-contrast" in post_meta["Description"]


# ---------------------------------------------------------------------------
# Fp/PS uncertainty TR gate (v0.46.2): the build-time decision to add
# fp_uncertainty/ps_uncertainty to dce_present must mirror the runtime
# decision in dce_compute._build_candidate_set so the sink isn't built
# without an upstream feed.
# ---------------------------------------------------------------------------


def test_2cxm_uncertainty_tr_gate_auto_short_tr() -> None:
    """auto-mode + TR ≤ 3.0 s → 2CXM in candidates → emit Fp/PS std."""
    from pmriprep.workflows.base import _emit_2cxm_uncertainty

    assert _emit_2cxm_uncertainty("auto", 2.5) is True
    assert _emit_2cxm_uncertainty("auto", 3.0) is True


def test_2cxm_uncertainty_tr_gate_auto_long_tr() -> None:
    """auto-mode + TR > 3.0 s → 2CXM excluded → don't emit Fp/PS std.

    Reproduces the original crash scenario: current institutional cohort
    has volume_tr=4.06 s; the prior (auto, _) gate built the sink but the
    producer never fed it → ``NodeExecutionError: _copy_perf_map missing
    in_file``.
    """
    from pmriprep.workflows.base import _emit_2cxm_uncertainty

    assert _emit_2cxm_uncertainty("auto", 4.06) is False
    assert _emit_2cxm_uncertainty("auto", 7.6) is False


def test_2cxm_uncertainty_tr_gate_explicit_2cxm() -> None:
    """Explicit ``model="2cxm"`` always emits — bypasses the TR check."""
    from pmriprep.workflows.base import _emit_2cxm_uncertainty

    assert _emit_2cxm_uncertainty("2cxm", 4.06) is True
    assert _emit_2cxm_uncertainty("2cxm", None) is True


def test_2cxm_uncertainty_tr_gate_other_models() -> None:
    """Non-2CXM single-model paths never emit 2CXM uncertainty."""
    from pmriprep.workflows.base import _emit_2cxm_uncertainty

    for model in ("tofts", "extended_tofts", "patlak"):
        assert _emit_2cxm_uncertainty(model, 2.5) is False, model


def test_2cxm_uncertainty_tr_gate_malformed_inputs() -> None:
    """Defensive: missing/malformed values fall through to no-emission."""
    from pmriprep.workflows.base import _emit_2cxm_uncertainty

    assert _emit_2cxm_uncertainty(None, 2.5) is False
    assert _emit_2cxm_uncertainty("auto", None) is False
    assert _emit_2cxm_uncertainty("auto", "not-a-number") is False  # type: ignore[arg-type]


def _build_dce_wf_for_tr_gate(
    tmp_path: Path, *, volume_tr: float, model: str, inference: str,
):
    """Build init_pmriprep_wf with a minimal DCE-only fixture at the given TR."""
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
            "RepetitionTime": volume_tr,
            "FlipAngle": 12.0,
            "MagneticFieldStrength": 3.0,
            "EchoTime": 0.0014,
        })
    )

    dce_cfg = DCEConfig()
    dce_cfg.model = model
    dce_cfg.inference = inference
    dce_cfg.no_t1 = True
    dce_cfg.b1_method = "none"

    return init_pmriprep_wf(
        bids_dir=str(bids),
        output_dir=str(tmp_path / "out"),
        subject_id="01",
        session_id="0001",
        skip_flags=PipelineSkipFlags(
            skip_anat=True, skip_dsc=True, skip_dce=False,
        ),
        anat_config=AnatConfig(),
        tumor_seg_config=TumorSegConfig(),
        dsc_config=DSCConfig(),
        dce_config=dce_cfg,
        work_dir=str(tmp_path / "work"),
    )


def _has_fp_uncertainty_sink(wf) -> bool:
    """True if any datasinks node ends with ``fp_uncertainty_sink``."""
    return any(
        name.endswith("fp_uncertainty_sink") for name in wf.list_node_names()
    )


def test_init_pmriprep_wf_auto_long_tr_no_fp_sink(tmp_path: Path) -> None:
    """Regression for crash.txt: auto + Bayesian + TR>3.0 → no fp/ps sink built.

    Reproduces the original ``NodeExecutionError: _copy_perf_map missing
    in_file`` failure mode: ``model="auto"``, ``inference="vi"``,
    ``RepetitionTime=4.06`` (current institutional cohort).
    """
    wf = _build_dce_wf_for_tr_gate(
        tmp_path, volume_tr=4.06, model="auto", inference="vi",
    )
    assert not _has_fp_uncertainty_sink(wf), (
        "auto + Bayesian + TR>3.0 must NOT build fp_uncertainty_sink "
        "(producer doesn't include 2CXM in candidate set at this TR; "
        "sink without feed → runtime crash)"
    )


def test_init_pmriprep_wf_auto_short_tr_builds_fp_sink(tmp_path: Path) -> None:
    """auto + Bayesian + TR≤3.0 → 2CXM in candidates → fp_uncertainty_sink built."""
    wf = _build_dce_wf_for_tr_gate(
        tmp_path, volume_tr=2.5, model="auto", inference="vi",
    )
    assert _has_fp_uncertainty_sink(wf), (
        "auto + Bayesian + TR≤3.0 must build fp_uncertainty_sink "
        "(producer includes 2CXM in candidate set; sink should consume it)"
    )
