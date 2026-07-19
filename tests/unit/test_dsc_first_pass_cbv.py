"""End-to-end + inertness + sidecar tests for the first-pass CBV output.

Covers the opt-in ``dsc.first_pass_cbv`` map through the ``ComputeDSCMaps``
interface (gamma-variate first-pass fit, WM-median normalization, the no-map
paths when a WM reference is unavailable/rejected, config OFF), the S1 inertness
invariant (nothing internal consumes the output), and the per-map sidecar
disclosures.
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pmriprep.interfaces.dsc_compute import ComputeDSCMaps


def _gamma(t, amplitude, t0, alpha, beta):
    dt = t - t0
    c = np.zeros_like(t, dtype=np.float64)
    m = dt > 0
    c[m] = amplitude * dt[m] ** alpha * np.exp(-dt[m] / beta)
    return c


@pytest.fixture
def dsc_inputs(tmp_path):
    """Minimal synthetic 4D DSC dataset + AIF + masks for a full run."""
    rng = np.random.default_rng(0)
    nx = ny = nz = 6
    T = 40
    t = np.arange(T, dtype=np.float64)
    aff = np.eye(4)

    conc = np.zeros((nx, ny, nz, T), dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                conc[i, j, k] = _gamma(t, rng.uniform(1.0, 2.0), 8.0, 3.0, 2.0)
                conc[i, j, k] += rng.normal(0.0, 0.02, T)

    conc_f = str(tmp_path / "conc.nii.gz")
    nib.save(nib.Nifti1Image(conc.astype(np.float32), aff), conc_f)
    mask = np.ones((nx, ny, nz), dtype=np.uint8)
    mask[0, :, :] = 0
    mask_f = str(tmp_path / "mask.nii.gz")
    nib.save(nib.Nifti1Image(mask, aff), mask_f)
    aif = _gamma(t, 3.0, 7.0, 3.0, 2.0)
    aif_f = str(tmp_path / "aif.txt")
    np.savetxt(aif_f, aif)
    maif_f = str(tmp_path / "maif.txt")
    np.savetxt(maif_f, aif)

    # Interior WM mask (>= 50 voxels) so the WM-FAST tier is chosen.
    wm = np.zeros((nx, ny, nz), dtype=np.uint8)
    wm[1:5, 1:5, 1:5] = 1
    wm_f = str(tmp_path / "wm.nii.gz")
    nib.save(nib.Nifti1Image(wm, aff), wm_f)

    return {
        "concentration_file": conc_f,
        "aif_file": aif_f,
        "measured_aif_file": maif_f,
        "brain_mask_file": mask_f,
        "wm_mask_file": wm_f,
        "affine": aff,
    }


def _run(dsc_inputs, tmp_path, **kw):
    node = ComputeDSCMaps(
        concentration_file=dsc_inputs["concentration_file"],
        aif_file=dsc_inputs["aif_file"],
        measured_aif_file=dsc_inputs["measured_aif_file"],
        brain_mask_file=dsc_inputs["brain_mask_file"],
        tr=1.5,
        method="oSVD",
        n_procs=1,
        **kw,
    )
    node.base_dir = str(tmp_path)
    import os
    cwd = os.getcwd()
    workdir = tmp_path / "run"
    workdir.mkdir(exist_ok=True)
    os.chdir(workdir)
    try:
        return node.run()
    finally:
        os.chdir(cwd)


def test_config_on_wm_median_normalized(dsc_inputs, tmp_path):
    """Default (WM mask present): map exists, WM median == 1.0, tier = WM-FAST."""
    res = _run(
        dsc_inputs, tmp_path,
        first_pass_cbv=True, wm_mask_dsc=dsc_inputs["wm_mask_file"],
    )
    out = res.outputs.out_rel_cbv_firstpass
    assert out and Path(out).exists()
    data = nib.load(out).get_fdata()
    wm = nib.load(dsc_inputs["wm_mask_file"]).get_fdata() > 0
    # rCBV = 1.0 at the reference median (over the fitted WM voxels; unfittable
    # voxels are NaN, so use nanmedian).
    assert abs(float(np.nanmedian(data[wm])) - 1.0) < 1e-3
    # Upper sentinel clamp holds on the finite (fitted) voxels.
    assert float(np.nanmax(data)) <= 10.0 + 1e-4

    qc = res.outputs.qc_metrics
    assert qc["firstpass_reference"] == "firstpass-WM-FAST"
    assert qc["firstpass_reference_n"] >= 50
    assert 0.0 <= qc["firstpass_unfittable_fraction"] <= 1.0


def test_saved_dsc_diagnostics_are_brain_masked(dsc_inputs, tmp_path):
    """Saved first-pass, K2, delay, and residue outputs have zero background."""
    res = _run(
        dsc_inputs,
        tmp_path,
        first_pass_cbv=True,
        wm_mask_dsc=dsc_inputs["wm_mask_file"],
        save_residue=True,
    )
    mask = nib.load(dsc_inputs["brain_mask_file"]).get_fdata() > 0

    for output_name in ("out_rel_cbv_firstpass", "out_k2", "out_delay"):
        data = nib.load(getattr(res.outputs, output_name)).get_fdata()
        assert np.all(data[~mask] == 0), output_name

    residues = nib.load(res.outputs.out_residues).get_fdata()
    assert np.all(residues[~mask, :] == 0)


def test_no_wm_produces_no_map(dsc_inputs, tmp_path):
    """No WM mask: the map has no valid anchor, so it is not produced.

    There is no whole-brain fallback (a whole-brain percentile / GMM-largest-
    component is a biased, cross-subject-unstable WM proxy on real data). The
    output is left Undefined (same as config-off) with an explanatory QC note.
    """
    from nipype.interfaces.base import isdefined
    res = _run(dsc_inputs, tmp_path, first_pass_cbv=True, wm_mask_dsc="")
    assert not isdefined(res.outputs.out_rel_cbv_firstpass)
    qc = res.outputs.qc_metrics
    assert qc["firstpass_reference"] == "unavailable-no-wm-segmentation"
    assert "firstpass_reference_n" not in qc
    assert "firstpass_low_confidence" not in qc


def test_tiny_wm_produces_no_map(dsc_inputs, tmp_path):
    """A WM mask with < 50 voxels is rejected: no valid reference, no map.

    The QC reason distinguishes a present-but-rejected reference from a wholly
    absent WM segmentation.
    """
    from nipype.interfaces.base import isdefined
    tiny = np.zeros((6, 6, 6), dtype=np.uint8)
    tiny[3, 3, 3] = 1  # 1 voxel
    tiny_f = str(tmp_path / "tiny_wm.nii.gz")
    nib.save(nib.Nifti1Image(tiny, dsc_inputs["affine"]), tiny_f)
    res = _run(dsc_inputs, tmp_path, first_pass_cbv=True, wm_mask_dsc=tiny_f)
    assert not isdefined(res.outputs.out_rel_cbv_firstpass)
    assert res.outputs.qc_metrics["firstpass_reference"] == "unavailable-wm-reference-rejected"


def test_no_bolus_wm_reference_rejected(dsc_inputs, tmp_path):
    """Pure-noise concentration: every gamma fit fails, so no reference, no map.

    Exercises the reference-rejected path via the unfittable route (all voxels
    NaN -> fewer than the minimum fitted WM voxels).
    """
    from nipype.interfaces.base import isdefined
    rng = np.random.default_rng(3)
    conc = nib.load(dsc_inputs["concentration_file"]).get_fdata()
    noise = rng.normal(0.0, 0.01, size=conc.shape).astype(np.float32)
    noise_f = str(tmp_path / "noise_conc.nii.gz")
    nib.save(nib.Nifti1Image(noise, dsc_inputs["affine"]), noise_f)
    inp = dict(dsc_inputs)
    inp["concentration_file"] = noise_f
    res = _run(inp, tmp_path, first_pass_cbv=True, wm_mask_dsc=dsc_inputs["wm_mask_file"])
    assert not isdefined(res.outputs.out_rel_cbv_firstpass)
    assert res.outputs.qc_metrics["firstpass_reference"] == "unavailable-wm-reference-rejected"


def test_config_off_produces_no_map(dsc_inputs, tmp_path):
    """Config OFF: no first-pass map, no error, no firstpass QC keys."""
    from nipype.interfaces.base import isdefined
    res = _run(dsc_inputs, tmp_path, first_pass_cbv=False)
    # Left Undefined (like out_delay); datasink isdefined guard skips it.
    assert not isdefined(res.outputs.out_rel_cbv_firstpass)
    qc = res.outputs.qc_metrics
    assert "firstpass_reference" not in qc
    assert "firstpass_low_confidence" not in qc


# ---------------------------------------------------------------------------
# Inertness (S1): the first-pass output feeds ONLY the public outputnode
# inside the preproc workflow — nothing internal (constraint / MTT /
# normalization) consumes it.
# ---------------------------------------------------------------------------
def test_first_pass_output_is_inert(tmp_path):
    """The first-pass output is consumed only by the public outputnode."""
    from tests.unit.test_workflow_graph_regression import (
        _dsc,
        _write_bids,
        extract_edges,
        flatten_collapsed_graph,
    )

    bids = _write_bids(tmp_path)
    g = flatten_collapsed_graph(_dsc(bids, first_pass_cbv=True))
    firstpass_edges = [
        e for e in extract_edges(g)
        if "firstpass" in e[1].lower() or "firstpass" in e[3].lower()
    ]
    # The compute node emits it; the only consumer is the public outputnode.
    assert firstpass_edges, "expected a first-pass edge in the graph"
    consumers = {dst_node for _, _, dst_node, _ in firstpass_edges}
    assert consumers == {"outputnode"}, (
        f"first-pass output consumed by non-public nodes: {consumers}"
    )
    # And no normalization/deconvolution node ingests a firstpass field.
    for _src_node, _src, dst_node, _dst in firstpass_edges:
        assert "normal" not in dst_node.lower()
        assert dst_node != "compute_maps"


# ---------------------------------------------------------------------------
# Sidecar disclosures (per-map JSON).
# ---------------------------------------------------------------------------
def test_first_pass_sidecar_disclosures(tmp_path):
    """The per-map sidecar carries the required first-pass disclosures + tags."""
    from pmriprep.workflows.dsc.datasinks import init_dsc_datasinks_wf

    bids = tmp_path / "bids"
    perf = bids / "sub-t01" / "perf"
    perf.mkdir(parents=True)
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )
    src = perf / "sub-t01_dsc.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4)), src)

    def _dummy(name):
        p = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4)), p)
        return str(p)

    out_dir = str(tmp_path / "derivatives")
    present = {
        "mtt_map", "tmax_map", "psr_map", "ph_map", "rel_cbv_firstpass_map",
    }
    wf = init_dsc_datasinks_wf(
        output_dir=out_dir, bids_dir=str(bids),
        subject_id="t01", session_id=None, present_inputs=present,
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_dsc = str(src)
    for m in ("mtt_map", "tmax_map", "psr_map", "ph_map", "rel_cbv_firstpass_map"):
        setattr(inp, m, _dummy(m))
    inp.qc_metrics = {
        "aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.85},
        "firstpass_reference": "firstpass-WM-FAST",
        "firstpass_reference_n": 1234,
        "firstpass_unfittable_fraction": 0.18,
        "firstpass_prefit_fwhm_mm": 5.0,
    }
    wf.run()

    perf_out = (
        Path(out_dir) / "pmriprep" / "sub-t01" / "perf"
        / "sub-t01_desc-dscRelFirstPassCbv_stat.json"
    )
    assert perf_out.exists(), "first-pass CBV sidecar not written"
    sc = json.loads(perf_out.read_text())

    # AIF-independent (tissue/tissue): no AIF dependency, safe under synthetic.
    assert sc["AIFDependent"] is False
    assert sc["TemplateAIFUsage"] == "safe"
    # Dynamic per-run reference + fit-quality keys.
    assert sc["FirstPassReference"] == "firstpass-WM-FAST"
    assert sc["FirstPassReferenceVoxelCount"] == 1234
    assert sc["FirstPassUnfittableFraction"] == 0.18
    assert sc["FirstPassPreSmoothingFWHMmm"] == 5.0
    # Removed keys must not resurface.
    assert "FirstPassFallbackFraction" not in sc
    assert "FirstPassLowConfidence" not in sc
    # Static disclosures.
    assert sc["CBVMethod"] == "gamma_variate_first_pass_area"
    assert "gamma" in sc["Estimator"].lower()
    assert "UnfittableVoxels" in sc
    assert "NotTumorCBV" in sc and "dscRelAUCCbv" in sc["NotTumorCBV"]
    assert "intact" in sc["ConcentrationInput"].lower()
    assert "none" in sc["AbsoluteCalibration"].lower()
    assert sc["Units"] == "arbitrary"


def test_first_pass_nii_written(tmp_path):
    """The BIDS-named NIfTI is written alongside the sidecar."""
    from pmriprep.workflows.dsc.datasinks import init_dsc_datasinks_wf

    bids = tmp_path / "bids"
    perf = bids / "sub-t01" / "perf"
    perf.mkdir(parents=True)
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )
    src = perf / "sub-t01_dsc.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4)), src)

    def _dummy(name):
        p = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4)), p)
        return str(p)

    out_dir = str(tmp_path / "derivatives")
    present = {"mtt_map", "tmax_map", "psr_map", "ph_map", "rel_cbv_firstpass_map"}
    wf = init_dsc_datasinks_wf(
        output_dir=out_dir, bids_dir=str(bids),
        subject_id="t01", session_id=None, present_inputs=present,
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_dsc = str(src)
    for m in ("mtt_map", "tmax_map", "psr_map", "ph_map", "rel_cbv_firstpass_map"):
        setattr(inp, m, _dummy(m))
    inp.qc_metrics = {
        "aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.85},
        "firstpass_reference": "firstpass-WM-FAST",
        "firstpass_reference_n": 500,
    }
    wf.run()
    nii = (
        Path(out_dir) / "pmriprep" / "sub-t01" / "perf"
        / "sub-t01_desc-dscRelFirstPassCbv_stat.nii.gz"
    )
    assert nii.exists()
