"""Signal CPI must reach production with unchanged inputs, fits and output units."""

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from pmriprep.config.dsc import DSCConfig
from pmriprep.interfaces.dsc_compute import ComputeDSCMaps
from pmriprep.models.dsc.signal_cpi.acquisition import (
    prepare_cpi_acquisition,
    fit_selected_voxels,
)
from pmriprep.models.dsc.signal_cpi.settings import CPIOptions
from pmriprep.models.dsc.signal_cpi.maps import CPIMaps
from pmriprep.utils.bsw_guidance import BSWGuidance
from pmriprep.utils.bsw_guidance_io import load_bsw_guidance, save_bsw_guidance
from pmriprep.utils.dsc_preload import K2CorrectionResult
from pmriprep.workflows.dsc._helpers import _apply_dsc_leakage_policy


def read_image(path: str) -> np.ndarray:
    """Read the numeric array of a native-grid test image."""
    image: Any = nib.load(path)
    return np.asarray(image.get_fdata(), dtype=float)


@pytest.fixture
def captured_maps(monkeypatch: pytest.MonkeyPatch) -> list[CPIMaps]:
    """Inspect the real controller return without retaining images on the interface."""
    from pmriprep.models.dsc.signal_cpi import maps

    original = maps.fit_cpi_maps
    captured: list[CPIMaps] = []

    def capture(*args: Any, **kwargs: Any) -> CPIMaps:
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(maps, "fit_cpi_maps", capture)
    return captured


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Any]:
    """Write portable images with acquisition timing supplied explicitly."""
    time = np.arange(30.0) * 3
    u = np.maximum(time - 9, 0)
    aif = 10 * u**2 * np.exp(-u / 3)
    tissue = 0.2 * u**2 * np.exp(-u / 5)
    raw = np.stack([tissue, tissue * 1.1])[:, None, None]
    signal = 100 * np.exp(-0.03 * raw)
    affine = np.diag([2.0, 2.0, 3.0, 1.0])
    result: dict[str, Any] = {
        "time": time,
        "aif": aif,
        "raw": raw,
        "signal": signal,
        "affine": affine,
    }
    for name, data in (
        ("raw", raw),
        ("signal", signal),
        ("mask", np.ones(raw.shape[:3])),
    ):
        path = tmp_path / (name + ".nii.gz")
        nib.save(nib.Nifti1Image(data, affine), path)
        result[name + "_file"] = str(path)
    np.savetxt(tmp_path / "aif.txt", aif)
    result["aif_file"] = str(tmp_path / "aif.txt")
    return result


def interface_for(files: dict[str, Any], **options: Any) -> ComputeDSCMaps:
    """Configure a minimal native-grid production compute node."""
    return ComputeDSCMaps(
        concentration_file=files["raw_file"],
        raw_concentration_file=files["raw_file"],
        signal_file=files["signal_file"],
        brain_mask_file=files["mask_file"],
        aif_file=files["aif_file"],
        measured_aif_file=files["aif_file"],
        method="cpi",
        te=0.03,
        tr=3.0,
        baseline_frames=4,
        leakage_method="none",
        cpi_control_points=5,
        cpi_dispersion="none",
        **options,
    )


def guidance_for(files: dict[str, Any]) -> BSWGuidance:
    """Supply distinct voxel coefficients and the same owned reference basis."""
    curve = files["raw"][0, 0, 0]
    integral = cumulative_trapezoid(curve, files["time"], initial=0)
    shape = files["raw"].shape[:3]
    return BSWGuidance(
        curve,
        integral,
        0.01 * integral[:, None],
        np.ones(shape, bool),
        2,
        None,
        np.array([0.001, -0.002]).reshape(shape),
        np.full(shape, 0.004),
        np.zeros(shape, np.uint8),
        4,
        3.0,
        28,
    )


@pytest.mark.parametrize(
    "inference,infer_delay", [("vi", True), ("vi", False), ("svb", False)]
)
def test_actual_interface_uses_signal_and_preserves_selected_fit(
    files: dict[str, Any],
    tmp_path: Path,
    inference: str,
    infer_delay: bool,
    monkeypatch: pytest.MonkeyPatch,
    captured_maps: list[CPIMaps],
) -> None:
    monkeypatch.chdir(tmp_path)
    interface = interface_for(
        files, inference=inference, cpi_infer_delay=infer_delay, cpi_fallback="none"
    )
    monkeypatch.setattr(
        interface,
        "_deconvolve_parametric",
        lambda *a, **k: pytest.fail("old CPI route"),
    )
    monkeypatch.setattr(
        interface, "_save_qc_fit_data", lambda *a, **k: pytest.fail("QC must not refit")
    )
    output = interface.run().outputs
    native = captured_maps[0]
    assert not any(isinstance(value, CPIMaps) for value in vars(interface).values())
    np.testing.assert_allclose(
        read_image(output.out_rel_cbf), native.cbf * 60, rtol=1e-6
    )
    np.testing.assert_allclose(
        read_image(output.out_cbf_std), native.cbf_std * 60, rtol=1e-6
    )
    np.testing.assert_allclose(read_image(output.out_mtt), native.mtt, rtol=1e-6)
    np.testing.assert_allclose(read_image(output.out_rel_cbv), native.cbv, rtol=1e-6)
    assert not np.any(read_image(output.out_fallback_mask))
    qc = json.loads(Path(output.out_qc_fit_data).read_text())
    assert (
        qc["fit_domain"] == "signal"
        and qc["prediction_source"] == "selected_voxel_posterior"
    )
    voxel = tuple(qc["voxel"])
    np.testing.assert_array_equal(qc["model_prediction"], native.predictions[voxel])
    np.testing.assert_array_equal(qc["raw_signal"], files["signal"][voxel])
    assert qc["r_squared"] == native.r_squared[voxel]
    timing = read_image(output.out_tmax)
    assert np.all(np.isfinite(timing)) if infer_delay else np.all(np.isnan(timing))
    if inference == "vi":
        prepared = prepare_cpi_acquisition(
            files["signal"],
            files["aif"],
            files["time"],
            0.03,
            leakage_method="none",
            baseline_frames=4,
            options=CPIOptions(
                control_points=5,
                infer_delay=infer_delay,
                dispersion="none",
                fallback="none",
            ),
        )
        for expected in fit_selected_voxels(prepared):
            i = expected.flat_index
            np.testing.assert_allclose(
                [native.cbf.flat[i], native.cbv.flat[i], native.mtt.flat[i]],
                [expected.cbf, expected.cbv, expected.mtt],
                rtol=1e-10,
                atol=1e-12,
            )


@pytest.mark.parametrize("fallback", ["osvd", "none"])
def test_rejected_production_voxels_follow_configured_fallback(
    files: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fallback: str,
    captured_maps: list[CPIMaps],
) -> None:
    monkeypatch.chdir(tmp_path)
    bad = files["signal"].copy()
    bad[..., 15] = np.nan
    nib.save(nib.Nifti1Image(bad, files["affine"]), files["signal_file"])
    interface = interface_for(files, cpi_fallback=fallback)
    output = interface.run().outputs
    cbf = read_image(output.out_rel_cbf)
    mask = read_image(output.out_fallback_mask)
    if fallback == "none":
        assert np.all(np.isnan(cbf)) and not np.any(mask)
    else:
        assert np.all(np.isfinite(cbf)) and np.all(mask == 1)
    assert np.all(np.isnan(read_image(output.out_cbf_std)))
    assert np.all(np.isfinite(read_image(output.out_rel_cbv_auc)))
    assert captured_maps[0].qc["n_fallback_requested"] == 2


@pytest.mark.parametrize("missing", [False, True])
def test_guidance_roundtrip_is_lossless(
    files: dict[str, Any], tmp_path: Path, missing: bool
) -> None:
    guidance = guidance_for(files)
    if missing:
        guidance = replace(
            guidance,
            reference_curve=None,
            reference_integral=None,
            reference_basis=None,
            reference_mask=None,
            reference_count=0,
            reference_reason="no_reference",
            k2_sd=np.full(guidance.k2.shape, np.nan),
            status=np.full(guidance.k2.shape, 2, np.uint8),
        )
    path = save_bsw_guidance(guidance, tmp_path / "guidance.npz", files["affine"])
    restored = load_bsw_guidance(
        path, affine=files["affine"], shape=files["raw"].shape, tr=3.0
    )
    for item in fields(guidance):
        before, after = getattr(guidance, item.name), getattr(restored, item.name)
        if isinstance(before, np.ndarray):
            np.testing.assert_array_equal(before, after)
        else:
            assert before == after
    with pytest.raises(ValueError, match="grid"):
        load_bsw_guidance(path, affine=np.eye(4), shape=files["raw"].shape, tr=3.0)
    with pytest.raises(ValueError, match="grid"):
        load_bsw_guidance(
            path, affine=files["affine"], shape=files["raw"].shape, tr=1.5
        )


def test_upstream_helper_saves_requested_exact_guidance(
    files: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    guidance = guidance_for(files)

    def correct(**kwargs: Any) -> K2CorrectionResult:
        assert kwargs["return_guidance"]
        np.testing.assert_array_equal(kwargs["signal"], files["signal"])
        return K2CorrectionResult(
            files["raw"],
            guidance.k2,
            np.ones(guidance.k2.shape),
            True,
            method="bsw",
            guidance=guidance,
        )

    monkeypatch.setattr("pmriprep.utils.dsc_preload.apply_k2_correction", correct)
    _, _, _, path = _apply_dsc_leakage_policy(
        files["raw_file"],
        files["mask_file"],
        0.03,
        3.0,
        3.0,
        "unknown",
        files["signal_file"],
        4,
        "bsw",
        return_guidance=True,
    )
    restored = load_bsw_guidance(
        path, affine=files["affine"], shape=files["raw"].shape, tr=3.0
    )
    np.testing.assert_array_equal(restored.reference_basis, guidance.reference_basis)
    np.testing.assert_array_equal(restored.k2_sd, guidance.k2_sd)


@pytest.mark.parametrize(
    "fault", ["no_te", "no_signal", "off_grid", "no_guidance", "wrong_engine"]
)
def test_cpi_rejects_missing_or_mismatched_contract(
    files: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    from nipype.interfaces.base import Undefined

    monkeypatch.chdir(tmp_path)
    interface = interface_for(files)
    if fault == "no_te":
        interface.inputs.te = Undefined
    elif fault == "no_signal":
        interface.inputs.signal_file = Undefined
    elif fault == "off_grid":
        nib.save(nib.Nifti1Image(files["signal"], np.eye(4)), files["signal_file"])
    elif fault == "no_guidance":
        interface.inputs.leakage_method = "bsw"
    else:
        interface.inputs.inference = "map"
    with pytest.raises(ValueError, match="CPI"):
        interface.run()


def test_approved_public_config_is_validated() -> None:
    from pmriprep.config.loader import build_dsc_config

    config = DSCConfig(method="cpi")
    assert (
        config.cpi_infer_delay
        and config.cpi_dispersion == "none"
        and config.cpi_fallback == "osvd"
    )
    assert not hasattr(config, "cpi_infer_cpt")
    with pytest.raises(TypeError):
        DSCConfig(cpi_infer_cpt=False)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="cpi_infer_cpt"):
        build_dsc_config({"dsc": {"cpi_infer_cpt": False}})
    assert (
        build_dsc_config({"dsc": {"cpi_control_points": "6"}}).cpi_control_points == 6
    )
    for options in (
        {"cpi_control_points": 4},
        {"cpi_control_points": 6.5},
        {"cpi_dispersion": "bad"},
        {"cpi_fallback": "bad"},
        {"inference": "map"},
    ):
        with pytest.raises((ValueError, TypeError)):
            DSCConfig(method="cpi", **options)
        with pytest.raises((ValueError, TypeError)):
            build_dsc_config({"dsc": {"method": "cpi", **options}})


def test_compute_delivers_original_signal_and_exact_bsw_guidance(
    files: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    captured_maps: list[CPIMaps],
) -> None:
    """Corrected concentration is reserved for fallback, never the signal fit."""
    from pmriprep.models.dsc.signal_cpi import maps
    from pmriprep.models.dsc.signal_cpi.acquisition import CPIAcquisition
    from pmriprep.models.dsc.signal_cpi.connection import GuidedVoxel

    monkeypatch.chdir(tmp_path)
    guidance = guidance_for(files)
    archive = save_bsw_guidance(guidance, tmp_path / "bsw.npz", files["affine"])
    corrected_file = tmp_path / "corrected.nii.gz"
    nib.save(nib.Nifti1Image(files["raw"] * 2, files["affine"]), corrected_file)
    k2_file = tmp_path / "k2.nii.gz"
    nib.save(nib.Nifti1Image(guidance.k2, files["affine"]), k2_file)
    interface = interface_for(files, cpi_fallback="none")
    interface.inputs.leakage_method = "bsw"
    interface.inputs.bsw_guidance_file = archive
    interface.inputs.k2_map_input_file = str(k2_file)
    interface.inputs.concentration_file = str(corrected_file)
    called = []

    def selected(a: CPIAcquisition, **kwargs: Any) -> Any:
        called.append(True)
        assert kwargs["executor"].resources.requested_processes == int(interface.inputs.n_procs)
        branch = a.branches[0]
        np.testing.assert_array_equal(branch.signals, files["signal"].reshape(2, -1))
        np.testing.assert_array_equal(branch.k2, guidance.k2.ravel())
        np.testing.assert_array_equal(branch.k2_sd, guidance.k2_sd.ravel())
        assert branch.reference is not None
        np.testing.assert_array_equal(
            branch.reference.mean, guidance.reference_integral
        )
        np.testing.assert_array_equal(branch.reference.basis, guidance.reference_basis)
        for index in range(2):
            yield GuidedVoxel(
                index,
                "retained",
                (),
                0.01,
                0.04,
                4.0,
                0.0,
                fitted_k2=0.003,
                cbf_std=0.001,
                family="cpi",
            )

    monkeypatch.setattr(maps, "fit_selected_voxels", selected)
    output = interface.run().outputs
    assert called == [True]
    np.testing.assert_allclose(read_image(output.out_k2), guidance.k2, rtol=1e-6)
    np.testing.assert_allclose(captured_maps[0].fitted_k2, 0.003)
    assert captured_maps[0].qc["cpi_leakage_method"] == "bsw"


def test_resource_flags_reach_cpi_controller(
    files: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interface scheduling flags must reach fitting rather than only Nipype."""
    from pmriprep.models.dsc.signal_cpi import maps

    monkeypatch.chdir(tmp_path)
    interface = interface_for(files, n_procs=4, sloppy=True, fitting_memory_gb=2.5)

    class Captured(Exception):
        pass

    def capture(a: Any, *args: Any, **kwargs: Any) -> Any:
        assert kwargs["n_procs"] == 4 and kwargs["memory_gb"] == 2.5
        assert all(b.sloppy for b in a.branches)
        assert a.options.fallback == "osvd"
        raise Captured

    monkeypatch.setattr(maps, "fit_cpi_maps", capture)
    with pytest.raises(Captured):
        interface.run()


@pytest.mark.parametrize("memory_gb,workers", [(1.5, 1), (3., 2)])
def test_cpi_process_pool_runs_inside_nipype_multiproc(
    files: dict[str, Any], tmp_path: Path, memory_gb: float, workers: int,
) -> None:
    """A scheduled compute node caps its owned workers at the memory allocation."""
    import nipype.pipeline.engine as pe

    interface = interface_for(files, n_procs=2, sloppy=True, cpi_fallback="none",
                              fitting_memory_gb=memory_gb)
    node = pe.Node(interface, name="maps", n_procs=2, mem_gb=memory_gb)
    workflow = pe.Workflow(name="cpi_process_test", base_dir=str(tmp_path))
    workflow.add_nodes([node])
    # A fresh worker isolates its memory budget from unrelated tests' resident data.
    graph = workflow.run(plugin="MultiProc", plugin_args={
        "n_procs": 2, "memory_gb": 4.0, "mp_context": "spawn",
    })
    completed = next(n for n in graph.nodes if n.name == "maps")
    qc = completed.result.outputs.qc_metrics
    assert qc["execution"]["processes"] == workers
    assert qc["execution"]["sloppy"] is True


@pytest.mark.parametrize("inference", ["vi", "svb"])
def test_empty_mask_never_fits_or_falls_back(
    files: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inference: str,
    captured_maps: list[CPIMaps],
) -> None:
    from pmriprep.models.dsc.signal_cpi import fallback

    monkeypatch.chdir(tmp_path)
    nib.save(
        nib.Nifti1Image(np.zeros(files["raw"].shape[:3]), files["affine"]),
        files["mask_file"],
    )
    monkeypatch.setattr(
        fallback,
        "svd_deconvolve_oscillation",
        lambda *a, **k: pytest.fail("empty mask refit"),
    )
    interface = interface_for(files, inference=inference)
    output = interface.run().outputs
    assert not np.any(read_image(output.out_rel_cbf))
    assert captured_maps[0].qc["n_mask_voxels"] == 0
    assert captured_maps[0].qc["n_fallback_requested"] == 0
    assert captured_maps[0].qc["n_missing"] == 0


def test_signal_qc_preserves_enhancing_voxel_selection(
    files: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    interface = interface_for(files)
    enhancing = np.zeros(files["raw"].shape[:3])
    enhancing[0] = 1
    path = tmp_path / "enhancing.nii.gz"
    nib.save(nib.Nifti1Image(enhancing, files["affine"]), path)
    interface.inputs.enhancing_mask_file = str(path)
    record = CPIMaps(
        cbf=np.ones_like(enhancing),
        cbv=np.ones_like(enhancing),
        mtt=np.ones_like(enhancing),
        delay=np.zeros_like(enhancing),
        cbf_std=np.zeros_like(enhancing),
        fallback_mask=np.zeros_like(enhancing, dtype=np.uint8),
        qc={},
        dispersion=np.zeros_like(enhancing),
        predictions=files["signal"],
        r_squared=np.ones_like(enhancing),
        converged=np.ones_like(enhancing, dtype=bool),
        fitted_k2=np.zeros_like(enhancing),
    )
    interface._results = {}
    interface._save_signal_cpi_qc(
        np.ones_like(enhancing),
        files["raw"],
        files["aif"],
        np.ones_like(enhancing, dtype=bool),
        3.0,
        files["signal"],
        record,
    )
    saved = json.loads(Path(interface._results["out_qc_fit_data"]).read_text())
    assert saved["voxel"] == [0, 0, 0]
    assert interface._qc_voxel_tier == "enhancing"
