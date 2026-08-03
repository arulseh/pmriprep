"""Workflow-level tests for preload provenance and leakage correction."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import nibabel as nib  # type: ignore[import-untyped]
from nipype.interfaces.base import isdefined
from nipype.pipeline import engine as pe
import numpy as np
import pytest

from pmriprep.utils.dsc_preload import K2CorrectionResult, PreloadClass, PreloadResult
from pmriprep.workflows.dsc.base import (
    _detect_and_correct_preload,
    init_dsc_deconvolution_wf,
    init_dsc_preload_wf,
    init_dsc_qc_wf,
)


def test_dsc_compute_node_reserves_internal_worker_count() -> None:
    """Nipype reserves the processors used by the internal fitting pool."""
    workflow = init_dsc_deconvolution_wf(
        dsc_map_kwargs={"n_procs": 6}, mem_gb=1.0,
    )
    assert workflow.get_node("compute_maps").n_procs == 6


def _write_fixtures(tmp_path: Path) -> tuple[str, str, str]:
    """Write minimal concentration, brain-mask, and raw-signal fixtures."""
    shape = (4, 4, 4, 20)
    rng = np.random.default_rng(0)
    conc = rng.normal(0.0, 0.01, size=shape).astype(np.float32)
    conc_path = tmp_path / "conc.nii.gz"
    nib.save(nib.Nifti1Image(conc, np.eye(4)), str(conc_path))

    mask = np.ones(shape[:3], dtype=np.float32)
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))

    signal_path = tmp_path / "signal.nii.gz"
    nib.save(nib.Nifti1Image(1000.0 * np.exp(-0.03 * conc), np.eye(4)), signal_path)

    return str(conc_path), str(mask_path), str(signal_path)


def _write_executable_fixtures(tmp_path: Path) -> tuple[str, str, str, str]:
    """Write a sufficiently supported bolus phantom for an executable workflow."""
    curve = np.array(
        [0, 0, 0, 0, 0, 0.2, 0.8, 1.8, 1.2, 0.7,
         0.45, 0.32, 0.25, 0.20, 0.16, 0.13, 0.11, 0.09, 0.08, 0.07],
        dtype=np.float32,
    )
    concentration = np.broadcast_to(curve, (8, 8, 8, curve.size)).copy()
    concentration_file = tmp_path / "executable_conc.nii.gz"
    nib.save(nib.Nifti1Image(concentration, np.eye(4)), concentration_file)

    mask_file = tmp_path / "executable_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.uint8), np.eye(4)), mask_file)

    signal_file = tmp_path / "executable_signal.nii.gz"
    signal = 1000.0 * np.exp(-0.03 * concentration)
    nib.save(nib.Nifti1Image(signal.astype(np.float32), np.eye(4)), signal_file)

    aif_file = tmp_path / "executable_aif.txt"
    np.savetxt(aif_file, curve)
    return str(concentration_file), str(mask_file), str(signal_file), str(aif_file)


@pytest.mark.parametrize("leakage_method", ["bsw", "none"])
def test_leakage_policy_executes_through_preload_and_deconvolution(
    tmp_path: Path, leakage_method: str
) -> None:
    """Each public leakage policy executes across the connected DSC seam."""
    concentration, mask, signal, aif = _write_executable_fixtures(tmp_path)
    workflow = pe.Workflow(name=f"smoke_{leakage_method}", base_dir=str(tmp_path))
    preload = init_dsc_preload_wf(
        preload="unknown",
        agent_standard_dose=0.1,
        baseline_frames=5,
        leakage_method=leakage_method,
        mem_gb=0.1,
        name="preload",
    )
    deconvolution = init_dsc_deconvolution_wf(
        dsc_map_kwargs={
            "method": "sSVD",
            "leakage_method": leakage_method,
            "n_procs": 1,
        },
        mem_gb=0.1,
        name="deconvolution",
    )
    preload.inputs.inputnode.concentration_file = concentration
    preload.inputs.inputnode.signal_file = signal
    preload.inputs.inputnode.brain_mask_file = mask
    preload.inputs.inputnode.te = 0.03
    preload.inputs.inputnode.field_strength = 3.0
    preload.inputs.inputnode.tr = 1.5
    preload.inputs.inputnode.baseline_frames = 5
    deconvolution.inputs.inputnode.aif_file = aif
    deconvolution.inputs.inputnode.measured_aif_file = aif
    deconvolution.inputs.inputnode.aif_detection_level = "synthetic"
    deconvolution.inputs.inputnode.baseline_frames = 5
    deconvolution.inputs.inputnode.brain_mask_file = mask
    deconvolution.inputs.inputnode.tr = 1.5
    deconvolution.inputs.inputnode.signal_file = signal
    deconvolution.inputs.inputnode.raw_concentration_file = concentration
    workflow.connect([
        (preload, deconvolution, [
            ("outputnode.corrected_concentration_file", "inputnode.concentration_file"),
            ("outputnode.k2_map_file", "inputnode.k2_map_input_file"),
        ]),
    ])

    graph = workflow.run(plugin="Linear")
    compute = next(node for node in graph.nodes if node.name == "compute_maps")
    outputs = compute.result.outputs
    assert Path(outputs.out_rel_cbv).is_file()
    if leakage_method == "bsw":
        assert isdefined(outputs.out_k2) and Path(outputs.out_k2).is_file()
    else:
        assert not isdefined(outputs.out_k2)


def test_leakage_qc_node_accepts_undefined_k2_output(tmp_path: Path) -> None:
    """The ``none`` policy may leave the connected K2 input undefined."""
    workflow = init_dsc_qc_wf(
        baseline_frames=5,
        subject_id="0001",
        session_id="0001",
        aif_source="auto",
    )
    node = workflow.get_node("dsc_leakage_qc")
    node.base_dir = str(tmp_path)
    node.inputs.preload_qc = {
        "k2_method": "none",
        "preload_correction_applied": False,
    }

    result = node.run()

    assert result.outputs.plot_file is None


def _fake_k2_result(shape: tuple[int, int, int]) -> K2CorrectionResult:
    """Stub K2 correction result with realistic QC metrics for the K2 classifier."""
    return K2CorrectionResult(
        concentration_corrected=np.zeros(shape + (20,), dtype=np.float32),
        k2_map=np.zeros(shape, dtype=np.float32),
        auc_tissue=np.zeros(shape, dtype=np.float32),
        correction_applied=True,
        qc_metrics={
            "k2_median": 0.01,
            "percent_leaky_voxels": 5.0,
            "n_voxels_corrected": 64,
        },
        method="bsw",
    )


def _advisory(cls: PreloadClass) -> PreloadResult:
    return PreloadResult(
        classification=cls,
        baseline_delta_r2=0.5,
        detection_method="baseline_delta_r2",
        confidence=0.8,
        details={"effective_thresholds": {"minimal": 0.5, "significant": 2.0}},
    )


@pytest.fixture(autouse=True)
def _chdir_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Confine ``_detect_and_correct_preload``'s ``Path.cwd()`` writes to tmp_path."""
    monkeypatch.chdir(tmp_path)


class TestPreloadProvenance:
    """Administered preload is either a supplied fraction or unknown."""

    def test_user_dose_is_recorded_without_classification(self, tmp_path: Path) -> None:
        conc, mask, signal = _write_fixtures(tmp_path)
        with mock.patch(
            "pmriprep.utils.dsc_preload.apply_k2_correction",
            return_value=_fake_k2_result((4, 4, 4)),
        ), mock.patch(
            "pmriprep.utils.dsc_preload.detect_preload",
            return_value=_advisory(PreloadClass.MINIMAL),
        ):
            _, _, qc = _detect_and_correct_preload(
                concentration_file=conc,
                brain_mask_file=mask,
                te=0.03,
                field_strength=3.0,
                tr=1.5,
                preload="0.3",
                signal_file=signal,
            )

        assert qc["preload_source"] == "user_specified"
        assert qc["preload_administered_dose_fraction"] == pytest.approx(0.3)
        assert "preload_classification" not in qc
        assert "preload_k2_classification" not in qc

    def test_refined_baseline_reaches_k2_and_provenance(
        self, tmp_path: Path,
    ) -> None:
        """K2 correction and QC use the baseline supplied by AIF refinement."""
        conc, mask, signal = _write_fixtures(tmp_path)
        with mock.patch(
            "pmriprep.utils.dsc_preload.apply_k2_correction",
            return_value=_fake_k2_result((4, 4, 4)),
        ) as apply_k2, mock.patch(
            "pmriprep.utils.dsc_preload.detect_preload",
            return_value=_advisory(PreloadClass.MINIMAL),
        ):
            _, _, qc = _detect_and_correct_preload(
                concentration_file=conc,
                brain_mask_file=mask,
                te=0.03,
                field_strength=3.0,
                tr=1.5,
                preload="unknown",
                signal_file=signal,
                baseline_frames=17,
            )

        assert apply_k2.call_args.kwargs["baseline_frames"] == 17
        np.testing.assert_allclose(
            apply_k2.call_args.kwargs["signal"], nib.load(signal).get_fdata()
        )
        assert qc["baseline_frames_used"] == 17

    def test_unknown_records_unknown_without_classification(self, tmp_path: Path) -> None:
        conc, mask, signal = _write_fixtures(tmp_path)
        with mock.patch(
            "pmriprep.utils.dsc_preload.apply_k2_correction",
            return_value=_fake_k2_result((4, 4, 4)),
        ), mock.patch(
            "pmriprep.utils.dsc_preload.detect_preload",
            return_value=_advisory(PreloadClass.MINIMAL),
        ):
            _, _, qc = _detect_and_correct_preload(
                concentration_file=conc,
                brain_mask_file=mask,
                te=0.03,
                field_strength=3.0,
                tr=1.5,
                preload="unknown",
                signal_file=signal,
            )

        assert qc["preload_source"] == "unknown"
        assert qc["preload_administered_dose_fraction"] is None
        assert "preload_classification" not in qc

    def test_none_returns_baseline_centered_concentration_without_k2(
        self, tmp_path: Path,
    ) -> None:
        """Disabling K2 correction preserves the common baseline convention."""
        conc, mask, signal = _write_fixtures(tmp_path)
        with mock.patch(
            "pmriprep.utils.dsc_preload.apply_k2_correction"
        ) as apply_k2, mock.patch(
            "pmriprep.utils.dsc_preload.detect_preload",
            return_value=_advisory(PreloadClass.MINIMAL),
        ):
            corrected, k2_map, qc = _detect_and_correct_preload(
                concentration_file=conc,
                brain_mask_file=mask,
                te=0.03,
                field_strength=3.0,
                tr=1.5,
                preload="unknown",
                signal_file=signal,
                leakage_method="none",
            )

        original = nib.load(conc).get_fdata()
        centered = nib.load(corrected).get_fdata()
        expected = original - np.mean(original[..., :5], axis=-1, keepdims=True)
        np.testing.assert_allclose(centered, expected, atol=1e-6)
        assert corrected != conc
        assert corrected.endswith("_baselinecentered.nii.gz")
        assert k2_map == ""
        assert qc["k2_method"] == "none"
        assert qc["preload_correction_applied"] is False
        assert qc["leakage_correction_disabled"] is True
        apply_k2.assert_not_called()
