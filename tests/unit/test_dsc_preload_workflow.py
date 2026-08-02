"""Workflow-level tests for preload provenance and leakage correction."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import nibabel as nib  # type: ignore[import-untyped]
import numpy as np
import pytest

from pmriprep.utils.dsc_preload import K2CorrectionResult, PreloadClass, PreloadResult
from pmriprep.workflows.dsc.base import (
    _detect_and_correct_preload,
    init_dsc_deconvolution_wf,
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
                preload="auto",
                signal_file=signal,
                baseline_frames=17,
            )

        assert apply_k2.call_args.kwargs["baseline_frames"] == 17
        np.testing.assert_allclose(
            apply_k2.call_args.kwargs["signal"], nib.load(signal).get_fdata()
        )
        assert qc["baseline_frames_used"] == 17

    def test_auto_records_unknown_without_classification(self, tmp_path: Path) -> None:
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
                preload="auto",
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
                preload="auto",
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
