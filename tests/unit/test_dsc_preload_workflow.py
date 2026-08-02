"""Workflow-level tests for preload classification precedence.

``_detect_and_correct_preload`` in ``pmriprep/workflows/dsc/base.py`` resolves
the effective preload class from three sources with strict precedence:

    user override > K2-primary > ΔR2* advisory

These tests exercise that path as a whole, complementing the unit
tests that cover ``classify_preload`` / ``classify_preload_adequacy_from_k2`` /
``preload_class_from_dose`` in isolation. These tests mock the four
``pmriprep.utils.dsc_preload`` entrypoints the function imports locally
(patch at the source module so the local import resolves to the mock) and
write small synthetic NIfTI fixtures to ``tmp_path`` so the real nibabel
I/O runs.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import nibabel as nib  # type: ignore[import-untyped]
import numpy as np
import pytest

from pmriprep.utils.dsc_preload import (
    K2Adequacy,
    K2CorrectionResult,
    PreloadClass,
    PreloadResult,
)
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


class TestPreloadPrecedence:
    """user override > K2-primary > ΔR2* advisory."""

    def test_user_override_wins_over_k2(self, tmp_path: Path) -> None:
        """preload='0.3' (user half-dose) overrides K2 classification."""
        conc, mask, signal = _write_fixtures(tmp_path)
        with mock.patch(
            "pmriprep.utils.dsc_preload.apply_k2_correction",
            return_value=_fake_k2_result((4, 4, 4)),
        ), mock.patch(
            "pmriprep.utils.dsc_preload.detect_preload",
            return_value=_advisory(PreloadClass.MINIMAL),
        ), mock.patch(
            "pmriprep.utils.dsc_preload.classify_preload_adequacy_from_k2",
            return_value=(K2Adequacy.PARTIAL, "high", "stub"),
        ), mock.patch(
            "pmriprep.utils.dsc_preload.preload_class_from_dose",
            return_value=PreloadClass.HALF_DOSE,
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

        assert qc["preload_classification_source"] == "user_specified"
        assert qc["preload_classification"] == PreloadClass.HALF_DOSE.value
        assert qc["preload_user_specified"] is True
        # K2 adequacy still recorded for reference, but not effective.
        assert qc["preload_k2_classification"] == K2Adequacy.PARTIAL.value
        assert "preload_k2_confidence" in qc

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
        ), mock.patch(
            "pmriprep.utils.dsc_preload.classify_preload_adequacy_from_k2",
            return_value=(K2Adequacy.ADEQUATE, "moderate", "stub"),
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

    def test_k2_primary_when_user_auto(self, tmp_path: Path) -> None:
        """preload='auto' + concrete K2 adequacy → effective = adequacy."""
        conc, mask, signal = _write_fixtures(tmp_path)
        with mock.patch(
            "pmriprep.utils.dsc_preload.apply_k2_correction",
            return_value=_fake_k2_result((4, 4, 4)),
        ), mock.patch(
            "pmriprep.utils.dsc_preload.detect_preload",
            return_value=_advisory(PreloadClass.MINIMAL),
        ), mock.patch(
            "pmriprep.utils.dsc_preload.classify_preload_adequacy_from_k2",
            return_value=(K2Adequacy.ADEQUATE, "moderate", "stub"),
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

        assert qc["preload_classification_source"] == "k2_primary"
        assert qc["preload_classification"] == K2Adequacy.ADEQUATE.value
        assert qc["preload_advisory_classification"] == PreloadClass.MINIMAL.value
        assert qc["preload_user_specified"] is False

    def test_k2_adequacy_drives_auto_classification(self, tmp_path: Path) -> None:
        """Auto path: K2 adequacy drives the effective classification; advisory is separate."""
        conc, mask, signal = _write_fixtures(tmp_path)
        with mock.patch(
            "pmriprep.utils.dsc_preload.apply_k2_correction",
            return_value=_fake_k2_result((4, 4, 4)),
        ), mock.patch(
            "pmriprep.utils.dsc_preload.detect_preload",
            return_value=_advisory(PreloadClass.NONE),
        ), mock.patch(
            "pmriprep.utils.dsc_preload.classify_preload_adequacy_from_k2",
            return_value=(K2Adequacy.ADEQUATE, "moderate", "stub"),
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

        assert qc["preload_classification"] == K2Adequacy.ADEQUATE.value
        assert qc["preload_advisory_classification"] == PreloadClass.NONE.value
        assert qc["preload_classification_source"] == "k2_primary"
        assert qc["preload_user_specified"] is False
        # Decouple removed the ΔR2*-vs-K2 disagreement key.
        assert "preload_k2_vs_advisory_disagreement" not in qc

    def test_k2_unavailable_skips_constraint(self, tmp_path: Path) -> None:
        """preload='auto' + K2 returns UNKNOWN → source='k2_unavailable'."""
        conc, mask, signal = _write_fixtures(tmp_path)
        with mock.patch(
            "pmriprep.utils.dsc_preload.apply_k2_correction",
            return_value=_fake_k2_result((4, 4, 4)),
        ), mock.patch(
            "pmriprep.utils.dsc_preload.detect_preload",
            return_value=_advisory(PreloadClass.MINIMAL),
        ), mock.patch(
            "pmriprep.utils.dsc_preload.classify_preload_adequacy_from_k2",
            return_value=(K2Adequacy.UNKNOWN, 0.0, "stub"),
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

        assert qc["preload_classification_source"] == "k2_unavailable"
        assert qc["preload_classification"] == K2Adequacy.UNKNOWN.value
        assert qc["preload_user_specified"] is False

    def test_signed_k2_uses_uncalibrated_source_label(self, tmp_path: Path) -> None:
        """Populated signed K2 is distinct from a missing reference."""
        conc, mask, signal = _write_fixtures(tmp_path)
        signed_result = _fake_k2_result((4, 4, 4))
        signed_result.qc_metrics = {
            "k2_signed": True,
            "k2_reference_available": True,
            "k2_median": -0.002,
            "percent_negative_k2": 60.0,
        }
        with mock.patch(
            "pmriprep.utils.dsc_preload.apply_k2_correction",
            return_value=signed_result,
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

        assert qc["preload_classification_source"] == "signed_k2_uncalibrated"
        assert qc["preload_k2_classification"] == K2Adequacy.UNKNOWN.value
        assert "not calibrated" in qc["preload_k2_reason"]
