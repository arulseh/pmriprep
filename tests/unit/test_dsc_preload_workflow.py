"""Workflow-level tests for preload classification precedence (v0.26.0 W3).

``_detect_and_correct_preload`` in ``pmriprep/workflows/dsc/base.py`` resolves
the effective preload class from three sources with strict precedence:

    user override > K2-primary > ΔR2* advisory

Prior to v0.26.0 no integration test exercised this path as a whole — unit
tests covered ``classify_preload`` / ``classify_preload_adequacy_from_k2`` /
``preload_class_from_dose`` in isolation only. These tests mock the four
``pmriprep.utils.dsc_preload`` entrypoints the function imports locally
(patch at the source module so the local import resolves to the mock) and
write small synthetic NIfTI fixtures to ``tmp_path`` so the real nibabel
I/O runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import nibabel as nib  # type: ignore[import-untyped]
import numpy as np
import pytest

from pmriprep.utils.dsc_preload import K2CorrectionResult, PreloadClass, PreloadResult
from pmriprep.workflows.dsc.base import _detect_and_correct_preload


def _write_fixtures(tmp_path: Path) -> tuple[str, str, str]:
    """Write minimal 4D concentration, 3D brain mask, and AIF text fixtures."""
    shape = (4, 4, 4, 20)
    rng = np.random.default_rng(0)
    conc = rng.normal(0.0, 0.01, size=shape).astype(np.float32)
    conc_path = tmp_path / "conc.nii.gz"
    nib.save(nib.Nifti1Image(conc, np.eye(4)), str(conc_path))

    mask = np.ones(shape[:3], dtype=np.float32)
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))

    aif = np.linspace(0.0, 1.0, shape[3])
    aif_path = tmp_path / "aif.txt"
    np.savetxt(aif_path, aif)

    return str(conc_path), str(mask_path), str(aif_path)


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
    """v0.26.0 W3: user override > K2-primary > ΔR2* advisory."""

    def test_user_override_wins_over_k2(self, tmp_path: Path) -> None:
        """preload='0.3' (user half-dose) overrides K2 classification."""
        conc, mask, aif = _write_fixtures(tmp_path)
        with mock.patch(
            "pmriprep.utils.dsc_preload.apply_k2_correction",
            return_value=_fake_k2_result((4, 4, 4)),
        ), mock.patch(
            "pmriprep.utils.dsc_preload.detect_preload",
            return_value=_advisory(PreloadClass.MINIMAL),
        ), mock.patch(
            "pmriprep.utils.dsc_preload.classify_preload_adequacy_from_k2",
            return_value=(PreloadClass.SIGNIFICANT, "high", "stub"),
        ), mock.patch(
            "pmriprep.utils.dsc_preload.preload_class_from_dose",
            return_value=PreloadClass.HALF_DOSE,
        ):
            _, _, qc = _detect_and_correct_preload(
                concentration_file=conc,
                aif_file=aif,
                brain_mask_file=mask,
                te=0.03,
                field_strength=3.0,
                tr=1.5,
                preload="0.3",
            )

        assert qc["preload_classification_source"] == "user_specified"
        assert qc["preload_classification"] == PreloadClass.HALF_DOSE.value
        assert qc["preload_user_specified"] is True
        # K2 class still recorded for reference, but not effective.
        assert qc["preload_k2_classification"] == PreloadClass.SIGNIFICANT.value
        assert "preload_k2_confidence" in qc

    def test_k2_primary_when_user_auto(self, tmp_path: Path) -> None:
        """preload='auto' + concrete K2 class → effective = K2 class."""
        conc, mask, aif = _write_fixtures(tmp_path)
        with mock.patch(
            "pmriprep.utils.dsc_preload.apply_k2_correction",
            return_value=_fake_k2_result((4, 4, 4)),
        ), mock.patch(
            "pmriprep.utils.dsc_preload.detect_preload",
            return_value=_advisory(PreloadClass.MINIMAL),
        ), mock.patch(
            "pmriprep.utils.dsc_preload.classify_preload_adequacy_from_k2",
            return_value=(PreloadClass.HALF_DOSE, "moderate", "stub"),
        ):
            _, _, qc = _detect_and_correct_preload(
                concentration_file=conc,
                aif_file=aif,
                brain_mask_file=mask,
                te=0.03,
                field_strength=3.0,
                tr=1.5,
                preload="auto",
            )

        assert qc["preload_classification_source"] == "k2_primary"
        assert qc["preload_classification"] == PreloadClass.HALF_DOSE.value
        assert qc["preload_advisory_classification"] == PreloadClass.MINIMAL.value
        assert qc["preload_user_specified"] is False

    def test_advisory_disagreement_logged(self, tmp_path: Path) -> None:
        """Advisory ΔR2* disagrees with K2 (both concrete) → K2 drives, disagreement flagged."""
        conc, mask, aif = _write_fixtures(tmp_path)
        with mock.patch(
            "pmriprep.utils.dsc_preload.apply_k2_correction",
            return_value=_fake_k2_result((4, 4, 4)),
        ), mock.patch(
            "pmriprep.utils.dsc_preload.detect_preload",
            return_value=_advisory(PreloadClass.NONE),
        ), mock.patch(
            "pmriprep.utils.dsc_preload.classify_preload_adequacy_from_k2",
            return_value=(PreloadClass.HALF_DOSE, "moderate", "stub"),
        ):
            _, _, qc = _detect_and_correct_preload(
                concentration_file=conc,
                aif_file=aif,
                brain_mask_file=mask,
                te=0.03,
                field_strength=3.0,
                tr=1.5,
                preload="auto",
            )

        assert qc["preload_k2_vs_advisory_disagreement"] is True
        assert qc["preload_classification"] == PreloadClass.HALF_DOSE.value
        assert qc["preload_advisory_classification"] == PreloadClass.NONE.value
        assert qc["preload_classification_source"] == "k2_primary"
        assert qc["preload_user_specified"] is False

    def test_k2_unavailable_skips_constraint(self, tmp_path: Path) -> None:
        """preload='auto' + K2 returns UNKNOWN → source='k2_unavailable'."""
        conc, mask, aif = _write_fixtures(tmp_path)
        with mock.patch(
            "pmriprep.utils.dsc_preload.apply_k2_correction",
            return_value=_fake_k2_result((4, 4, 4)),
        ), mock.patch(
            "pmriprep.utils.dsc_preload.detect_preload",
            return_value=_advisory(PreloadClass.MINIMAL),
        ), mock.patch(
            "pmriprep.utils.dsc_preload.classify_preload_adequacy_from_k2",
            return_value=(PreloadClass.UNKNOWN, 0.0, "stub"),
        ):
            _, _, qc = _detect_and_correct_preload(
                concentration_file=conc,
                aif_file=aif,
                brain_mask_file=mask,
                te=0.03,
                field_strength=3.0,
                tr=1.5,
                preload="auto",
            )

        assert qc["preload_classification_source"] == "k2_unavailable"
        assert qc["preload_classification"] == PreloadClass.UNKNOWN.value
        assert qc["preload_user_specified"] is False
