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
from typing import Any
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
    _augment_qc_positive_tail,
    _detect_and_correct_preload,
)


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
    """user override > K2-primary > ΔR2* advisory."""

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
            return_value=(K2Adequacy.PARTIAL, "high", "stub"),
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
        # K2 adequacy still recorded for reference, but not effective.
        assert qc["preload_k2_classification"] == K2Adequacy.PARTIAL.value
        assert "preload_k2_confidence" in qc

    def test_k2_primary_when_user_auto(self, tmp_path: Path) -> None:
        """preload='auto' + concrete K2 adequacy → effective = adequacy."""
        conc, mask, aif = _write_fixtures(tmp_path)
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
                aif_file=aif,
                brain_mask_file=mask,
                te=0.03,
                field_strength=3.0,
                tr=1.5,
                preload="auto",
            )

        assert qc["preload_classification_source"] == "k2_primary"
        assert qc["preload_classification"] == K2Adequacy.ADEQUATE.value
        assert qc["preload_advisory_classification"] == PreloadClass.MINIMAL.value
        assert qc["preload_user_specified"] is False

    def test_k2_adequacy_drives_auto_classification(self, tmp_path: Path) -> None:
        """Auto path: K2 adequacy drives the effective classification; advisory is separate."""
        conc, mask, aif = _write_fixtures(tmp_path)
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
                aif_file=aif,
                brain_mask_file=mask,
                te=0.03,
                field_strength=3.0,
                tr=1.5,
                preload="auto",
            )

        assert qc["preload_classification"] == K2Adequacy.ADEQUATE.value
        assert qc["preload_advisory_classification"] == PreloadClass.NONE.value
        assert qc["preload_classification_source"] == "k2_primary"
        assert qc["preload_user_specified"] is False
        # Decouple removed the ΔR2*-vs-K2 disagreement key.
        assert "preload_k2_vs_advisory_disagreement" not in qc

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
            return_value=(K2Adequacy.UNKNOWN, 0.0, "stub"),
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
        assert qc["preload_classification"] == K2Adequacy.UNKNOWN.value
        assert qc["preload_user_specified"] is False


def _write_tail_fixtures(tmp_path: Path, tail_ratio: float, k2_value: float, n_tumor: int = 12):
    """Concentration + K2 map + lesion mask for _augment_qc_positive_tail.

    Each tumor voxel: linear rise to peak (10) at t=5, then plateau at
    ``tail_ratio * peak`` — so late/peak == ``tail_ratio``. K2 map is uniform
    ``k2_value`` (0.0 = BSW-blind, >0 = corrected).
    """
    shape = (4, 4, 4, 20)
    conc = np.zeros(shape, dtype=np.float32)
    curve = np.zeros(20, dtype=np.float32)
    curve[:6] = np.linspace(0.0, 10.0, 6)
    curve[6:] = tail_ratio * 10.0
    tumor = np.zeros(shape[:3], dtype=np.float32)
    coords = [tuple(c) for c in np.argwhere(np.ones(shape[:3]))[:n_tumor]]
    for (x, y, z) in coords:
        conc[x, y, z, :] = curve
        tumor[x, y, z] = 1.0
    conc_path = tmp_path / "conc.nii.gz"
    nib.save(nib.Nifti1Image(conc, np.eye(4)), str(conc_path))
    k2_path = tmp_path / "k2map.nii.gz"
    nib.save(nib.Nifti1Image(np.full(shape[:3], k2_value, np.float32), np.eye(4)), str(k2_path))
    lesion_path = tmp_path / "lesion.nii.gz"
    nib.save(nib.Nifti1Image(tumor, np.eye(4)), str(lesion_path))
    return str(conc_path), str(k2_path), str(lesion_path)


class TestPositiveTailDetector:
    """cbv_positive_tail_fraction: fires on positive tail AND k2==0 over tumor."""

    def _run(self, tmp_path, tail_ratio, k2_value, with_mask=True):
        conc, k2, lesion = _write_tail_fixtures(tmp_path, tail_ratio, k2_value)
        qc = _augment_qc_positive_tail(
            qc_dict={"k2_median": 0.002},
            concentration_file=conc,
            k2_map_file=k2,
            lesion_mask_dsc=(lesion if with_mask else None),
        )
        return qc

    def test_preexisting_qc_preserved(self, tmp_path: Path) -> None:
        qc = self._run(tmp_path, tail_ratio=0.5, k2_value=0.0)
        assert qc["k2_median"] == 0.002  # augment, don't replace

    def test_positive_tail_with_zero_k2_fires(self, tmp_path: Path) -> None:
        qc = self._run(tmp_path, tail_ratio=0.5, k2_value=0.0)
        assert qc["cbv_positive_tail_fraction"] == pytest.approx(1.0)

    def test_positive_tail_but_k2_corrected_is_silent(self, tmp_path: Path) -> None:
        # Non-zero K2 = BSW applied a correction here → not the blind regime.
        qc = self._run(tmp_path, tail_ratio=0.5, k2_value=0.02)
        assert qc["cbv_positive_tail_fraction"] == pytest.approx(0.0)

    def test_mild_tail_below_tau_is_silent(self, tmp_path: Path) -> None:
        qc = self._run(tmp_path, tail_ratio=0.04, k2_value=0.0)  # < TAIL_RATIO_TAU 0.08
        assert qc["cbv_positive_tail_fraction"] == pytest.approx(0.0)

    def test_no_lesion_mask_defaults_zero(self, tmp_path: Path) -> None:
        qc = self._run(tmp_path, tail_ratio=0.5, k2_value=0.0, with_mask=False)
        assert qc["cbv_positive_tail_fraction"] == pytest.approx(0.0)
