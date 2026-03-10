"""Tests for FLAIR selection logic in discovery module."""

import json
from pathlib import Path

import pytest

from pmriprep.workflows.discovery import _select_best_flair


class TestSelectBestFlair:
    """Tests for _select_best_flair helper."""

    def test_single_candidate_returns_directly(self, tmp_path: Path) -> None:
        """Single FLAIR should be returned without any sidecar check."""
        flair = tmp_path / "sub-01_ses-01_run-1_FLAIR.nii.gz"
        flair.touch()
        assert _select_best_flair([flair]) == flair

    def test_prefers_3d_from_sidecar(self, tmp_path: Path) -> None:
        """When JSON sidecar has MRAcquisitionType=3D, prefer that file."""
        flair_2d = tmp_path / "sub-01_ses-01_acq-2d_run-1_FLAIR.nii.gz"
        flair_3d = tmp_path / "sub-01_ses-01_acq-3d_run-1_FLAIR.nii.gz"
        flair_2d.touch()
        flair_3d.touch()

        # Write 2D sidecar
        sidecar_2d = tmp_path / "sub-01_ses-01_acq-2d_run-1_FLAIR.json"
        sidecar_2d.write_text(json.dumps({"MRAcquisitionType": "2D"}))

        # Write 3D sidecar
        sidecar_3d = tmp_path / "sub-01_ses-01_acq-3d_run-1_FLAIR.json"
        sidecar_3d.write_text(json.dumps({"MRAcquisitionType": "3D"}))

        # 2D listed first — should still pick 3D
        result = _select_best_flair([flair_2d, flair_3d])
        assert result == flair_3d

    def test_prefers_acq_3d_in_filename_when_no_sidecar(
        self, tmp_path: Path
    ) -> None:
        """When no sidecars exist, fall back to acq-3d in filename."""
        flair_2d = tmp_path / "sub-01_ses-01_acq-2d_run-1_FLAIR.nii.gz"
        flair_3d = tmp_path / "sub-01_ses-01_acq-3d_run-1_FLAIR.nii.gz"
        flair_2d.touch()
        flair_3d.touch()

        result = _select_best_flair([flair_2d, flair_3d])
        assert result == flair_3d

    def test_fallback_returns_first_candidate(self, tmp_path: Path) -> None:
        """When no 3D indicator exists, return first candidate."""
        flair_a = tmp_path / "sub-01_ses-01_run-1_FLAIR.nii.gz"
        flair_b = tmp_path / "sub-01_ses-01_run-2_FLAIR.nii.gz"
        flair_a.touch()
        flair_b.touch()

        result = _select_best_flair([flair_a, flair_b])
        assert result == flair_a

    def test_sidecar_3d_takes_precedence_over_filename(
        self, tmp_path: Path
    ) -> None:
        """Sidecar MRAcquisitionType=3D should win over acq-3d filename."""
        # File named acq-3d but sidecar says 2D
        flair_fake3d = tmp_path / "sub-01_ses-01_acq-3d_run-1_FLAIR.nii.gz"
        flair_fake3d.touch()
        sidecar_fake = tmp_path / "sub-01_ses-01_acq-3d_run-1_FLAIR.json"
        sidecar_fake.write_text(json.dumps({"MRAcquisitionType": "2D"}))

        # File with no acq label but sidecar says 3D
        flair_real3d = tmp_path / "sub-01_ses-01_run-2_FLAIR.nii.gz"
        flair_real3d.touch()
        sidecar_real = tmp_path / "sub-01_ses-01_run-2_FLAIR.json"
        sidecar_real.write_text(json.dumps({"MRAcquisitionType": "3D"}))

        result = _select_best_flair([flair_fake3d, flair_real3d])
        assert result == flair_real3d

    def test_handles_malformed_sidecar(self, tmp_path: Path) -> None:
        """Malformed JSON sidecar should be skipped gracefully."""
        flair_a = tmp_path / "sub-01_ses-01_run-1_FLAIR.nii.gz"
        flair_b = tmp_path / "sub-01_ses-01_acq-3d_run-2_FLAIR.nii.gz"
        flair_a.touch()
        flair_b.touch()

        # Write invalid JSON for first file
        sidecar_a = tmp_path / "sub-01_ses-01_run-1_FLAIR.json"
        sidecar_a.write_text("not valid json{{{")

        result = _select_best_flair([flair_a, flair_b])
        assert result == flair_b  # falls through to acq-3d filename match

    def test_nii_extension_without_gz(self, tmp_path: Path) -> None:
        """Works with .nii files (not just .nii.gz)."""
        flair = tmp_path / "sub-01_ses-01_run-1_FLAIR.nii"
        flair.touch()

        sidecar = tmp_path / "sub-01_ses-01_run-1_FLAIR.json"
        sidecar.write_text(json.dumps({"MRAcquisitionType": "3D"}))

        assert _select_best_flair([flair]) == flair
