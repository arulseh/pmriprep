"""Tests for FLAIR selection logic in discovery module."""

import json
from pathlib import Path
from typing import Optional

import pytest

from pmriprep.workflows.discovery import _select_best_flair


class _MockLayout:
    """Minimal stand-in: returns the per-file JSON sidecar dict."""

    def get_metadata(self, path: str) -> Optional[dict]:
        sidecar = Path(path).with_name(
            Path(path).name.replace(".nii.gz", ".json").replace(".nii", ".json")
        )
        if not sidecar.is_file():
            return None
        try:
            return json.loads(sidecar.read_text())
        except json.JSONDecodeError:
            return None


class TestSelectBestFlair:
    """Tests for _select_best_flair helper."""

    def test_single_candidate_returns_directly(self, tmp_path: Path) -> None:
        """Single FLAIR should be returned without any sidecar check."""
        flair = tmp_path / "sub-01_ses-01_run-1_FLAIR.nii.gz"
        flair.touch()
        assert _select_best_flair([flair], _MockLayout()) == flair

    def test_prefers_3d_from_sidecar(self, tmp_path: Path) -> None:
        """When JSON sidecar has MRAcquisitionType=3D, prefer that file."""
        flair_2d = tmp_path / "sub-01_ses-01_acq-2d_run-1_FLAIR.nii.gz"
        flair_3d = tmp_path / "sub-01_ses-01_acq-3d_run-1_FLAIR.nii.gz"
        flair_2d.touch()
        flair_3d.touch()

        (tmp_path / "sub-01_ses-01_acq-2d_run-1_FLAIR.json").write_text(
            json.dumps({"MRAcquisitionType": "2D"})
        )
        (tmp_path / "sub-01_ses-01_acq-3d_run-1_FLAIR.json").write_text(
            json.dumps({"MRAcquisitionType": "3D"})
        )

        result = _select_best_flair([flair_2d, flair_3d], _MockLayout())
        assert result == flair_3d

    def test_prefers_acq_3d_in_filename_when_no_sidecar(
        self, tmp_path: Path
    ) -> None:
        """When no sidecars exist, fall back to acq-3d in filename."""
        flair_2d = tmp_path / "sub-01_ses-01_acq-2d_run-1_FLAIR.nii.gz"
        flair_3d = tmp_path / "sub-01_ses-01_acq-3d_run-1_FLAIR.nii.gz"
        flair_2d.touch()
        flair_3d.touch()

        result = _select_best_flair([flair_2d, flair_3d], _MockLayout())
        assert result == flair_3d

    def test_fallback_returns_first_candidate(self, tmp_path: Path) -> None:
        """When no 3D indicator exists, return first candidate."""
        flair_a = tmp_path / "sub-01_ses-01_run-1_FLAIR.nii.gz"
        flair_b = tmp_path / "sub-01_ses-01_run-2_FLAIR.nii.gz"
        flair_a.touch()
        flair_b.touch()

        result = _select_best_flair([flair_a, flair_b], _MockLayout())
        assert result == flair_a

    def test_sidecar_3d_takes_precedence_over_filename(
        self, tmp_path: Path
    ) -> None:
        """Sidecar MRAcquisitionType=3D should win over acq-3d filename."""
        flair_fake3d = tmp_path / "sub-01_ses-01_acq-3d_run-1_FLAIR.nii.gz"
        flair_fake3d.touch()
        (tmp_path / "sub-01_ses-01_acq-3d_run-1_FLAIR.json").write_text(
            json.dumps({"MRAcquisitionType": "2D"})
        )

        flair_real3d = tmp_path / "sub-01_ses-01_run-2_FLAIR.nii.gz"
        flair_real3d.touch()
        (tmp_path / "sub-01_ses-01_run-2_FLAIR.json").write_text(
            json.dumps({"MRAcquisitionType": "3D"})
        )

        result = _select_best_flair([flair_fake3d, flair_real3d], _MockLayout())
        assert result == flair_real3d

    def test_handles_malformed_sidecar(self, tmp_path: Path) -> None:
        """Malformed JSON sidecar should be skipped gracefully."""
        flair_a = tmp_path / "sub-01_ses-01_run-1_FLAIR.nii.gz"
        flair_b = tmp_path / "sub-01_ses-01_acq-3d_run-2_FLAIR.nii.gz"
        flair_a.touch()
        flair_b.touch()

        (tmp_path / "sub-01_ses-01_run-1_FLAIR.json").write_text(
            "not valid json{{{"
        )

        result = _select_best_flair([flair_a, flair_b], _MockLayout())
        assert result == flair_b  # falls through to acq-3d filename match

    def test_nii_extension_without_gz(self, tmp_path: Path) -> None:
        """Works with .nii files (not just .nii.gz)."""
        flair = tmp_path / "sub-01_ses-01_run-1_FLAIR.nii"
        flair.touch()

        (tmp_path / "sub-01_ses-01_run-1_FLAIR.json").write_text(
            json.dumps({"MRAcquisitionType": "3D"})
        )

        assert _select_best_flair([flair], _MockLayout()) == flair
