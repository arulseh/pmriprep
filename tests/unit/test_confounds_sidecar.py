"""Tests for confounds timeseries sidecar provenance."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


class TestWriteConfoundsSidecar:
    def test_no_provenance_args(self, tmp_path: Path) -> None:
        from pmriprep.utils.outputs import write_confounds_sidecar

        out = tmp_path / "confounds.json"
        write_confounds_sidecar(out)
        data = json.loads(out.read_text())

        assert "trans_x" in data
        assert "framewise_displacement" in data
        assert "GeneratedBy" in data
        assert "Sources" not in data

    def test_with_source(self, tmp_path: Path) -> None:
        from pmriprep.utils.outputs import write_confounds_sidecar

        bids_dir = tmp_path / "rawdata"
        bids_dir.mkdir()
        source = str(bids_dir / "sub-01" / "ses-01" / "perf" / "sub-01_ses-01_dsc.nii.gz")

        out = tmp_path / "confounds.json"
        write_confounds_sidecar(out, source_path=source, bids_dir=str(bids_dir))
        data = json.loads(out.read_text())

        assert "GeneratedBy" in data
        assert len(data["GeneratedBy"]) == 1
        assert data["GeneratedBy"][0]["Name"] == "pmriprep"
        assert len(data["Sources"]) == 1
        assert data["Sources"][0].startswith("bids:raw:")
        assert "trans_x" in data
        assert "rot_z" in data

    def test_column_keys_unchanged(self, tmp_path: Path) -> None:
        from pmriprep.utils.outputs import write_confounds_sidecar

        out = tmp_path / "confounds.json"
        write_confounds_sidecar(out)
        data = json.loads(out.read_text())

        expected_columns = {
            "trans_x", "trans_y", "trans_z",
            "rot_x", "rot_y", "rot_z",
            "framewise_displacement",
        }
        assert expected_columns <= set(data.keys())
        assert "rmsd" not in data


class TestParToConfounds:
    def test_rejects_wrong_column_count(self, tmp_path: Path) -> None:
        from pmriprep.utils.outputs import par_to_confounds_tsv

        par = tmp_path / "motion.par"
        np.savetxt(par, np.zeros((3, 5)))
        with pytest.raises(ValueError, match="exactly 6 columns"):
            par_to_confounds_tsv(par, tmp_path / "confounds.tsv")

    def test_rejects_nonfinite_values(self, tmp_path: Path) -> None:
        from pmriprep.utils.outputs import par_to_confounds_tsv

        par = tmp_path / "motion.par"
        values = np.zeros((3, 6))
        values[1, 2] = np.nan
        np.savetxt(par, values)
        with pytest.raises(ValueError, match="non-finite"):
            par_to_confounds_tsv(par, tmp_path / "confounds.tsv")
