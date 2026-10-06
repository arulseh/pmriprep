"""Tests for ``_classify_b1_data`` after the AFI 3D-split removal."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pytest

from pmriprep.workflows.discovery import _classify_b1_data


class _MockLayout:
    """Returns the per-file JSON sidecar dict (no BIDS inheritance)."""

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


def _write_nii(path: Path, shape: tuple) -> None:
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4)), path)


def test_afi_4d_detection(tmp_path: Path) -> None:
    """4D NIfTI with shape (...,2) is detected as ``afi_mode='4d'``."""
    afi = tmp_path / "sub-01_acq-1_TB1AFI.nii.gz"
    _write_nii(afi, (4, 4, 4, 2))
    (tmp_path / "sub-01_acq-1_TB1AFI.json").write_text(
        json.dumps({
            "RepetitionTimeExcitation": [0.020, 0.100],
            "FlipAngle": 60,
        })
    )
    result = _classify_b1_data([afi], _MockLayout())
    assert result["afi_mode"] == "4d"
    assert result["afi_4d_file"] == afi
    assert result["afi_tr1_ms"] == pytest.approx(20.0)
    assert result["afi_tr2_ms"] == pytest.approx(100.0)
    assert result["afi_flip_angle"] == pytest.approx(60.0)
    assert result["afi_4d_candidates"] == [afi]


def test_repeated_4d_afi_candidates_are_retained(tmp_path: Path) -> None:
    """Discovery preserves competing 4D AFI inputs for preflight rejection."""
    candidates = [
        tmp_path / "sub-01_run-1_TB1AFI.nii.gz",
        tmp_path / "sub-01_run-2_TB1AFI.nii.gz",
    ]
    for path in candidates:
        _write_nii(path, (4, 4, 4, 2))
        path.with_name(path.name.replace(".nii.gz", ".json")).write_text(
            json.dumps({
                "RepetitionTimeExcitation": [0.020, 0.100],
                "FlipAngle": 60,
            })
        )

    result = _classify_b1_data(candidates, _MockLayout())

    assert result["afi_4d_file"] == candidates[0]
    assert result["afi_4d_candidates"] == candidates


def test_two_separate_3d_files_no_longer_paired(tmp_path: Path) -> None:
    """Two 3D AFI files must NOT be paired into 3D mode (returns afi_mode=None)."""
    a = tmp_path / "sub-01_acq-tr1_AFI.nii.gz"
    b = tmp_path / "sub-01_acq-tr2_AFI.nii.gz"
    _write_nii(a, (4, 4, 4))
    _write_nii(b, (4, 4, 4))
    (tmp_path / "sub-01_acq-tr1_AFI.json").write_text(
        json.dumps({"RepetitionTime": 0.020, "FlipAngle": 60})
    )
    (tmp_path / "sub-01_acq-tr2_AFI.json").write_text(
        json.dumps({"RepetitionTime": 0.100, "FlipAngle": 60})
    )

    result = _classify_b1_data([a, b], _MockLayout())
    assert result["afi_mode"] is None
    assert result["afi_4d_file"] is None
    assert result["b1_method_detected"] is None


def test_tfl_b1map_famp_detected(tmp_path: Path) -> None:
    """Siemens tfl_b1map acq-famp file is detected as precomputed."""
    famp = tmp_path / "sub-01_acq-famp_TB1TFL.nii.gz"
    anat = tmp_path / "sub-01_acq-anat_TB1TFL.nii.gz"
    _write_nii(famp, (4, 4, 4))
    _write_nii(anat, (4, 4, 4))
    (tmp_path / "sub-01_acq-famp_TB1TFL.json").write_text(
        json.dumps({"FlipAngle": 80})
    )
    (tmp_path / "sub-01_acq-anat_TB1TFL.json").write_text(
        json.dumps({"FlipAngle": 8})
    )
    result = _classify_b1_data([anat, famp], _MockLayout())
    assert result["afi_mode"] == "precomputed"
    assert result["b1_method_detected"] == "precomputed"
    assert result["b1_map_encoding"] == "actual_flip_deci_deg"
    assert result["b1_precomputed_file"] == famp
    assert result["b1_nominal_prep_fa"] == pytest.approx(80.0)
    assert result["b1_precomputed_candidates"] == [famp]


def test_competing_precomputed_maps_are_retained(tmp_path: Path) -> None:
    """All eligible precomputed maps reach the ambiguity gate."""
    famp = tmp_path / "sub-01_acq-famp_TB1TFL.nii.gz"
    tb1map = tmp_path / "sub-01_TB1map.nii.gz"
    for path in (famp, tb1map):
        _write_nii(path, (4, 4, 4))
    (tmp_path / "sub-01_acq-famp_TB1TFL.json").write_text(
        json.dumps({"FlipAngle": 80})
    )
    (tmp_path / "sub-01_TB1map.json").write_text(
        json.dumps({"Units": "p.u."})
    )

    result = _classify_b1_data([famp, tb1map], _MockLayout())

    assert result["b1_precomputed_file"] == famp
    assert result["b1_precomputed_candidates"] == [famp, tb1map]


def test_tfl_b1map_low_flip_angle_raises(tmp_path: Path) -> None:
    """A famp file with FlipAngle<20° is treated as the anat companion."""
    famp = tmp_path / "sub-01_acq-famp_TB1TFL.nii.gz"
    _write_nii(famp, (4, 4, 4))
    (tmp_path / "sub-01_acq-famp_TB1TFL.json").write_text(
        json.dumps({"FlipAngle": 10})
    )
    with pytest.raises(ValueError, match="Use the acq-famp file"):
        _classify_b1_data([famp], _MockLayout())


def test_tfl_b1map_imagetype_fallback_detects_famp(tmp_path: Path) -> None:
    """A TB1TFL file without 'famp' in name is detected via ImageType."""
    f = tmp_path / "sub-01_run-1_TB1TFL.nii.gz"
    _write_nii(f, (4, 4, 4))
    (tmp_path / "sub-01_run-1_TB1TFL.json").write_text(
        json.dumps({
            "ImageType": ["DERIVED", "PRIMARY", "FLIP ANGLE MAP"],
            "FlipAngle": 80,
        })
    )
    result = _classify_b1_data([f], _MockLayout())
    assert result["afi_mode"] == "precomputed"
    assert result["b1_method_detected"] == "precomputed"
    assert result["b1_map_encoding"] == "actual_flip_deci_deg"


def test_tb1map_percent_units_detected(tmp_path: Path) -> None:
    """A vendor-neutral TB1map with Units='p.u.' → percent encoding, no FA raise."""
    m = tmp_path / "sub-01_TB1map.nii.gz"
    _write_nii(m, (4, 4, 4))
    (tmp_path / "sub-01_TB1map.json").write_text(
        json.dumps({"Units": "p.u.", "FlipAngle": 8})
    )
    result = _classify_b1_data([m], _MockLayout())
    assert result["afi_mode"] == "precomputed"
    assert result["b1_method_detected"] == "precomputed"
    assert result["b1_map_encoding"] == "percent"
    assert result["b1_precomputed_file"] == m
    # FlipAngle < 20 must NOT raise for a TB1map (only the TB1TFL famp guard does).
    assert result["b1_nominal_prep_fa"] == pytest.approx(8.0)


def test_tb1map_deg_units_detected(tmp_path: Path) -> None:
    """TB1map with Units='deg' → actual_flip_deg encoding."""
    m = tmp_path / "sub-01_TB1map.nii.gz"
    _write_nii(m, (4, 4, 4))
    (tmp_path / "sub-01_TB1map.json").write_text(
        json.dumps({"Units": "deg", "FlipAngle": 70})
    )
    result = _classify_b1_data([m], _MockLayout())
    assert result["b1_map_encoding"] == "actual_flip_deg"


def test_tb1map_no_units_is_auto(tmp_path: Path) -> None:
    """TB1map without Units → encoding hint 'auto' (interface infers from pixels)."""
    m = tmp_path / "sub-01_TB1map.nii.gz"
    _write_nii(m, (4, 4, 4))
    (tmp_path / "sub-01_TB1map.json").write_text(json.dumps({"FlipAngle": 70}))
    result = _classify_b1_data([m], _MockLayout())
    assert result["afi_mode"] == "precomputed"
    assert result["b1_map_encoding"] == "auto"
