"""Tests for explicit volume- and pulse-TR metadata contracts."""

from __future__ import annotations

import json
from pathlib import Path

from pmriprep.workflows.base import _read_bids_sidecar


def _write_sidecar(tmp_path: Path, metadata: dict[str, float]) -> Path:
    image = tmp_path / "sub-01_dce.nii.gz"
    image.with_name("sub-01_dce.json").write_text(json.dumps(metadata))
    return image


def test_read_sidecar_keeps_volume_and_pulse_tr_distinct(tmp_path: Path) -> None:
    image = _write_sidecar(
        tmp_path,
        {"RepetitionTime": 7.0, "RepetitionTimeExcitation": 0.004},
    )
    metadata = _read_bids_sidecar(image)
    assert metadata["volume_tr"] == 7.0
    assert metadata["pulse_tr"] == 0.004
    assert "tr" not in metadata


def test_read_sidecar_uses_volume_tr_for_2d_pulse_tr(tmp_path: Path) -> None:
    image = _write_sidecar(tmp_path, {"RepetitionTime": 1.5})
    metadata = _read_bids_sidecar(image)
    assert metadata["volume_tr"] == 1.5
    assert metadata["pulse_tr"] == 1.5
    assert "tr" not in metadata
