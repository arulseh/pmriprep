"""Smoke + edge-case tests for the layout-backed discovery path."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from bids.layout import BIDSLayout
from pmriprep.workflows.discovery import discover_session_data


def _write_dd(root: Path) -> None:
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "Test", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )


def _make_t1w(anat: Path, sub: str, ses: str | None) -> Path:
    prefix = f"sub-{sub}" + (f"_ses-{ses}" if ses else "")
    nii = anat / f"{prefix}_T1w.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), nii)
    (anat / f"{prefix}_T1w.json").write_text(
        json.dumps({"RepetitionTime": 2.0, "EchoTime": 0.003})
    )
    return nii


def test_discovery_returns_path_objects(tmp_path: Path) -> None:
    """Discovery must wrap layout's str paths back into ``pathlib.Path``."""
    bids = tmp_path / "bids"
    anat = bids / "sub-01" / "anat"
    anat.mkdir(parents=True)
    _write_dd(bids)
    _make_t1w(anat, "01", None)

    layout = BIDSLayout(str(bids), validate=False)
    sd = discover_session_data(bids, "01", None, layout=layout)
    assert isinstance(sd.modalities.t1w_pre, Path)


def test_discovery_anat_only_session(tmp_path: Path) -> None:
    """A session with only anat/ must yield dce/dsc=None without raising."""
    bids = tmp_path / "bids"
    anat = bids / "sub-01" / "ses-0001" / "anat"
    anat.mkdir(parents=True)
    _write_dd(bids)
    _make_t1w(anat, "01", "0001")

    layout = BIDSLayout(str(bids), validate=False)
    sd = discover_session_data(bids, "01", "0001", layout=layout)
    assert sd.modalities.dce is None
    assert sd.modalities.dsc is None
    assert sd.pipelines.dce is False
    assert sd.pipelines.dsc is False


def test_discovery_session_less_dataset(tmp_path: Path) -> None:
    """``session=None`` must still return discovered T1w."""
    bids = tmp_path / "bids"
    anat = bids / "sub-01" / "anat"
    anat.mkdir(parents=True)
    _write_dd(bids)
    _make_t1w(anat, "01", None)

    layout = BIDSLayout(str(bids), validate=False)
    sd = discover_session_data(bids, "01", None, layout=layout)
    assert sd.modalities.t1w_pre is not None
    assert sd.modalities.t1w_pre.name == "sub-01_T1w.nii.gz"


def _touch_nii(p: Path, sidecar: dict | None = None) -> None:
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), p)
    if sidecar is not None:
        p.with_name(p.name.replace(".nii.gz", ".json")).write_text(
            json.dumps(sidecar)
        )


def test_find_t1w_precontrast_acquisition_preference(tmp_path: Path) -> None:
    """Multiple precon T1w → MPRAGE wins over SE."""
    from pmriprep.workflows.discovery import _find_t1w_precontrast

    bids = tmp_path / "bids"
    anat = bids / "sub-01" / "anat"
    anat.mkdir(parents=True)
    _write_dd(bids)
    _touch_nii(anat / "sub-01_acq-se_T1w.nii.gz", {})
    _touch_nii(anat / "sub-01_acq-mprage_T1w.nii.gz", {})

    layout = BIDSLayout(str(bids), validate=False)
    chosen = _find_t1w_precontrast(layout, "01", None)
    assert chosen is not None and "mprage" in chosen.name


def test_find_t1w_ce_gad_fallback(tmp_path: Path) -> None:
    """Prefer ce-GAD; fall back to other ce- agents when no GAD present."""
    from pmriprep.workflows.discovery import _find_t1w_ce_gad

    bids = tmp_path / "bids"
    anat = bids / "sub-01" / "anat"
    anat.mkdir(parents=True)
    _write_dd(bids)
    _touch_nii(anat / "sub-01_ce-DOTAREM_T1w.nii.gz", {})
    _touch_nii(anat / "sub-01_ce-GAD_T1w.nii.gz", {})

    layout = BIDSLayout(str(bids), validate=False)
    chosen = _find_t1w_ce_gad(layout, "01", None)
    assert chosen is not None and "GAD" in chosen.name

    # Remove GAD; DOTAREM should now win as the fallback.
    (anat / "sub-01_ce-GAD_T1w.nii.gz").unlink()
    layout2 = BIDSLayout(str(bids), validate=False)
    chosen2 = _find_t1w_ce_gad(layout2, "01", None)
    assert chosen2 is not None and "DOTAREM" in chosen2.name


def test_find_dsc_prefers_bold_or_perf_named_files(tmp_path: Path) -> None:
    """When multiple *dsc* files exist, prefer one with _bold or perf in name."""
    from pmriprep.workflows.discovery import _find_dsc

    bids = tmp_path / "bids"
    perf = bids / "sub-01" / "perf"
    perf.mkdir(parents=True)
    _write_dd(bids)
    _touch_nii(perf / "sub-01_dsc.nii.gz", {})
    _touch_nii(perf / "sub-01_dsc_bold.nii.gz", {})

    layout = BIDSLayout(str(bids), validate=False)
    chosen = _find_dsc(layout, "01", None)
    assert chosen is not None and "_bold" in chosen.name


