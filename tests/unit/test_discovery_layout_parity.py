"""Smoke + edge-case tests for the layout-backed discovery path."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

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


def test_discovery_marks_mismatched_t1w_pair_ineligible_for_subtraction(
    tmp_path: Path,
) -> None:
    """Independent T1w availability does not imply a valid subtraction pair."""
    bids = tmp_path / "bids"
    anat = bids / "sub-01" / "anat"
    anat.mkdir(parents=True)
    _write_dd(bids)
    _touch_nii(anat / "sub-01_acq-mprage_T1w.nii.gz", {})
    _touch_nii(anat / "sub-01_acq-spgr_ce-GAD_T1w.nii.gz", {})

    layout = BIDSLayout(str(bids), validate=False)
    session = discover_session_data(bids, "01", None, layout=layout)

    assert session.modalities.t1w_pre is not None
    assert session.modalities.t1w_ce_gad is not None
    assert session.modalities.t1w_subtraction_compatible is False


@pytest.mark.parametrize(
    "names, code",
    [
        (["sub-01_dsc", "sub-01_dsc_bold"], "DSC_AMBIGUOUS_INPUT"),
        (["sub-01_dsc", "sub-01_dsc_e2"], "DSC_AMBIGUOUS_INPUT"),
        (["sub-01_run-1_dsc", "sub-01_run-2_dsc"], "DSC_AMBIGUOUS_INPUT"),
        (["sub-01_echo-1_dsc", "sub-01_echo-2_dsc"], "DSC_MULTI_ECHO_UNSUPPORTED"),
        (["sub-01_run-1_bold", "sub-01_run-2_bold"], "DSC_AMBIGUOUS_INPUT"),
    ],
)
def test_discovery_rejects_multiple_dsc_candidates(
    tmp_path: Path, names: list[str], code: str,
) -> None:
    """Discovery retains candidates; enabled DSC fails before processing."""
    from pmriprep.config.dce import DCEConfig
    from pmriprep.config.dsc import DSCConfig
    from pmriprep.utils.preflight import PreflightError
    from pmriprep.workflows.base import (
        PipelineSkipFlags, decide_pipelines, _prepare_master_preflight_state,
    )

    bids = tmp_path / "bids"
    perf = bids / "sub-01" / "perf"
    perf.mkdir(parents=True)
    _write_dd(bids)
    for name in names:
        _touch_nii(perf / f"{name}.nii.gz", {})
    dce = perf / "sub-01_dce.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 50), dtype=np.float32), np.eye(4)), dce)
    dce.with_name("sub-01_dce.json").write_text(json.dumps({
        "EchoTime": 0.002, "RepetitionTime": 7.0,
        "RepetitionTimeExcitation": 0.004, "FlipAngle": 15.0,
        "MagneticFieldStrength": 3.0, "ScanningSequence": "GR",
        "SequenceVariant": "SP",
    }))
    session_data = discover_session_data(bids, "01", None)
    assert session_data.modalities.dsc is None
    assert len(session_data.modalities.dsc_candidates) == 2
    for skip in (True, False):
        decision = decide_pipelines(
            session_data, PipelineSkipFlags(skip_dsc=skip, skip_tumor_seg=True),
        )
        assert decision.run_dsc is not skip
        assert decision.run_dce is True
        if skip:
            _prepare_master_preflight_state(
                session_data=session_data, decision=decision, dsc_config=DSCConfig(),
                dce_config=DCEConfig(), user_specified=None,
            )
    with pytest.raises(PreflightError, match=code) as caught:
        _prepare_master_preflight_state(
            session_data=session_data, decision=decision, dsc_config=DSCConfig(),
            dce_config=DCEConfig(), user_specified=None,
        )
    for name in names:
        assert f"{name}.nii.gz" in str(caught.value)


def test_find_dsc_isolates_sessions_and_ignores_other_modalities(tmp_path: Path) -> None:
    """One DSC per session remains unambiguous beside DCE and BOLD files."""
    from pmriprep.workflows.discovery import _find_dsc_candidates

    bids = tmp_path / "bids"
    bids.mkdir()
    _write_dd(bids)
    for session in ("01", "02"):
        perf = bids / "sub-01" / f"ses-{session}" / "perf"
        perf.mkdir(parents=True)
        for suffix in ("dsc", "dce", "bold"):
            _touch_nii(perf / f"sub-01_ses-{session}_{suffix}.nii.gz", {})
    layout = BIDSLayout(str(bids), validate=False)
    for session in ("01", "02"):
        candidates = _find_dsc_candidates(layout, "01", session)
        assert len(candidates) == 1
        assert candidates[0].name == f"sub-01_ses-{session}_dsc.nii.gz"
