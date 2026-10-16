"""DCE discovery and skip-aware preflight regression tests."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pmriprep.config.dce import DCEConfig
from pmriprep.config.dsc import DSCConfig
from pmriprep.utils.preflight import PreflightError
from pmriprep.workflows.base import (
    PipelineSkipFlags,
    _prepare_master_preflight_state,
    decide_pipelines,
)
from pmriprep.workflows.discovery import discover_session_data


def _write_series(path: Path, *, dsc: bool = False) -> None:
    """Create a portable dynamic acquisition and valid metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 50), dtype=np.float32), np.eye(4)), path)
    path.with_name(path.name.replace(".nii.gz", ".json")).write_text(json.dumps({
        "ContrastBolusAgent": "Magnevist",
        "EchoTime": 0.035 if dsc else 0.002,
        "RepetitionTime": 1.5 if dsc else 7.0,
        "RepetitionTimeExcitation": 1.5 if dsc else 0.004,
        "FlipAngle": 60.0 if dsc else 15.0,
        "MagneticFieldStrength": 3.0,
        "ScanningSequence": ["GR", "EP"] if dsc else "GR",
        "SequenceVariant": "SP",
    }))


def _write_map(path: Path, metadata: dict[str, object]) -> None:
    """Create a portable 3D quantitative-input image and sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4)),
        path,
    )
    path.with_name(path.name.replace(".nii.gz", ".json")).write_text(
        json.dumps(metadata)
    )


@pytest.mark.parametrize(
    "names, code",
    [
        (["perf/sub-01_run-1_dce", "perf/sub-01_run-2_dce"], "DCE_AMBIGUOUS_INPUT"),
        (["perf/sub-01_acq-a_dce", "anat/sub-01_acq-b_dce"], "DCE_AMBIGUOUS_INPUT"),
        (["perf/sub-01_dce", "perf/sub-01_dce_e2"], "DCE_AMBIGUOUS_INPUT"),
        (["perf/sub-01_echo-1_dce", "perf/sub-01_echo-2_dce"], "DCE_MULTI_ECHO_UNSUPPORTED"),
    ],
)
def test_dce_ambiguity_respects_skip_flags(
    tmp_path: Path, names: list[str], code: str,
) -> None:
    """Multiple DCE acquisitions stop DCE but allow DSC-only processing."""
    (tmp_path / "dataset_description.json").write_text(json.dumps({
        "Name": "Portable input", "BIDSVersion": "1.9.0",
    }))
    for name in names:
        _write_series(tmp_path / "sub-01" / f"{name}.nii.gz")
    _write_series(tmp_path / "sub-01/perf/sub-01_dsc.nii.gz", dsc=True)
    session = discover_session_data(tmp_path, "01", None)
    assert session.modalities.dce is None
    assert len(session.modalities.dce_candidates) == 2
    assert session.modalities.dsc is not None
    for skip in (True, False):
        decision = decide_pipelines(
            session, PipelineSkipFlags(skip_dce=skip, skip_tumor_seg=True),
        )
        assert decision.run_dce is not skip
        assert decision.run_dsc

        def prepare() -> None:
            _prepare_master_preflight_state(
                session_data=session, decision=decision,
                dsc_config=DSCConfig(), dce_config=DCEConfig(), user_specified=None,
            )

        if skip:
            prepare()
        else:
            with pytest.raises(PreflightError, match=code) as caught:
                prepare()
            for name in names:
                assert f"{name}.nii.gz" in str(caught.value)


def test_single_dce_with_echo_entity_is_rejected(tmp_path: Path) -> None:
    """A separately exported echo cannot bypass multi-echo validation."""
    (tmp_path / "dataset_description.json").write_text(json.dumps({
        "Name": "Portable input", "BIDSVersion": "1.9.0",
    }))
    _write_series(tmp_path / "sub-01/perf/sub-01_echo-1_dce.nii.gz")
    session = discover_session_data(tmp_path, "01", None)
    for skip in (True, False):
        decision = decide_pipelines(
            session, PipelineSkipFlags(skip_dce=skip, skip_tumor_seg=True),
        )

        def prepare() -> None:
            _prepare_master_preflight_state(
                session_data=session, decision=decision,
                dsc_config=DSCConfig(), dce_config=DCEConfig(), user_specified=None,
            )

        if skip:
            prepare()
        else:
            with pytest.raises(PreflightError, match="DCE_MULTI_ECHO_UNSUPPORTED"):
                prepare()


def test_dce_selection_isolated_by_subject_and_session(tmp_path: Path) -> None:
    """One DCE in each session is valid despite other subjects and sessions."""
    (tmp_path / "dataset_description.json").write_text(json.dumps({
        "Name": "Portable input", "BIDSVersion": "1.9.0",
    }))
    for subject in ("01", "02"):
        for session in ("01", "02"):
            _write_series(tmp_path / f"sub-{subject}/ses-{session}/anat/sub-{subject}_ses-{session}_dce.nii.gz")
    for subject in ("01", "02"):
        for session in ("01", "02"):
            data = discover_session_data(tmp_path, subject, session)
            assert len(data.modalities.dce_candidates) == 1
            assert data.modalities.dce == data.modalities.dce_candidates[0]
            _prepare_master_preflight_state(
                session_data=data,
                decision=decide_pipelines(data, PipelineSkipFlags(skip_tumor_seg=True)),
                dsc_config=DSCConfig(), dce_config=DCEConfig(), user_specified=None,
            )


def test_competing_b1_maps_fail_only_when_dce_uses_them(tmp_path: Path) -> None:
    """Discovery retains repeated B1 maps and preflight respects DCE policy."""
    (tmp_path / "dataset_description.json").write_text(json.dumps({
        "Name": "Portable input", "BIDSVersion": "1.9.0",
    }))
    _write_series(tmp_path / "sub-01/perf/sub-01_dce.nii.gz")
    for run in (1, 2):
        _write_map(
            tmp_path / f"sub-01/fmap/sub-01_run-{run}_TB1map.nii.gz",
            {"Units": "p.u."},
        )

    session = discover_session_data(tmp_path, "01", None)
    assert len(session.modalities.b1_precomputed_candidates) == 2

    enabled = decide_pipelines(
        session, PipelineSkipFlags(skip_tumor_seg=True)
    )
    with pytest.raises(PreflightError, match="DCE_B1_AMBIGUOUS_INPUT"):
        _prepare_master_preflight_state(
            session_data=session,
            decision=enabled,
            dsc_config=DSCConfig(),
            dce_config=DCEConfig(),
            user_specified=None,
        )

    disabled = decide_pipelines(
        session,
        PipelineSkipFlags(skip_dce=True, skip_tumor_seg=True),
    )
    _prepare_master_preflight_state(
        session_data=session,
        decision=disabled,
        dsc_config=DSCConfig(),
        dce_config=DCEConfig(),
        user_specified=None,
    )


def test_competing_vfa_runs_fail_preflight(tmp_path: Path) -> None:
    """VFA images from distinct run groups are not silently pooled."""
    (tmp_path / "dataset_description.json").write_text(json.dumps({
        "Name": "Portable input", "BIDSVersion": "1.9.0",
    }))
    _write_series(tmp_path / "sub-01/perf/sub-01_dce.nii.gz")
    for run, flip_angle in ((1, 5.0), (2, 15.0)):
        _write_map(
            tmp_path
            / f"sub-01/anat/sub-01_run-{run}_flip-{run}_VFA.nii.gz",
            {"FlipAngle": flip_angle, "RepetitionTime": 0.004},
        )

    session = discover_session_data(tmp_path, "01", None)
    decision = decide_pipelines(
        session, PipelineSkipFlags(skip_tumor_seg=True)
    )

    with pytest.raises(PreflightError, match="VFA_AMBIGUOUS_INPUT"):
        _prepare_master_preflight_state(
            session_data=session,
            decision=decision,
            dsc_config=DSCConfig(),
            dce_config=DCEConfig(),
            user_specified=None,
        )
