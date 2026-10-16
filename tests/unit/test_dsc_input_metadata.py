"""Input metadata regressions through the shared reader and master preflight."""

import json
from pathlib import Path
from typing import Any

import pytest

from pmriprep.config.dce import DCEConfig
from pmriprep.config.dsc import DSCConfig
from pmriprep.utils.preflight import (
    PreflightError, validate_dce_metadata, validate_vfa_metadata,
)
from pmriprep.workflows.base import (
    PipelineDecision,
    _prepare_master_preflight_state,
    _read_bids_sidecar,
)
from pmriprep.workflows.discovery import (
    ModalityAvailability,
    PipelineAvailability,
    SessionData,
)


@pytest.mark.parametrize(
    "echo_time, code",
    [
        ([0.015, 0.035], "DSC_MULTI_ECHO_UNSUPPORTED"),
        ([], "DSC_MULTI_ECHO_UNSUPPORTED"),
        (None, "DSC_INVALID_TE"),
        (True, "DSC_INVALID_TE"),
        ("bad", "DSC_INVALID_TE"),
        ({"echo": 0.035}, "DSC_INVALID_TE"),
        (float("nan"), "DSC_INVALID_TE"),
        (float("inf"), "DSC_INVALID_TE"),
        (0.035, None),
    ],
)
def test_master_preflight_handles_echo_time(
    tmp_path: Path, echo_time: Any, code: str | None,
) -> None:
    """Invalid supplied metadata produces a clean error before graph construction."""
    image = tmp_path / "sub-01_dsc.nii.gz"
    image.touch()
    image.with_name("sub-01_dsc.json").write_text(json.dumps({
        "EchoTime": echo_time,
        "RepetitionTime": 1.5,
        "FlipAngle": 60.0,
        "MagneticFieldStrength": 3.0,
        "ScanningSequence": ["GR", "EP"],
    }))
    session = SessionData(
        subject="01", session=None,
        modalities=ModalityAvailability(dsc=image, dsc_n_volumes=50),
        pipelines=PipelineAvailability(dsc=True),
    )

    def prepare() -> None:
        _prepare_master_preflight_state(
            session_data=session, decision=PipelineDecision(run_dsc=True),
            dsc_config=DSCConfig(), dce_config=DCEConfig(), user_specified=None,
        )

    if code:
        with pytest.raises(PreflightError, match=code):
            prepare()
    else:
        prepare()


@pytest.mark.parametrize("echo_time", [0.002, [0.002, 0.004], "bad"])
def test_shared_reader_preserves_dce_validation(tmp_path: Path, echo_time: Any) -> None:
    """Valid DCE timing is unchanged; invalid supplied TE cannot become absent TE."""
    image = tmp_path / "sub-01_dce.nii.gz"
    image.with_name("sub-01_dce.json").write_text(json.dumps({
        "ContrastBolusAgent": "Magnevist",
        "EchoTime": echo_time, "RepetitionTime": 7.0,
        "RepetitionTimeExcitation": 0.004, "FlipAngle": 15.0,
        "MagneticFieldStrength": 3.0, "ScanningSequence": "GR",
        "SequenceVariant": "SP",
    }))
    metadata = _read_bids_sidecar(image)
    errors = [m.code for m in validate_dce_metadata(metadata, DCEConfig()) if m.level == "error"]
    if isinstance(echo_time, float):
        assert errors == []
        assert metadata["te"] == echo_time
        assert metadata["pulse_tr"] == 0.004
        assert metadata["volume_tr"] == 7.0
        assert metadata["flip_angle"] == 15.0
    elif isinstance(echo_time, list):
        assert "DCE_MULTI_ECHO_UNSUPPORTED" in errors
    else:
        assert "DCE_INVALID_TE" in errors


def test_shared_reader_preserves_vfa_metadata(tmp_path: Path) -> None:
    """VFA pulse timing and flip angles retain their existing validation."""
    metadata = []
    for angle in (2, 15):
        image = tmp_path / f"sub-01_flip-{angle}_VFA.nii.gz"
        image.with_name(image.name.replace(".nii.gz", ".json")).write_text(json.dumps({
            "EchoTime": 0.002, "RepetitionTime": 0.004, "FlipAngle": angle,
        }))
        metadata.append(_read_bids_sidecar(image))
    messages = validate_vfa_metadata(
        [item["flip_angle"] for item in metadata],
        [item["pulse_tr"] for item in metadata],
    )
    assert not [message for message in messages if message.level == "error"]
    assert [item["te"] for item in metadata] == [0.002, 0.002]
