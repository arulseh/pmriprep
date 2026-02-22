"""Unit tests for pmriprep.utils.preflight validation module."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from pmriprep.config.dce import DCEConfig
from pmriprep.utils.preflight import (
    PreflightError,
    PreflightMessage,
    PreflightReport,
    validate_afi_metadata,
    validate_dce_metadata,
    validate_dsc_metadata,
    validate_sidecar_exists,
    validate_vfa_metadata,
    run_preflight,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_dsc_sidecar() -> dict:
    """Return a valid DSC sidecar dict."""
    return {
        "te": 0.030,
        "tr": 1.5,
        "volume_tr": 1.5,
        "pulse_tr": 1.5,
        "field_strength": 3.0,
        "flip_angle": 90.0,
        "r1_relaxivity": None,
        "contrast_agent": None,
    }


def _valid_dce_sidecar() -> dict:
    """Return a valid DCE sidecar dict."""
    return {
        "te": 0.002,
        "tr": 0.004,
        "volume_tr": 7.0,
        "pulse_tr": 0.004,
        "field_strength": 3.0,
        "flip_angle": 15.0,
        "r1_relaxivity": 5.0,
        "contrast_agent": "Gadovist",
    }


def _make_modalities(**kwargs):
    """Create a mock ModalityAvailability."""
    defaults = {
        "dsc": None,
        "dce": None,
        "vfa": [],
        "afi": [],
        "afi_tr1_ms": None,
        "afi_tr2_ms": None,
    }
    defaults.update(kwargs)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


def _make_decision(run_dsc: bool = False, run_dce: bool = False):
    """Create a mock PipelineDecision."""
    mock = MagicMock()
    mock.run_dsc = run_dsc
    mock.run_dce = run_dce
    return mock


def _make_session_data(modalities=None):
    """Create a mock SessionData."""
    mock = MagicMock()
    mock.modalities = modalities or _make_modalities()
    return mock


# ===========================================================================
# PreflightReport tests
# ===========================================================================

class TestPreflightReport:
    def test_has_errors_empty(self):
        report = PreflightReport()
        assert not report.has_errors

    def test_has_errors_with_warnings_only(self):
        report = PreflightReport(messages=[
            PreflightMessage("warning", "TEST", "a warning"),
        ])
        assert not report.has_errors

    def test_has_errors_with_error(self):
        report = PreflightReport(messages=[
            PreflightMessage("warning", "W1", "warning"),
            PreflightMessage("error", "E1", "error msg"),
        ])
        assert report.has_errors

    def test_raise_if_errors_no_errors(self):
        report = PreflightReport(messages=[
            PreflightMessage("warning", "W1", "ok"),
        ])
        report.raise_if_errors()  # should not raise

    def test_raise_if_errors_with_errors(self):
        report = PreflightReport(messages=[
            PreflightMessage("error", "E1", "first error"),
            PreflightMessage("warning", "W1", "a warning"),
            PreflightMessage("error", "E2", "second error"),
        ])
        with pytest.raises(PreflightError, match="2 error"):
            report.raise_if_errors()

    def test_log_all(self, caplog):
        report = PreflightReport(messages=[
            PreflightMessage("info", "I1", "info msg"),
            PreflightMessage("warning", "W1", "warn msg"),
            PreflightMessage("error", "E1", "err msg"),
        ])
        with caplog.at_level("DEBUG", logger="pmriprep.preflight"):
            report.log_all()
        assert "I1" in caplog.text
        assert "W1" in caplog.text
        assert "E1" in caplog.text


# ===========================================================================
# validate_sidecar_exists tests
# ===========================================================================

class TestValidateSidecarExists:
    def test_sidecar_exists(self, tmp_path):
        nii = tmp_path / "sub-01_bold.nii.gz"
        nii.touch()
        (tmp_path / "sub-01_bold.json").touch()
        msgs = validate_sidecar_exists(nii, "DSC")
        assert len(msgs) == 0

    def test_sidecar_missing_dsc(self, tmp_path):
        nii = tmp_path / "sub-01_bold.nii.gz"
        nii.touch()
        msgs = validate_sidecar_exists(nii, "DSC")
        assert len(msgs) == 1
        assert msgs[0].level == "error"
        assert msgs[0].code == "DSC_NO_SIDECAR"

    def test_sidecar_missing_vfa_is_warning(self, tmp_path):
        nii = tmp_path / "sub-01_FA15.nii.gz"
        nii.touch()
        msgs = validate_sidecar_exists(nii, "VFA")
        assert len(msgs) == 1
        assert msgs[0].level == "warning"


# ===========================================================================
# validate_dsc_metadata tests
# ===========================================================================

class TestValidateDSCMetadata:
    def test_all_valid(self):
        msgs = validate_dsc_metadata(_valid_dsc_sidecar())
        # Should have no errors (may have no messages at all)
        assert not any(m.level == "error" for m in msgs)

    def test_missing_te(self):
        sc = _valid_dsc_sidecar()
        sc["te"] = None
        msgs = validate_dsc_metadata(sc)
        assert any(m.code == "DSC_MISSING_TE" for m in msgs)

    def test_negative_te(self):
        sc = _valid_dsc_sidecar()
        sc["te"] = -0.01
        msgs = validate_dsc_metadata(sc)
        assert any(m.code == "DSC_INVALID_TE" for m in msgs)

    def test_zero_te(self):
        sc = _valid_dsc_sidecar()
        sc["te"] = 0.0
        msgs = validate_dsc_metadata(sc)
        assert any(m.code == "DSC_INVALID_TE" for m in msgs)

    def test_te_in_milliseconds(self):
        """TE=30 (ms instead of s) should trigger range error."""
        sc = _valid_dsc_sidecar()
        sc["te"] = 30.0  # ms, not s
        msgs = validate_dsc_metadata(sc)
        assert any(m.code == "DSC_TE_RANGE" for m in msgs)
        assert "seconds" in msgs[0].message.lower() or "BIDS" in msgs[0].message

    def test_te_too_small(self):
        sc = _valid_dsc_sidecar()
        sc["te"] = 0.001  # 1ms — too short for DSC
        msgs = validate_dsc_metadata(sc)
        assert any(m.code == "DSC_TE_RANGE" for m in msgs)

    def test_missing_tr(self):
        sc = _valid_dsc_sidecar()
        sc["tr"] = None
        msgs = validate_dsc_metadata(sc)
        assert any(m.code == "DSC_MISSING_TR" for m in msgs)

    def test_negative_tr(self):
        sc = _valid_dsc_sidecar()
        sc["tr"] = -1.0
        msgs = validate_dsc_metadata(sc)
        assert any(m.code == "DSC_INVALID_TR" for m in msgs)

    def test_coarse_tr_warning(self):
        sc = _valid_dsc_sidecar()
        sc["tr"] = 4.0
        msgs = validate_dsc_metadata(sc)
        assert any(m.code == "DSC_TR_COARSE" for m in msgs)
        assert all(m.level == "warning" for m in msgs if m.code == "DSC_TR_COARSE")

    def test_missing_field_strength(self):
        sc = _valid_dsc_sidecar()
        sc["field_strength"] = None
        msgs = validate_dsc_metadata(sc)
        assert any(m.code == "DSC_MISSING_FIELD_STRENGTH" for m in msgs)


# ===========================================================================
# validate_dce_metadata tests
# ===========================================================================

class TestValidateDCEMetadata:
    def test_all_valid(self):
        sc = _valid_dce_sidecar()
        cfg = DCEConfig()
        msgs = validate_dce_metadata(sc, cfg)
        assert not any(m.level == "error" for m in msgs)

    def test_missing_flip_angle(self):
        sc = _valid_dce_sidecar()
        sc["flip_angle"] = None
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert any(m.code == "DCE_MISSING_FLIP_ANGLE" for m in msgs)

    def test_negative_flip_angle(self):
        sc = _valid_dce_sidecar()
        sc["flip_angle"] = -5.0
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert any(m.code == "DCE_INVALID_FLIP_ANGLE" for m in msgs)

    def test_flip_angle_in_radians(self):
        """FlipAngle=0.26 (radians ~15deg) should trigger >90 or <=0 error."""
        sc = _valid_dce_sidecar()
        sc["flip_angle"] = 95.0  # invalid
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert any(m.code == "DCE_INVALID_FLIP_ANGLE" for m in msgs)

    def test_missing_all_tr(self):
        sc = _valid_dce_sidecar()
        sc["volume_tr"] = None
        sc["pulse_tr"] = None
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert any(m.code == "DCE_MISSING_TR" for m in msgs)

    def test_tr_swapped(self):
        sc = _valid_dce_sidecar()
        sc["volume_tr"] = 0.003  # volume < pulse
        sc["pulse_tr"] = 7.0
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert any(m.code == "DCE_TR_SWAPPED" for m in msgs)

    def test_no_volume_tr_3d_dce(self):
        """Only pulse TR < 0.1s, no volume TR → error."""
        sc = _valid_dce_sidecar()
        sc["volume_tr"] = None
        sc["pulse_tr"] = 0.004
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert any(m.code == "DCE_NO_VOLUME_TR" for m in msgs)

    def test_no_pulse_tr_long_volume(self):
        """Only volume TR > 1.0s, no pulse TR → error."""
        sc = _valid_dce_sidecar()
        sc["pulse_tr"] = None
        sc["volume_tr"] = 7.0
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert any(m.code == "DCE_NO_PULSE_TR" for m in msgs)

    def test_r1_override_info(self):
        sc = _valid_dce_sidecar()
        cfg = DCEConfig(r1_relaxivity=5.0)
        msgs = validate_dce_metadata(sc, cfg)
        assert any(m.code == "DCE_R1_OVERRIDE" and m.level == "info" for m in msgs)

    def test_r1_override_out_of_range(self):
        sc = _valid_dce_sidecar()
        cfg = DCEConfig(r1_relaxivity=50.0)
        msgs = validate_dce_metadata(sc, cfg)
        assert any(m.code == "DCE_R1_OUT_OF_RANGE" for m in msgs)

    def test_missing_field_strength_with_agent(self):
        sc = _valid_dce_sidecar()
        sc["field_strength"] = None
        sc["r1_relaxivity"] = None
        sc["contrast_agent"] = "Gadovist"
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert any(m.code == "DCE_R1_NO_FIELD_STRENGTH" for m in msgs)

    def test_missing_field_strength_no_agent(self):
        sc = _valid_dce_sidecar()
        sc["field_strength"] = None
        sc["r1_relaxivity"] = None
        sc["contrast_agent"] = None
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert any(m.code == "DCE_MISSING_FIELD_STRENGTH" for m in msgs)

    def test_unknown_agent(self):
        sc = _valid_dce_sidecar()
        sc["r1_relaxivity"] = None
        sc["contrast_agent"] = None
        sc["field_strength"] = 3.0
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert any(m.code == "DCE_UNKNOWN_AGENT" for m in msgs)

    def test_r1_override_skips_agent_warnings(self):
        """When r1 is user-supplied, no warnings about agents or field strength."""
        sc = _valid_dce_sidecar()
        sc["field_strength"] = None
        sc["r1_relaxivity"] = None
        sc["contrast_agent"] = None
        cfg = DCEConfig(r1_relaxivity=4.0)
        msgs = validate_dce_metadata(sc, cfg)
        codes = {m.code for m in msgs}
        assert "DCE_MISSING_FIELD_STRENGTH" not in codes
        assert "DCE_UNKNOWN_AGENT" not in codes
        assert "DCE_R1_NO_FIELD_STRENGTH" not in codes


# ===========================================================================
# validate_vfa_metadata tests
# ===========================================================================

class TestValidateVFAMetadata:
    def test_valid_vfa(self):
        msgs = validate_vfa_metadata([5.0, 15.0], [0.004, 0.004])
        assert not any(m.level == "error" for m in msgs)
        assert len(msgs) == 0

    def test_single_angle(self):
        msgs = validate_vfa_metadata([15.0], [0.004])
        assert any(m.code == "VFA_FEW_ANGLES" for m in msgs)

    def test_identical_angles(self):
        msgs = validate_vfa_metadata([15.0, 15.0], [0.004, 0.004])
        assert any(m.code == "VFA_FEW_ANGLES" for m in msgs)

    def test_missing_tr(self):
        msgs = validate_vfa_metadata([5.0, 15.0], [0.004, None])
        assert any(m.code == "VFA_MISSING_TR" for m in msgs)

    def test_poor_conditioning(self):
        msgs = validate_vfa_metadata([14.0, 15.0], [0.004, 0.004])
        assert any(m.code == "VFA_POOR_CONDITIONING" for m in msgs)


# ===========================================================================
# validate_afi_metadata tests
# ===========================================================================

class TestValidateAFIMetadata:
    def test_valid_afi(self):
        msgs = validate_afi_metadata(20.0, 100.0)
        assert len(msgs) == 0

    def test_missing_tr1(self):
        msgs = validate_afi_metadata(None, 100.0)
        assert any(m.code == "AFI_MISSING_TR" for m in msgs)

    def test_missing_tr2(self):
        msgs = validate_afi_metadata(20.0, None)
        assert any(m.code == "AFI_MISSING_TR" for m in msgs)

    def test_tr_order_wrong(self):
        msgs = validate_afi_metadata(100.0, 20.0)
        assert any(m.code == "AFI_TR_ORDER" for m in msgs)

    def test_tr_equal(self):
        msgs = validate_afi_metadata(50.0, 50.0)
        assert any(m.code == "AFI_TR_ORDER" for m in msgs)


# ===========================================================================
# run_preflight orchestrator tests
# ===========================================================================

class TestRunPreflight:
    def test_dsc_only(self, tmp_path):
        """DSC-only pipeline validates DSC metadata."""
        dsc_file = tmp_path / "sub-01_dsc.nii.gz"
        dsc_file.touch()
        (tmp_path / "sub-01_dsc.json").touch()

        mods = _make_modalities(dsc=dsc_file)
        session = _make_session_data(mods)
        decision = _make_decision(run_dsc=True)

        report = run_preflight(
            session_data=session,
            decision=decision,
            dsc_sidecar=_valid_dsc_sidecar(),
            dce_sidecar=None,
            dce_config=None,
        )
        # Valid sidecar → no errors
        assert not report.has_errors

    def test_dce_only(self, tmp_path):
        """DCE-only pipeline validates DCE metadata."""
        dce_file = tmp_path / "sub-01_dce.nii.gz"
        dce_file.touch()
        (tmp_path / "sub-01_dce.json").touch()

        mods = _make_modalities(dce=dce_file)
        session = _make_session_data(mods)
        decision = _make_decision(run_dce=True)

        report = run_preflight(
            session_data=session,
            decision=decision,
            dsc_sidecar=None,
            dce_sidecar=_valid_dce_sidecar(),
            dce_config=DCEConfig(),
        )
        assert not report.has_errors

    def test_neither_pipeline(self):
        """No pipelines → no messages."""
        session = _make_session_data()
        decision = _make_decision()
        report = run_preflight(
            session_data=session,
            decision=decision,
            dsc_sidecar=None,
            dce_sidecar=None,
            dce_config=None,
        )
        assert len(report.messages) == 0

    def test_dsc_missing_te_stops(self, tmp_path):
        """Missing TE should produce an error."""
        dsc_file = tmp_path / "sub-01_dsc.nii.gz"
        dsc_file.touch()
        (tmp_path / "sub-01_dsc.json").touch()

        mods = _make_modalities(dsc=dsc_file)
        session = _make_session_data(mods)
        decision = _make_decision(run_dsc=True)

        sc = _valid_dsc_sidecar()
        sc["te"] = None
        report = run_preflight(
            session_data=session,
            decision=decision,
            dsc_sidecar=sc,
            dce_sidecar=None,
            dce_config=None,
        )
        assert report.has_errors

    def test_vfa_validation_included(self, tmp_path):
        """VFA validation runs when VFA data is present."""
        dce_file = tmp_path / "sub-01_dce.nii.gz"
        dce_file.touch()
        (tmp_path / "sub-01_dce.json").touch()

        mods = _make_modalities(
            dce=dce_file,
            vfa=[Path("/fake/fa5.nii.gz"), Path("/fake/fa15.nii.gz")],
        )
        session = _make_session_data(mods)
        decision = _make_decision(run_dce=True)

        report = run_preflight(
            session_data=session,
            decision=decision,
            dsc_sidecar=None,
            dce_sidecar=_valid_dce_sidecar(),
            dce_config=DCEConfig(),
            vfa_flip_angles=[15.0, 15.0],  # identical → warning
            vfa_trs=[0.004, 0.004],
        )
        assert any(m.code == "VFA_FEW_ANGLES" for m in report.messages)

    def test_afi_validation_included(self, tmp_path):
        """AFI validation runs when AFI data is present."""
        dce_file = tmp_path / "sub-01_dce.nii.gz"
        dce_file.touch()
        (tmp_path / "sub-01_dce.json").touch()

        mods = _make_modalities(
            dce=dce_file,
            afi=[Path("/fake/afi1.nii.gz")],
            afi_tr1_ms=None,  # missing
            afi_tr2_ms=100.0,
        )
        session = _make_session_data(mods)
        decision = _make_decision(run_dce=True)

        report = run_preflight(
            session_data=session,
            decision=decision,
            dsc_sidecar=None,
            dce_sidecar=_valid_dce_sidecar(),
            dce_config=DCEConfig(),
        )
        assert any(m.code == "AFI_MISSING_TR" for m in report.messages)


# ===========================================================================
# DCEConfig r1_relaxivity tests
# ===========================================================================

class TestDCEConfigR1:
    def test_default_none(self):
        cfg = DCEConfig()
        assert cfg.r1_relaxivity is None

    def test_set_float(self):
        cfg = DCEConfig(r1_relaxivity=5.0)
        assert cfg.r1_relaxivity == 5.0

    def test_coerce_from_string(self):
        cfg = DCEConfig(r1_relaxivity="4.5")  # type: ignore[arg-type]
        assert cfg.r1_relaxivity == 4.5

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="r1_relaxivity"):
            DCEConfig(r1_relaxivity="not_a_number")  # type: ignore[arg-type]


# ===========================================================================
# Config loader integration tests
# ===========================================================================

class TestConfigLoaderR1:
    def test_merge_cli_dce_r1(self):
        from pmriprep.config.loader import merge_cli_config
        cli_args = {"dce_r1": 5.0}
        file_config = {"dsc": {}, "dce": {}, "anat": {}, "tumor_seg": {}}
        merged = merge_cli_config(cli_args, file_config)
        assert merged["dce"]["r1_relaxivity"] == 5.0

    def test_merge_cli_dce_r1_none_no_override(self):
        from pmriprep.config.loader import merge_cli_config
        cli_args = {"dce_r1": None}
        file_config = {"dsc": {}, "dce": {"r1_relaxivity": 4.0}, "anat": {}, "tumor_seg": {}}
        merged = merge_cli_config(cli_args, file_config)
        assert merged["dce"]["r1_relaxivity"] == 4.0

    def test_build_dce_config_with_r1(self):
        from pmriprep.config.loader import build_dce_config
        cfg = build_dce_config({"dce": {"r1_relaxivity": "3.8"}})
        assert cfg.r1_relaxivity == 3.8

    def test_build_dce_config_without_r1(self):
        from pmriprep.config.loader import build_dce_config
        cfg = build_dce_config({"dce": {}})
        assert cfg.r1_relaxivity is None
