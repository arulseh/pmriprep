"""Unit tests for pmriprep.utils.preflight validation module."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pmriprep.config.dce import DCEConfig
from pmriprep.utils.preflight import (
    PreflightError,
    PreflightMessage,
    PreflightReport,
    validate_afi_metadata,
    validate_dce_metadata,
    validate_dce_temporal_sufficiency,
    validate_dsc_metadata,
    validate_dsc_temporal_sufficiency,
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
        "volume_tr": 1.5,
        "pulse_tr": 1.5,
        "field_strength": 3.0,
        "flip_angle": 90.0,
        "r1_relaxivity": None,
        "contrast_agent": None,
        "scanning_sequence": frozenset({"GR", "EP"}),
        "sequence_variant": None,
        "echo_entity": False,
    }


def _valid_dce_sidecar() -> dict:
    """Return a valid DCE sidecar dict."""
    return {
        "te": 0.002,
        "volume_tr": 7.0,
        "pulse_tr": 0.004,
        "field_strength": 3.0,
        "flip_angle": 15.0,
        "r1_relaxivity": 5.0,
        "contrast_agent": "Gadovist",
        "scanning_sequence": frozenset({"GR"}),
        "sequence_variant": frozenset({"SP"}),
        "echo_entity": False,
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
        "afi_mode": None,
        "afi_4d_file": None,
        "b1_method_detected": None,
        "b1_precomputed_file": None,
        "b1_precomputed_candidates": [],
        "afi_4d_candidates": [],
        "dsc_n_volumes": None,
        "dce_n_volumes": None,
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

    @pytest.mark.parametrize(
        ("field", "value", "code", "level", "message_fragment"),
        [
            pytest.param("te", None, "DSC_MISSING_TE", "error", None, id="missing-te"),
            pytest.param("te", -0.01, "DSC_INVALID_TE", "error", None, id="negative-te"),
            pytest.param("te", 0.0, "DSC_INVALID_TE", "error", None, id="zero-te"),
            pytest.param("te", 30.0, "DSC_TE_RANGE", "error", "seconds", id="te-in-ms"),
            pytest.param("te", 0.001, "DSC_TE_RANGE", "error", None, id="te-too-short"),
            pytest.param("volume_tr", None, "DSC_MISSING_TR", "error", None, id="missing-tr"),
            pytest.param("volume_tr", -1.0, "DSC_INVALID_TR", "error", None, id="negative-tr"),
            pytest.param("volume_tr", 4.0, "DSC_TR_COARSE", "warning", None, id="coarse-tr"),
            pytest.param(
                "field_strength", None, "DSC_MISSING_FIELD_STRENGTH", "error", None,
                id="missing-field-strength",
            ),
        ],
    )
    def test_scalar_metadata_validation(
        self,
        field: str,
        value: float | None,
        code: str,
        level: str,
        message_fragment: str | None,
    ) -> None:
        sc = _valid_dsc_sidecar()
        sc[field] = value
        msgs = validate_dsc_metadata(sc)
        matching = [message for message in msgs if message.code == code]
        assert len(matching) == 1
        assert matching[0].level == level
        if message_fragment is not None:
            assert message_fragment in matching[0].message.lower()

    # --- Sequence type validation ---

    def test_gre_epi_no_sequence_error(self):
        sc = _valid_dsc_sidecar()
        sc["scanning_sequence"] = frozenset({"GR", "EP"})
        msgs = validate_dsc_metadata(sc)
        seq_codes = {m.code for m in msgs if "ECHO" in m.code or "SPIN" in m.code
                     or "HYBRID" in m.code or "SEQUENCE_TYPE" in m.code}
        assert len(seq_codes) == 0

    @pytest.mark.parametrize("tokens", [{"SE"}, {"SE", "EP"}])
    def test_spin_echo_error(self, tokens: set[str]) -> None:
        sc = _valid_dsc_sidecar()
        sc["scanning_sequence"] = frozenset(tokens)
        msgs = validate_dsc_metadata(sc)
        assert any(m.code == "DSC_SPIN_ECHO_UNSUPPORTED" for m in msgs)
        assert any(m.level == "error" for m in msgs
                   if m.code == "DSC_SPIN_ECHO_UNSUPPORTED")

    @pytest.mark.parametrize("tokens", [{"GR", "SE"}, {"GR", "SE", "EP"}])
    def test_hybrid_gr_se_error(self, tokens: set[str]) -> None:
        sc = _valid_dsc_sidecar()
        sc["scanning_sequence"] = frozenset(tokens)
        msgs = validate_dsc_metadata(sc)
        assert any(m.code == "DSC_HYBRID_SEQUENCE" for m in msgs)
        assert any(m.level == "error" for m in msgs
                   if m.code == "DSC_HYBRID_SEQUENCE")
        with pytest.raises(PreflightError, match="DSC_HYBRID_SEQUENCE"):
            PreflightReport(messages=msgs).raise_if_errors()

    def test_missing_scanning_sequence_info(self):
        sc = _valid_dsc_sidecar()
        sc["scanning_sequence"] = None
        msgs = validate_dsc_metadata(sc)
        assert any(m.code == "DSC_SEQUENCE_TYPE_UNKNOWN" for m in msgs)
        assert any(m.level == "info" for m in msgs
                   if m.code == "DSC_SEQUENCE_TYPE_UNKNOWN")

    def test_multi_echo_entity_error(self):
        sc = _valid_dsc_sidecar()
        sc["echo_entity"] = True
        msgs = validate_dsc_metadata(sc)
        assert any(m.code == "DSC_MULTI_ECHO_UNSUPPORTED" for m in msgs)
        assert any(m.level == "error" for m in msgs
                   if m.code == "DSC_MULTI_ECHO_UNSUPPORTED")

    def test_no_echo_entity_no_error(self):
        sc = _valid_dsc_sidecar()
        sc["echo_entity"] = False
        msgs = validate_dsc_metadata(sc)
        assert not any(m.code == "DSC_MULTI_ECHO_UNSUPPORTED" for m in msgs)


# ===========================================================================
# validate_dce_metadata tests
# ===========================================================================

class TestValidateDCEMetadata:
    def test_all_valid(self):
        sc = _valid_dce_sidecar()
        cfg = DCEConfig()
        msgs = validate_dce_metadata(sc, cfg)
        assert not any(m.level == "error" for m in msgs)

    @pytest.mark.parametrize(
        ("updates", "code"),
        [
            pytest.param({"flip_angle": None}, "DCE_MISSING_FLIP_ANGLE", id="missing-flip-angle"),
            pytest.param({"flip_angle": -5.0}, "DCE_INVALID_FLIP_ANGLE", id="negative-flip-angle"),
            pytest.param({"flip_angle": 95.0}, "DCE_INVALID_FLIP_ANGLE", id="flip-angle-over-90"),
            pytest.param(
                {"volume_tr": None, "pulse_tr": None}, "DCE_MISSING_TR", id="missing-all-tr",
            ),
            pytest.param(
                {"volume_tr": 0.003, "pulse_tr": 7.0}, "DCE_TR_SWAPPED", id="swapped-tr",
            ),
            pytest.param(
                {"volume_tr": None, "pulse_tr": 0.004}, "DCE_NO_VOLUME_TR",
                id="missing-volume-tr",
            ),
            pytest.param(
                {"pulse_tr": None, "volume_tr": 7.0}, "DCE_NO_PULSE_TR",
                id="missing-pulse-tr",
            ),
        ],
    )
    def test_scalar_metadata_validation(self, updates: dict, code: str) -> None:
        sc = _valid_dce_sidecar()
        sc.update(updates)
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert any(message.code == code for message in msgs)

    # --- Sequence type validation ---

    def test_valid_spgr_no_sequence_errors(self):
        sc = _valid_dce_sidecar()
        sc["scanning_sequence"] = frozenset({"GR"})
        sc["sequence_variant"] = frozenset({"SP"})
        msgs = validate_dce_metadata(sc, DCEConfig())
        seq_codes = {m.code for m in msgs if "SPIN" in m.code or "IR_" in m.code
                     or "BSSFP" in m.code or "UNSPOILED" in m.code
                     or "EPI_READOUT" in m.code or "SEQUENCE_TYPE" in m.code
                     or "HYBRID" in m.code}
        assert len(seq_codes) == 0

    @pytest.mark.parametrize("tokens", [{"SE"}, {"SE", "EP"}])
    def test_dce_spin_echo_error(self, tokens: set[str]) -> None:
        sc = _valid_dce_sidecar()
        sc["scanning_sequence"] = frozenset(tokens)
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert any(m.code == "DCE_SPIN_ECHO_UNSUPPORTED" for m in msgs)
        assert any(m.level == "error" for m in msgs
                   if m.code == "DCE_SPIN_ECHO_UNSUPPORTED")

    @pytest.mark.parametrize("tokens", [{"GR", "SE"}, {"GR", "SE", "EP"}])
    def test_dce_hybrid_gr_se_error(self, tokens: set[str]) -> None:
        sc = _valid_dce_sidecar()
        sc["scanning_sequence"] = frozenset(tokens)
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert any(m.code == "DCE_HYBRID_SEQUENCE" and m.level == "error" for m in msgs)
        with pytest.raises(PreflightError, match="DCE_HYBRID_SEQUENCE"):
            PreflightReport(messages=msgs).raise_if_errors()

    def test_dce_ir_error(self):
        sc = _valid_dce_sidecar()
        sc["scanning_sequence"] = frozenset({"GR", "IR"})
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert any(m.code == "DCE_IR_UNSUPPORTED" for m in msgs)
        assert any(m.level == "error" for m in msgs
                   if m.code == "DCE_IR_UNSUPPORTED")

    def test_dce_bssfp_error(self):
        sc = _valid_dce_sidecar()
        sc["scanning_sequence"] = frozenset({"GR"})
        sc["sequence_variant"] = frozenset({"SS"})
        sc["te"] = 0.002
        sc["pulse_tr"] = 0.004  # TE/TR = 0.5
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert any(m.code == "DCE_BSSFP_UNSUPPORTED" for m in msgs)
        assert any(m.level == "error" for m in msgs
                   if m.code == "DCE_BSSFP_UNSUPPORTED")

    def test_dce_unspoiled_gr_warning(self):
        sc = _valid_dce_sidecar()
        sc["scanning_sequence"] = frozenset({"GR"})
        sc["sequence_variant"] = frozenset({"SS"})
        sc["te"] = 0.002
        sc["pulse_tr"] = 0.008  # TE/TR = 0.25, below bSSFP threshold
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert any(m.code == "DCE_UNSPOILED_GR" for m in msgs)
        assert any(m.level == "warning" for m in msgs
                   if m.code == "DCE_UNSPOILED_GR")
        assert not any(m.code == "DCE_BSSFP_UNSUPPORTED" for m in msgs)

    def test_dce_unspoiled_gr_missing_pulse_tr(self):
        sc = _valid_dce_sidecar()
        sc["scanning_sequence"] = frozenset({"GR"})
        sc["sequence_variant"] = frozenset({"SS"})
        sc["pulse_tr"] = None
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert any(m.code == "DCE_UNSPOILED_GR" for m in msgs)
        assert not any(m.code == "DCE_BSSFP_UNSUPPORTED" for m in msgs)

    def test_dce_epi_readout_info(self):
        sc = _valid_dce_sidecar()
        sc["scanning_sequence"] = frozenset({"GR", "EP"})
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert any(m.code == "DCE_EPI_READOUT" for m in msgs)
        assert any(m.level == "info" for m in msgs
                   if m.code == "DCE_EPI_READOUT")

    def test_dce_missing_scanning_sequence_info(self):
        sc = _valid_dce_sidecar()
        sc["scanning_sequence"] = None
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert any(m.code == "DCE_SEQUENCE_TYPE_UNKNOWN" for m in msgs)
        assert any(m.level == "info" for m in msgs
                   if m.code == "DCE_SEQUENCE_TYPE_UNKNOWN")

    def test_dce_missing_variant_no_variant_messages(self):
        sc = _valid_dce_sidecar()
        sc["scanning_sequence"] = frozenset({"GR"})
        sc["sequence_variant"] = None
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert not any(m.code in ("DCE_BSSFP_UNSUPPORTED", "DCE_UNSPOILED_GR")
                       for m in msgs)

    def test_dce_bssfp_missing_te_falls_to_unspoiled(self):
        sc = _valid_dce_sidecar()
        sc["scanning_sequence"] = frozenset({"GR"})
        sc["sequence_variant"] = frozenset({"SS"})
        sc["te"] = None
        sc["pulse_tr"] = 0.004
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert any(m.code == "DCE_UNSPOILED_GR" for m in msgs)
        assert not any(m.code == "DCE_BSSFP_UNSUPPORTED" for m in msgs)

    def test_dce_se_ir_fires_both_errors(self):
        sc = _valid_dce_sidecar()
        sc["scanning_sequence"] = frozenset({"SE", "IR"})
        msgs = validate_dce_metadata(sc, DCEConfig())
        codes = {m.code for m in msgs}
        assert "DCE_SPIN_ECHO_UNSUPPORTED" in codes
        assert "DCE_IR_UNSUPPORTED" in codes

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

    @pytest.mark.parametrize("agent", ["Gadovist", None], ids=["known-agent", "no-agent"])
    def test_missing_field_strength(self, agent: str | None) -> None:
        sc = _valid_dce_sidecar()
        sc["field_strength"] = None
        sc["r1_relaxivity"] = None
        sc["contrast_agent"] = agent
        msgs = validate_dce_metadata(sc, DCEConfig())
        fs_msgs = [m for m in msgs if m.code == "DCE_MISSING_FIELD_STRENGTH"]
        assert len(fs_msgs) == 1
        assert fs_msgs[0].level == "error"

    def test_unknown_agent(self):
        sc = _valid_dce_sidecar()
        sc["r1_relaxivity"] = None
        sc["contrast_agent"] = None
        sc["field_strength"] = 3.0
        msgs = validate_dce_metadata(sc, DCEConfig())
        assert any(m.code == "DCE_R1_UNAVAILABLE" and m.level == "error" for m in msgs)

    def test_r1_override_skips_agent_warnings(self):
        """When r1 is user-supplied, agent warnings are skipped but field strength is still required."""
        sc = _valid_dce_sidecar()
        sc["field_strength"] = None
        sc["r1_relaxivity"] = None
        sc["contrast_agent"] = None
        cfg = DCEConfig(r1_relaxivity=4.0)
        msgs = validate_dce_metadata(sc, cfg)
        codes = {m.code for m in msgs}
        assert "DCE_MISSING_FIELD_STRENGTH" in codes
        assert "DCE_R1_UNAVAILABLE" not in codes


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
        assert any(
            m.code == "VFA_FEW_ANGLES" and m.level == "error" for m in msgs
        )

    @pytest.mark.parametrize("angle", [0.0, -5.0, 120.0, float("nan")])
    def test_invalid_flip_angle(self, angle):
        msgs = validate_vfa_metadata([angle, 15.0], [0.004, 0.004])
        assert any(
            m.code == "VFA_INVALID_FLIP_ANGLE" and m.level == "error"
            for m in msgs
        )

    def test_identical_angles(self):
        msgs = validate_vfa_metadata([15.0, 15.0], [0.004, 0.004])
        assert any(
            m.code == "VFA_FEW_ANGLES" and m.level == "error" for m in msgs
        )

    def test_repeated_angles_in_one_group_are_allowed(self):
        msgs = validate_vfa_metadata(
            [5.0, 5.0, 15.0],
            [0.004, 0.004, 0.004],
            [
                Path("sub-01_run-1_flip-1_VFA.nii.gz"),
                Path("sub-01_run-1_flip-2_VFA.nii.gz"),
                Path("sub-01_run-1_flip-3_VFA.nii.gz"),
            ],
        )
        assert not any(message.level == "error" for message in msgs)

    @pytest.mark.parametrize("entity", ["acq", "run", "echo"])
    def test_competing_acquisition_entities(self, entity):
        files = [
            Path(f"sub-01_{entity}-1_flip-1_VFA.nii.gz"),
            Path(f"sub-01_{entity}-2_flip-2_VFA.nii.gz"),
        ]
        msgs = validate_vfa_metadata(
            [5.0, 15.0], [0.004, 0.004], files
        )
        assert any(
            m.code == "VFA_AMBIGUOUS_INPUT" and m.level == "error"
            for m in msgs
        )

    def test_inconsistent_tr(self):
        msgs = validate_vfa_metadata([5.0, 15.0], [0.004, 0.006])
        assert any(
            m.code == "VFA_INCONSISTENT_TR" and m.level == "error"
            for m in msgs
        )

    def test_missing_flip_angle(self):
        msgs = validate_vfa_metadata([5.0, None], [0.004, 0.004])
        assert any(
            m.code == "VFA_MISSING_FLIP_ANGLE" and m.level == "error"
            for m in msgs
        )

    def test_missing_tr(self):
        msgs = validate_vfa_metadata([5.0, 15.0], [0.004, None])
        assert any(
            m.code == "VFA_MISSING_TR" and m.level == "error" for m in msgs
        )

    def test_poor_conditioning(self):
        msgs = validate_vfa_metadata([14.0, 15.0], [0.004, 0.004])
        assert any(m.code == "VFA_POOR_CONDITIONING" for m in msgs)


# ===========================================================================
# validate_afi_metadata tests
# ===========================================================================

@pytest.mark.parametrize(
    ("tr1", "tr2", "expected"),
    [
        pytest.param(20.0, 100.0, None, id="valid"),
        pytest.param(None, 100.0, "AFI_MISSING_TR", id="missing-tr1"),
        pytest.param(20.0, None, "AFI_MISSING_TR", id="missing-tr2"),
        pytest.param(100.0, 20.0, "AFI_TR_ORDER", id="reversed"),
        pytest.param(50.0, 50.0, "AFI_TR_ORDER", id="equal"),
    ],
)
def test_validate_afi_metadata(
    tr1: float | None,
    tr2: float | None,
    expected: str | None,
) -> None:
    codes = {message.code for message in validate_afi_metadata(tr1, tr2)}
    assert (expected is None and not codes) or expected in codes


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
            vfa_flip_angles=[15.0, 15.0],
            vfa_trs=[0.004, 0.004],
        )
        assert any(m.code == "VFA_FEW_ANGLES" for m in report.messages)
        assert report.has_errors

    def test_vfa_validation_skipped_when_t1_mapping_disabled(self, tmp_path):
        """Unused discovered VFA metadata does not warn or fail a no-T1 run."""
        dce_file = tmp_path / "sub-01_dce.nii.gz"
        dce_file.touch()
        mods = _make_modalities(
            dce=dce_file,
            vfa=[Path("/fake/only_vfa.nii.gz")],
        )

        report = run_preflight(
            session_data=_make_session_data(mods),
            decision=_make_decision(run_dce=True),
            dsc_sidecar=None,
            dce_sidecar=_valid_dce_sidecar(),
            dce_config=DCEConfig(no_t1=True),
            vfa_flip_angles=[None],
            vfa_trs=[None],
        )

        assert not any(message.code.startswith("VFA_") for message in report.messages)

    def test_afi_validation_included(self, tmp_path):
        """AFI validation runs when AFI data is selected for use."""
        dce_file = tmp_path / "sub-01_dce.nii.gz"
        dce_file.touch()
        (tmp_path / "sub-01_dce.json").touch()

        mods = _make_modalities(
            dce=dce_file,
            afi=[Path("/fake/afi1.nii.gz")],
            afi_4d_file=Path("/fake/afi1.nii.gz"),
            b1_method_detected="afi",
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

    def test_unused_afi_metadata_is_not_validated(self, tmp_path):
        """AFI metadata cannot block a run with B1+ correction disabled."""
        dce_file = tmp_path / "sub-01_dce.nii.gz"
        dce_file.touch()
        mods = _make_modalities(
            dce=dce_file,
            afi=[Path("/fake/afi1.nii.gz")],
            afi_4d_file=Path("/fake/afi1.nii.gz"),
            afi_tr1_ms=None,
            afi_tr2_ms=100.0,
        )

        report = run_preflight(
            session_data=_make_session_data(mods),
            decision=_make_decision(run_dce=True),
            dsc_sidecar=None,
            dce_sidecar=_valid_dce_sidecar(),
            dce_config=DCEConfig(b1_method="none"),
        )

        assert not any(message.code.startswith("AFI_") for message in report.messages)

    @pytest.mark.parametrize(
        ("method", "detected", "candidate_field"),
        [
            ("auto", "precomputed", "b1_precomputed_candidates"),
            ("precomputed", "precomputed", "b1_precomputed_candidates"),
            ("afi", "precomputed", "afi_4d_candidates"),
        ],
    )
    def test_competing_effective_b1_inputs_fail_closed(
        self, tmp_path, method, detected, candidate_field,
    ):
        """Only candidates for the effective B1 method are ambiguity-gated."""
        dce_file = tmp_path / "sub-01_dce.nii.gz"
        dce_file.touch()
        first = tmp_path / "sub-01_run-1_TB1map.nii.gz"
        second = tmp_path / "sub-01_run-2_TB1map.nii.gz"
        values = {
            "dce": dce_file,
            "b1_method_detected": detected,
            "b1_precomputed_file": first,
            "afi": [first, second],
            "afi_4d_file": first,
            candidate_field: [first, second],
        }

        report = run_preflight(
            session_data=_make_session_data(_make_modalities(**values)),
            decision=_make_decision(run_dce=True),
            dsc_sidecar=None,
            dce_sidecar=_valid_dce_sidecar(),
            dce_config=DCEConfig(b1_method=method),
        )

        assert any(
            message.code == "DCE_B1_AMBIGUOUS_INPUT"
            for message in report.messages
        )
        assert report.has_errors

    def test_competing_b1_inputs_do_not_block_disabled_or_skipped_dce(
        self, tmp_path,
    ):
        """Unused B1 candidates do not affect disabled or skipped DCE routes."""
        dce_file = tmp_path / "sub-01_dce.nii.gz"
        dce_file.touch()
        candidates = [Path("/fake/b1a.nii.gz"), Path("/fake/b1b.nii.gz")]
        mods = _make_modalities(
            dce=dce_file,
            b1_method_detected="precomputed",
            b1_precomputed_file=candidates[0],
            b1_precomputed_candidates=candidates,
        )
        for run_dce, method in ((True, "none"), (False, "auto")):
            report = run_preflight(
                session_data=_make_session_data(mods),
                decision=_make_decision(run_dce=run_dce),
                dsc_sidecar=None,
                dce_sidecar=_valid_dce_sidecar(),
                dce_config=DCEConfig(b1_method=method),
            )
            assert not any(
                message.code == "DCE_B1_AMBIGUOUS_INPUT"
                for message in report.messages
            )

    @pytest.mark.parametrize(
        ("method", "modalities", "code"),
        [
            (
                "precomputed",
                {"b1_precomputed_file": None},
                "DCE_B1_PRECOMPUTED_MISSING",
            ),
            (
                "afi",
                {"afi": [], "afi_mode": None, "afi_4d_file": None},
                "DCE_B1_AFI_MISSING",
            ),
            (
                "afi",
                {
                    "afi": [Path("/fake/afi.nii.gz")],
                    "afi_mode": "split",
                    "afi_4d_file": None,
                },
                "DCE_B1_AFI_MISSING",
            ),
        ],
    )
    def test_forced_b1_method_requires_matching_session_input(
        self, tmp_path, method, modalities, code,
    ):
        """Forced B1 modes fail preflight instead of crashing a Nipype node."""
        dce_file = tmp_path / "sub-01_dce.nii.gz"
        dce_file.touch()
        (tmp_path / "sub-01_dce.json").touch()
        mods = _make_modalities(dce=dce_file, **modalities)

        report = run_preflight(
            session_data=_make_session_data(mods),
            decision=_make_decision(run_dce=True),
            dsc_sidecar=None,
            dce_sidecar=_valid_dce_sidecar(),
            dce_config=DCEConfig(b1_method=method),
        )

        assert any(message.code == code for message in report.messages)
        assert report.has_errors

    @pytest.mark.parametrize("method", ["auto", "none"])
    def test_optional_b1_modes_do_not_require_session_input(self, tmp_path, method):
        """Automatic and explicitly disabled B1 modes remain optional."""
        dce_file = tmp_path / "sub-01_dce.nii.gz"
        dce_file.touch()
        (tmp_path / "sub-01_dce.json").touch()

        report = run_preflight(
            session_data=_make_session_data(_make_modalities(dce=dce_file)),
            decision=_make_decision(run_dce=True),
            dsc_sidecar=None,
            dce_sidecar=_valid_dce_sidecar(),
            dce_config=DCEConfig(b1_method=method),
        )

        assert not any(message.code.startswith("DCE_B1_") for message in report.messages)


# ===========================================================================
# DCEConfig r1_relaxivity tests
# ===========================================================================

class TestDCEConfigR1:
    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            pytest.param({}, None, id="default"),
            pytest.param({"r1_relaxivity": 5.0}, 5.0, id="float"),
            pytest.param({"r1_relaxivity": "4.5"}, 4.5, id="numeric-string"),
        ],
    )
    def test_valid_values(self, kwargs: dict, expected: float | None) -> None:
        cfg = DCEConfig(**kwargs)
        assert cfg.r1_relaxivity == expected

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


# ===========================================================================
# Agent-specific preflight guardrails
# ===========================================================================

def _make_dsc_config(**overrides):
    """Create a DSCConfig with sensible defaults for testing."""
    from pmriprep.config.dsc import DSCConfig
    defaults = dict(
        preload="unknown",
        dose=1.0,
        agent_standard_dose=0.1,
    )
    defaults.update(overrides)
    return DSCConfig(**defaults)


class TestDSCAgentGuardrails:

    def test_high_r1_no_preload_warning(self):
        sc = _valid_dsc_sidecar()
        sc["contrast_agent"] = "gadopiclenol"
        sc["flip_angle"] = 30.0
        cfg = _make_dsc_config(preload=0.0, agent_standard_dose=0.05)
        msgs = validate_dsc_metadata(sc, dsc_config=cfg)
        codes = [m.code for m in msgs]
        assert "DSC_HIGH_R1_NO_PRELOAD" in codes
        msg = next(m for m in msgs if m.code == "DSC_HIGH_R1_NO_PRELOAD")
        assert "reduce the non-enhancing reference support" in msg.message
        assert "unvalidated for agents" not in msg.message

    def test_high_r1_gadoquatrane_warning(self):
        sc = _valid_dsc_sidecar()
        sc["contrast_agent"] = "gadoquatrane"
        sc["flip_angle"] = 30.0
        cfg = _make_dsc_config(preload="unknown", agent_standard_dose=0.04)
        msgs = validate_dsc_metadata(sc, dsc_config=cfg)
        codes = [m.code for m in msgs]
        assert "DSC_HIGH_R1_NO_PRELOAD" in codes

    def test_high_r1_with_leakage_disabled_does_not_describe_bsw(self):
        """An intentional correction opt-out reports retained contamination."""
        sc = _valid_dsc_sidecar()
        sc["contrast_agent"] = "gadopiclenol"
        sc["flip_angle"] = 30.0
        cfg = _make_dsc_config(
            preload=0.0,
            agent_standard_dose=0.05,
            leakage_method="none",
        )

        msgs = validate_dsc_metadata(sc, dsc_config=cfg)

        msg = next(m for m in msgs if m.code == "DSC_HIGH_R1_NO_PRELOAD")
        assert "disabled by configuration" in msg.message
        assert "BSW" not in msg.message
        assert "unvalidated for agents" not in msg.message

    def test_high_r1_rejects_unknown_leakage_policy(self):
        """A future policy cannot silently inherit the disabled-policy text."""
        sc = _valid_dsc_sidecar()
        sc["contrast_agent"] = "gadopiclenol"
        cfg = _make_dsc_config(preload=0.0, agent_standard_dose=0.05)
        cfg.leakage_method = "experimental"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="Unsupported DSC leakage_method"):
            validate_dsc_metadata(sc, dsc_config=cfg)

    def test_high_r1_unknown_preload_warning(self):
        sc = _valid_dsc_sidecar()
        sc["contrast_agent"] = "gadopiclenol"
        sc["flip_angle"] = 30.0
        cfg = _make_dsc_config(preload="unknown", agent_standard_dose=0.05)
        msgs = validate_dsc_metadata(sc, dsc_config=cfg)
        codes = [m.code for m in msgs]
        assert "DSC_HIGH_R1_NO_PRELOAD" in codes

    def test_high_r1_with_preload_no_warning(self):
        sc = _valid_dsc_sidecar()
        sc["contrast_agent"] = "gadopiclenol"
        sc["flip_angle"] = 30.0
        cfg = _make_dsc_config(preload=1.0, agent_standard_dose=0.05)
        msgs = validate_dsc_metadata(sc, dsc_config=cfg)
        codes = [m.code for m in msgs]
        assert "DSC_HIGH_R1_NO_PRELOAD" not in codes

    def test_high_r1_user_no_preload_omits_cta(self):
        # User explicitly set preload=0: warning still fires (real shine-through
        # risk) but must not tell them to set the option they already set.
        sc = _valid_dsc_sidecar()
        sc["contrast_agent"] = "gadopiclenol"
        sc["flip_angle"] = 30.0
        cfg = _make_dsc_config(preload=0.0, agent_standard_dose=0.05)
        msgs = validate_dsc_metadata(sc, dsc_config=cfg)
        msg = next(m for m in msgs if m.code == "DSC_HIGH_R1_NO_PRELOAD")
        assert "Use --dsc-preload" not in msg.message
        assert "without confirmed preload" not in msg.message
        assert "no preload (user-specified)" in msg.message

    def test_high_r1_unknown_preload_includes_cta(self):
        # Unknown preload keeps the confirmation call-to-action.
        sc = _valid_dsc_sidecar()
        sc["contrast_agent"] = "gadopiclenol"
        sc["flip_angle"] = 30.0
        cfg = _make_dsc_config(preload="unknown", agent_standard_dose=0.05)
        msgs = validate_dsc_metadata(sc, dsc_config=cfg)
        msg = next(m for m in msgs if m.code == "DSC_HIGH_R1_NO_PRELOAD")
        assert "Use --dsc-preload to confirm preload status." in msg.message
        assert "without confirmed preload" in msg.message

    def test_high_fa_user_no_preload_omits_cta(self):
        # Same fix for the high-FA advisory.
        sc = _valid_dsc_sidecar()
        sc["flip_angle"] = 60.0
        cfg = _make_dsc_config(preload=0.0)
        msgs = validate_dsc_metadata(sc, dsc_config=cfg)
        msg = next(m for m in msgs if m.code == "DSC_HIGH_FA_NO_PRELOAD")
        assert "Use --dsc-preload" not in msg.message
        assert "no preload (user-specified)" in msg.message

    def test_high_fa_no_preload_warning(self):
        sc = _valid_dsc_sidecar()
        sc["flip_angle"] = 60.0
        cfg = _make_dsc_config(preload=0.0)
        msgs = validate_dsc_metadata(sc, dsc_config=cfg)
        codes = [m.code for m in msgs]
        assert "DSC_HIGH_FA_NO_PRELOAD" in codes

    def test_high_fa_with_preload_no_warning(self):
        sc = _valid_dsc_sidecar()
        sc["flip_angle"] = 60.0
        cfg = _make_dsc_config(preload=1.0)
        msgs = validate_dsc_metadata(sc, dsc_config=cfg)
        codes = [m.code for m in msgs]
        assert "DSC_HIGH_FA_NO_PRELOAD" not in codes

    def test_normal_fa_no_warning(self):
        sc = _valid_dsc_sidecar()
        sc["flip_angle"] = 30.0
        cfg = _make_dsc_config(preload=0.0)
        msgs = validate_dsc_metadata(sc, dsc_config=cfg)
        codes = [m.code for m in msgs]
        assert "DSC_HIGH_FA_NO_PRELOAD" not in codes

    def test_low_dose_warning(self):
        sc = _valid_dsc_sidecar()
        sc["contrast_agent"] = "gadopiclenol"
        sc["flip_angle"] = 30.0
        cfg = _make_dsc_config(dose=0.5, agent_standard_dose=0.05)
        msgs = validate_dsc_metadata(sc, dsc_config=cfg)
        codes = [m.code for m in msgs]
        assert "DSC_LOW_EFFECTIVE_DOSE" in codes

    def test_low_dose_gadoquatrane_warning(self):
        sc = _valid_dsc_sidecar()
        sc["contrast_agent"] = "gadoquatrane"
        sc["flip_angle"] = 30.0
        cfg = _make_dsc_config(dose=1.0, agent_standard_dose=0.04)
        msgs = validate_dsc_metadata(sc, dsc_config=cfg)
        codes = [m.code for m in msgs]
        assert "DSC_LOW_EFFECTIVE_DOSE" in codes

    def test_full_dose_no_warning(self):
        sc = _valid_dsc_sidecar()
        sc["contrast_agent"] = "gadobutrol"
        sc["flip_angle"] = 30.0
        cfg = _make_dsc_config(dose=1.0, agent_standard_dose=0.1)
        msgs = validate_dsc_metadata(sc, dsc_config=cfg)
        codes = [m.code for m in msgs]
        assert "DSC_LOW_EFFECTIVE_DOSE" not in codes

    def test_standard_agent_no_warnings(self):
        sc = _valid_dsc_sidecar()
        sc["contrast_agent"] = "gadobutrol"
        sc["flip_angle"] = 30.0
        cfg = _make_dsc_config(preload="unknown")
        msgs = validate_dsc_metadata(sc, dsc_config=cfg)
        agent_codes = {m.code for m in msgs if m.code.startswith("DSC_H") or m.code.startswith("DSC_LOW")}
        assert len(agent_codes) == 0

    def test_no_agent_no_warnings(self):
        sc = _valid_dsc_sidecar()
        sc["contrast_agent"] = None
        cfg = _make_dsc_config(preload="unknown")
        msgs = validate_dsc_metadata(sc, dsc_config=cfg)
        agent_codes = {m.code for m in msgs if "AGENT" in m.code or "R1" in m.code or "DOSE" in m.code}
        assert len(agent_codes) == 0

    def test_no_config_no_crash(self):
        sc = _valid_dsc_sidecar()
        sc["contrast_agent"] = "gadopiclenol"
        msgs = validate_dsc_metadata(sc, dsc_config=None)
        codes = [m.code for m in msgs]
        assert "DSC_HIGH_R1_NO_PRELOAD" not in codes
        assert "DSC_HIGH_FA_NO_PRELOAD" not in codes
        assert "DSC_LOW_EFFECTIVE_DOSE" not in codes


class TestDCEAgentGuardrails:

    def test_high_r1_measured_aif_warning(self):
        sc = _valid_dce_sidecar()
        sc["contrast_agent"] = "gadopiclenol"
        cfg = DCEConfig(aif_method="auto")
        msgs = validate_dce_metadata(sc, cfg)
        codes = [m.code for m in msgs]
        assert "DCE_HIGH_R1_AIF_SATURATION" in codes

    def test_high_r1_population_aif_no_warning(self):
        sc = _valid_dce_sidecar()
        sc["contrast_agent"] = "gadopiclenol"
        cfg = DCEConfig(aif_method="population")
        msgs = validate_dce_metadata(sc, cfg)
        codes = [m.code for m in msgs]
        assert "DCE_HIGH_R1_AIF_SATURATION" not in codes

    def test_high_r1_external_aif_no_warning(self):
        sc = _valid_dce_sidecar()
        sc["contrast_agent"] = "gadopiclenol"
        cfg = DCEConfig(aif_method="auto", aif_file=Path("/tmp/aif.txt"))
        msgs = validate_dce_metadata(sc, cfg)
        codes = [m.code for m in msgs]
        assert "DCE_HIGH_R1_AIF_SATURATION" not in codes

    def test_low_dose_warning(self):
        """Gd-DTPA quarter-dose: 0.1 * 0.25 * 3.7 = 0.093 < 0.15."""
        sc = _valid_dce_sidecar()
        sc["contrast_agent"] = "gd-dtpa"
        cfg = DCEConfig(dose=0.25)
        msgs = validate_dce_metadata(sc, cfg)
        codes = [m.code for m in msgs]
        assert "DCE_LOW_EFFECTIVE_DOSE" in codes

    def test_high_r1_standard_dose_no_warning(self):
        """Gadopiclenol at standard dose (0.05 mmol/kg): 0.05 × 11.6 = 0.58 > 0.15."""
        sc = _valid_dce_sidecar()
        sc["contrast_agent"] = "gadopiclenol"
        cfg = DCEConfig(dose=1.0, aif_method="population")
        msgs = validate_dce_metadata(sc, cfg)
        codes = [m.code for m in msgs]
        assert "DCE_LOW_EFFECTIVE_DOSE" not in codes

    def test_full_dose_no_warning(self):
        sc = _valid_dce_sidecar()
        sc["contrast_agent"] = "gadobutrol"
        cfg = DCEConfig(dose=1.0)
        msgs = validate_dce_metadata(sc, cfg)
        codes = [m.code for m in msgs]
        assert "DCE_LOW_EFFECTIVE_DOSE" not in codes


# ===========================================================================
# Temporal sufficiency tests
# ===========================================================================

@pytest.mark.parametrize(
    ("modality", "n_volumes", "tr", "expected", "forbidden", "error"),
    [
        pytest.param(
            "dsc", 20, 1.5, ("TOO_FEW_VOLUMES", "SHORT_DURATION"), (),
            "TOO_FEW_VOLUMES", id="dsc-too-few-and-short",
        ),
        pytest.param(
            "dsc", 30, 1.5, ("MARGINAL_VOLUMES", "MARGINAL_DURATION"),
            ("TOO_FEW_VOLUMES",), None, id="dsc-threshold-marginal",
        ),
        pytest.param(
            "dsc", 35, None, ("MARGINAL_VOLUMES",), ("DURATION",), None,
            id="dsc-marginal-without-tr",
        ),
        pytest.param(
            "dsc", 40, 1.5, (), ("VOLUME", "DURATION"), None,
            id="dsc-exact-upper-boundaries",
        ),
        pytest.param(
            "dsc", 20, 2.0, ("TOO_FEW_VOLUMES", "MARGINAL_DURATION"),
            ("SHORT_DURATION",), "TOO_FEW_VOLUMES", id="dsc-exact-minimum-duration",
        ),
        pytest.param(
            "dsc", 20, None, ("TOO_FEW_VOLUMES",), ("DURATION",),
            "TOO_FEW_VOLUMES", id="dsc-too-few-without-tr",
        ),
        pytest.param(
            "dce", 10, 7.0, ("TOO_FEW_VOLUMES",), (), "TOO_FEW_VOLUMES",
            id="dce-too-few",
        ),
        pytest.param(
            "dce", 15, 7.0, ("MARGINAL_VOLUMES", "MARGINAL_DURATION"),
            ("TOO_FEW_VOLUMES",), None, id="dce-threshold-marginal",
        ),
        pytest.param(
            "dce", 18, None, ("MARGINAL_VOLUMES",), ("DURATION",), None,
            id="dce-marginal-without-tr",
        ),
        pytest.param(
            "dce", 20, 6.0, (), ("VOLUME", "DURATION"), None,
            id="dce-exact-upper-boundaries",
        ),
        pytest.param("dce", 5, 7.0, ("SHORT_DURATION",), (), None, id="dce-short"),
        pytest.param(
            "dce", 10, 6.0, ("TOO_FEW_VOLUMES", "MARGINAL_DURATION"),
            ("SHORT_DURATION",), "TOO_FEW_VOLUMES", id="dce-exact-minimum-duration",
        ),
        pytest.param(
            "dce", 10, None, ("TOO_FEW_VOLUMES",), ("DURATION",),
            "TOO_FEW_VOLUMES", id="dce-too-few-without-tr",
        ),
    ],
)
def test_temporal_sufficiency_boundaries(
    modality: str,
    n_volumes: int,
    tr: float | None,
    expected: tuple[str, ...],
    forbidden: tuple[str, ...],
    error: str | None,
) -> None:
    """DSC and DCE retain their distinct volume and duration boundaries."""
    if modality == "dsc":
        messages = validate_dsc_temporal_sufficiency(n_volumes, tr=tr)
    else:
        messages = validate_dce_temporal_sufficiency(n_volumes, volume_tr=tr)

    prefix = modality.upper()
    codes = {message.code for message in messages}
    assert {f"{prefix}_{suffix}" for suffix in expected} <= codes
    assert not any(fragment in code for fragment in forbidden for code in codes)
    if error is not None:
        assert f"{prefix}_{error}" in {
            message.code for message in messages if message.level == "error"
        }


class TestWarningStrings:

    def test_all_warnings(self):
        report = PreflightReport(messages=[
            PreflightMessage("warning", "DSC_HIGH_R1_NO_PRELOAD", "msg1"),
            PreflightMessage("error", "DSC_MISSING_TE", "msg2"),
            PreflightMessage("warning", "DCE_HIGH_R1_AIF_SATURATION", "msg3"),
        ])
        result = report.warning_strings()
        assert len(result) == 2
        assert "[DSC_HIGH_R1_NO_PRELOAD]" in result[0]
        assert "[DCE_HIGH_R1_AIF_SATURATION]" in result[1]

    def test_prefix_filter(self):
        report = PreflightReport(messages=[
            PreflightMessage("warning", "DSC_HIGH_R1_NO_PRELOAD", "msg1"),
            PreflightMessage("warning", "DCE_HIGH_R1_AIF_SATURATION", "msg2"),
        ])
        dsc = report.warning_strings(prefix="DSC_")
        dce = report.warning_strings(prefix="DCE_")
        assert len(dsc) == 1
        assert len(dce) == 1

    def test_empty(self):
        report = PreflightReport()
        assert report.warning_strings() == []
