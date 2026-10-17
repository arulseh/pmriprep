"""Pipeline decisions, summaries, interface exports and device preflight."""

import unittest.mock
from unittest.mock import MagicMock

import pytest

from pmriprep.config.anat import AnatConfig
from pmriprep.config.tumor_seg import TumorSegConfig
from pmriprep.workflows.base import (
    _resolve_anat_brain_extraction_tool,
    PipelineDecision,
    PipelineSkipFlags,
    decide_pipelines,
    get_pipeline_summary,
)
from pmriprep.workflows.discovery import (
    ModalityAvailability,
    PipelineAvailability,
    SessionData,
)


class TestPipelineDecisions:
    """Test pipeline decision logic."""

    def test_anat_always_runs_by_default(self) -> None:
        """Verify anat runs unless explicitly skipped."""
        session_data = SessionData(
            subject="0001",
            session="0001",
            modalities=ModalityAvailability(),
            pipelines=PipelineAvailability(),
        )
        skip_flags = PipelineSkipFlags()

        decision = decide_pipelines(session_data, skip_flags)

        assert decision.run_anat is True

    def test_anat_skipped_with_flag(self) -> None:
        """Verify anat skipped when flag is set."""
        session_data = SessionData(
            subject="0001",
            session="0001",
            modalities=ModalityAvailability(),
            pipelines=PipelineAvailability(),
        )
        skip_flags = PipelineSkipFlags(skip_anat=True)

        decision = decide_pipelines(session_data, skip_flags)

        assert decision.run_anat is False
        assert "skip-anat" in str(decision.anat_reason).lower()

    def test_dsc_requires_data(self) -> None:
        """Verify DSC doesn't run without data."""
        session_data = SessionData(
            subject="0001",
            session="0001",
            modalities=ModalityAvailability(),
            pipelines=PipelineAvailability(dsc=False, dsc_reason="No DSC data"),
        )
        skip_flags = PipelineSkipFlags()

        decision = decide_pipelines(session_data, skip_flags)

        assert decision.run_dsc is False

    def test_dce_requires_data(self) -> None:
        """Verify DCE doesn't run without data."""
        session_data = SessionData(
            subject="0001",
            session="0001",
            modalities=ModalityAvailability(),
            pipelines=PipelineAvailability(dce=False, dce_reason="No DCE data"),
        )
        skip_flags = PipelineSkipFlags()

        decision = decide_pipelines(session_data, skip_flags)

        assert decision.run_dce is False

    def test_tumor_seg_requires_four_modalities(self) -> None:
        """Verify tumor segmentation doesn't run without all 4 modalities."""
        session_data = SessionData(
            subject="0001",
            session="0001",
            modalities=ModalityAvailability(),
            pipelines=PipelineAvailability(
                tumor_seg=False,
                tumor_seg_reason="Missing modalities for tumor segmentation",
            ),
        )
        skip_flags = PipelineSkipFlags()

        decision = decide_pipelines(session_data, skip_flags)

        assert decision.run_tumor_seg is False


class TestPipelineSummary:
    """Test pipeline summary generation."""

    def test_summary_contains_all_pipelines(self) -> None:
        """Verify summary includes all pipeline statuses."""
        mock_workflow = MagicMock()
        mock_workflow.pipeline_decision = PipelineDecision(
            run_anat=True,
            run_tumor_seg=False,
            run_dsc=True,
            run_dce=False,
            tumor_seg_reason="Missing modalities",
            dce_reason="No DCE data",
        )

        summary = get_pipeline_summary(mock_workflow)

        assert "Anatomical" in summary
        assert "Tumor segmentation" in summary
        assert "DSC" in summary
        assert "DCE" in summary
        assert "WILL RUN" in summary
        assert "SKIPPED" in summary


class TestInterfaceExports:
    """Public AIF aliases resolve to their modality-specific interfaces."""

    def test_aif_interface_exports(self) -> None:
        import pmriprep.interfaces as interfaces
        from pmriprep.interfaces import dsc_aif, dce_aif

        assert interfaces.DSCAIFDetection is dsc_aif.AIFDetection
        assert interfaces.DCEAIFDetection is dce_aif.AIFDetection


class TestDevicePreflight:
    """Preflight GPU/device checks for tumor segmentation."""

    @pytest.fixture
    def tumor_seg_session(self) -> SessionData:
        """SessionData with all 4 modalities so tumor_seg is eligible."""
        return SessionData(
            subject="0001",
            session="0001",
            modalities=ModalityAvailability(),
            pipelines=PipelineAvailability(tumor_seg=True, tumor_seg_reason=""),
        )

    def test_cuda_unavailable_skips(
        self, tumor_seg_session: SessionData, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify tumor seg is skipped when CUDA is requested but unavailable."""
        monkeypatch.setattr("pmriprep.workflows.base.shutil.which", lambda _: "/bin/tool")
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        config = TumorSegConfig(device="cuda")
        with unittest.mock.patch.dict("sys.modules", {"torch": mock_torch}):
            decision = decide_pipelines(
                tumor_seg_session, PipelineSkipFlags(), config
            )
        assert decision.run_tumor_seg is False
        assert "No CUDA GPU" in (decision.tumor_seg_reason or "")

    def test_conda_env_bypasses_check(self, tumor_seg_session: SessionData) -> None:
        """Verify device check is skipped when conda_env is set."""
        config = TumorSegConfig(device="mps", conda_env="tumor-seg-runtime")
        decision = decide_pipelines(
            tumor_seg_session, PipelineSkipFlags(), config
        )
        assert decision.run_tumor_seg is True

    def test_mps_unavailable_skips(
        self, tumor_seg_session: SessionData, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify MPS requires support from the configured runtime."""
        monkeypatch.setattr("pmriprep.workflows.base.shutil.which", lambda _: "/bin/tool")
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        config = TumorSegConfig(device="mps")
        with unittest.mock.patch.dict("sys.modules", {"torch": mock_torch}):
            decision = decide_pipelines(
                tumor_seg_session, PipelineSkipFlags(), config
            )
        assert decision.run_tumor_seg is False
        assert "MPS is unavailable" in (decision.tumor_seg_reason or "")

    def test_cpu_always_passes(
        self, tumor_seg_session: SessionData, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify CPU device always passes the check."""
        monkeypatch.setattr("pmriprep.workflows.base.shutil.which", lambda _: "/bin/tool")
        config = TumorSegConfig(device="cpu")
        decision = decide_pipelines(
            tumor_seg_session, PipelineSkipFlags(), config
        )
        assert decision.run_tumor_seg is True

    def test_missing_hdglio_skips(self, tumor_seg_session: SessionData) -> None:
        """Missing optional HD-GLIO is reported and skipped before execution."""
        config = TumorSegConfig(device="cpu")
        with unittest.mock.patch(
            "pmriprep.workflows.base.shutil.which", return_value=None
        ):
            decision = decide_pipelines(
                tumor_seg_session, PipelineSkipFlags(), config
            )
        assert decision.run_tumor_seg is False
        assert decision.tumor_seg_reason == (
            "HD-GLIO is not installed; tumor segmentation is unavailable."
        )

    def test_missing_hdbet_selects_ants(self) -> None:
        """Missing optional host HD-BET resolves to the required ANTs tool."""
        config = AnatConfig(tool="hdbet")
        with unittest.mock.patch(
            "pmriprep.workflows.base.shutil.which", return_value=None
        ):
            resolved = _resolve_anat_brain_extraction_tool(config)
        assert resolved.tool == "ants"
        assert config.tool == "hdbet"

    def test_hdbet_conda_env_remains_strict(self) -> None:
        """An explicit HD-BET environment is not replaced from host state."""
        config = AnatConfig(tool="hdbet", hdbet_conda_env="hdbet-runtime")
        with unittest.mock.patch(
            "pmriprep.workflows.base.shutil.which", return_value=None
        ):
            resolved = _resolve_anat_brain_extraction_tool(config)
        assert resolved.tool == "hdbet"

    def test_no_config_passes(self, tumor_seg_session: SessionData) -> None:
        """Verify tumor seg runs when no config is provided."""
        decision = decide_pipelines(tumor_seg_session, PipelineSkipFlags())
        assert decision.run_tumor_seg is True
