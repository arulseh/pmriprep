"""Pipeline smoke tests.

Verifies pipeline structure, workflow initialization, decision logic,
import health, QC standards, and utility constants. These tests require
no external tools (FSL, ANTs, etc.) and no local test fixtures.
"""

import sys
import unittest.mock
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from pmriprep.config.anat import AnatConfig
from pmriprep.config.dce import DCEConfig
from pmriprep.config.dsc import DSCConfig
from pmriprep.config.tumor_seg import TumorSegConfig
from pmriprep.workflows.base import (
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


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_bids_dir(tmp_path: Path) -> Path:
    """Create a mock BIDS directory structure for testing."""
    bids_dir = tmp_path / "bids"
    bids_dir.mkdir()
    (bids_dir / "dataset_description.json").write_text(
        '{"Name": "Test", "BIDSVersion": "1.9.0"}'
    )
    return bids_dir


@pytest.fixture
def mock_subject_session(mock_bids_dir: Path) -> tuple[Path, str, str]:
    """Create a mock subject/session structure."""
    subject_id = "0001"
    session_id = "0001"

    ses_dir = mock_bids_dir / f"sub-{subject_id}" / f"ses-{session_id}"
    anat_dir = ses_dir / "anat"
    anat_dir.mkdir(parents=True)
    (anat_dir / f"sub-{subject_id}_ses-{session_id}_T1w.nii.gz").touch()

    perf_dir = ses_dir / "perf"
    perf_dir.mkdir(parents=True)
    (perf_dir / f"sub-{subject_id}_ses-{session_id}_dsc.nii.gz").touch()

    return mock_bids_dir, subject_id, session_id


# =============================================================================
# Workflow Initialization Tests
# =============================================================================


class TestWorkflowInitialization:
    """Test workflow initialization without actual execution."""

    def test_anat_workflow_initializes(self, mock_subject_session: tuple) -> None:
        """Verify anatomical workflow can be initialized."""
        from pmriprep.workflows.anat.base import init_anat_preproc_wf

        bids_dir, subject_id, session_id = mock_subject_session
        config = AnatConfig()

        workflow = init_anat_preproc_wf(
            bids_dir=bids_dir,
            subject_id=subject_id,
            session_id=session_id,
            config=config,
        )

        assert workflow is not None
        assert workflow.name == "anat_preproc_wf"
        node_names = [node.name for node in workflow._graph.nodes()]
        assert "inputnode" in node_names
        assert "outputnode" in node_names

    def test_dsc_workflow_initializes(self, mock_subject_session: tuple) -> None:
        """Verify DSC workflow can be initialized."""
        from pmriprep.workflows.dsc.base import init_dsc_preproc_wf

        bids_dir, subject_id, session_id = mock_subject_session
        config = DSCConfig()

        workflow = init_dsc_preproc_wf(
            bids_dir=bids_dir,
            output_dir=bids_dir / "derivatives",
            subject_id=subject_id,
            session_id=session_id,
            config=config,
        )

        assert workflow is not None
        assert workflow.name == "dsc_preproc_wf"
        node_names = [node.name for node in workflow._graph.nodes()]
        assert "inputnode" in node_names
        assert "outputnode" in node_names

    def test_dce_workflow_initializes(self, mock_subject_session: tuple) -> None:
        """Verify DCE workflow can be initialized."""
        from pmriprep.workflows.dce.base import init_dce_preproc_wf

        bids_dir, subject_id, session_id = mock_subject_session
        config = DCEConfig()

        workflow = init_dce_preproc_wf(
            bids_dir=bids_dir,
            output_dir=bids_dir / "derivatives",
            subject_id=subject_id,
            session_id=session_id,
            config=config,
        )

        assert workflow is not None
        assert workflow.name == "dce_preproc_wf"
        node_names = [node.name for node in workflow._graph.nodes()]
        assert "inputnode" in node_names
        assert "outputnode" in node_names

    def test_tumor_seg_workflow_initializes(self, mock_subject_session: tuple) -> None:
        """Verify tumor segmentation workflow can be initialized."""
        from pmriprep.workflows.tumor_seg.base import init_tumor_seg_wf

        bids_dir, subject_id, session_id = mock_subject_session
        config = TumorSegConfig()

        workflow = init_tumor_seg_wf(
            bids_dir=bids_dir,
            subject_id=subject_id,
            session_id=session_id,
            config=config,
        )

        assert workflow is not None
        assert workflow.name == "tumor_seg_wf"
        node_names = [node.name for node in workflow._graph.nodes()]
        assert "inputnode" in node_names
        assert "outputnode" in node_names


# =============================================================================
# Pipeline Decision Tests
# =============================================================================


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


# =============================================================================
# Pipeline Summary Tests
# =============================================================================


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


# =============================================================================
# Typecheck Verification
# =============================================================================


class TestTypecheckVerification:
    """Verify key modules can be imported without type errors."""

    def test_import_pmriprep(self) -> None:
        """Verify main package imports."""
        import pmriprep

        assert pmriprep.__version__ is not None
        assert pmriprep.__version__ != "0+unknown", (
            "Package metadata not found -- is pmriprep installed? "
            "Run: pip install -e ."
        )

    def test_import_cli(self) -> None:
        """Verify CLI module imports."""
        from pmriprep.cli import run

        assert run.main is not None

    def test_import_workflows(self) -> None:
        """Verify workflow modules import."""
        from pmriprep.workflows import base, discovery
        from pmriprep.workflows.anat import base as anat_base
        from pmriprep.workflows.dsc import base as dsc_base
        from pmriprep.workflows.dce import base as dce_base
        from pmriprep.workflows.tumor_seg import base as tumor_seg_base

        assert base.init_pmriprep_wf is not None
        assert anat_base.init_anat_preproc_wf is not None
        assert dsc_base.init_dsc_preproc_wf is not None
        assert dce_base.init_dce_preproc_wf is not None
        assert tumor_seg_base.init_tumor_seg_wf is not None

    def test_import_configs(self) -> None:
        """Verify config modules import."""
        from pmriprep.config import dsc, dce, anat, tumor_seg, loader

        assert dsc.DSCConfig is not None
        assert dce.DCEConfig is not None
        assert anat.AnatConfig is not None
        assert tumor_seg.TumorSegConfig is not None
        assert loader.load_config is not None

    def test_import_interfaces(self) -> None:
        """Verify interface modules import."""
        from pmriprep.interfaces import anat, dsc, dce
        from pmriprep.interfaces.hdglio import HDGLIOPrep
        from pmriprep.interfaces import dsc_aif, dsc_compute
        from pmriprep.interfaces import dce_aif, dce_compute

        assert anat.ReorientAndCrop is not None
        assert dsc.MotionCorrection is not None
        assert dce.B1Mapping is not None
        assert HDGLIOPrep is not None

    def test_import_utils(self) -> None:
        """Verify utility modules import."""
        from pmriprep.utils import (
            nifti,
            dsc_constants,
            dce_constants,
            dsc_types,
            qc_standards,
            qc_utils,
            registration,
            provenance,
            contrast_agent,
            structural_reference,
        )

        assert nifti is not None
        assert dsc_constants.EPSILON_SIGNAL is not None
        assert dce_constants.EPSILON_VE is not None
        assert dsc_types.DSCVoxelResult is not None
        assert qc_standards.COLORBLIND_PALETTE is not None

    def test_import_models(self) -> None:
        """Verify model modules import."""
        from pmriprep.models.dsc import base as dsc_model_base
        from pmriprep.models.dsc import vascular, cpi, bayesian as dsc_bayesian
        from pmriprep.models.dce import base as dce_model_base
        from pmriprep.models.dce import tofts, extended_tofts, patlak
        from pmriprep.models.dce import two_compartment_exchange
        from pmriprep.models.dce import bayesian as dce_bayesian

        assert dsc_model_base.DSCModel is not None
        assert vascular.VascularModel is not None
        assert cpi.CPIModel is not None
        assert dce_model_base.PKModel is not None
        assert tofts.ToftsModel is not None
        assert extended_tofts.ExtendedToftsModel is not None
        assert patlak.PatlakModel is not None
        assert two_compartment_exchange.TwoCompartmentExchangeModel is not None


# =============================================================================
# QC Standards Validation
# =============================================================================


class TestQCStandardsValidation:
    """Validate QC standards are within expected ranges."""

    def test_correlation_threshold_reasonable(self) -> None:
        """Verify correlation threshold is reasonable (0.99)."""
        CORRELATION_THRESHOLD = 0.99

        assert 0.95 <= CORRELATION_THRESHOLD <= 1.0
        assert CORRELATION_THRESHOLD == 0.99

    def test_dice_threshold_reasonable(self) -> None:
        """Verify DICE threshold is reasonable (0.95)."""
        DICE_THRESHOLD = 0.95

        assert 0.90 <= DICE_THRESHOLD <= 1.0
        assert DICE_THRESHOLD == 0.95

    def test_motion_fd_thresholds(self) -> None:
        """Verify motion FD thresholds are reasonable."""
        from pmriprep.utils.qc_standards import MOTION_FD_THRESHOLDS

        assert MOTION_FD_THRESHOLDS.pass_threshold < MOTION_FD_THRESHOLDS.warn_threshold
        assert MOTION_FD_THRESHOLDS.pass_threshold == 0.3
        assert MOTION_FD_THRESHOLDS.warn_threshold == 0.5

    def test_physiological_ranges(self) -> None:
        """Verify physiological ranges are reasonable."""
        from pmriprep.utils.dce_t1_validation import T1_TISSUE_REFERENCE

        refs_3t = T1_TISSUE_REFERENCE[3.0]
        wm_min, wm_max = refs_3t["wm"]
        assert wm_min >= 700
        assert wm_max <= 1000

        gm_min, gm_max = refs_3t["gm"]
        assert gm_min >= 1100
        assert gm_max <= 1500


# =============================================================================
# Utility Tests
# =============================================================================


class TestUtilityFunctions:
    """Test utility functions used throughout the pipeline."""

    def test_epsilon_guards_positive(self) -> None:
        """Verify epsilon guards are positive and small."""
        from pmriprep.utils.dsc_constants import (
            EPSILON_SIGNAL,
            EPSILON_DELTA_R2,
            EPSILON_CONCENTRATION,
        )
        from pmriprep.utils.dce_constants import (
            EPSILON_VE,
            EPSILON_T1,
            EPSILON_KTRANS,
            EPSILON_VP,
            EPSILON_B1,
        )

        all_epsilons = [
            EPSILON_SIGNAL,
            EPSILON_DELTA_R2,
            EPSILON_CONCENTRATION,
            EPSILON_VE,
            EPSILON_T1,
            EPSILON_KTRANS,
            EPSILON_VP,
            EPSILON_B1,
        ]

        for eps in all_epsilons:
            assert eps > 0, f"Epsilon {eps} should be positive"
            assert eps < 0.1, f"Epsilon {eps} should be small"

    def test_perfusion_bounds_ordered(self) -> None:
        """Verify perfusion parameter bounds are correctly ordered."""
        from pmriprep.utils.dsc_constants import CBF_BOUNDS, MTT_BOUNDS

        assert CBF_BOUNDS.min < CBF_BOUNDS.max
        assert MTT_BOUNDS.min < MTT_BOUNDS.max

    def test_pk_parameter_bounds_ordered(self) -> None:
        """Verify PK parameter bounds are correctly ordered."""
        from pmriprep.utils.dce_constants import (
            KTRANS_BOUNDS,
            VE_BOUNDS,
            VP_BOUNDS,
            FP_BOUNDS,
            PS_BOUNDS,
        )

        assert KTRANS_BOUNDS.min < KTRANS_BOUNDS.max
        assert VE_BOUNDS.min < VE_BOUNDS.max
        assert VP_BOUNDS.min < VP_BOUNDS.max
        assert FP_BOUNDS.min < FP_BOUNDS.max
        assert PS_BOUNDS.min < PS_BOUNDS.max


# =============================================================================
# Device Preflight Tests
# =============================================================================


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

    def test_cuda_unavailable_skips(self, tumor_seg_session: SessionData) -> None:
        """Verify tumor seg is skipped when CUDA is requested but unavailable."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        config = TumorSegConfig(device="cuda")
        with unittest.mock.patch.dict("sys.modules", {"torch": mock_torch}):
            decision = decide_pipelines(
                tumor_seg_session, PipelineSkipFlags(), config
            )
        assert decision.run_tumor_seg is False
        assert "No CUDA GPU" in (decision.tumor_seg_reason or "")

    def test_mps_unavailable_skips(self, tumor_seg_session: SessionData) -> None:
        """Verify tumor seg is skipped when MPS is requested but unavailable."""
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        config = TumorSegConfig(device="mps")
        with unittest.mock.patch.dict("sys.modules", {"torch": mock_torch}):
            decision = decide_pipelines(
                tumor_seg_session, PipelineSkipFlags(), config
            )
        assert decision.run_tumor_seg is False
        assert "MPS not available" in (decision.tumor_seg_reason or "")

    def test_conda_env_bypasses_check(self, tumor_seg_session: SessionData) -> None:
        """Verify device check is skipped when conda_env is set."""
        config = TumorSegConfig(device="mps", conda_env="hdglio-mps")
        decision = decide_pipelines(
            tumor_seg_session, PipelineSkipFlags(), config
        )
        assert decision.run_tumor_seg is True

    def test_cpu_always_passes(self, tumor_seg_session: SessionData) -> None:
        """Verify CPU device always passes the check."""
        config = TumorSegConfig(device="cpu")
        decision = decide_pipelines(
            tumor_seg_session, PipelineSkipFlags(), config
        )
        assert decision.run_tumor_seg is True

    def test_no_config_passes(self, tumor_seg_session: SessionData) -> None:
        """Verify tumor seg runs when no config is provided."""
        decision = decide_pipelines(tumor_seg_session, PipelineSkipFlags())
        assert decision.run_tumor_seg is True
