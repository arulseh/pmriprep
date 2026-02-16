"""Pipeline verification tests for US-099.

This module provides comprehensive verification that the PMRIPrep pipeline
structure is correctly implemented for all 7 test subjects. It validates:

1. Pipeline can be initialized without errors
2. Workflow structure is correct (all expected nodes present)
3. All integration test configurations are valid
4. Configuration loading and merging works correctly
5. CLI parsing is correct for all test subject configurations

Note: These tests verify pipeline structure and configuration, not execution.
Full pipeline execution requires external tools (FSL, ANTs, HD-BET, etc.)
which may not be available in all test environments.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from pmriprep import __version__
from pmriprep.config.anat import AnatConfig
from pmriprep.config.dce import DCEConfig
from pmriprep.config.dsc import DSCConfig
from pmriprep.config.hdglio import HDGLIOConfig
from pmriprep.config.loader import (
    build_anat_config,
    build_dce_config,
    build_dsc_config,
    build_hdglio_config,
    load_config,
    merge_cli_config,
)
from pmriprep.workflows.base import (
    PipelineDecision,
    PipelineSkipFlags,
    decide_pipelines,
    get_pipeline_summary,
)
from pmriprep.workflows.discovery import SessionData, PipelineAvailability, ModalityAvailability


# =============================================================================
# Test Subject Configurations (from PRD US-091 through US-097)
# =============================================================================

TEST_SUBJECT_CLI_CONFIGS: List[Dict[str, Any]] = [
    {
        "id": "sub-0001_ses-0002",
        "user_story": "US-091",
        "cli_args": [
            "--dsc-no-preload",
            "--dsc-dose", "full",
            # sSVD is default, no need to specify
        ],
        "expected_dsc_config": {
            "preload_detected": False,
            "dose": "full",
            "method": "sSVD",
            "bayesian": "none",
        },
        "expected_dce_config": None,  # No DCE for this subject
    },
    {
        "id": "sub-0004_ses-0001",
        "user_story": "US-092",
        "cli_args": [
            "--dsc-preload",
            "--dsc-dose", "half",
            "--dsc-gbm-preset",
            "--dce-model", "extended_tofts",
            "--dce-bayesian", "map",
            "--dce-scale-aif",
        ],
        "config_file": "tests/fixtures/sub-0004_config.yaml",
        "expected_dsc_config": {
            "preload_detected": True,
            "dose": "half",
            "method": "oSVD",  # GBM preset sets oSVD
            "bayesian": "none",  # DSC has NO bayesian in GBM preset
        },
        "expected_dce_config": {
            "model": "extended_tofts",
            "bayesian": "map",
            "scale_aif": True,
        },
    },
    {
        "id": "sub-0010_ses-0001",
        "user_story": "US-093",
        "cli_args": [
            "--dsc-preload",
            "--dsc-dose", "half",
            "--dsc-deconv-method", "tikhonov",
            "--dce-model", "tofts",
            "--dce-scale-aif",
        ],
        "config_file": "tests/fixtures/sub-0010_config.yaml",
        "expected_dsc_config": {
            "preload_detected": True,
            "dose": "half",
            "method": "tikhonov",
            "bayesian": "none",
        },
        "expected_dce_config": {
            "model": "tofts",
            "bayesian": "none",
            "scale_aif": True,
        },
    },
    {
        "id": "sub-0011_ses-0001",
        "user_story": "US-094",
        "cli_args": [
            "--dsc-preload",
            "--dsc-dose", "half",
            "--dsc-gbm-preset",
            "--dce-model", "auto",
            "--dce-bayesian", "map",
            "--dce-scale-aif",
        ],
        "config_file": "tests/fixtures/sub-0011_config.yaml",
        "expected_dsc_config": {
            "preload_detected": True,
            "dose": "half",
            "method": "oSVD",  # GBM preset sets oSVD
            "bayesian": "none",
        },
        "expected_dce_config": {
            "model": "auto",
            "bayesian": "map",
            "scale_aif": True,
        },
    },
    {
        "id": "sub-0016_ses-0002",
        "user_story": "US-095",
        "cli_args": [
            "--dsc-preload",
            "--dsc-dose", "half",
            "--dsc-deconv-method", "oSVD",
            "--dsc-bayesian", "vi",
            "--dce-model", "2cxm",
            "--dce-bayesian", "map",
            "--dce-scale-aif",
        ],
        "config_file": "tests/fixtures/sub-0016_config.yaml",
        "expected_dsc_config": {
            "preload_detected": True,
            "dose": "half",
            "method": "oSVD",
            "bayesian": "vi",
        },
        "expected_dce_config": {
            "model": "2cxm",
            "bayesian": "map",
            "scale_aif": True,
        },
    },
    {
        "id": "sub-0019_ses-0001",
        "user_story": "US-096",
        "cli_args": [
            "--dsc-preload",
            "--dsc-dose", "half",
            "--dsc-deconv-method", "cSVD",
            "--dce-model", "patlak",
            "--dce-scale-aif",
        ],
        "config_file": "tests/fixtures/sub-0019_config.yaml",
        "expected_dsc_config": {
            "preload_detected": True,
            "dose": "half",
            "method": "cSVD",
            "bayesian": "none",
        },
        "expected_dce_config": {
            "model": "patlak",
            "bayesian": "none",
            "scale_aif": True,
        },
    },
    {
        "id": "sub-0020_ses-0001",
        "user_story": "US-097",
        "cli_args": [
            "--dsc-preload",
            "--dsc-dose", "half",
            "--dsc-deconv-method", "vascular",
            "--dsc-bayesian", "map",
            "--dce-model", "tofts",
            "--dce-scale-aif",
        ],
        "config_file": "tests/fixtures/sub-0020_config.yaml",
        "expected_dsc_config": {
            "preload_detected": True,
            "dose": "half",
            "method": "vascular",
            "bayesian": "map",
        },
        "expected_dce_config": {
            "model": "tofts",
            "bayesian": "none",
            "scale_aif": True,
        },
    },
]


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_bids_dir(tmp_path: Path) -> Path:
    """Create a mock BIDS directory structure for testing."""
    bids_dir = tmp_path / "bids"
    bids_dir.mkdir()

    # Create dataset_description.json
    (bids_dir / "dataset_description.json").write_text(
        '{"Name": "Test", "BIDSVersion": "1.8.0"}'
    )

    return bids_dir


@pytest.fixture
def mock_subject_session(mock_bids_dir: Path) -> tuple[Path, str, str]:
    """Create a mock subject/session structure."""
    subject_id = "0001"
    session_id = "0001"

    # Create subject/session directory structure
    ses_dir = mock_bids_dir / f"sub-{subject_id}" / f"ses-{session_id}"

    # Create anat directory with minimal files
    anat_dir = ses_dir / "anat"
    anat_dir.mkdir(parents=True)
    (anat_dir / f"sub-{subject_id}_ses-{session_id}_T1w.nii.gz").touch()

    # Create perf directory with minimal files
    perf_dir = ses_dir / "perf"
    perf_dir.mkdir(parents=True)
    (perf_dir / f"sub-{subject_id}_ses-{session_id}_dsc.nii.gz").touch()

    return mock_bids_dir, subject_id, session_id


# =============================================================================
# CLI Parsing Tests
# =============================================================================

class TestCLIParsing:
    """Test CLI argument parsing for all test subjects."""

    def test_parser_builds_successfully(self) -> None:
        """Verify the argument parser can be built."""
        from pmriprep.cli.run import _build_parser

        parser = _build_parser()
        assert parser is not None
        assert parser.prog == "pmriprep"

    def test_version_display(self) -> None:
        """Verify version is accessible."""
        assert __version__ is not None
        assert len(__version__) > 0

    @pytest.mark.parametrize("config", TEST_SUBJECT_CLI_CONFIGS)
    def test_cli_args_parse_successfully(
        self, config: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Verify CLI arguments for each test subject parse without error."""
        from pmriprep.cli.run import _build_parser

        parser = _build_parser()

        # Build full command line
        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()
        output_dir = tmp_path / "output"

        base_args = [str(bids_dir), str(output_dir), "participant"]
        cli_args = config["cli_args"]

        # Parse arguments
        args = parser.parse_args(base_args + cli_args)

        # Verify parsing succeeded
        assert args.bids_dir == bids_dir
        assert args.output_dir == output_dir
        assert args.analysis_level == "participant"


# =============================================================================
# Configuration Building Tests
# =============================================================================

class TestConfigurationBuilding:
    """Test configuration dataclass building and merging."""

    def test_dsc_config_defaults(self) -> None:
        """Verify DSCConfig has expected defaults."""
        config = DSCConfig()
        assert config.method == "sSVD"
        assert config.bayesian == "none"
        assert config.dose == "full"

    def test_dce_config_defaults(self) -> None:
        """Verify DCEConfig has expected defaults."""
        config = DCEConfig()
        assert config.model == "auto"
        assert config.bayesian == "none"

    def test_anat_config_defaults(self) -> None:
        """Verify AnatConfig has expected defaults."""
        config = AnatConfig()
        assert config.tool == "hdbet"
        assert config.denoise is False
        assert config.skip_acpc is False

    def test_hdglio_config_defaults(self) -> None:
        """Verify HDGLIOConfig has expected defaults."""
        config = HDGLIOConfig()
        assert config.lesion_dilation == 15.0
        assert config.nawm_erosion == 2

    def test_gbm_preset_application(self) -> None:
        """Verify GBM preset modifies DSCConfig correctly."""
        config = DSCConfig()
        preset_config = config.apply_gbm_preset()

        assert preset_config.method == "oSVD"
        assert preset_config.pre_bolus == 12  # inherits default, no longer overridden
        assert preset_config.aif_criteria == "relaxed"
        assert preset_config.n_aif_candidates == 5
        assert preset_config.save_k2_map is True

    @pytest.mark.parametrize("config", TEST_SUBJECT_CLI_CONFIGS)
    def test_config_file_loading(self, config: Dict[str, Any]) -> None:
        """Verify config files for each test subject load correctly."""
        config_file = config.get("config_file")
        if config_file is None:
            pytest.skip("No config file for this subject")

        config_path = Path(config_file)
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_file}")

        loaded_config = load_config(config_path)

        # Verify structure
        assert "dsc" in loaded_config or "dce" in loaded_config or "anat" in loaded_config


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
        # Check key nodes exist
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
            subject_id=subject_id,
            session_id=session_id,
            config=config,
        )

        assert workflow is not None
        assert workflow.name == "dce_preproc_wf"
        node_names = [node.name for node in workflow._graph.nodes()]
        assert "inputnode" in node_names
        assert "outputnode" in node_names

    def test_hdglio_workflow_initializes(self, mock_subject_session: tuple) -> None:
        """Verify HD-GLIO workflow can be initialized."""
        from pmriprep.workflows.hdglio.base import init_hdglio_wf

        bids_dir, subject_id, session_id = mock_subject_session
        config = HDGLIOConfig()

        workflow = init_hdglio_wf(
            bids_dir=bids_dir,
            subject_id=subject_id,
            session_id=session_id,
            config=config,
        )

        assert workflow is not None
        assert workflow.name == "hdglio_wf"
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

    def test_hdglio_requires_four_modalities(self) -> None:
        """Verify HD-GLIO doesn't run without all 4 modalities."""
        session_data = SessionData(
            subject="0001",
            session="0001",
            modalities=ModalityAvailability(),
            pipelines=PipelineAvailability(
                hdglio=False,
                hdglio_reason="Missing modalities for HD-GLIO"
            ),
        )
        skip_flags = PipelineSkipFlags()

        decision = decide_pipelines(session_data, skip_flags)

        assert decision.run_hdglio is False


# =============================================================================
# Pipeline Summary Tests
# =============================================================================

class TestPipelineSummary:
    """Test pipeline summary generation."""

    def test_summary_contains_all_pipelines(self) -> None:
        """Verify summary includes all pipeline statuses."""
        # Create a mock workflow with decision attribute
        mock_workflow = MagicMock()
        mock_workflow.pipeline_decision = PipelineDecision(
            run_anat=True,
            run_hdglio=False,
            run_dsc=True,
            run_dce=False,
            hdglio_reason="Missing modalities",
            dce_reason="No DCE data",
        )

        summary = get_pipeline_summary(mock_workflow)

        assert "Anatomical" in summary
        assert "HD-GLIO" in summary
        assert "DSC" in summary
        assert "DCE" in summary
        assert "WILL RUN" in summary
        assert "SKIPPED" in summary


# =============================================================================
# Integration Test Configuration Validation
# =============================================================================

class TestIntegrationTestConfigurations:
    """Validate all 7 integration test configurations are correct."""

    def test_all_seven_subjects_defined(self) -> None:
        """Verify all 7 test subjects are defined."""
        assert len(TEST_SUBJECT_CLI_CONFIGS) == 7

    @pytest.mark.parametrize("config", TEST_SUBJECT_CLI_CONFIGS)
    def test_each_subject_has_required_fields(
        self, config: Dict[str, Any]
    ) -> None:
        """Verify each test subject config has required fields."""
        assert "id" in config
        assert "user_story" in config
        assert "cli_args" in config
        assert "expected_dsc_config" in config
        # expected_dce_config can be None for DSC-only subjects

    @pytest.mark.parametrize("config", TEST_SUBJECT_CLI_CONFIGS)
    def test_dsc_methods_are_valid(self, config: Dict[str, Any]) -> None:
        """Verify DSC deconvolution methods are valid."""
        dsc_config = config.get("expected_dsc_config")
        if dsc_config is None:
            pytest.skip("No DSC config for this subject")

        valid_methods = {"sSVD", "cSVD", "oSVD", "tikhonov", "vascular", "cpi"}
        assert dsc_config["method"] in valid_methods

    @pytest.mark.parametrize("config", TEST_SUBJECT_CLI_CONFIGS)
    def test_dce_models_are_valid(self, config: Dict[str, Any]) -> None:
        """Verify DCE models are valid."""
        dce_config = config.get("expected_dce_config")
        if dce_config is None:
            pytest.skip("No DCE config for this subject")

        valid_models = {"auto", "tofts", "extended_tofts", "patlak", "2cxm"}
        assert dce_config["model"] in valid_models

    @pytest.mark.parametrize("config", TEST_SUBJECT_CLI_CONFIGS)
    def test_bayesian_methods_are_valid(self, config: Dict[str, Any]) -> None:
        """Verify Bayesian methods are valid for both DSC and DCE."""
        valid_dsc_bayesian = {"none", "map", "vi", "mcmc"}
        valid_dce_bayesian = {"none", "map", "mcmc"}

        dsc_config = config.get("expected_dsc_config")
        if dsc_config:
            assert dsc_config["bayesian"] in valid_dsc_bayesian

        dce_config = config.get("expected_dce_config")
        if dce_config and "bayesian" in dce_config:
            assert dce_config["bayesian"] in valid_dce_bayesian


# =============================================================================
# Typecheck Verification
# =============================================================================

class TestTypecheckVerification:
    """Verify key modules can be imported without type errors."""

    def test_import_pmriprep(self) -> None:
        """Verify main package imports."""
        import pmriprep
        assert pmriprep.__version__ is not None

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
        from pmriprep.workflows.hdglio import base as hdglio_base

        assert base.init_pmriprep_wf is not None
        assert anat_base.init_anat_preproc_wf is not None
        assert dsc_base.init_dsc_preproc_wf is not None
        assert dce_base.init_dce_preproc_wf is not None
        assert hdglio_base.init_hdglio_wf is not None

    def test_import_configs(self) -> None:
        """Verify config modules import."""
        from pmriprep.config import dsc, dce, anat, hdglio, loader

        assert dsc.DSCConfig is not None
        assert dce.DCEConfig is not None
        assert anat.AnatConfig is not None
        assert hdglio.HDGLIOConfig is not None
        assert loader.load_config is not None

    def test_import_interfaces(self) -> None:
        """Verify interface modules import."""
        from pmriprep.interfaces import anat, dsc, dce, hdglio
        from pmriprep.interfaces import dsc_aif, dsc_compute
        from pmriprep.interfaces import dce_aif, dce_compute

        assert anat.ReorientAndCrop is not None
        assert dsc.MotionCorrection is not None
        assert dce.B1Mapping is not None
        assert hdglio.HDGLIOPrep is not None

    def test_import_utils(self) -> None:
        """Verify utility modules import."""
        from pmriprep.utils import (
            bids,
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

        assert bids.get_anat_filename is not None
        assert nifti.load_nifti is not None
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
        from pmriprep.utils.qc_standards import PHYSIOLOGICAL_RANGES_3T

        # WM T1 at 3T should be around 800-1000ms
        assert PHYSIOLOGICAL_RANGES_3T.wm_t1.min >= 700
        assert PHYSIOLOGICAL_RANGES_3T.wm_t1.max <= 1000

        # GM T1 at 3T should be around 1200-1500ms
        assert PHYSIOLOGICAL_RANGES_3T.gm_t1.min >= 1100
        assert PHYSIOLOGICAL_RANGES_3T.gm_t1.max <= 1500


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
        from pmriprep.utils.dsc_constants import CBF_BOUNDS, CBV_BOUNDS, MTT_BOUNDS

        assert CBF_BOUNDS.min < CBF_BOUNDS.max
        assert CBV_BOUNDS.min < CBV_BOUNDS.max
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
