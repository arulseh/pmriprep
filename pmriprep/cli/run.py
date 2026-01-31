"""PMRIPrep CLI entry point."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pmriprep import __version__

# Set up module-level logger
logger = logging.getLogger("pmriprep")


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for pmriprep."""
    parser = argparse.ArgumentParser(
        prog="pmriprep",
        description="PMRIPrep: Perfusion MRI Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  pmriprep /data/bids /data/derivatives participant
  pmriprep /data/bids /data/derivatives participant --participant-label sub-01 sub-02
  pmriprep /data/bids /data/derivatives participant -w /tmp/work --nprocs 8
""",
    )

    # Version
    parser.add_argument(
        "--version",
        action="version",
        version=f"pmriprep {__version__}",
    )

    # Positional arguments (BIDS-App standard)
    parser.add_argument(
        "bids_dir",
        type=Path,
        help="The root directory of the BIDS dataset.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="The output directory for derivatives.",
    )
    parser.add_argument(
        "analysis_level",
        choices=["participant"],
        help="Level of analysis (currently only 'participant' supported).",
    )

    # Standard BIDS-App arguments
    parser.add_argument(
        "--participant-label",
        "--participant_label",
        dest="participant_label",
        nargs="+",
        metavar="LABEL",
        help="A space-separated list of participant labels to process "
        "(without 'sub-' prefix). If not provided, all participants are processed.",
    )
    parser.add_argument(
        "--session-label",
        "--session_label",
        dest="session_label",
        nargs="+",
        metavar="LABEL",
        help="A space-separated list of session labels to process "
        "(without 'ses-' prefix). If not provided, all sessions are processed.",
    )

    # Resource arguments
    parser.add_argument(
        "--nprocs",
        type=int,
        default=1,
        metavar="N",
        help="Maximum number of processors to use (default: 1).",
    )
    parser.add_argument(
        "--mem-gb",
        "--mem_gb",
        dest="mem_gb",
        type=float,
        metavar="GB",
        help="Upper bound memory limit in gigabytes.",
    )
    parser.add_argument(
        "-w",
        "--work-dir",
        "--work_dir",
        dest="work_dir",
        type=Path,
        metavar="PATH",
        help="Path to the working directory for intermediate files.",
    )

    # Configuration file
    parser.add_argument(
        "--config-file",
        "--config_file",
        dest="config_file",
        type=Path,
        metavar="PATH",
        help="Path to a YAML or JSON configuration file for advanced options.",
    )

    # Pipeline component skip flags
    skip_group = parser.add_argument_group(
        "Pipeline components",
        "Flags to skip specific pipeline components.",
    )
    skip_group.add_argument(
        "--skip-anat",
        "--skip_anat",
        dest="skip_anat",
        action="store_true",
        default=False,
        help="Skip anatomical preprocessing pipeline.",
    )
    skip_group.add_argument(
        "--skip-tumor-seg",
        "--skip_tumor_seg",
        dest="skip_tumor_seg",
        action="store_true",
        default=False,
        help="Skip HD-GLIO tumor segmentation.",
    )
    skip_group.add_argument(
        "--skip-dsc",
        "--skip_dsc",
        dest="skip_dsc",
        action="store_true",
        default=False,
        help="Skip DSC (Dynamic Susceptibility Contrast) perfusion pipeline.",
    )
    skip_group.add_argument(
        "--skip-dce",
        "--skip_dce",
        dest="skip_dce",
        action="store_true",
        default=False,
        help="Skip DCE (Dynamic Contrast Enhanced) perfusion pipeline.",
    )

    # DSC options
    dsc_group = parser.add_argument_group(
        "DSC options",
        "Options for DSC (Dynamic Susceptibility Contrast) perfusion processing.",
    )
    dsc_group.add_argument(
        "--dsc-deconv-method",
        "--dsc_deconv_method",
        dest="dsc_deconv_method",
        choices=["sSVD", "cSVD", "oSVD", "tikhonov", "vascular", "cpi"],
        default="sSVD",
        help="Deconvolution method for DSC (default: sSVD).",
    )
    dsc_group.add_argument(
        "--dsc-bayesian",
        "--dsc_bayesian",
        dest="dsc_bayesian",
        choices=["none", "map", "vi", "mcmc"],
        default="none",
        help="Bayesian inference method for DSC (default: none).",
    )
    # Mutually exclusive preload flags
    preload_group = dsc_group.add_mutually_exclusive_group()
    preload_group.add_argument(
        "--dsc-preload",
        "--dsc_preload",
        dest="dsc_preload",
        action="store_true",
        default=None,
        help="Indicate contrast preload is present.",
    )
    preload_group.add_argument(
        "--dsc-no-preload",
        "--dsc_no_preload",
        dest="dsc_preload",
        action="store_false",
        help="Indicate no contrast preload is present.",
    )
    dsc_group.add_argument(
        "--dsc-dose",
        "--dsc_dose",
        dest="dsc_dose",
        choices=["half", "full", "double"],
        default="full",
        help="Contrast agent dose for DSC (default: full).",
    )
    dsc_group.add_argument(
        "--dsc-gbm-preset",
        "--dsc_gbm_preset",
        dest="dsc_gbm_preset",
        action="store_true",
        default=False,
        help="Apply GBM preset: oSVD method, pre_bolus=10, aif_criteria=relaxed, "
        "n_aif_candidates=5, save_k2_map=true.",
    )
    dsc_group.add_argument(
        "--dsc-aif-method",
        "--dsc_aif_method",
        dest="dsc_aif_method",
        choices=["auto", "population"],
        default="auto",
        help="AIF detection method for DSC (default: auto).",
    )
    dsc_group.add_argument(
        "--dsc-absolute-quant",
        "--dsc_absolute_quant",
        dest="dsc_absolute_quant",
        action="store_true",
        default=False,
        help="Enable absolute quantification for DSC (requires relaxivity).",
    )
    dsc_group.add_argument(
        "--dsc-use-gpu",
        "--dsc_use_gpu",
        dest="dsc_use_gpu",
        action="store_true",
        default=False,
        help="Enable GPU acceleration for DSC processing.",
    )

    # DCE options
    dce_group = parser.add_argument_group(
        "DCE options",
        "Options for DCE (Dynamic Contrast Enhanced) perfusion processing.",
    )
    dce_group.add_argument(
        "--dce-model",
        "--dce_model",
        dest="dce_model",
        choices=["auto", "tofts", "extended_tofts", "patlak", "2cxm"],
        default="auto",
        help="Pharmacokinetic model for DCE (default: auto).",
    )
    dce_group.add_argument(
        "--dce-bayesian",
        "--dce_bayesian",
        dest="dce_bayesian",
        choices=["none", "map", "mcmc"],
        default="none",
        help="Bayesian inference method for DCE (default: none).",
    )
    dce_group.add_argument(
        "--dce-aif-method",
        "--dce_aif_method",
        dest="dce_aif_method",
        choices=["auto", "population"],
        default="auto",
        help="AIF detection method for DCE (default: auto).",
    )
    dce_group.add_argument(
        "--dce-scale-aif",
        "--dce_scale_aif",
        dest="dce_scale_aif",
        action="store_true",
        default=False,
        help="Scale AIF amplitude to population reference.",
    )
    dce_group.add_argument(
        "--dce-no-b1",
        "--dce_no_b1",
        dest="dce_no_b1",
        action="store_true",
        default=False,
        help="Skip B1+ mapping (use nominal flip angles).",
    )
    dce_group.add_argument(
        "--dce-no-t1",
        "--dce_no_t1",
        dest="dce_no_t1",
        action="store_true",
        default=False,
        help="Skip T1 mapping (use default T1=1200ms).",
    )
    dce_group.add_argument(
        "--dce-use-gpu",
        "--dce_use_gpu",
        dest="dce_use_gpu",
        action="store_true",
        default=False,
        help="Enable GPU acceleration for DCE processing.",
    )

    # Anatomical options
    anat_group = parser.add_argument_group(
        "Anatomical options",
        "Options for anatomical preprocessing.",
    )
    anat_group.add_argument(
        "--brain-extraction-tool",
        "--brain_extraction_tool",
        dest="brain_extraction_tool",
        choices=["hdbet", "ants"],
        default="hdbet",
        help="Tool for brain extraction (default: hdbet).",
    )
    anat_group.add_argument(
        "--denoise",
        action="store_true",
        default=False,
        help="Apply ANTs denoising to anatomical images.",
    )
    anat_group.add_argument(
        "--skip-acpc",
        "--skip_acpc",
        dest="skip_acpc",
        action="store_true",
        default=False,
        help="Skip ACPC alignment to MNI template.",
    )

    # HD-GLIO options
    hdglio_group = parser.add_argument_group(
        "HD-GLIO options",
        "Options for HD-GLIO tumor segmentation.",
    )
    hdglio_group.add_argument(
        "--hdglio-conda-env",
        "--hdglio_conda_env",
        dest="hdglio_conda_env",
        type=str,
        default=None,
        help=(
            "Conda environment name for hd_glio_predict "
            "(e.g., 'hdglio-mps' for Apple Silicon). Required when HD-GLIO "
            "is installed in a separate environment."
        ),
    )
    hdglio_group.add_argument(
        "--hdglio-device",
        "--hdglio_device",
        dest="hdglio_device",
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device for HD-GLIO inference (default: cuda).",
    )
    hdglio_group.add_argument(
        "--atlases-dir",
        "--atlases_dir",
        dest="atlases_dir",
        type=str,
        default=None,
        help=(
            "Path to MNI lobar atlas directory for NAWM region restriction. "
            "Enables atlas-based NAWM selection instead of hemisphere-only."
        ),
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main entry point for pmriprep CLI.

    Parameters
    ----------
    argv : list of str, optional
        Command line arguments. If None, uses sys.argv[1:].
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Validate bids_dir exists
    if not args.bids_dir.is_dir():
        parser.error(f"bids_dir does not exist: {args.bids_dir}")

    # Create output_dir if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create work_dir if specified and doesn't exist
    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)

    # Validate config_file exists if specified
    if args.config_file is not None and not args.config_file.is_file():
        parser.error(f"config_file does not exist: {args.config_file}")

    # Run the workflow
    run_pmriprep_wf(args)


def setup_logging(verbose: bool = False) -> None:
    """Set up logging for PMRIPrep.

    Parameters
    ----------
    verbose : bool
        If True, set logging level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _args_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert argparse Namespace to a dictionary.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.

    Returns
    -------
    dict
        Dictionary of argument values.
    """
    return vars(args)


def _find_subjects(
    bids_dir: Path,
    participant_label: Optional[List[str]] = None,
) -> List[str]:
    """Find subject directories in a BIDS dataset.

    Parameters
    ----------
    bids_dir : Path
        Path to BIDS dataset root directory.
    participant_label : list of str, optional
        List of participant labels to process. If None, find all subjects.

    Returns
    -------
    list of str
        List of subject IDs (without 'sub-' prefix).
    """
    if participant_label is not None:
        # Validate requested subjects exist
        subjects = []
        for label in participant_label:
            sub_dir = bids_dir / f"sub-{label}"
            if sub_dir.is_dir():
                subjects.append(label)
            else:
                logger.warning(f"Subject directory not found: {sub_dir}")
        return subjects

    # Find all subjects
    subjects = []
    for sub_dir in sorted(bids_dir.glob("sub-*")):
        if sub_dir.is_dir():
            # Extract subject ID (without 'sub-' prefix)
            subject_id = sub_dir.name[4:]  # Remove 'sub-' prefix
            subjects.append(subject_id)

    return subjects


def _find_sessions(
    bids_dir: Path,
    subject_id: str,
    session_label: Optional[List[str]] = None,
) -> List[Optional[str]]:
    """Find session directories for a subject.

    Parameters
    ----------
    bids_dir : Path
        Path to BIDS dataset root directory.
    subject_id : str
        Subject ID (without 'sub-' prefix).
    session_label : list of str, optional
        List of session labels to process. If None, find all sessions.

    Returns
    -------
    list of str or None
        List of session IDs (without 'ses-' prefix). Returns [None] if
        no sessions exist (single-session dataset).
    """
    sub_dir = bids_dir / f"sub-{subject_id}"

    # Check if this is a session-less dataset
    ses_dirs = list(sub_dir.glob("ses-*"))
    if not ses_dirs:
        # No sessions - return [None] to indicate single-session
        return [None]

    if session_label is not None:
        # Validate requested sessions exist
        sessions: List[Optional[str]] = []
        for label in session_label:
            ses_dir = sub_dir / f"ses-{label}"
            if ses_dir.is_dir():
                sessions.append(label)
            else:
                logger.warning(
                    f"Session directory not found for sub-{subject_id}: {ses_dir}"
                )
        return sessions if sessions else [None]

    # Find all sessions
    sessions = []
    for ses_dir in sorted(sub_dir.glob("ses-*")):
        if ses_dir.is_dir():
            # Extract session ID (without 'ses-' prefix)
            session_id = ses_dir.name[4:]  # Remove 'ses-' prefix
            sessions.append(session_id)

    return sessions if sessions else [None]


def run_pmriprep_wf(args: argparse.Namespace) -> None:
    """Run the PMRIPrep workflow.

    This function is the main workflow execution logic for PMRIPrep.
    It handles:
    1. Parsing CLI arguments and loading config file
    2. Merging CLI args with config file (CLI overrides)
    3. Building DSCConfig, DCEConfig, AnatConfig, HDGLIOConfig from merged config
    4. Initializing master workflow with configs
    5. Configuring Nipype execution (nprocs, work_dir, etc.)
    6. Running workflow with proper error handling

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    from pmriprep.config.dsc import DSCConfig
    from pmriprep.config.loader import (
        build_anat_config,
        build_dce_config,
        build_dsc_config,
        build_hdglio_config,
        load_config,
        merge_cli_config,
    )
    from pmriprep.workflows.base import (
        PipelineSkipFlags,
        get_pipeline_summary,
        init_pmriprep_wf,
    )

    # Set up logging
    setup_logging(verbose=getattr(args, "verbose", False))

    logger.info(f"PMRIPrep version {__version__}")
    logger.info(f"BIDS directory: {args.bids_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Convert args to dict for config merging
    cli_args = _args_to_dict(args)

    # Load config file if specified
    if args.config_file is not None:
        logger.info(f"Loading configuration from: {args.config_file}")
        file_config = load_config(args.config_file)
    else:
        file_config = {"dsc": {}, "dce": {}, "anat": {}, "hdglio": {}}

    # Merge CLI args with file config (CLI takes precedence)
    merged_config = merge_cli_config(cli_args, file_config)

    # Build configuration objects
    dsc_config = build_dsc_config(merged_config)
    dce_config = build_dce_config(merged_config)
    anat_config = build_anat_config(merged_config)
    hdglio_config = build_hdglio_config(merged_config)

    # Apply GBM preset if requested
    if args.dsc_gbm_preset:
        logger.info("Applying GBM clinical preset to DSC configuration")
        dsc_config = dsc_config.apply_gbm_preset()

    # Build skip flags
    skip_flags = PipelineSkipFlags(
        skip_anat=args.skip_anat,
        skip_tumor_seg=args.skip_tumor_seg,
        skip_dsc=args.skip_dsc,
        skip_dce=args.skip_dce,
    )

    # Find subjects to process
    subjects = _find_subjects(args.bids_dir, args.participant_label)

    if not subjects:
        logger.error("No subjects found to process")
        sys.exit(1)

    logger.info(f"Found {len(subjects)} subject(s) to process")

    # Configure Nipype
    _configure_nipype(args)

    # Import nipype after configuration
    import nipype.pipeline.engine as pe  # type: ignore[import-untyped]

    # Process each subject/session
    for subject_id in subjects:
        sessions = _find_sessions(args.bids_dir, subject_id, args.session_label)

        for session_id in sessions:
            session_str = f"ses-{session_id}" if session_id else "no-session"
            logger.info(f"Processing sub-{subject_id}/{session_str}")

            try:
                # Initialize the master workflow
                workflow = init_pmriprep_wf(
                    bids_dir=args.bids_dir,
                    output_dir=args.output_dir,
                    subject_id=subject_id,
                    session_id=session_id,
                    skip_flags=skip_flags,
                    anat_config=anat_config,
                    hdglio_config=hdglio_config,
                    dsc_config=dsc_config,
                    dce_config=dce_config,
                    work_dir=args.work_dir,
                    name=f"pmriprep_sub{subject_id}_{session_str}_wf",
                )

                # Log pipeline summary
                summary = get_pipeline_summary(workflow)
                for line in summary.split("\n"):
                    logger.info(line)

                # Set working directory
                if args.work_dir is not None:
                    workflow.base_dir = str(args.work_dir)
                else:
                    workflow.base_dir = str(args.output_dir / "work")

                # Run the workflow
                logger.info(f"Starting workflow execution for sub-{subject_id}")
                _run_workflow(workflow, args)
                logger.info(
                    f"Successfully completed processing for sub-{subject_id}/{session_str}"
                )

            except Exception as e:
                logger.error(
                    f"Error processing sub-{subject_id}/{session_str}: {e}"
                )
                # Continue to next subject/session instead of failing completely
                if getattr(args, "stop_on_error", False):
                    raise
                continue

    logger.info("PMRIPrep processing complete")


def _configure_nipype(args: argparse.Namespace) -> None:
    """Configure Nipype execution settings.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments containing:
        - nprocs: Number of processors
        - mem_gb: Memory limit in GB
        - work_dir: Working directory
    """
    from nipype import config as nipype_config  # type: ignore[import-untyped]
    from nipype import logging as nipype_logging  # type: ignore[import-untyped]

    # Create output and log directories if they don't exist
    if args.output_dir:
        log_dir = args.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

    # Configure execution settings
    nipype_config.enable_debug_mode()
    nipype_config.update_config(
        {
            "execution": {
                "stop_on_first_crash": False,
                "remove_unnecessary_outputs": True,
                "crashfile_format": "txt",
                "keep_inputs": False,
                "hash_method": "content",
            },
            "logging": {
                "log_directory": str(args.output_dir / "logs")
                if args.output_dir
                else "",
                "log_to_file": True,
            },
        }
    )

    # Set up nipype logging
    nipype_logging.update_logging(nipype_config)

    # Log configuration
    logger.debug(f"Nipype configured with nprocs={args.nprocs}")
    if args.mem_gb:
        logger.debug(f"Memory limit: {args.mem_gb} GB")


def _run_workflow(
    workflow: Any,  # pe.Workflow but avoiding circular import
    args: argparse.Namespace,
) -> None:
    """Execute a Nipype workflow with the specified settings.

    Parameters
    ----------
    workflow : pe.Workflow
        The Nipype workflow to run.
    args : argparse.Namespace
        Parsed command line arguments.
    """
    # Determine execution plugin based on nprocs
    if args.nprocs > 1:
        # Use MultiProc plugin for parallel execution
        plugin = "MultiProc"
        plugin_args = {"n_procs": args.nprocs}

        # Add memory limit if specified
        if args.mem_gb is not None:
            # Convert GB to bytes for nipype
            plugin_args["memory_gb"] = args.mem_gb

        logger.info(f"Running workflow with {args.nprocs} processors")
        workflow.run(plugin=plugin, plugin_args=plugin_args)
    else:
        # Single processor - use linear execution
        logger.info("Running workflow in single-processor mode")
        workflow.run(plugin="Linear")


if __name__ == "__main__":
    main()
