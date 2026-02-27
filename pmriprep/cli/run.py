"""PMRIPrep CLI entry point."""

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from time import strftime
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

import toml  # type: ignore[import-untyped]

from pmriprep import __version__
from pmriprep.cli import logging as plogger
from pmriprep.cli.logging import IMPORTANT

# Module-level logger — uses the centralized ``cli`` logger for top-level
# messages that should be IMPORTANT-level by default.
logger = plogger.cli


def _sanitize_for_toml(obj: Any) -> Any:
    """Convert Path objects to strings and drop None values for TOML."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_toml(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_toml(item) for item in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for pmriprep."""
    parser = argparse.ArgumentParser(
        prog="pmriprep",
        usage="%(prog)s bids_dir output_dir {participant} [options]",
        description=(
            "PMRIPrep: Perfusion MRI Preprocessing Pipeline\n\n"
            "Automated preprocessing and quantification of DSC and DCE\n"
            "perfusion MRI data. Reads BIDS-formatted input and produces\n"
            "BIDS-Derivatives compliant output with QC reports."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  pmriprep /data/bids /data/out participant
  pmriprep /data/bids /data/out participant --participant-label 01 02
  pmriprep /data/bids /data/out participant -w /tmp/work --nprocs 8

  # Use a config file for advanced options:
  pmriprep /data/bids /data/out participant --config-file my_config.yaml
""",
    )

    # Version
    parser.add_argument(
        "--version",
        action="version",
        version=f"pmriprep {__version__}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose_count",
        action="count",
        default=0,
        help="Increases log verbosity for each occurrence, debug level is -vvv.",
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
        nargs="+",
        metavar="LABEL",
        help="A space-separated list of participant labels to process "
        "(without 'sub-' prefix). If not provided, all participants are processed.",
    )
    parser.add_argument(
        "--session-label",
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
        "--omp-nthreads",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of threads per-process for ANTs/FSL. "
        "Controls OMP_NUM_THREADS and ITK thread settings.",
    )
    parser.add_argument(
        "--mem-gb",
        type=float,
        metavar="GB",
        help="Upper bound memory limit in gigabytes.",
    )
    parser.add_argument(
        "-w",
        "--work-dir",
        type=Path,
        metavar="PATH",
        help="Path to the working directory for intermediate files.",
    )

    # Configuration file
    parser.add_argument(
        "--config-file",
        type=Path,
        metavar="PATH",
        help="Path to a YAML or JSON configuration file for advanced options. "
        "See README for full parameter reference.",
    )

    # Pipeline component skip flags
    skip_group = parser.add_argument_group(
        "Pipeline components",
        "Flags to skip specific pipeline components.",
    )
    skip_group.add_argument(
        "--skip-anat",
        action="store_true",
        default=False,
        help="Skip anatomical preprocessing pipeline.",
    )
    skip_group.add_argument(
        "--skip-tumor-seg",
        action="store_true",
        default=False,
        help="Skip HD-GLIO tumor segmentation.",
    )
    skip_group.add_argument(
        "--skip-dsc",
        action="store_true",
        default=False,
        help="Skip DSC (Dynamic Susceptibility Contrast) perfusion pipeline.",
    )
    skip_group.add_argument(
        "--skip-dce",
        action="store_true",
        default=False,
        help="Skip DCE (Dynamic Contrast Enhanced) perfusion pipeline.",
    )

    # Device options
    device_group = parser.add_argument_group(
        "Device options",
        "Hardware device selection for neural network tools.",
    )
    device_group.add_argument(
        "--hdbet-device",
        choices=["auto", "cuda", "mps", "cpu"],
        default=None,
        help=argparse.SUPPRESS,
    )
    device_group.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Enable GPU acceleration for DSC and DCE processing.",
    )

    # DSC options
    dsc_group = parser.add_argument_group(
        "DSC options",
        "Options for DSC (Dynamic Susceptibility Contrast) perfusion processing.",
    )
    dsc_group.add_argument(
        "--dsc-deconv-method",
        choices=["sSVD", "cSVD", "oSVD", "tikhonov", "vascular", "cpi"],
        default=None,
        help="Deconvolution method for DSC (default: oSVD).",
    )
    dsc_group.add_argument(
        "--dsc-bayesian",
        choices=["none", "map", "vi", "mcmc"],
        default=None,
        help="Bayesian inference method for DSC: none, map (maximum a posteriori), "
        "vi (variational), mcmc (Markov chain Monte Carlo). Default: none.",
    )
    dsc_group.add_argument(
        "--dsc-preload",
        action="store_const",
        const=True,
        default=None,
        help="Indicate contrast preload is present (e.g., prior DCE injection).",
    )
    dsc_group.add_argument(
        "--dsc-dose",
        choices=["half", "full", "double"],
        default=None,
        help="Contrast agent dose for DSC (default: full).",
    )
    dsc_group.add_argument(
        "--dsc-aif-method",
        choices=["auto", "synthetic"],
        default=None,
        help="AIF detection method for DSC (default: auto).",
    )

    # DCE options
    dce_group = parser.add_argument_group(
        "DCE options",
        "Options for DCE (Dynamic Contrast Enhanced) perfusion processing.",
    )
    dce_group.add_argument(
        "--dce-model",
        choices=["auto", "tofts", "extended_tofts", "patlak", "2cxm"],
        default=None,
        help="Pharmacokinetic model for DCE. 'auto' performs per-voxel BIC "
        "model selection (candidate set gated by temporal resolution), "
        "producing a hybrid map where each voxel uses its best-fitting "
        "model. (default: auto).",
    )
    dce_group.add_argument(
        "--dce-bayesian",
        choices=["none", "map", "vi", "mcmc"],
        default=None,
        help="Bayesian inference method for DCE: none, map (maximum a posteriori), "
        "vi (variational), mcmc (Markov chain Monte Carlo). Default: none.",
    )
    dce_group.add_argument(
        "--dce-aif-method",
        choices=["auto", "population"],
        default=None,
        help="AIF detection method for DCE (default: auto).",
    )
    dce_group.add_argument(
        "--dce-aif-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="External AIF curve file (.txt with one value per line, or .npz with 'Cp' or 'aif' key). "
             "Values must be plasma concentration in mM, temporally aligned with DCE volumes. "
             "Bypasses AIF detection. Number of values must match DCE volumes.",
    )
    dce_group.add_argument(
        "--dce-dose",
        choices=["half", "full", "double"],
        default=None,
        help="Contrast agent dose for DCE (default: full).",
    )
    dce_group.add_argument(
        "--dce-r1",
        type=float,
        default=None,
        metavar="R1",
        help=(
            "Override r1 relaxivity (s^-1/mM) for DCE concentration "
            "conversion. Bypasses contrast-agent lookup from BIDS sidecar. "
            "Typical values: Gd-DTPA 3.8, Gadovist 5.0 at 3T."
        ),
    )
    dce_group.add_argument(
        "--dce-b1-method",
        choices=["auto", "afi", "tfl_b1map", "none"],
        default=None,
        help="B1+ correction method: auto (detect from data), afi (Yarnykh AFI), "
             "tfl_b1map (pre-computed flip angle map), none (skip B1+ correction).",
    )
    dce_group.add_argument(
        "--dce-no-t1",
        action="store_const",
        const=True,
        default=None,
        help="Skip T1 mapping (use field-strength-appropriate default T1).",
    )

    # Anatomical options
    anat_group = parser.add_argument_group(
        "Anatomical options",
        "Options for anatomical preprocessing.",
    )
    anat_group.add_argument(
        "--brain-extraction-tool",
        choices=["hdbet", "ants"],
        default=None,
        help="Tool for brain extraction (default: hdbet).",
    )
    anat_group.add_argument(
        "--denoise",
        action="store_const",
        const=True,
        default=None,
        help="Apply ANTs denoising to anatomical images.",
    )
    anat_group.add_argument(
        "--hdbet-conda-env",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    # Tumor segmentation options
    tumor_seg_group = parser.add_argument_group(
        "Tumor segmentation options",
        "Options for tumor segmentation.",
    )
    tumor_seg_group.add_argument(
        "--tumor-seg-conda-env",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    tumor_seg_group.add_argument(
        "--tumor-seg-device",
        choices=["cuda", "mps", "cpu"],
        default=None,
        help=argparse.SUPPRESS,
    )

    # Other options (NiPreps parity)
    g_other = parser.add_argument_group(
        "Other options",
        "Additional workflow control options.",
    )
    g_other.add_argument(
        "--stop-on-first-crash",
        action="store_true",
        default=False,
        help="Force stopping on first crash, even if a work directory was specified.",
    )
    g_other.add_argument(
        "--write-graph",
        action="store_true",
        default=False,
        help="Write workflow graph.",
    )
    g_other.add_argument(
        "--resource-monitor",
        action="store_true",
        default=False,
        help="Enable Nipype's resource monitoring to keep track of memory and CPU usage.",
    )
    g_other.add_argument(
        "--report-only",
        action="store_true",
        default=False,
        help="Regenerate HTML QC reports from existing outputs without "
        "re-running any pipeline. Requires a previous run's output.",
    )
    g_other.add_argument(
        "--clean-workdir",
        action="store_true",
        default=False,
        help="Delete working directory contents after processing all subjects.",
    )
    g_other.add_argument(
        "--sloppy",
        action="store_true",
        default=False,
        help="Use low-quality, fast settings for testing. NOT FOR PRODUCTION USE.",
    )
    g_other.add_argument(
        "--random-seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Initialize random seed for reproducible results. "
        "Full reproducibility requires --omp-nthreads 1 and --nprocs 1.",
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

    # Validate dce_aif_file exists if specified
    if args.dce_aif_file is not None and not args.dce_aif_file.is_file():
        parser.error(f"--dce-aif-file does not exist: {args.dce_aif_file}")

    # Resolve all Path arguments to absolute paths so nipype nodes
    # can locate them regardless of their execution working directory
    for attr in ("bids_dir", "output_dir", "work_dir", "config_file", "dce_aif_file"):
        val = getattr(args, attr, None)
        if val is not None:
            setattr(args, attr, val.resolve())

    # Run the workflow
    run_pmriprep_wf(args)


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


def _run_report_only(
    args: argparse.Namespace,
    subjects: List[str],
) -> None:
    """Regenerate HTML reports from existing QC outputs on disk.

    For each subject/session, auto-detects which pipelines ran by checking
    for QC JSON files and calls ``generate_unified_report()`` directly
    without any Nipype involvement.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    subjects : list of str
        Subject IDs to generate reports for.
    """
    from pmriprep.utils.report_utils import (
        _PIPELINE_QC_SUFFIXES,
        generate_unified_report,
    )

    for subject_id in subjects:
        sessions = _find_sessions(args.bids_dir, subject_id, args.session_label)

        for session_id in sessions:
            session_str = f"ses-{session_id}" if session_id else "no-session"

            # Build paths
            ses_dir = args.output_dir / "pmriprep" / f"sub-{subject_id}"
            if session_id:
                ses_dir = ses_dir / f"ses-{session_id}"
            figures_dir = ses_dir / "figures"

            # Build BIDS prefix
            prefix = f"sub-{subject_id}"
            if session_id:
                prefix += f"_ses-{session_id}"

            # Auto-detect which pipelines ran from QC JSONs on disk
            ran_pipelines: Dict[str, bool] = {}
            for pipeline, suffix in _PIPELINE_QC_SUFFIXES.items():
                qc_path = ses_dir / f"{prefix}_desc-{suffix}_quality.json"
                ran_pipelines[pipeline] = qc_path.exists()

            if not any(ran_pipelines.values()):
                logger.warning(
                    "No QC JSON files found for sub-%s/%s in %s — skipping",
                    subject_id,
                    session_str,
                    ses_dir,
                )
                continue

            detected = [k for k, v in ran_pipelines.items() if v]
            logger.log(
                IMPORTANT,
                "sub-%s/%s: regenerating report (detected: %s)",
                subject_id,
                session_str,
                ", ".join(detected),
            )

            output_path = ses_dir / f"{prefix}.html"
            generate_unified_report(
                subject_id=subject_id,
                session_id=session_id or "",
                output_path=str(output_path),
                ses_dir=str(ses_dir),
                figures_dir=str(figures_dir),
                ran_pipelines=ran_pipelines,
            )

            logger.log(IMPORTANT, "Report written: %s", output_path)


def run_pmriprep_wf(args: argparse.Namespace) -> None:
    """Run the PMRIPrep workflow.

    This function is the main workflow execution logic for PMRIPrep.
    It handles:
    1. Parsing CLI arguments and loading config file
    2. Merging CLI args with config file (CLI overrides)
    3. Building DSCConfig, DCEConfig, AnatConfig, TumorSegConfig from merged config
    4. Initializing master workflow with configs
    5. Configuring Nipype execution (nprocs, work_dir, etc.)
    6. Running workflow with proper error handling

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    # Compute log level from -v count: (none)=25, -v=20, -vv=15, -vvv=10
    log_level = int(max(IMPORTANT - 5 * args.verbose_count, logging.DEBUG))

    # Console-only logging before per-subject FileHandler is set up
    plogger.init(log_level=log_level)

    logger.log(IMPORTANT, f"PMRIPrep version {__version__}")
    logger.log(IMPORTANT, f"BIDS directory: {args.bids_dir}")
    logger.log(IMPORTANT, f"Output directory: {args.output_dir}")

    # Set per-process thread limits (OMP/ITK) before any computation
    if args.omp_nthreads is not None:
        import os

        omp = str(args.omp_nthreads)
        os.environ["OMP_NUM_THREADS"] = omp
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = omp
        logger.info("Per-process threads: OMP_NUM_THREADS=%s", omp)

    # Set random seed for reproducibility (numpy + stdlib random)
    if args.random_seed is not None:
        import random

        import numpy as np

        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        logger.info("Random seed: %d", args.random_seed)

    # Find subjects to process
    subjects = _find_subjects(args.bids_dir, args.participant_label)

    if not subjects:
        logger.error("No subjects found to process")
        sys.exit(1)

    logger.log(IMPORTANT, f"Found {len(subjects)} subject(s) to process")

    # --report-only: regenerate HTML from existing QC outputs and exit
    if args.report_only:
        _run_report_only(args, subjects)
        return

    # --- Full pipeline path: load configs, build workflows, run Nipype ---
    from pmriprep.config.loader import (
        build_anat_config,
        build_dce_config,
        build_dsc_config,
        build_tumor_seg_config,
        load_config,
        merge_cli_config,
    )
    from pmriprep.workflows.base import (
        PipelineSkipFlags,
        get_pipeline_summary,
        init_pmriprep_wf,
    )

    # Generate run UUID once per invocation (all subjects share it)
    run_uuid = f"{strftime('%Y%m%d-%H%M%S')}_{uuid4()}"

    # Convert args to dict for config merging
    cli_args = _args_to_dict(args)

    # Load config file if specified
    if args.config_file is not None:
        logger.info(f"Loading configuration from: {args.config_file}")
        file_config = load_config(args.config_file)
    else:
        file_config = {"dsc": {}, "dce": {}, "anat": {}, "tumor_seg": {}}

    # Merge CLI args with file config (CLI takes precedence)
    merged_config = merge_cli_config(cli_args, file_config)

    # Build configuration objects
    dsc_config = build_dsc_config(merged_config)
    dce_config = build_dce_config(merged_config)
    anat_config = build_anat_config(merged_config)
    tumor_seg_config = build_tumor_seg_config(merged_config)

    # Propagate --nprocs as fallback for num_processes (intra-node parallelism)
    if dsc_config.num_processes == 0:
        dsc_config.num_processes = args.nprocs
    if dce_config.num_processes == 0:
        dce_config.num_processes = args.nprocs

    # Build skip flags
    skip_flags = PipelineSkipFlags(
        skip_anat=args.skip_anat,
        skip_tumor_seg=args.skip_tumor_seg,
        skip_dsc=args.skip_dsc,
        skip_dce=args.skip_dce,
    )

    if args.sloppy:
        logger.warning("Sloppy mode enabled — results are for testing only")

    # Resolve work_dir once for downstream use
    work_dir = args.work_dir if args.work_dir is not None else args.output_dir / "work"

    # Process each subject
    for subject_id in subjects:
        sessions = _find_sessions(args.bids_dir, subject_id, args.session_label)

        for session_id in sessions:
            session_str = f"ses-{session_id}" if session_id else "no-session"

            # Per-session log directory
            log_dir = (
                args.output_dir
                / "pmriprep"
                / f"sub-{subject_id}"
                / "log"
                / run_uuid
            )
            if session_id:
                log_dir = log_dir / f"ses-{session_id}"

            # Re-initialize loggers with per-session FileHandler + nipype config
            plogger.init(log_level=log_level, log_dir=log_dir)
            _configure_nipype(args, log_dir=log_dir)

            # Save config TOML snapshot (sanitize Path→str, drop None values)
            config_snapshot = _sanitize_for_toml({
                "pmriprep": {"version": __version__, "run_uuid": run_uuid},
                "execution": {
                    "nprocs": args.nprocs,
                    "omp_nthreads": args.omp_nthreads,
                    "mem_gb": args.mem_gb,
                    "random_seed": args.random_seed,
                    "sloppy": args.sloppy,
                    "clean_workdir": args.clean_workdir,
                },
                "dsc": asdict(dsc_config),
                "dce": asdict(dce_config),
                "anat": asdict(anat_config),
                "tumor_seg": asdict(tumor_seg_config),
            })
            (log_dir / "pmriprep.toml").write_text(toml.dumps(config_snapshot))

            logger.log(IMPORTANT, f"Processing sub-{subject_id}/{session_str}")

            try:
                # Initialize the master workflow
                workflow = init_pmriprep_wf(
                    bids_dir=args.bids_dir,
                    output_dir=args.output_dir,
                    subject_id=subject_id,
                    session_id=session_id,
                    skip_flags=skip_flags,
                    anat_config=anat_config,
                    tumor_seg_config=tumor_seg_config,
                    dsc_config=dsc_config,
                    dce_config=dce_config,
                    work_dir=work_dir,
                    random_seed=args.random_seed,
                    name=f"pmriprep_sub{subject_id}_{session_str}_wf",
                )

                # Write workflow graph if requested
                if args.write_graph:
                    workflow.write_graph(
                        graph2use="colored", dotfilename="graph"
                    )

                # Log pipeline summary
                summary = get_pipeline_summary(workflow)
                for line in summary.split("\n"):
                    logger.info(line)

                # Set working directory
                workflow.base_dir = str(work_dir)

                # Run the workflow
                logger.log(
                    IMPORTANT,
                    f"Starting workflow execution for sub-{subject_id}",
                )
                _run_workflow(workflow, args)
                logger.log(
                    IMPORTANT,
                    f"Successfully completed processing for "
                    f"sub-{subject_id}/{session_str}",
                )

            except Exception as e:
                logger.error(
                    f"Error processing sub-{subject_id}/{session_str}: {e}"
                )
                # Continue to next subject/session instead of failing completely
                if args.stop_on_first_crash:
                    raise
                continue

            # Post-run crash file reporting for this session
            crashfiles = list(log_dir.glob("crash-*.txt"))
            if crashfiles:
                logger.error(
                    "Crash files found for sub-%s/%s:",
                    subject_id,
                    session_str,
                )
                for cf in crashfiles:
                    logger.error("  %s", cf.name)
                    try:
                        content = cf.read_text()
                        lines = content.splitlines()[:50]
                        for line in lines:
                            logger.error("    %s", line)
                    except Exception:
                        logger.error("    (could not read crash file)")

    # Clean working directory after all subjects are processed
    if args.clean_workdir and work_dir.exists():
        import shutil

        logger.log(IMPORTANT, "Clearing working directory: %s", work_dir)
        try:
            shutil.rmtree(work_dir)
        except OSError as e:
            logger.warning("Failed to clean work directory: %s", e)

    logger.log(IMPORTANT, "PMRIPrep processing complete")


def _configure_nipype(
    args: argparse.Namespace, *, log_dir: Path
) -> None:
    """Configure Nipype execution settings.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    log_dir : Path
        Per-subject log directory for crash dumps and Nipype logs.
    """
    import os

    from nipype import config as nipype_config  # type: ignore[import-untyped]
    from nipype import logging as nipype_logging  # type: ignore[import-untyped]

    # Ensure FSL output type is set
    if "FSLOUTPUTTYPE" not in os.environ:
        os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
        logger.debug("Set FSLOUTPUTTYPE=NIFTI_GZ")

    nipype_config.update_config(
        {
            "execution": {
                "crashdump_dir": str(log_dir),
                "crashfile_format": "txt",
                "stop_on_first_crash": args.stop_on_first_crash,
                "remove_unnecessary_outputs": True,
                "keep_inputs": False,
                "hash_method": "content",
            },
            "logging": {
                "log_directory": str(log_dir),
                "log_to_file": True,
            },
        }
    )

    # Enable resource monitoring if requested
    if args.resource_monitor:
        nipype_config.enable_resource_monitor()

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
