"""PMRIPrep CLI entry point."""

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from time import strftime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
from uuid import uuid4

import toml  # type: ignore[import-untyped]

from pmriprep import __version__
from pmriprep.cli import logging as plogger
from pmriprep.cli.logging import IMPORTANT
from pmriprep.config import DOSE_PRESETS

if TYPE_CHECKING:
    from bids.layout import BIDSLayout

# Module-level logger — uses the centralized ``cli`` logger for top-level
# messages that should be IMPORTANT-level by default.
logger = plogger.cli


def _clean_work_dir(work_dir: Path) -> Optional[Path]:
    """Remove the working directory after processing.

    Parameters
    ----------
    work_dir : pathlib.Path
        User-selected working directory.

    Returns
    -------
    pathlib.Path or None
        Removed directory, or ``None`` when it did not exist.
    """
    import shutil

    if not work_dir.exists():
        return None
    shutil.rmtree(work_dir)
    return work_dir


def _dose_type(value: str) -> float:
    """Parse a dose argument as a named preset or positive float.

    Parameters
    ----------
    value : str
        Dose string from CLI (e.g. ``"half"``, ``"1.0"``, ``"0.75"``).

    Returns
    -------
    float
        Dose multiplier (e.g. ``0.5``, ``1.0``, ``0.75``).

    Raises
    ------
    argparse.ArgumentTypeError
        If *value* is not a known preset or a finite positive number.
    """
    import math

    if value in DOSE_PRESETS:
        return DOSE_PRESETS[value]
    try:
        fval = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid dose value: {value!r}. "
            f"Use a named preset ({', '.join(sorted(DOSE_PRESETS))}) "
            f"or a positive number (e.g. 0.75)."
        )
    if not math.isfinite(fval) or fval <= 0:
        raise argparse.ArgumentTypeError(
            f"dose must be a finite positive number, got {fval}"
        )
    return fval


def _preload_type(value: str) -> str | float:
    """Parse a --dsc-preload argument as 'auto' or a float dose multiplier.

    Parameters
    ----------
    value : str
        Either the literal ``"auto"`` or a float string representing the
        administered preload as a fraction of the agent's standard dose
        (e.g., ``"0.5"`` for half-dose preload).

    Returns
    -------
    str or float
        ``"auto"`` unchanged, or the parsed float value in [0.0, 2.0].

    Raises
    ------
    argparse.ArgumentTypeError
        If *value* is neither ``"auto"`` nor a finite float in [0.0, 2.0].
    """
    import math

    if value == "auto":
        return "auto"
    try:
        fval = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--dsc-preload must be 'auto' or a float dose multiplier, "
            f"got {value!r}. Example: --dsc-preload 0.5 for half-dose preload."
        )
    if not math.isfinite(fval):
        raise argparse.ArgumentTypeError(
            f"--dsc-preload must be finite, got {fval}"
        )
    if not (0.0 <= fval <= 2.0):
        raise argparse.ArgumentTypeError(
            f"--dsc-preload must be in [0.0, 2.0], got {fval}"
        )
    return fval


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
            "data. Reads BIDS-formatted input and produces\n"
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
    parser.add_argument(
        "--bids-validation",
        "--bids_validation",
        action="store_true",
        default=False,
        help="Run the bids-validator pre-flight check on the input dataset. "
        "Disabled by default because perfusion MRI suffixes (_dsc, _dce) are "
        "not yet in the stable BIDS specification.",
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
        help="Per-process thread limit (OMP/MKL/ITK).",
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
        help="Path where intermediate results should be stored.",
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
        help="Skip DSC (Dynamic Susceptibility Contrast) pipeline.",
    )
    skip_group.add_argument(
        "--skip-dce",
        action="store_true",
        default=False,
        help="Skip DCE (Dynamic Contrast Enhanced) pipeline.",
    )

    # Device options
    device_group = parser.add_argument_group()
    device_group.add_argument(
        "--hdbet-device",
        choices=["auto", "cuda", "mps", "cpu"],
        default=None,
        help=argparse.SUPPRESS,
    )
    # DSC options
    dsc_group = parser.add_argument_group(
        "DSC options",
        "Options for DSC (Dynamic Susceptibility Contrast) processing.",
    )
    dsc_group.add_argument(
        "--dsc-deconv-method",
        choices=["sSVD", "oSVD", "tikhonov", "vascular", "cpi"],
        default=None,
        help="Deconvolution method for DSC (default: oSVD).",
    )
    dsc_group.add_argument(
        "--dsc-inference",
        choices=["nlls", "map", "vi", "svb"],
        metavar="{nlls,vi,svb}",
        default=None,
        help="Inference engine for parametric DSC methods (cpi, vascular): "
        "vi (default, linearized variational inference), "
        "svb (spatial variational Bayes with a boundary-corrected M-prior), "
        "nlls (nonlinear least squares). "
        "SVD methods ignore this option.",
    )
    dsc_group.add_argument(
        "--dsc-preload",
        type=_preload_type,
        default=None,
        metavar="{auto|FLOAT}",
        help="DSC preload as 'auto' (classify preload adequacy from the K2 "
        "leakage residuals) or a float administered-dose multiplier relative "
        "to the agent's standard dose (e.g., 0.5 for half-dose preload). "
        "Accepts 0.0 to 2.0. Internally binned to a class for K2 constraint "
        "tuning.",
    )
    dsc_group.add_argument(
        "--dsc-dose",
        type=_dose_type,
        default=None,
        metavar="DOSE",
        help="Dose multiplier (default: 1.0). "
        "Accepts a float or presets: half, full, double.",
    )
    dsc_group.add_argument(
        "--dsc-agent-standard-dose",
        type=float,
        default=None,
        metavar="DOSE",
        help="Override agent standard dose in mmol Gd/kg "
        "(auto-detected from ContrastBolusAgent sidecar field when available). "
        "Default: 0.1.",
    )
    dsc_group.add_argument(
        "--dsc-aif-method",
        choices=["auto", "synthetic"],
        default=None,
        help="AIF detection method for DSC (default: auto).",
    )
    dsc_group.add_argument(
        "--dsc-aif-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="External AIF curve file (.txt with one value per line, .npz with 'aif' key, "
             "or .npy). Values must be \u0394R2* concentration in s\u207b\u00b9, temporally aligned "
             "with DSC volumes. Bypasses AIF detection.",
    )

    dsc_group.add_argument(
        "--dsc-registration",
        choices=["flirt", "flirt_bbr"],
        default=None,
        help=argparse.SUPPRESS,
    )

    # DCE options
    dce_group = parser.add_argument_group(
        "DCE options",
        "Options for DCE (Dynamic Contrast Enhanced) processing.",
    )
    dce_group.add_argument(
        "--dce-model",
        choices=["auto", "tofts", "extended_tofts", "patlak", "2cxm"],
        default=None,
        help="Pharmacokinetic model for DCE. 'auto' performs per-voxel BIC "
        "model selection from Patlak, Extended Tofts, and 2CXM "
        "(default: auto).",
    )
    dce_group.add_argument(
        "--dce-inference",
        choices=["nlls", "vi", "map"],
        metavar="{nlls,vi}",
        default=None,
        help="Inference engine for DCE: nlls (default, scipy "
        "least_squares), vi (variational Bayes).",
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
        help="External AIF curve file (.txt with one value per line, .npz with 'Cp' or 'aif' key, "
             "or .npy). Values must be plasma concentration in mM, temporally aligned with DCE volumes. "
             "Bypasses AIF detection. Number of values must match DCE volumes.",
    )
    dce_group.add_argument(
        "--dce-dose",
        type=_dose_type,
        default=None,
        metavar="DOSE",
        help="Dose multiplier (default: 1.0). "
        "Accepts a float or presets: half, full, double.",
    )
    dce_group.add_argument(
        "--dce-r1",
        type=float,
        default=None,
        metavar="R1",
        help=(
            "Override r1 relaxivity (s^-1/mM) for DCE concentration "
            "conversion. Bypasses contrast-agent lookup from BIDS sidecar. "
            "If not specified, looked up from ContrastBolusAgent in BIDS "
            "sidecar; falls back to Gd-DTPA (3.8 at 3T) if agent unknown. "
            "Typical values: Gd-DTPA 3.8, Gadovist 5.0 at 3T."
        ),
    )
    dce_group.add_argument(
        "--dce-b1-method",
        choices=["auto", "afi", "precomputed", "none"],
        default=None,
        help="B1+ correction method: auto (detect from data), afi (Yarnykh AFI), "
             "precomputed (pre-computed B1+ map), none (skip B1+ correction).",
    )
    dce_group.add_argument(
        "--dce-b1-map-encoding",
        choices=["auto", "actual_flip_deci_deg", "actual_flip_deg", "percent", "ratio"],
        default=None,
        help="Pixel encoding of a pre-computed B1+ map: auto (infer from BIDS "
             "suffix/Units, else pixel median), actual_flip_deci_deg (Siemens tfl), "
             "actual_flip_deg, percent (BIDS p.u.), ratio.",
    )
    dce_group.add_argument(
        "--dce-no-t1",
        action="store_const",
        const=True,
        default=None,
        help="Skip T1 mapping (use field-strength-appropriate default T1). "
             "B1+ correction remains controlled by --dce-b1-method.",
    )
    dce_group.add_argument(
        "--dce-t1-map",
        type=Path,
        default=None,
        help="External T1 map file (NIfTI, units: milliseconds). Bypasses "
             "VFA T1 fitting. Useful for MP2RAGE, MOLLI, or ground-truth T1 maps.",
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
    tumor_seg_group = parser.add_argument_group()
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
        help="Stop on first crash.",
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
        help="Delete the working directory after processing.",
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
    g_other.add_argument(
        "--use-plugin",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a YAML/JSON file specifying the nipype execution "
        "plugin (e.g. SLURM, SGE, PBS). Overrides the default "
        "MultiProc/Linear selection. File must contain top-level keys "
        "'plugin' (string) and optional 'plugin_args' (mapping). See "
        "nipype docs for plugin-specific args.",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the pmriprep CLI entry point.

    Parameters
    ----------
    argv : list of str, optional
        Command line arguments. If None, uses sys.argv[1:].
    """
    import shlex

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Reconstruct CLI command for QC report reproducibility
    cmd_parts = sys.argv.copy() if argv is None else ["pmriprep"] + list(argv)
    cmd_parts[0] = "pmriprep"  # Normalize entry point (not container path)
    args.command_string = shlex.join(cmd_parts)

    # Validate bids_dir exists
    if not args.bids_dir.is_dir():
        parser.error(f"bids_dir does not exist: {args.bids_dir}")

    # Run BIDS validator (warn-and-skip if binary missing; sys.exit on failure).
    from pmriprep.utils.bids import validate_bids_input

    validate_bids_input(args.bids_dir, skip=not args.bids_validation)

    # Create output_dir if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create work_dir if specified and doesn't exist
    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)

    # Validate config_file exists if specified
    if args.config_file is not None and not args.config_file.is_file():
        parser.error(f"config_file does not exist: {args.config_file}")

    # Validate dce_aif_file exists if specified
    _VALID_AIF_EXTENSIONS = {".txt", ".tsv", ".npz", ".npy"}
    if args.dce_aif_file is not None:
        if not args.dce_aif_file.is_file():
            parser.error(f"--dce-aif-file does not exist: {args.dce_aif_file}")
        if args.dce_aif_file.suffix.lower() not in _VALID_AIF_EXTENSIONS:
            parser.error(
                f"--dce-aif-file has unsupported extension '{args.dce_aif_file.suffix}'. "
                f"Supported: {sorted(_VALID_AIF_EXTENSIONS)}"
            )

    # Validate dsc_aif_file exists if specified
    if args.dsc_aif_file is not None:
        if not args.dsc_aif_file.is_file():
            parser.error(f"--dsc-aif-file does not exist: {args.dsc_aif_file}")
        if args.dsc_aif_file.suffix.lower() not in _VALID_AIF_EXTENSIONS:
            parser.error(
                f"--dsc-aif-file has unsupported extension '{args.dsc_aif_file.suffix}'. "
                f"Supported: {sorted(_VALID_AIF_EXTENSIONS)}"
            )

    # Resolve all Path arguments to absolute paths so nipype nodes
    # can locate them regardless of their execution working directory
    for attr in ("bids_dir", "output_dir", "work_dir", "config_file", "dce_aif_file", "dsc_aif_file"):
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
    layout: Optional["BIDSLayout"] = None,
) -> List[str]:
    """Return participant IDs in ``bids_dir`` (sub- prefix stripped).

    Delegates to :func:`pmriprep.utils.bids.collect_participants_pmriprep`,
    which wraps :func:`niworkflows.utils.bids.collect_participants` and
    accepts a prebuilt :class:`bids.layout.BIDSLayout` for reuse.
    """
    from pmriprep.utils.bids import collect_participants_pmriprep

    return collect_participants_pmriprep(
        bids_dir, participant_label=participant_label, layout=layout
    )


def _find_sessions(
    bids_dir: Path,
    subject_id: str,
    session_label: Optional[List[str]] = None,
    layout: Optional["BIDSLayout"] = None,
) -> List[Optional[str]]:
    """Return session IDs (ses- prefix stripped) for ``subject_id``.

    Returns ``[None]`` for session-less datasets so the caller's per-session
    loop iterates exactly once.
    """
    from pmriprep.utils.bids import collect_sessions

    return collect_sessions(
        bids_dir, subject_id, session_label=session_label, layout=layout
    )


def _run_report_only(
    args: argparse.Namespace,
    subjects: List[str],
    layout: Optional["BIDSLayout"] = None,
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
        _PIPELINE_PRESENCE_PROBES,
        generate_unified_report,
    )

    for subject_id in subjects:
        sessions = _find_sessions(
            args.bids_dir, subject_id, args.session_label, layout=layout
        )

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

            # Auto-detect which pipelines ran from canonical per-image
            # presence-probe sidecars on disk.
            ran_pipelines: Dict[str, bool] = {}
            for pipeline, anchor_pattern in _PIPELINE_PRESENCE_PROBES.items():
                # Presence = a probe sidecar exists, contrast-enhanced or not.
                # A `_ce-GAD_*` anat sidecar still means anat ran (postcon-only
                # acquisitions have no native T1w); choosing the right file
                # among variants is `_resolve_bids_anchor`'s job downstream.
                pattern = anchor_pattern.format(prefix=prefix)
                ran_pipelines[pipeline] = any(ses_dir.glob(pattern))

            if not any(ran_pipelines.values()):
                logger.warning(
                    "No pipeline outputs found for sub-%s/%s in %s — skipping",
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
                command_string="(report regenerated)",
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

    # Set per-process thread limits (OMP/ITK/MKL) before any computation.
    # Without this, OMP defaults to cpu_count per worker — with nprocs=32
    # on a 112-CPU machine, that's 3584 OMP threads total, exceeding
    # thread limits and causing pthread_create EAGAIN.
    import os

    if args.omp_nthreads is not None:
        omp = args.omp_nthreads
    else:
        n_cpus = os.cpu_count() or 1
        omp = max(1, min(8, n_cpus - 1))
        logger.info(
            "Auto-calculated per-process threads: %d (cpus=%d, nprocs=%d)",
            omp, n_cpus, args.nprocs,
        )

    os.environ["OMP_NUM_THREADS"] = str(omp)
    os.environ["MKL_NUM_THREADS"] = str(omp)
    os.environ["NUMEXPR_NUM_THREADS"] = str(omp)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(omp)
    # nnU-Net multiprocessing workers (process count, NOT thread count —
    # not governed by OMP). Must be 1 to prevent fork storms in HD-BET/HD-GLIO.
    os.environ["nnUNet_def_n_proc"] = "1"
    logger.info("Per-process threads: OMP_NUM_THREADS=%s", omp)

    # Set random seed for reproducibility (numpy + stdlib random)
    if args.random_seed is not None:
        import random

        import numpy as np

        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        logger.info("Random seed: %d", args.random_seed)

    # Build a single BIDSLayout for the whole invocation; reused for subject
    # enumeration, per-session discovery, and downstream finders. validate=False
    # because the JS bids-validator already ran (or was skipped) above.
    from bids.layout import BIDSLayout

    bids_layout = BIDSLayout(str(args.bids_dir), validate=False)

    # Find subjects to process
    subjects = _find_subjects(
        args.bids_dir, args.participant_label, layout=bids_layout
    )

    if not subjects:
        logger.error("No subjects found to process")
        sys.exit(1)

    logger.log(IMPORTANT, f"Found {len(subjects)} subject(s) to process")

    # --report-only: regenerate HTML from existing QC outputs and exit
    if args.report_only:
        _run_report_only(args, subjects, layout=bids_layout)
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
    from pmriprep.utils.naming import workflow_dirname
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

    # Track which parameters were explicitly specified by the user
    # (via CLI or config file) vs falling back to pipeline defaults.
    # Key presence in cli_args (non-None) or file_config counts as
    # user-specified, even if the value equals the pipeline default.
    user_specified: dict[str, bool] = {
        "dce_dose": (
            cli_args.get("dce_dose") is not None
            or "dose" in file_config.get("dce", {})
        ),
        "dsc_dose": (
            cli_args.get("dsc_dose") is not None
            or "dose" in file_config.get("dsc", {})
        ),
        "dsc_agent_standard_dose": (
            cli_args.get("dsc_agent_standard_dose") is not None
            or "agent_standard_dose" in file_config.get("dsc", {})
        ),
        "dsc_preload": (
            cli_args.get("dsc_preload") is not None
            or "preload" in file_config.get("dsc", {})
        ),
    }

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
        sessions = _find_sessions(
            args.bids_dir,
            subject_id,
            args.session_label,
            layout=bids_layout,
        )

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
                    mem_limit_gb=args.mem_gb,
                    command_string=args.command_string,
                    user_specified=user_specified,
                    name=workflow_dirname(subject_id, session_id),
                    layout=bids_layout,
                )

                # Log pipeline summary
                summary = get_pipeline_summary(workflow)
                for line in summary.split("\n"):
                    logger.info(line)

                # Write the workflow graph after construction.
                if args.write_graph:
                    try:
                        workflow.write_graph(
                            graph2use="colored", dotfilename="graph.dot"
                        )
                    except OSError as e:
                        # graphviz not installed - dot file was still written
                        logger.warning(f"Workflow graph .dot written but PNG failed: {e}")

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
    if args.clean_workdir:
        logger.log(IMPORTANT, "Clearing working directory: %s", work_dir)
        try:
            _clean_work_dir(work_dir)
        except OSError as e:
            logger.warning(
                "Failed to clean work directory %s: %s",
                work_dir,
                e,
            )

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

    # Monkey-patch nipype's file logging to use plain FileHandler.
    # ConcurrentRotatingFileHandler uses SoftFileLock which breaks in forked
    # worker processes (MultiProc plugin, ProcessPoolExecutor for GPU ops),
    # silently dropping log entries.
    import types

    def _plain_file_logging(self: Any) -> None:
        config = self._config
        log_filename = os.path.join(
            config.get("logging", "log_directory"), "pypeline.log"
        )
        hdlr = logging.FileHandler(log_filename, mode="a")
        hdlr.setFormatter(logging.Formatter(fmt=self.fmt, datefmt=self.datefmt))
        self._logger.addHandler(hdlr)
        self._utlogger.addHandler(hdlr)
        self._iflogger.addHandler(hdlr)
        self._fmlogger.addHandler(hdlr)
        self._hdlr = hdlr

    nipype_logging.enable_file_logging = types.MethodType(
        _plain_file_logging, nipype_logging
    )

    # Set up nipype logging
    nipype_logging.update_logging(nipype_config)

    # Log configuration
    logger.debug(f"Nipype configured with nprocs={args.nprocs}")
    if args.mem_gb:
        logger.debug(f"Memory limit: {args.mem_gb} GB")


def _load_plugin_config(path: Path) -> tuple[str, dict[str, Any]]:
    """Read a nipype execution plugin config file.

    Accepts YAML (``.yaml``/``.yml``) or JSON (``.json``); the format is
    sniffed by file extension. Expected schema::

        plugin: SLURM
        plugin_args:
          template: '#SBATCH --partition=normal'
          sbatch_args: '--mem=64G --time=4:00:00'

    Parameters
    ----------
    path : Path
        Path to the plugin config file.

    Returns
    -------
    tuple
        ``(plugin_name, plugin_args)`` ready to pass to
        ``workflow.run(...)``.
    """
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        import json

        data = json.loads(text)
    else:
        raise ValueError(
            f"--use-plugin file must be .yaml/.yml/.json (got {path.suffix})"
        )

    if not isinstance(data, dict) or "plugin" not in data:
        raise ValueError(
            f"--use-plugin file {path} must contain a top-level 'plugin' key"
        )

    plugin_value = data["plugin"]
    if not isinstance(plugin_value, str):
        raise ValueError(
            f"--use-plugin 'plugin' must be a string (got "
            f"{type(plugin_value).__name__})"
        )

    plugin_args_raw = data.get("plugin_args") or {}
    if not isinstance(plugin_args_raw, dict):
        raise ValueError(
            f"--use-plugin 'plugin_args' must be a mapping (got "
            f"{type(plugin_args_raw).__name__})"
        )

    return plugin_value, dict(plugin_args_raw)


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
    from pmriprep.utils.processes import (
        enable_child_subreaper,
        terminate_descendant_processes,
    )

    enable_child_subreaper()

    def _run_with_cleanup(plugin: str, plugin_args: dict[str, Any] | None = None) -> None:
        try:
            workflow.run(plugin=plugin, plugin_args=plugin_args)
        except BaseException as error:
            try:
                terminate_descendant_processes(
                    interrupted=isinstance(error, KeyboardInterrupt),
                )
            except Exception as cleanup_error:
                logger.warning(
                    "Failed to fully reap workflow child processes: %s",
                    cleanup_error,
                )
            raise

    # User-supplied plugin config takes precedence over the auto
    # MultiProc/Linear selection. Useful for cluster schedulers
    # (SLURM/SGE/PBS/LSF) that need scheduler-specific args.
    user_plugin = getattr(args, "use_plugin", None)
    if user_plugin is not None:
        plugin, plugin_args = _load_plugin_config(user_plugin)
        logger.info(
            "Running workflow with user-supplied plugin %s (from %s)",
            plugin, user_plugin,
        )
        _run_with_cleanup(plugin, plugin_args)
        return

    # Determine execution plugin based on nprocs
    if args.nprocs > 1:
        # Use MultiProc plugin for parallel execution
        plugin = "MultiProc"
        plugin_args = {"n_procs": args.nprocs}

        # Add memory limit if specified
        if args.mem_gb is not None:
            # Convert GB to bytes for nipype
            plugin_args["memory_gb"] = args.mem_gb

        # Tell nipype how many GPU slots are available when any node uses GPU.
        # GPU nodes include: HD-BET brain extraction (--hdbet-device cuda/mps)
        # and HD-GLIO tumor segmentation (--tumor-seg-device cuda/mps).
        # nipype's gpu_count() relies on nvidia-smi which may not be on PATH
        # inside containers (e.g., Apptainer --contain).  Without this,
        # MultiProc._prerun_check raises "Insufficient GPU resources" because
        # n_gpu_procs defaults to 0 while nodes with use_gpu=True are present.
        # Check if any GPU nodes will be present. CLI arg defaults are None
        # while config defaults are "auto"/"cuda", so use the CLI value when
        # explicitly set, otherwise fall back to "auto" (the config default).
        from pmriprep.utils.device import resolve_device_uses_gpu

        _hdbet_device = getattr(args, "hdbet_device", None) or "auto"
        _tumor_device = getattr(args, "tumor_seg_device", None) or "auto"

        _needs_gpu = (
            resolve_device_uses_gpu(_hdbet_device)  # HD-BET brain extraction
            or resolve_device_uses_gpu(_tumor_device)  # HD-GLIO tumor seg
        )

        if _needs_gpu:
            from pmriprep.utils.gpu import detect_gpu_count

            n_gpus = detect_gpu_count()
            if n_gpus > 0:
                plugin_args["n_gpu_procs"] = n_gpus
                logger.info("Detected %d GPU(s) for scheduling", n_gpus)
            else:
                plugin_args["n_gpu_procs"] = 1
                logger.warning(
                    "GPU detection returned 0 but GPU nodes are present; "
                    "defaulting to 1 GPU slot"
                )

        logger.info(f"Running workflow with {args.nprocs} processors")
        _run_with_cleanup(plugin, plugin_args)
    else:
        # Single processor - use linear execution
        logger.info("Running workflow in single-processor mode")
        _run_with_cleanup("Linear")


if __name__ == "__main__":
    main()
