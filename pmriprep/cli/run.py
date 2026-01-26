"""PMRIPrep CLI entry point."""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from pmriprep import __version__


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

    # TODO: Implement workflow execution (US-085)
    raise NotImplementedError("Workflow execution not yet implemented - see US-085")


if __name__ == "__main__":
    main()
