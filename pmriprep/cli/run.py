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
