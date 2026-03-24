"""Logging configuration for PMRIPrep.

Centralizes all logging setup, matching fMRIPrep's ``config.loggers`` pattern.
Provides custom log levels (IMPORTANT, VERBOSE), a console handler on stdout,
and an optional per-subject FileHandler that writes ``pmriprep.log`` to disk.

.. warning::
    This module shadows the stdlib ``logging`` module within ``pmriprep.cli``.
    Other modules in this package must use ``import logging as stdlib_logging``
    or import the stdlib before this module if they need it directly.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Custom log levels (fMRIPrep convention)
IMPORTANT = 25
VERBOSE = 15
logging.addLevelName(IMPORTANT, "IMPORTANT")
logging.addLevelName(VERBOSE, "VERBOSE")

# Log format matching fMRIPrep style
LOG_FORMAT = "%(asctime)s %(name)-2s %(levelname)-2s:\n\t %(message)s"
LOG_DATEFMT = "%y%m%d-%H:%M:%S"

# Logger instances (fMRIPrep pattern: module-level, accessible everywhere)
cli = logging.getLogger("cli")
cli.propagate = False  # prevent double-write via root logger
workflow = logging.getLogger("nipype.workflow")
interface = logging.getLogger("nipype.interface")
utils = logging.getLogger("nipype.utils")


def init(
    log_level: int = IMPORTANT, log_dir: Optional[Path] = None
) -> None:
    """Initialize all loggers.

    Sets the console handler level to *log_level* so that only messages at or
    above that severity are shown on the terminal.  The FileHandler (when
    *log_dir* is given) is always set to DEBUG so the log file captures
    everything.

    Args:
        log_level: Console logging level (default: 25/IMPORTANT).
        log_dir: Directory for log files. If provided, a FileHandler is added
            that writes ``pmriprep.log`` alongside Nipype's ``pypeline.log``.
    """
    # Console handler (stdout, like fMRIPrep).
    # Logger level is set to DEBUG so messages reach both handlers;
    # the console handler gates output via its own level.
    _has_console = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        for h in cli.handlers
    )
    if not _has_console:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT))
        handler.setLevel(log_level)
        cli.addHandler(handler)
    else:
        # Update existing console handler level on re-init
        for h in cli.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(
                h, logging.FileHandler
            ):
                h.setLevel(log_level)

    # Set logger levels to DEBUG so messages reach both handlers
    for lgr in (cli, workflow, interface, utils):
        lgr.setLevel(logging.DEBUG)

    # FileHandlers for pmriprep.log — captures cli + interface messages.
    # Close any existing FileHandlers from a previous subject before adding new.
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "pmriprep.log", mode="w")
        fh.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT))
        fh.setLevel(logging.DEBUG)  # capture everything to file

        # cli logger (named "cli", not under pmriprep hierarchy)
        for h in cli.handlers[:]:
            if isinstance(h, logging.FileHandler):
                h.close()
                cli.removeHandler(h)
        cli.addHandler(fh)

        # pmriprep parent logger — captures pmriprep.interfaces.*, pmriprep.utils.*, etc.
        pmriprep_logger = logging.getLogger("pmriprep")
        pmriprep_logger.setLevel(logging.DEBUG)
        for h in pmriprep_logger.handlers[:]:
            if isinstance(h, logging.FileHandler):
                h.close()
                pmriprep_logger.removeHandler(h)
        pmriprep_logger.addHandler(fh)
