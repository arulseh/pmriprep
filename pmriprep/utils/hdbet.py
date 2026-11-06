"""Shared HD-BET brain extraction subprocess wrapper.

Consolidates HD-BET invocation logic used by DSC and DCE brain
extraction interfaces.  Handles device auto-resolution, conda
environment wrapping, subprocess timeout, error logging, and
mask derivation from the brain-extracted output.
"""

import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from pmriprep.utils.processes import run_isolated_command

logger = logging.getLogger(__name__)

_TTA_TIMEOUT_MULTIPLIER = 8


def hdbet_thread_budget(device: str, limit: int) -> int:
    """Match HD-BET's CPU reservation and launcher thread limit.

    Parameters
    ----------
    device : str
        Resolved Torch device: cpu, cuda or mps.
    limit : int
        Positive per-process CPU limit within the run budget.

    Returns
    -------
    int
        Full limit for CPU inference, or at most two CPUs for GPU inference.
    """
    return min(limit, 2) if device in {"cuda", "mps"} else limit


def build_hdbet_command(
    in_file: str, out_brain: str, device: str, use_tta: bool, conda_env: str = "",
) -> list[str]:
    """Build a bounded single-image HD-BET command in its Python environment.

    Parameters
    ----------
    in_file, out_brain : str
        Input image and output brain-image paths.
    device : str
        Resolved Torch device.
    use_tta : bool
        Whether to enable mirrored predictions.
    conda_env : str
        Optional HD-BET environment. Otherwise use the installed ``hd-bet``
        Python entry point's interpreter, falling back to the current Python.

    Returns
    -------
    list of str
        Subprocess arguments, including the exported per-process thread budget.
    """
    if conda_env:
        command = ["conda", "run", "-n", conda_env, "--no-capture-output", "python"]
    else:
        command = [sys.executable]
        executable = shutil.which("hd-bet")
        if executable:
            try:
                with open(executable, "rb") as stream:
                    shebang = stream.readline(512).decode("utf-8").strip()
                interpreter = shlex.split(shebang[2:]) if shebang.startswith("#!") else []
                if interpreter and any("python" in Path(part).name for part in interpreter):
                    command = interpreter
            except (OSError, UnicodeError, ValueError):
                pass
    threads = hdbet_thread_budget(device, max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    command += [str(Path(__file__).with_name("hdbet_runner.py")),
                "-i", str(in_file), "-o", str(out_brain), "-device", device,
                "--threads", str(threads)]
    if not use_tta:
        command.append("--disable_tta")
    return command


def resolve_hdbet_execution(
    device: str,
    sloppy: bool = False,
) -> tuple[str, bool]:
    """Resolve the HD-BET device and test-time augmentation policy.

    Parameters
    ----------
    device : str
        Requested Torch device: ``"auto"``, ``"cuda"``, ``"mps"``, or
        ``"cpu"``.
    sloppy : bool
        Whether fast testing settings are requested.

    Returns
    -------
    resolved_device : str
        Concrete Torch device.
    use_tta : bool
        True when mirrored test-time augmentation should be enabled.

    Raises
    ------
    ValueError
        If the device is not ``auto``, ``cpu``, ``cuda`` or ``mps``.
    """
    if device == "auto":
        from pmriprep.utils.device import detect_torch_device

        device = detect_torch_device()
    if device not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported HD-BET device: {device!r}")
    return device, device in {"cuda", "mps"} and not sloppy


def run_hdbet(
    in_file: str,
    out_brain: str,
    device: str = "auto",
    conda_env: str = "",
    timeout: int = 300,
    sloppy: bool = False,
) -> tuple[bool, Optional[str]]:
    """Run HD-BET brain extraction as a subprocess.

    Builds and executes the bounded HD-BET runner, optionally via a conda
    environment.  On success, derives a binary brain mask by binarizing
    the brain-extracted output (more consistent than HD-BET's own
    ``_mask.nii.gz``).

    Parameters
    ----------
    in_file : str
        Path to the input NIfTI image.
    out_brain : str
        Path for the brain-extracted output NIfTI.
    device : str
        Torch device for HD-BET inference.  ``"auto"`` resolves
        via :func:`~pmriprep.utils.device.detect_torch_device`.
    conda_env : str
        Conda environment name.  When non-empty, HD-BET is
        invoked via ``conda run -n <env>``.
    timeout : int
        Base subprocess timeout in seconds (default 300 = 5 min). When TTA is
        enabled, this limit is multiplied by eight to cover the mirrored
        predictions.
    sloppy : bool
        Disable test-time augmentation for faster testing. TTA is also
        disabled on CPU and otherwise enabled on CUDA or MPS.

    Returns
    -------
    tuple
        Tuple of ``(succeeded, mask_path)``:
            - *succeeded*: True if HD-BET produced a valid brain image.
            - *mask_path*: Path to the derived binary mask, or None on
              failure.
    """
    device, use_tta = resolve_hdbet_execution(device, sloppy)
    effective_timeout = timeout * (_TTA_TIMEOUT_MULTIPLIER if use_tta else 1)

    logger.info(
        "Starting HD-BET on %s with test-time augmentation %s",
        device,
        "enabled" if use_tta else "disabled",
    )

    cmd = build_hdbet_command(in_file, out_brain, device, use_tta, conda_env)

    try:
        result = run_isolated_command(
            cmd,
            timeout=effective_timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("HD-BET timed out after %d seconds", effective_timeout)
        return False, None
    except (FileNotFoundError, OSError) as e:
        # OSError covers Apptainer/Singularity missing-executable cases
        logger.warning("HD-BET executable not found: %s", e)
        return False, None

    if result.returncode != 0:
        logger.warning(
            "HD-BET failed (exit %d): %s",
            result.returncode,
            result.stderr.strip() if result.stderr else "no stderr",
        )
        return False, None

    # Verify output exists
    brain_path = Path(out_brain)
    if not brain_path.exists():
        logger.warning("HD-BET brain output not found at %s", out_brain)
        return False, None

    # Empty brain guard: HD-BET may exit 0 but produce all-zero output
    brain_img = nib.load(str(brain_path))
    brain_data = brain_img.get_fdata()  # type: ignore[attr-defined]
    if np.max(brain_data) == 0:
        logger.warning("HD-BET produced all-zero brain output")
        return False, None

    # Derive binary mask by binarizing brain output (more consistent
    # than HD-BET's own _mask.nii.gz which uses internal thresholding)
    mask_data = (brain_data > 0).astype(np.uint8)
    mask_path = str(
        brain_path.parent
        / brain_path.name.replace(".nii.gz", "_mask.nii.gz")
    )
    nib.save(
        nib.Nifti1Image(mask_data, brain_img.affine, brain_img.header),  # type: ignore[attr-defined]
        mask_path,
    )

    logger.info("HD-BET completed successfully")
    return True, mask_path
