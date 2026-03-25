"""DSC brain extraction interface with HD-BET and BET fallback.

Provides a proper SimpleInterface for DSC brain extraction, replacing the
previous Function node wrapper.  This ensures correct nipype GPU slot
scheduling via the ``use_gpu`` trait and provides a robust fallback chain:

1. **HD-BET** — deep learning, most accurate for pathological brains
2. **FSL BET -F** — functional/EPI mode, handles T2* contrast and distortion
3. **Intensity threshold** — 10th percentile, last resort

BET ``-F`` (not ``-R``) is used for DSC because ``-R``'s center-of-gravity
iteration can fail with EPI susceptibility distortion, while ``-F`` is
specifically designed for EPI cleanup.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

logger = logging.getLogger(__name__)


class DSCBrainExtractInputSpec(BaseInterfaceInputSpec):
    """Input specification for DSCBrainExtract."""

    dsc_mean_file = File(
        exists=True,
        mandatory=True,
        desc="Pre-bolus mean DSC image for brain extraction",
    )
    hdbet_device = traits.Enum(
        "auto", "cuda", "mps", "cpu",
        usedefault=True,
        desc="Device for HD-BET: auto, cuda, mps, or cpu (default: auto)",
    )
    hdbet_conda_env = traits.Str(
        desc="Conda environment name for HD-BET.",
    )
    use_gpu = traits.Bool(
        False,
        usedefault=True,
        desc="Whether this node uses GPU (for nipype scheduling)",
    )
    bet_frac = traits.Float(
        0.3,
        usedefault=True,
        desc="BET fractional intensity threshold (default: 0.3 for DSC)",
    )


class DSCBrainExtractOutputSpec(TraitedSpec):
    """Output specification for DSCBrainExtract."""

    dsc_brain_file = File(desc="Brain-extracted DSC mean image")
    dsc_brain_mask_file = File(desc="Binary brain mask in DSC native space")
    method_used = traits.Str(desc="Extraction method that succeeded")


class DSCBrainExtract(SimpleInterface):
    """Extract brain from DSC pre-bolus mean image.

    Uses a three-level fallback chain to ensure brain extraction always
    succeeds:

    1. **HD-BET** — deep learning skull stripping (Isensee et al., 2019).
       Most accurate, handles pathological anatomy.
    2. **FSL BET -F** — functional/EPI mode.  Designed for T2*-weighted
       EPI data with susceptibility distortion.
    3. **Intensity threshold** — 10th percentile of nonzero voxels.
       Crude but always succeeds.

    The brain mask is used for registration quality (T1w → DSC) and as a
    search region for AIF detection.  The final processing mask comes from
    transforming the T1w brain mask through registration.
    """

    input_spec = DSCBrainExtractInputSpec
    output_spec = DSCBrainExtractOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        """Execute DSC brain extraction with fallback chain."""
        dsc_mean = Path(self.inputs.dsc_mean_file)

        out_dir = Path.cwd()
        out_brain = str(out_dir / "dsc_brain.nii.gz")
        out_mask = str(out_dir / "dsc_brain_mask.nii.gz")

        method_used = "hdbet"

        # Try HD-BET first
        success = self._run_hdbet(dsc_mean, out_brain, out_mask)
        if not success:
            logger.warning("HD-BET failed, falling back to FSL BET -F")
            method_used = "bet"
            success = self._run_bet(dsc_mean, out_brain, out_mask)
        if not success:
            logger.warning("FSL BET failed, falling back to intensity threshold")
            method_used = "threshold"
            self._run_threshold(dsc_mean, out_brain, out_mask)

        self._results["dsc_brain_file"] = out_brain
        self._results["dsc_brain_mask_file"] = out_mask
        self._results["method_used"] = method_used

        return runtime

    def _run_hdbet(
        self, dsc_mean: Path, out_brain: str, out_mask: str,
    ) -> bool:
        """Run HD-BET via shared utility."""
        from nipype.interfaces.base import isdefined

        from pmriprep.utils.hdbet import run_hdbet

        conda_env = (
            self.inputs.hdbet_conda_env
            if isdefined(self.inputs.hdbet_conda_env)
            else ""
        )

        succeeded, mask_path = run_hdbet(
            in_file=str(dsc_mean),
            out_brain=out_brain,
            device=self.inputs.hdbet_device,
            conda_env=conda_env,
        )
        if succeeded and mask_path:
            if str(mask_path) != out_mask:
                shutil.copy2(mask_path, out_mask)
        return succeeded

    def _run_bet(
        self, dsc_mean: Path, out_brain: str, out_mask: str,
    ) -> bool:
        """Run FSL BET with -F flag for functional/EPI data.

        Uses ``-F`` for EPI cleanup and ``-f 0.3`` for the low tissue
        contrast typical of DSC pre-bolus mean images.  Does NOT use
        ``-R`` (robust) because its center-of-gravity iteration can fail
        with EPI susceptibility distortion.

        Always derives mask by binarizing the brain output rather than
        relying on BET's ``-m`` flag, because ``-F`` and ``-m`` interaction
        can produce inconsistent masks.
        """
        cmd = [
            "bet",
            str(dsc_mean),
            out_brain,
            "-F",
            "-f", str(self.inputs.bet_frac),
        ]

        try:
            subprocess.run(
                cmd, check=True, capture_output=True, text=True,
            )
            logger.info("FSL BET -F completed successfully")

            # Derive mask by binarizing brain output
            if Path(out_brain).exists():
                brain_img = nib.load(out_brain)
                brain_data = brain_img.get_fdata()
                mask_data = (brain_data > 0).astype(np.uint8)
                nib.save(
                    nib.Nifti1Image(
                        mask_data, brain_img.affine, brain_img.header,
                    ),
                    out_mask,
                )
                return True

            return False

        except subprocess.CalledProcessError as e:
            logger.warning(
                "FSL BET failed: %s", e.stderr.strip() if e.stderr else str(e),
            )
            return False
        except (FileNotFoundError, OSError) as e:
            logger.warning("FSL bet command not found: %s", e)
            return False

    def _run_threshold(
        self, dsc_mean: Path, out_brain: str, out_mask: str,
    ) -> None:
        """Create brain mask using intensity thresholding (last resort).

        Uses the 10th percentile of nonzero voxels as threshold.
        """
        logger.info("Using threshold-based brain extraction (fallback)")

        dsc_img = nib.load(str(dsc_mean))
        dsc_data = dsc_img.get_fdata()

        nonzero = dsc_data[dsc_data > 0]
        threshold = float(np.percentile(nonzero, 10)) if len(nonzero) > 0 else 0.0

        mask_data = (dsc_data > threshold).astype(np.uint8)
        brain_data = dsc_data * mask_data

        nib.save(
            nib.Nifti1Image(
                brain_data.astype(np.float32), dsc_img.affine, dsc_img.header,
            ),
            out_brain,
        )
        nib.save(
            nib.Nifti1Image(mask_data, dsc_img.affine, dsc_img.header),
            out_mask,
        )
