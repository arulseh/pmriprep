"""Tumor segmentation configuration dataclass."""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class TumorSegConfig:
    """Configuration for tumor segmentation.

    This dataclass contains all parameters for deep learning tumor
    segmentation, including mask processing, GPU settings, and tool
    selection.

    Attributes
    ----------
    tool : Literal["hdglio"]
        Segmentation tool to use. Currently only HD-GLIO is supported.
    lesion_dilation : float
        Dilation radius (mm) for lesion mask output.
    nawm_erosion : int
        Erosion iterations for NAWM mask (voxels).
    nawm_min_voxels : int
        Minimum voxels required in NAWM ROI.
    device : Literal["cuda", "mps", "cpu"]
        Device for inference: cuda (NVIDIA GPU), mps (Apple Silicon), or cpu.
    conda_env : str, optional
        Conda environment name where the segmentation tool is installed.
        When set, the command is wrapped with ``conda run -n <env>``.
        Required when the tool has conflicting dependencies (e.g.,
        "hdglio-mps" on Apple Silicon, "hdbet-env" on Linux).
    """

    # Tool selection (for future extensibility)
    tool: Literal["hdglio"] = "hdglio"

    # Mask processing settings
    lesion_dilation: float = 15.0
    nawm_erosion: int = 2
    nawm_min_voxels: int = 500

    # GPU settings
    device: Literal["cuda", "mps", "cpu"] = "cuda"
    conda_env: Optional[str] = None
