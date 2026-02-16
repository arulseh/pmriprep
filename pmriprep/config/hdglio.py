"""HD-GLIO tumor segmentation configuration dataclass."""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class HDGLIOConfig:
    """Configuration for HD-GLIO tumor segmentation.

    This dataclass contains all parameters for HD-GLIO deep learning
    tumor segmentation, including mask processing and GPU settings.

    Attributes
    ----------
    lesion_dilation : float
        Dilation radius (mm) for lesion mask output.
    nawm_erosion : int
        Erosion iterations for NAWM mask (voxels).
    nawm_min_voxels : int
        Minimum voxels required in NAWM ROI.
    device : Literal["cuda", "mps", "cpu"]
        Device for HD-GLIO inference: cuda (NVIDIA GPU), mps (Apple Silicon), or cpu.
    conda_env : str, optional
        Conda environment name where hd_glio_predict is installed. When set,
        the command is wrapped with ``conda run -n <env>``. Required when
        HD-GLIO has conflicting dependencies (e.g., "hdglio-mps" on Apple
        Silicon, "hdbet-env" on Linux).
    """

    # Mask processing settings
    lesion_dilation: float = 15.0
    nawm_erosion: int = 2
    nawm_min_voxels: int = 500

    # GPU settings
    device: Literal["cuda", "mps", "cpu"] = "cuda"
    conda_env: Optional[str] = None
