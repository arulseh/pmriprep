"""HD-GLIO tumor segmentation configuration dataclass."""

from dataclasses import dataclass
from typing import Literal


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
    """

    # Mask processing settings
    lesion_dilation: float = 15.0
    nawm_erosion: int = 2
    nawm_min_voxels: int = 500

    # GPU settings
    device: Literal["cuda", "mps", "cpu"] = "cuda"
