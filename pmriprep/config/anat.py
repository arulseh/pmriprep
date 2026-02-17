"""Anatomical preprocessing configuration dataclass."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class AnatConfig:
    """Configuration for anatomical preprocessing.

    This dataclass contains all parameters for anatomical preprocessing,
    including brain extraction tool selection, preprocessing options,
    and registration settings.

    Attributes
    ----------
    tool : Literal["hdbet", "ants"]
        Brain extraction tool: hdbet (HD-BET) or ants (ANTs).
    denoise : bool
        Apply ANTs denoising to anatomical images.
    use_lesion_mask : bool
        Use lesion mask for registration cost function weighting.
        Default False: 6-DOF FLIRT is robust to focal lesions.
        Enable for large (>3cm) enhancing tumors, resection cavities, or BBR.
    """

    # Brain extraction settings
    tool: Literal["hdbet", "ants"] = "hdbet"

    # Processing settings
    denoise: bool = False

    # Registration settings
    use_lesion_mask: bool = False

    # Cross-modal coregistration settings
    coregister_modalities: bool = True  # Enable cross-modal coregistration

    # Subtraction image settings
    create_subtraction: bool = True  # Create post-pre subtraction image

