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
    cost_function : str
        Cost function for registration (default: normmi).
    use_lesion_mask : bool
        Use lesion mask for registration cost function weighting.
    """

    # Brain extraction settings
    tool: Literal["hdbet", "ants"] = "hdbet"

    # Processing settings
    denoise: bool = False

    # Registration settings
    cost_function: str = "normmi"
    use_lesion_mask: bool = True

    # Cross-modal coregistration settings
    coregister_modalities: bool = True  # Enable cross-modal coregistration

    # Subtraction image settings
    create_subtraction: bool = True  # Create post-pre subtraction image

    # Postcon-only mode handling
    skip_fast_if_postcon_only: bool = True  # Skip FAST when only T1 postcon available
