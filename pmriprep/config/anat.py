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
    skip_acpc : bool
        Skip ACPC alignment to MNI template.
    cost_function : str
        Cost function for registration (default: normmi).
    use_lesion_mask : bool
        Use lesion mask for registration cost function weighting.
    """

    # Brain extraction settings
    tool: Literal["hdbet", "ants"] = "hdbet"

    # Processing settings
    denoise: bool = False
    skip_acpc: bool = False

    # Registration settings
    cost_function: str = "normmi"
    use_lesion_mask: bool = True
