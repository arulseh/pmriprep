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
    """

    # Brain extraction settings
    tool: Literal["hdbet", "ants"] = "hdbet"

    # Processing settings
    denoise: bool = False

    # Subtraction image settings
    create_subtraction: bool = True  # Create post-pre subtraction image

