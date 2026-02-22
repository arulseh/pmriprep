"""Anatomical preprocessing configuration dataclass."""

from dataclasses import dataclass
from typing import Literal, Optional


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
    hdbet_conda_env : Optional[str]
        Conda environment name for HD-BET (e.g. ``'hdbet-env'``).
        When ``None`` (the default), HD-BET is called directly without
        ``conda run`` wrapping — this is the expected Docker behaviour.
    """

    # Brain extraction settings
    tool: Literal["hdbet", "ants"] = "hdbet"
    hdbet_conda_env: Optional[str] = None

    # Processing settings
    denoise: bool = False

    # Subtraction image settings
    create_subtraction: bool = True  # Create post-pre subtraction image

    # Testing / debug
    sloppy: bool = False  # Use fast, low-quality settings for testing

