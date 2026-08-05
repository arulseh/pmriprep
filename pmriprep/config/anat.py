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
    hdbet_device : Literal["auto", "cuda", "mps", "cpu"]
        Device for HD-BET: ``'auto'`` detects CUDA > MPS > CPU.
    denoise : bool
        Apply ANTs denoising to anatomical images.
    hdbet_conda_env : Optional[str]
        Conda environment name for HD-BET (e.g. ``'hdbet-runtime'``).
        When ``None`` (the default), HD-BET is called directly without
        ``conda run`` wrapping — this is the expected Docker behaviour.
    """

    # Brain extraction settings
    tool: Literal["hdbet", "ants"] = "hdbet"
    hdbet_device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    hdbet_conda_env: Optional[str] = None

    # Processing settings
    denoise: bool = False

    # Subtraction image settings
    create_subtraction: bool = False  # Create a post-contrast minus pre-contrast T1w subtraction image

    # Testing / debug
    sloppy: bool = False  # Use fast, low-quality settings for testing

    def __post_init__(self) -> None:
        """Validate public anatomical configuration values."""
        if self.tool not in {"hdbet", "ants"}:
            raise ValueError(f"Invalid tool={self.tool!r}. Valid: ['ants', 'hdbet']")
        if self.hdbet_device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError(
                f"Invalid hdbet_device={self.hdbet_device!r}. "
                "Valid: ['auto', 'cpu', 'cuda', 'mps']"
            )
        for option in ("denoise", "create_subtraction", "sloppy"):
            if not isinstance(getattr(self, option), bool):
                raise TypeError(f"{option} must be a boolean")
