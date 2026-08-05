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
    nawm_erosion : int
        Erosion iterations for NAWM mask (voxels).
    nawm_min_voxels : int
        Minimum voxels required in NAWM ROI.
    device : Literal["cuda", "mps", "cpu"]
        Execution device requested from the configured HD-GLIO environment.
        ``mps`` requires an HD-GLIO installation that supports that backend.
    conda_env : str, optional
        Conda environment name where the segmentation tool is installed.
        When set, the command is wrapped with ``conda run -n <env>``.
        Required when the tool has conflicting dependencies (e.g., a
        dedicated ``"tumor-seg-runtime"`` environment).
    """

    # Tool selection (for future extensibility)
    tool: Literal["hdglio"] = "hdglio"

    # Mask processing settings
    nawm_erosion: int = 2
    nawm_min_voxels: int = 500

    # GPU settings
    device: Literal["cuda", "mps", "cpu"] = "cuda"
    conda_env: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate public tumor-segmentation configuration values."""
        if self.tool != "hdglio":
            raise ValueError("Invalid tool={!r}. Valid: ['hdglio']".format(self.tool))
        if self.device not in {"cuda", "mps", "cpu"}:
            raise ValueError(
                f"Invalid device={self.device!r}. Valid: ['cpu', 'cuda', 'mps']"
            )
        for option in ("nawm_erosion", "nawm_min_voxels"):
            value = getattr(self, option)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{option} must be an integer")
            if value < 0:
                raise ValueError(f"{option} must be non-negative")
