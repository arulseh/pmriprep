"""Validated CPI policy choices, separate from numerical model settings."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CPIOptions:
    """Acquisition-level CPI options.

    Parameters
    ----------
    control_points : int
        Zero chooses automatically; a manual count must be in [5,12] and also
        feasible for the acquisition's frame count and knot-time geometry.
    infer_delay : bool
        Infer arrival, or condition it at zero and report it as unestimated.
    dispersion : {"auto", "none", "gamma"}
        Select transport using ordinary-VI scores, omit it, or require gamma.
    fallback : {"osvd", "none"}
        Replace rejected fits using oSVD, or preserve missing measurements.
    """

    control_points: int = 0
    infer_delay: bool = True
    dispersion: Literal["auto", "none", "gamma"] = "none"
    fallback: Literal["osvd", "none"] = "osvd"

    def __post_init__(self) -> None:
        """Reject unsupported choices without silently coercing configuration."""
        if isinstance(self.control_points, bool) or not isinstance(self.control_points, int):
            raise TypeError("cpi_control_points must be an integer")
        if self.control_points != 0 and not 5 <= self.control_points <= 12:
            raise ValueError("cpi_control_points must be 0 or an integer from 5 to 12")
        if not isinstance(self.infer_delay, bool):
            raise TypeError("cpi_infer_delay must be a boolean")
        if self.dispersion not in ("auto", "none", "gamma"):
            raise ValueError("cpi_dispersion must be auto, none or gamma")
        if self.fallback not in ("osvd", "none"):
            raise ValueError("cpi_fallback must be osvd or none")
