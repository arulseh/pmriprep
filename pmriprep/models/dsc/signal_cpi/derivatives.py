"""Domain-safe fourth-order signal derivatives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .probability import Array


@dataclass(frozen=True)
class Stencil:
    """Four evaluation points and their unchanged finite-difference rule."""

    points: Array
    step: float
    central: bool

    def apply(self, values: Array, center: Array) -> Array:
        """Combine evaluations in the same order as the scalar reference."""
        a, b, c, d = values
        if self.central:
            return np.asarray((a - 8 * b + 8 * c - d) / (12 * self.step))
        return np.asarray(
            (-25 * center + 48 * a - 36 * b + 16 * c - 3 * d) / (12 * self.step)
        )


def plan_stencil(
    mean: Array,
    index: int,
    scale: float,
    valid_mean: Callable[[Array], bool],
) -> Stencil:
    """Plan the scalar reference's central/one-sided/halved-step evaluations."""
    step = np.finfo(float).eps ** 0.2 * scale
    if not np.isfinite(step) or step <= 0:
        raise ValueError("invalid parameter scale for derivative")
    for _ in range(32):
        offset = np.zeros_like(mean)
        offset[index] = step
        central = np.array([mean + k * offset for k in (-2, -1, 1, 2)])
        if all(valid_mean(point) for point in central):
            return Stencil(central, step, True)
        for sign in (1, -1):
            points = np.array([mean + sign * k * offset for k in (1, 2, 3, 4)])
            if all(valid_mean(point) for point in points):
                return Stencil(points, sign * step, False)
        step /= 2
    raise ValueError("parameter domain too narrow for derivative")


def fourth_order_partial(
    function: Callable[[Array], Array],
    mean: Array,
    value: Array,
    index: int,
    scale: float,
    valid_mean: Callable[[Array], bool],
) -> Array:
    """Differentiate an array function with a domain-safe fourth-order stencil.

    Parameters
    ----------
    function : callable
        Vector or matrix function of the parameter vector.
    mean, value : ndarray
        Evaluation point and corresponding function value.
    index : int
        Parameter coordinate to differentiate.
    scale : float
        Positive characteristic size in that parameter's units.
    valid_mean : callable
        Domain in which the function can be evaluated.

    Returns
    -------
    ndarray
        Partial derivative with the same shape as the function value.

    Raises
    ------
    ValueError
        If the derivative scale is not finite and positive, or the parameter
        domain is too narrow for a stencil.
    """
    step = np.finfo(float).eps ** 0.2 * scale
    if not np.isfinite(step) or step <= 0:
        raise ValueError("invalid parameter scale for derivative")
    for _ in range(32):
        offset = np.zeros_like(mean)
        offset[index] = step
        central = [mean + k * offset for k in (-2, -1, 1, 2)]
        if all(valid_mean(point) for point in central):
            fm2, fm1, fp1, fp2 = (function(point) for point in central)
            return np.asarray((fm2 - 8 * fm1 + 8 * fp1 - fp2) / (12 * step))
        for sign in (1, -1):
            points = [mean + sign * k * offset for k in (1, 2, 3, 4)]
            if all(valid_mean(point) for point in points):
                f1, f2, f3, f4 = (function(point) for point in points)
                return np.asarray(
                    (-25 * value + 48 * f1 - 36 * f2 + 16 * f3 - 3 * f4)
                    / (12 * sign * step)
                )
        step /= 2
    raise ValueError("parameter domain too narrow for derivative")
