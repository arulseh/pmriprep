"""Conditional BSW coefficient guidance and empirical reference uncertainty."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt
from scipy.integrate import cumulative_trapezoid

from pmriprep.utils.dsc_constants import EPSILON_CONCENTRATION

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


class BSWGuidanceStatus(IntEnum):
    """Availability of a conditional coefficient prior, not a fit-quality rank."""

    AVAILABLE = 0
    OUTSIDE_SUPPORT = 1
    REFERENCE_UNAVAILABLE = 2
    NONFINITE_TISSUE = 3
    COEFFICIENT_UNAVAILABLE = 4
    STANDARD_ERROR_UNAVAILABLE = 5
    SUPPLIED_COEFFICIENT = 6


@dataclass(frozen=True)
class BSWGuidance:
    """BSW quantities on the exact correction grid and selected support.

    Attributes
    ----------
    reference_curve, reference_integral : ndarray or None
        Exact correction reference in delta-R2-star and its seconds integral.
    reference_basis : ndarray or None
        Shape (frame, rank); its product with its transpose is the empirical
        covariance of an iid reference-region mean. None means unavailable,
        while a zero-column array means measured zero empirical variation.
    reference_mask : ndarray or None
        Exact automatically selected voxel membership on the input grid.
    reference_count : int
        Selected sample count; zero when membership is unavailable.
    reference_reason : str or None
        Reason reference guidance is unavailable, independent of correction QC.
    k2, k2_sd : ndarray
        Original signed coefficient and conditional OLS standard error in s^-1.
        Unavailable standard errors are NaN, not an invented noise floor.
    status : ndarray
        Per-voxel BSWGuidanceStatus; not a measure of accuracy or convergence.
    baseline_frames : int
        Actual baseline-centering window used by the correction.
    tr : float
        Uniform acquisition interval in seconds.
    degrees_of_freedom : int
        Number of observations minus the two regression coefficients.

    Notes
    -----
    The regression reuses tissue observations. Reference covariance conditions
    on selected membership and ignores inter-voxel correlation. Neither
    quantity establishes independent measurement uncertainty or coverage.
    """

    reference_curve: FloatArray | None
    reference_integral: FloatArray | None
    reference_basis: FloatArray | None
    reference_mask: BoolArray | None
    reference_count: int
    reference_reason: str | None
    k2: FloatArray
    k2_sd: FloatArray
    status: npt.NDArray[np.uint8]
    baseline_frames: int
    tr: float
    degrees_of_freedom: int

    @property
    def available(self) -> BoolArray:
        """Return voxels with a finite coefficient and positive conditional SE."""
        return np.asarray(self.status == BSWGuidanceStatus.AVAILABLE, dtype=np.bool_)


def build_bsw_guidance(
    concentration: npt.NDArray[np.floating],
    k2: npt.NDArray[np.floating],
    *,
    tr: float,
    baseline_frames: int,
    brain_mask: BoolArray | None,
    reference_curve: npt.NDArray[np.floating] | None,
    reference_mask: BoolArray | None,
    reference_failure: str | None = None,
    supplied_coefficient: bool = False,
) -> BSWGuidance:
    """Describe existing BSW results without changing their means or correction.

    Parameters
    ----------
    concentration : ndarray
        Baseline-centered delta-R2-star, shape (x, y, z, frame).
    k2 : ndarray
        Coefficients already used by the correction, on the spatial grid.
    tr : float
        Positive finite sample interval in seconds.
    baseline_frames : int
        Actual baseline window, after the correction's clamping.
    brain_mask : ndarray or None
        Original regression support; None uses the correction's signal floor.
    reference_curve : ndarray or None
        Exact mean used for the correction, not a newly selected mean.
    reference_mask : ndarray or None
        Membership of that mean; None cannot supply empirical covariance.
    reference_failure : str or None, optional
        Existing selector's reason when no reference was available.
    supplied_coefficient : bool, optional
        True when coefficients were supplied rather than estimated here.

    Returns
    -------
    BSWGuidance
        Separate coefficient/reference guidance and explicit availability.
    """
    values = np.asarray(concentration, dtype=np.float64)
    coefficients = np.asarray(k2, dtype=np.float64).copy()
    if values.ndim != 4 or values.shape[-1] < 4 or coefficients.shape != values.shape[:3]:
        raise ValueError("matching 4D concentration and 3D coefficient maps required")
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError("positive finite TR in seconds required")
    if (isinstance(baseline_frames, (bool, np.bool_))
            or not isinstance(baseline_frames, (int, np.integer))
            or not 0 < baseline_frames < values.shape[-1]):
        raise ValueError("interior integer baseline window required")
    for mask in (brain_mask, reference_mask):
        if mask is not None and (mask.shape != values.shape[:3] or mask.dtype != np.bool_):
            raise ValueError("boolean masks must match the concentration grid")
    support = (np.max(values, axis=-1) > EPSILON_CONCENTRATION
               if brain_mask is None else brain_mask)
    status = np.full(values.shape[:3], BSWGuidanceStatus.OUTSIDE_SUPPORT, dtype=np.uint8)
    status[support] = BSWGuidanceStatus.REFERENCE_UNAVAILABLE
    sd = np.full(values.shape[:3], np.nan)
    membership = None if reference_mask is None else reference_mask.copy()
    count = 0 if membership is None else int(np.count_nonzero(membership))
    curve = None if reference_curve is None else np.array(reference_curve, dtype=np.float64, copy=True)
    integral = basis = None
    reason = reference_failure
    dof = values.shape[-1] - 2
    if curve is None:
        reason = reason or "reference_unavailable"
    elif curve.shape != (values.shape[-1],) or not np.all(np.isfinite(curve)):
        reason = "invalid_reference_curve"
    else:
        integral = np.asarray(cumulative_trapezoid(curve, dx=tr, initial=0.0))
        design = np.column_stack((curve, -integral))
        if not np.all(np.isfinite(integral)):
            reason = "nonfinite_reference_integral"
        elif np.linalg.matrix_rank(design) != 2:
            reason = "rank_deficient_reference"
        elif membership is None:
            reason = "reference_samples_unavailable"
        elif count < 2:
            reason = "insufficient_reference_samples"
        elif not np.all(np.isfinite(values[membership])):
            reason = "nonfinite_reference_samples"
        else:
            samples = values[membership]
            if not np.array_equal(np.asarray(concentration)[membership].mean(axis=0), curve):
                raise ValueError("reference membership does not reproduce the correction mean")
            integrals = cumulative_trapezoid(samples, dx=tr, axis=1, initial=0.0)
            centered = integrals - integrals.mean(axis=0)
            _, singular, vh = np.linalg.svd(centered, full_matrices=False)
            keep = singular > np.finfo(float).eps * max(centered.shape) * singular.max()
            basis = vh[keep].T * (singular[keep] / np.sqrt(count * (count - 1)))
            basis[0] = 0.0
            reason = None
            if supplied_coefficient:
                status[support] = BSWGuidanceStatus.SUPPLIED_COEFFICIENT
            else:
                finite_tissue = np.all(np.isfinite(values), axis=-1)
                status[support & ~finite_tissue] = BSWGuidanceStatus.NONFINITE_TISSUE
                fitted = support & finite_tissue
                status[fitted] = BSWGuidanceStatus.COEFFICIENT_UNAVAILABLE
                fitted &= np.isfinite(coefficients)
                if np.any(fitted):
                    observations = values[fitted]
                    beta = np.linalg.lstsq(design, observations.T, rcond=None)[0]
                    if not np.allclose(beta[1], coefficients[fitted], rtol=1e-10, atol=1e-14):
                        raise ValueError("coefficient provenance differs from the BSW regression")
                    residual = observations - (design @ beta).T
                    inverse = np.linalg.pinv(design)
                    standard_error = np.sqrt(
                        np.sum(residual ** 2, axis=1) / dof * np.sum(inverse[1] ** 2)
                    )
                    usable = np.isfinite(standard_error) & (standard_error > 0)
                    sd[fitted] = np.where(usable, standard_error, np.nan)
                    status[fitted] = np.where(
                        usable, BSWGuidanceStatus.AVAILABLE,
                        BSWGuidanceStatus.STANDARD_ERROR_UNAVAILABLE,
                    )
    return BSWGuidance(
        curve, integral, basis, membership, count, reason, coefficients,
        sd, status, int(baseline_frames), float(tr), dof,
    )
