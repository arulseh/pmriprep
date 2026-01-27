"""Control Point Interpolation (CPI) model for DSC perfusion.

Implements a non-parametric model where the residue function is represented
as a PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) spline through
N control points. This approach allows flexible residue shapes while
maintaining physiological constraints.

The CPI model is particularly useful when the underlying transit time
distribution is unknown or doesn't follow a standard parametric form.

References:
    Mouridsen K, Christensen S, Gyldensted L, Østergaard L. (2006).
    "Automatic selection of arterial input function using cluster analysis."
    Magnetic Resonance in Medicine 55(3):524-531.

    Mehndiratta A, MacIntosh BJ, Crane DE, et al. (2013).
    "A control point interpolation method for the non-parametric quantification
    of cerebral haemodynamics from dynamic susceptibility contrast MRI."
    NeuroImage 64:560-570.
"""

from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt
from scipy import interpolate  # type: ignore[import-untyped]

from pmriprep.models.dsc.base import (
    DSCFitResult,
    DSCModel,
    DSCModelParameter,
)
from pmriprep.utils.dsc_constants import (
    CBF_BOUNDS,
    EPSILON_CONCENTRATION,
)


# Default bounds for CPI control points
CPI_N_MIN: int = 4
CPI_N_MAX: int = 12
CPI_DEFAULT_N: int = 6


class CPIModel(DSCModel):
    """Control Point Interpolation model for non-parametric residue estimation.

    The residue function is represented as a PCHIP spline through N control
    points. Control points are evenly spaced in time (or optimally placed),
    and the spline interpolation ensures a smooth residue curve.

    Constraints:
        - R(0) = CBF: The first control point determines CBF
        - Monotonicity: R(t) is monotonically decreasing (or non-increasing)
        - Non-negativity: R(t) >= 0 for all t

    The model automatically selects the number of control points based on
    bolus duration, or uses a user-specified value.

    Attributes:
        n_control_points: Number of control points in the spline (4-12).
        time_range: Time range (t_min, t_max) for control point placement.

    Parameters:
        - r0: First control point = CBF (mL/100g/min)
        - r1, r2, ..., r(N-1): Subsequent control points (mL/100g/min)
          These are constrained to be <= previous point (monotonicity)

    Example:
        For N=6 control points:
        - Parameters: [r0, r1, r2, r3, r4, r5]
        - r0 = CBF (maximum of residue function)
        - Control point times: evenly spaced from 0 to t_max

    References:
        Mehndiratta A, et al. (2013). "A control point interpolation method
        for the non-parametric quantification of cerebral haemodynamics from
        dynamic susceptibility contrast MRI." NeuroImage 64:560-570.
    """

    def __init__(
        self,
        n_control_points: int = CPI_DEFAULT_N,
        time_range: Optional[tuple[float, float]] = None,
    ) -> None:
        """Initialize the CPI model.

        Args:
            n_control_points: Number of control points (4-12, default 6).
            time_range: Optional (t_min, t_max) for control point placement.
                        If None, defaults to (0, 30) seconds.
        """
        # Validate and clamp n_control_points
        self._n_control_points = max(CPI_N_MIN, min(CPI_N_MAX, n_control_points))

        # Set time range for control point placement
        self._time_range = time_range if time_range is not None else (0.0, 30.0)

        # Pre-compute control point times (evenly spaced)
        self._control_point_times = np.linspace(
            self._time_range[0],
            self._time_range[1],
            self._n_control_points,
        )

    @property
    def name(self) -> str:
        """Model name for identification."""
        return f"cpi_{self._n_control_points}"

    @property
    def n_parameters(self) -> int:
        """Number of free parameters (N control points)."""
        return self._n_control_points

    @property
    def n_control_points(self) -> int:
        """Number of control points in the spline."""
        return self._n_control_points

    @property
    def control_point_times(self) -> npt.NDArray[np.floating]:
        """Times at which control points are placed (seconds)."""
        return self._control_point_times

    @property
    def time_range(self) -> tuple[float, float]:
        """Time range (t_min, t_max) for control points."""
        return self._time_range

    def get_parameter_definitions(self) -> list[DSCModelParameter]:
        """Get parameter definitions for CPI control points.

        Returns:
            List of DSCModelParameter objects for each control point.
            The first control point (r0) has the full CBF range.
            Subsequent points have decreasing upper bounds to encourage
            monotonicity.
        """
        params = []

        for i in range(self._n_control_points):
            # Time at this control point
            t = self._control_point_times[i]

            # First control point is CBF (full range)
            if i == 0:
                params.append(
                    DSCModelParameter(
                        name="r0",
                        bounds=(0.0, CBF_BOUNDS.max),
                        typical_range=(10.0, 100.0),
                        log_transform=False,
                        description=f"Control point at t={t:.1f}s (CBF)",
                        units="mL/100g/min",
                    )
                )
            else:
                # Subsequent control points
                # Upper bound progressively decreases to encourage monotonicity
                # but actual bounds allow full range for flexibility
                params.append(
                    DSCModelParameter(
                        name=f"r{i}",
                        bounds=(0.0, CBF_BOUNDS.max),
                        typical_range=(0.0, 100.0 - i * (80.0 / self._n_control_points)),
                        log_transform=False,
                        description=f"Control point at t={t:.1f}s",
                        units="mL/100g/min",
                    )
                )

        return params

    def get_parameter_names(self) -> list[str]:
        """Get list of control point parameter names.

        Returns:
            List: ['r0', 'r1', 'r2', ..., 'r(N-1)']
        """
        return [f"r{i}" for i in range(self._n_control_points)]

    def enforce_constraints(
        self, params: Sequence[float]
    ) -> npt.NDArray[np.floating]:
        """Enforce monotonicity and non-negativity constraints on control points.

        Constraints:
            - Non-negativity: r[i] >= 0 for all i
            - Monotonicity: r[i] <= r[i-1] for i > 0

        Args:
            params: Control point values (may violate constraints).

        Returns:
            Constrained control point values.
        """
        constrained = np.array(params, dtype=np.float64)

        # Non-negativity
        constrained = np.maximum(constrained, 0.0)

        # Monotonicity (cumulative minimum from start)
        for i in range(1, len(constrained)):
            if constrained[i] > constrained[i - 1]:
                constrained[i] = constrained[i - 1]

        return constrained

    def residue_function(
        self,
        params: Sequence[float],
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute the residue function R(t) using PCHIP interpolation.

        The residue is a smooth spline through the control points,
        constrained to be monotonically decreasing and non-negative.

        Args:
            params: Control point values [r0, r1, ..., r(N-1)].
            time_points: Time points (seconds) at which to evaluate R(t).

        Returns:
            Residue function values R(t) at each time point.
        """
        # Enforce constraints
        control_values = self.enforce_constraints(params)

        # Create PCHIP interpolant
        # PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) preserves
        # monotonicity and is shape-preserving
        try:
            pchip = interpolate.PchipInterpolator(
                self._control_point_times,
                control_values,
                extrapolate=True,
            )

            # Evaluate at requested time points
            residue = pchip(time_points)
        except ValueError:
            # Fallback: linear interpolation if PCHIP fails
            residue = np.interp(
                time_points,
                self._control_point_times,
                control_values,
            )

        # Enforce non-negativity on output (numerical safety)
        residue = np.maximum(residue, 0.0)

        # Handle extrapolation beyond control point range
        # For t > t_max: extrapolate towards zero
        # For t < t_min: use r0 (CBF)
        if len(time_points) > 0:
            t_max = self._time_range[1]
            beyond_mask = time_points > t_max

            if np.any(beyond_mask):
                # Exponential decay beyond last control point
                # Use last two control points to estimate decay rate
                if self._n_control_points >= 2:
                    r_last = float(control_values[-1])
                    r_prev = float(control_values[-2])
                    dt = self._control_point_times[-1] - self._control_point_times[-2]

                    # Decay rate from last two points
                    if r_prev > EPSILON_CONCENTRATION and r_last > EPSILON_CONCENTRATION:
                        decay_rate = np.log(r_prev / max(r_last, EPSILON_CONCENTRATION)) / dt
                        decay_rate = max(0.0, decay_rate)  # Ensure non-negative decay
                    else:
                        decay_rate = 0.5  # Default decay rate

                    # Exponential extrapolation
                    t_excess = time_points[beyond_mask] - t_max
                    residue[beyond_mask] = r_last * np.exp(-decay_rate * t_excess)

        return residue

    def tissue_curve(
        self,
        params: Sequence[float],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute tissue concentration curve by convolving AIF with residue.

        C_tissue(t) = AIF(t) ⊗ R(t)

        Uses trapezoidal integration for the convolution.

        Args:
            params: Control point values [r0, r1, ..., r(N-1)].
            aif: Arterial input function concentration values.
            time_points: Time points (seconds) at which to evaluate C(t).

        Returns:
            Tissue concentration values C(t) at each time point.
        """
        n_times = len(time_points)
        if len(aif) != n_times:
            raise ValueError(
                f"AIF length ({len(aif)}) must match time_points ({n_times})"
            )

        if n_times < 2:
            return np.zeros(n_times, dtype=np.float64)

        # Get time step (assume uniform spacing)
        dt = float(time_points[1] - time_points[0])

        # Compute residue function
        residue = self.residue_function(params, time_points)

        # Convolution: C_tissue(t) = ∫₀ᵗ AIF(τ) × R(t-τ) dτ
        tissue = np.zeros(n_times, dtype=np.float64)

        for i in range(1, n_times):
            # R(t_i - t_j) for j = 0..i
            r_values = residue[:i+1][::-1]  # R[i], R[i-1], ..., R[0]
            aif_values = aif[:i+1]  # AIF[0], AIF[1], ..., AIF[i]

            # Trapezoidal rule
            integrand = aif_values * r_values
            tissue[i] = np.trapezoid(integrand, dx=dt)

        return tissue

    def get_cbf(self, params: Sequence[float]) -> float:
        """Get CBF from the first control point.

        Args:
            params: Control point values.

        Returns:
            CBF value (mL/100g/min).
        """
        return float(params[0]) if len(params) > 0 else 0.0

    def get_cbv(
        self,
        params: Sequence[float],
        time_points: Optional[npt.NDArray[np.floating]] = None,
    ) -> float:
        """Compute CBV by integrating the residue function.

        CBV = ∫₀^∞ R(t) dt ≈ ∫₀^t_max R(t) dt

        Args:
            params: Control point values.
            time_points: Optional time points for integration.
                        If None, uses control point times extended.

        Returns:
            CBV value (mL/100g).
        """
        if time_points is None:
            # Use fine-grained time points for accurate integration
            time_points = np.linspace(
                self._time_range[0],
                self._time_range[1] * 1.5,  # Extend for tail
                200,
            )

        residue = self.residue_function(params, time_points)
        dt = float(time_points[1] - time_points[0])

        # Integrate: CBV = ∫R(t)dt
        # Units: (mL/100g/min) × s = mL/100g × (s/min)
        # Divide by 60 to convert min to s
        cbv = float(np.trapezoid(residue, dx=dt)) / 60.0

        return cbv

    def get_mtt(
        self,
        params: Sequence[float],
        time_points: Optional[npt.NDArray[np.floating]] = None,
    ) -> float:
        """Compute MTT from central volume theorem.

        MTT = CBV / CBF

        Args:
            params: Control point values.
            time_points: Optional time points for CBV integration.

        Returns:
            MTT value (seconds).
        """
        cbf = self.get_cbf(params)
        if cbf < EPSILON_CONCENTRATION:
            return 0.0

        cbv = self.get_cbv(params, time_points)

        # MTT = CBV / CBF
        # CBV in mL/100g, CBF in mL/100g/min
        # MTT = (mL/100g) / (mL/100g/min) = min
        # Multiply by 60 to get seconds
        mtt = (cbv / cbf) * 60.0

        return mtt


def select_n_control_points(
    bolus_duration_frames: int,
    tr_seconds: float,
    min_n: int = CPI_N_MIN,
    max_n: int = CPI_N_MAX,
) -> int:
    """Automatically select number of control points based on bolus duration.

    The rule is: N ≈ bolus_duration / (2 × TR) + 2
    This ensures adequate sampling of the residue function while avoiding
    overfitting.

    Args:
        bolus_duration_frames: Duration of bolus passage in frames.
        tr_seconds: Repetition time in seconds.
        min_n: Minimum number of control points (default 4).
        max_n: Maximum number of control points (default 12).

    Returns:
        Recommended number of control points.
    """
    # Calculate bolus duration in seconds
    bolus_duration_seconds = bolus_duration_frames * tr_seconds

    # N ≈ bolus_duration / (2 × TR) + 2
    # This gives roughly one control point per 2 TR during the bolus,
    # plus 2 extra for the tail
    n_recommended = int(bolus_duration_frames / 2.0) + 2

    # Clamp to valid range
    n_clamped = max(min_n, min(max_n, n_recommended))

    return n_clamped


def estimate_bolus_duration(
    concentration: npt.NDArray[np.floating],
    threshold_fraction: float = 0.1,
) -> int:
    """Estimate bolus passage duration from concentration curve.

    Finds the number of frames where concentration exceeds a threshold
    fraction of the peak value.

    Args:
        concentration: Concentration time course.
        threshold_fraction: Fraction of peak defining bolus region (default 0.1).

    Returns:
        Estimated bolus duration in frames.
    """
    if len(concentration) == 0:
        return 10  # Default fallback

    peak = float(np.max(concentration))
    if peak < EPSILON_CONCENTRATION:
        return 10  # Default fallback

    threshold = peak * threshold_fraction

    # Find frames above threshold
    above_threshold = concentration > threshold
    n_frames_above = int(np.sum(above_threshold))

    # Return at least 5 frames
    return max(5, n_frames_above)


def select_n_by_bic(
    concentration: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    n_candidates: Sequence[int] = (4, 5, 6, 7, 8, 9, 10),
    noise_std: Optional[float] = None,
) -> tuple[int, list[tuple[int, float, float]]]:
    """Select optimal number of control points using BIC.

    Bayesian Information Criterion balances goodness-of-fit against model
    complexity. Lower BIC indicates a better model.

    BIC = n × ln(RSS/n) + k × ln(n)

    where:
        n = number of observations (time points)
        RSS = residual sum of squares
        k = number of parameters (control points)

    Args:
        concentration: Measured tissue concentration curve.
        aif: Arterial input function.
        time_points: Time points (seconds).
        n_candidates: Candidate numbers of control points to evaluate.
        noise_std: Optional noise standard deviation for weighting.

    Returns:
        Tuple of:
            - Optimal number of control points
            - List of (n, BIC, RSS) for each candidate
    """
    n_obs = len(concentration)
    results: list[tuple[int, float, float]] = []

    for n in n_candidates:
        if n < CPI_N_MIN or n > CPI_N_MAX:
            continue

        # Create model with this number of control points
        t_range = (float(time_points[0]), float(time_points[-1]))
        model = CPIModel(n_control_points=n, time_range=t_range)

        # Fit model (simple least squares for speed)
        fit_result = _fit_cpi_simple(model, concentration, aif, time_points)

        if fit_result is None:
            continue

        # Calculate BIC
        rss = fit_result.residual_sum_squares
        k = n  # Number of parameters

        if rss > 0:
            bic = n_obs * np.log(rss / n_obs) + k * np.log(n_obs)
        else:
            bic = float("-inf")  # Perfect fit (unlikely)

        results.append((n, float(bic), float(rss)))

    if not results:
        # Fallback to default
        return CPI_DEFAULT_N, []

    # Find minimum BIC
    best_n = min(results, key=lambda x: x[1])[0]

    return best_n, results


def _fit_cpi_simple(
    model: CPIModel,
    concentration: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
) -> Optional[DSCFitResult]:
    """Simple least-squares fit of CPI model for BIC calculation.

    Uses scipy.optimize.minimize with L-BFGS-B for bounded optimization.

    Args:
        model: CPIModel instance.
        concentration: Measured tissue concentration.
        aif: Arterial input function.
        time_points: Time points (seconds).

    Returns:
        DSCFitResult with fitted parameters, or None if fitting failed.
    """
    from scipy import optimize  # type: ignore[import-untyped]

    n_params = model.n_parameters
    bounds = model.get_bounds()
    x0 = model.get_initial_guess()

    def objective(params: npt.NDArray[np.floating]) -> float:
        """Residual sum of squares objective."""
        # Enforce constraints
        constrained = model.enforce_constraints(params)

        # Compute model prediction
        predicted = model.tissue_curve(constrained, aif, time_points)

        # Sum of squared residuals
        residuals = concentration - predicted
        return float(np.sum(residuals**2))

    try:
        result = optimize.minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500},
        )

        if result.success or result.fun < np.inf:
            fitted_params = model.enforce_constraints(result.x)
            fitted_tissue = model.tissue_curve(fitted_params, aif, time_points)

            # Calculate R²
            ss_res = result.fun
            ss_tot = float(np.sum((concentration - np.mean(concentration))**2))
            r_squared = 1.0 - (ss_res / max(ss_tot, EPSILON_CONCENTRATION))

            return DSCFitResult(
                parameters=model.params_to_dict(fitted_params),
                r_squared=r_squared,
                residual_sum_squares=float(result.fun),
                model_name=model.name,
                n_parameters=n_params,
                n_observations=len(concentration),
                converged=result.success,
                fitting_method="nlls",
                fitted_residue=model.residue_function(fitted_params, time_points),
                fitted_tissue_curve=fitted_tissue,
            )

    except (ValueError, RuntimeError):
        pass

    return None
