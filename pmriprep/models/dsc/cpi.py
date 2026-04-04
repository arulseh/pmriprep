"""Control Point Interpolation (CPI) model for DSC perfusion.

Implements a non-parametric model where the residue function is represented
as a PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) spline through
N control points. This approach allows flexible residue shapes while
maintaining physiological constraints.

The CPI model is particularly useful when the underlying transit time
distribution is unknown or doesn't follow a standard parametric form.

The parameterization follows Mehndiratta et al. (2013) / Quantiphyse:
- R(t) is **dimensionless** (0-1), with R(0) = 1.0 fixed
- CBF is a **separate** scaling parameter
- C_tissue(t) = CBF × [A_trap @ R_norm]

References:
    Mouridsen K, Christensen S, Gyldensted L, Østergaard L. (2006).
    "Automatic selection of arterial input function using cluster analysis."
    Magnetic Resonance in Medicine 55(3):524-531.

    Mehndiratta A, MacIntosh BJ, Crane DE, et al. (2013).
    "A control point interpolation method for the non-parametric quantification
    of cerebral haemodynamics from dynamic susceptibility contrast MRI."
    NeuroImage 64:560-570.
"""

from typing import Sequence

import numpy as np
import numpy.typing as npt
from scipy import interpolate  # type: ignore[import-untyped]

from pmriprep.models.bayesian_utils import TransformType
from pmriprep.models.dsc.base import (
    DSCFitResult,
    DSCModel,
    DSCModelParameter,
    ParamsLike,
)
from pmriprep.utils.dsc_constants import (
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

    Following Mehndiratta et al. (2013) / Quantiphyse parameterization:
        - R(t) is dimensionless (0-1), with R(0) = 1.0 fixed
        - CBF is a separate scaling parameter (relative units, s⁻¹)
        - C(t) = CBF × [A_trap @ R_norm]

    Constraints:
        - R(0) = 1.0: Fixed (not a free parameter)
        - Monotonicity: R(t) is monotonically decreasing (or non-increasing)
        - Non-negativity: R(t) >= 0 for all t

    The model automatically selects the number of control points based on
    bolus duration, or uses a user-specified value.

    Attributes:
        n_control_points: Number of control points in the spline (4-12).
        time_range: Time range (t_min, t_max) for control point placement.

    Parameters:
        - cbf: CBF scaling parameter (relative units, s⁻¹)
        - r1, r2, ..., r(N-1): Control point fractions (0-1) of R(0)=1.0
          These are constrained to be <= previous point (monotonicity)

    Example:
        For N=6 control points:
        - Parameters: [cbf, r1, r2, r3, r4, r5]
        - cbf = flow scaling (max of unnormalized residue)
        - R(0) = 1.0 (fixed, not a parameter)
        - Control point times: uniformly spaced from 0 to t_max

    References:
        Mehndiratta A, et al. (2013). "A control point interpolation method
        for the non-parametric quantification of cerebral haemodynamics from
        dynamic susceptibility contrast MRI." NeuroImage 64:560-570.
    """

    def __init__(
        self,
        n_control_points: int = CPI_DEFAULT_N,
        time_range: tuple[float, float] | None = None,
    ) -> None:
        """Initialize the CPI model.

        Args:
            n_control_points: Number of control points (4-12, default 6).
                This controls the shape of R(t). The total number of free
                parameters is n_control_points: 1 for CBF + (N-1) fractions.
            time_range: Optional (t_min, t_max) for control point placement.
                        If None, defaults to (0, 30) seconds.
        """
        # Validate and clamp n_control_points
        self._n_control_points = max(CPI_N_MIN, min(CPI_N_MAX, n_control_points))

        # Set time range for control point placement
        self._time_range = time_range if time_range is not None else (0.0, 30.0)

        # Pre-compute control point times (uniform spacing, per Mehndiratta 2013)
        self._control_point_times = np.linspace(
            self._time_range[0],
            self._time_range[1],
            self._n_control_points,
        )

        # Cache for the Toeplitz matrix (built on first use)
        self._toeplitz_cache: npt.NDArray[np.floating] | None = None
        self._toeplitz_aif_id: int | None = None

        # When True, enforce_constraints uses hard monotonicity projection.
        # Set by _fit_cpi_simple for NLLS fitting; False for Bayesian paths.
        self._hard_monotonicity: bool = False

    @property
    def name(self) -> str:
        """Model name for identification."""
        return f"cpi_{self._n_control_points}"

    @property
    def n_parameters(self) -> int:
        """Number of free parameters (CBF + N-1 fractions)."""
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
        """Get parameter definitions for CPI model.

        Returns:
            List of DSCModelParameter objects:
            - cbf: CBF scaling (relative s⁻¹ units)
            - r1..r_{N-1}: Fractions of R(0)=1.0 at each control point
        """
        params = []

        # CBF scaling parameter (relative units, s⁻¹)
        # Actual CPI CBF values are 25-100 (same scale as SVD methods),
        # matching VascularModel CBF_BOUNDS=(0, 200).
        params.append(
            DSCModelParameter(
                name="cbf",
                bounds=(0.0, 200.0),
                typical_range=(10.0, 100.0),
                transform=TransformType.LOG,
                description="CBF scaling (relative)",
                units="s⁻¹",
            )
        )

        # Weakly informative priors on fractions — monotonicity is enforced
        # by a separate soft prior (compute_monotonicity_prior), not by
        # biasing each fraction's marginal toward an exponential template.
        for i in range(1, self._n_control_points):
            t = self._control_point_times[i]
            params.append(
                DSCModelParameter(
                    name=f"r{i}",
                    bounds=(0.0, 1.0),
                    typical_range=(0.0, 1.0),
                    transform=TransformType.IDENTITY,
                    description=f"Control point fraction at t={t:.1f}s",
                    units="fraction",
                )
            )

        return params

    def get_initial_guess(self) -> list[float]:
        """Get initial parameter guess with exponential decay fractions.

        Returns:
            List: [cbf_init, exp(-1/τ), exp(-2/τ), ...] where τ = N/2.
            CBF initial guess is set to a small value appropriate for
            relative ΔR2* units. Exponential decay matches cerebral
            physiology better than the linear midpoint default.
        """
        x0: list[float] = [55.0]  # CBF initial guess (midpoint of typical_range)
        tau = self._n_control_points / 2.0
        for i in range(1, self._n_control_points):
            x0.append(float(np.exp(-i / tau)))
        return x0

    def get_parameter_names(self) -> list[str]:
        """Get list of parameter names.

        Returns:
            List: ['cbf', 'r1', 'r2', ..., 'r(N-1)']
        """
        return ["cbf"] + [f"r{i}" for i in range(1, self._n_control_points)]

    def enforce_constraints(
        self, params: ParamsLike
    ) -> npt.NDArray[np.floating]:
        """Convert fraction parameterization to dimensionless R(t) values.

        In Bayesian mode (default): clips to [0, 1] only. Monotonicity
        is handled by the soft prior (compute_monotonicity_prior).

        In NLLS mode (_hard_monotonicity=True): also enforces monotonic
        non-increase via sequential capping.

        Args:
            params: [cbf, frac1, frac2, ...] where cbf is the scaling
                parameter and frac_i are fractions of R(0)=1.0.

        Returns:
            Dimensionless control point values (R(0)=1.0), length =
            n_control_points.
        """
        if self._hard_monotonicity:
            return self.enforce_constraints_hard(params)

        control_values = np.zeros(self._n_control_points, dtype=np.float64)
        control_values[0] = 1.0

        for i in range(1, self._n_control_points):
            control_values[i] = float(np.clip(params[i], 0.0, 1.0))

        return control_values

    def enforce_constraints_hard(
        self, params: ParamsLike
    ) -> npt.NDArray[np.floating]:
        """Convert fractions to R(t) values with hard monotonicity projection.

        Used by _fit_cpi_simple (NLLS) where hard projection is appropriate
        for L-BFGS-B box-constrained optimization.

        Args:
            params: [cbf, frac1, frac2, ...].

        Returns:
            Monotonically non-increasing control point values.
        """
        control_values = np.zeros(self._n_control_points, dtype=np.float64)
        control_values[0] = 1.0

        prev_r = 1.0
        for i in range(1, self._n_control_points):
            frac = float(np.clip(params[i], 0.0, 1.0))
            control_values[i] = min(frac, prev_r)
            prev_r = control_values[i]

        return control_values

    def residue_function(
        self,
        params: ParamsLike,
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute the dimensionless residue function R(t) using PCHIP.

        The residue is a smooth spline through the control points,
        constrained to be monotonically decreasing and non-negative.
        R(0) = 1.0 and all values are in [0, 1].

        Args:
            params: Parameter values [cbf, r1, ..., r(N-1)].
            time_points: Time points (seconds) at which to evaluate R(t).

        Returns:
            Dimensionless residue function values R(t) at each time point.
        """
        # Enforce constraints (returns dimensionless values with R(0)=1.0)
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
            residue: npt.NDArray[np.floating] = np.asarray(
                pchip(time_points), dtype=np.float64
            )
        except ValueError:
            # Fallback: linear interpolation if PCHIP fails
            residue = np.interp(
                time_points,
                self._control_point_times,
                control_values,
            )

        # Clip to [0, 1] — PCHIP can overshoot above 1.0 when control
        # points are non-monotonic (soft monotonicity allows this).
        residue = np.clip(residue, 0.0, 1.0)

        # Handle extrapolation beyond control point range
        # For t > t_max: extrapolate towards zero
        # For t < t_min: use R(0) = 1.0
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
                    # The clip on line below serves as a secondary safety net,
                    # bounding extreme log ratios from near-zero control points.
                    if r_prev > EPSILON_CONCENTRATION and r_last > EPSILON_CONCENTRATION:
                        decay_rate = np.log(r_prev / max(r_last, EPSILON_CONCENTRATION)) / dt
                        decay_rate = float(np.clip(decay_rate, 0.01, 1.0))
                    else:
                        decay_rate = 0.1  # Default: 10s time constant

                    # Exponential extrapolation
                    t_excess = time_points[beyond_mask] - t_max
                    residue[beyond_mask] = r_last * np.exp(-decay_rate * t_excess)

        return residue

    def compute_monotonicity_prior(
        self,
        params: ParamsLike,
        sigma_delta: float = 0.15,
        violation_scale: float = 10.0,
    ) -> float:
        """Soft monotonicity prior on CPI control point fractions.

        Penalizes violations (r_i > r_{i-1}) more steeply than normal
        decrements. Used by the Bayesian fitter in the log-posterior.

        Args:
            params: Parameter values [cbf, r1, ..., r(N-1)].
            sigma_delta: Prior std for expected decrement between points.
            violation_scale: How much more steeply to penalize violations.

        Returns:
            Log-prior contribution (negative for violations).
        """
        log_p = 0.0
        prev = 1.0  # R(0) = 1.0
        for i in range(1, self._n_control_points):
            ri = float(np.clip(params[i], 0.0, 1.0))
            delta = prev - ri  # Should be >= 0 for monotonicity
            if delta >= 0:
                log_p -= 0.5 * (delta / sigma_delta) ** 2
            else:
                # Violations penalized violation_scale^2 times more steeply
                log_p -= 0.5 * (delta / (sigma_delta / violation_scale)) ** 2
            prev = ri
        return log_p

    def compute_monotonicity_precision(
        self,
        sigma_delta: float = 0.15,
    ) -> npt.NDArray[np.floating]:
        """Build symmetric monotonicity prior precision matrix for VI.

        The symmetric penalty sum((r_{i-1} - r_i)^2 / sigma_delta^2)
        expands to a tridiagonal matrix on the fraction parameters only
        (CBF row/column is zero).

        Args:
            sigma_delta: Prior std for decrement penalty.

        Returns:
            n_parameters x n_parameters precision matrix.
        """
        n = self._n_control_points  # = n_parameters
        lam = np.zeros((n, n), dtype=np.float64)
        tau = 1.0 / (sigma_delta ** 2)

        # First pair: R(0)=1.0 (fixed) vs r1. Since R(0) is not a
        # parameter, the penalty (1 - r1)^2 / sigma^2 adds tau to
        # position [1,1] (the r1 diagonal).
        lam[1, 1] += tau

        # Subsequent pairs: (r_{i-1} - r_i)^2 / sigma^2
        for i in range(2, n):
            lam[i - 1, i - 1] += tau
            lam[i, i] += tau
            lam[i - 1, i] -= tau
            lam[i, i - 1] -= tau

        # CBF (index 0) has no monotonicity penalty
        return lam

    def compute_nlls_seed(
        self,
        concentration: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating] | None:
        """Compute NLLS seed for Bayesian VI initialization.

        CPI's high-dimensional spline space requires a good starting point
        for VI to prevent noise precision collapse. Runs a quick L-BFGS-B
        fit and returns the parameter array.

        Args:
            concentration: Measured tissue concentration curve.
            aif: Arterial input function.
            time_points: Time points (seconds).

        Returns:
            Parameter array from NLLS fit, or None if fitting failed.
        """
        result = _fit_cpi_simple(
            self, concentration, aif, time_points, sloppy=True
        )
        if result is not None and (result.converged or result.r_squared > 0.5):
            return np.array(
                self.dict_to_params(result.parameters), dtype=np.float64
            )
        return None

    def compute_fitted_residue(
        self,
        params: ParamsLike,
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute CBF-scaled residue for Bayesian fitted_residue output.

        CPI's residue_function() returns dimensionless R(t) in [0,1],
        but downstream code expects fitted_residue = CBF × R(t) so that
        CBF = max(residue) and CBV = trapezoid(residue, dx=dt).

        Args:
            params: Parameter values [cbf, r1, ..., r(N-1)].
            time_points: Time points (seconds).

        Returns:
            CBF-scaled residue function (CBF × R_norm).
        """
        cbf = max(float(params[0]), 0.0)
        return cbf * self.residue_function(params, time_points)

    def tissue_curve(
        self,
        params: ParamsLike,
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute tissue concentration curve by convolving AIF with residue.

        C_tissue(t) = CBF × [A_trap @ R_norm]

        Uses trapezoidal Toeplitz matrix for convolution, consistent with
        the SVD deconvolution path (Sourbron & Luypaert PMB 2007).

        Args:
            params: Parameter values [cbf, r1, ..., r(N-1)].
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

        # Extract CBF scaling
        cbf = max(float(params[0]), 0.0)

        # Compute dimensionless residue function R(t) with R(0)=1.0
        residue = self.residue_function(params, time_points)

        # Build or reuse Toeplitz matrix (includes dt in its elements)
        aif_id = id(aif)
        if self._toeplitz_cache is None or self._toeplitz_aif_id != aif_id:
            from pmriprep.utils.dsc_deconvolution import build_toeplitz_matrix
            dt = float(time_points[1] - time_points[0])
            self._toeplitz_cache = build_toeplitz_matrix(aif, dt)
            self._toeplitz_aif_id = aif_id

        # C(t) = CBF × (A_trap @ R_norm)
        # Toeplitz already includes dt, so no additional multiplication
        tissue = cbf * (self._toeplitz_cache @ residue)

        return tissue

    def get_cbf(self, params: ParamsLike) -> float:
        """Get CBF from the scaling parameter.

        Args:
            params: Parameter values [cbf, r1, ..., r(N-1)].

        Returns:
            CBF value (relative, s⁻¹).
        """
        return float(params[0]) if len(params) > 0 else 0.0

    def get_cbv(
        self,
        params: ParamsLike,
        time_points: npt.NDArray[np.floating] | None = None,
    ) -> float:
        """Compute CBV by integrating the unnormalized residue function.

        CBV = ∫₀^∞ CBF × R_norm(t) dt = CBF × ∫R_norm(t) dt

        Since CBF is in relative s⁻¹ units, CBV = CBF × ∫R_norm dt
        has units of (dimensionless), matching the SVD path where
        CBV = trapezoid(residue, dx=dt).

        Args:
            params: Parameter values [cbf, r1, ..., r(N-1)].
            time_points: Optional time points for integration.
                        If None, uses control point times extended.

        Returns:
            CBV value (relative, dimensionless).
        """
        if time_points is None:
            # Use fine-grained time points for accurate integration
            time_points = np.linspace(
                self._time_range[0],
                self._time_range[1] * 1.5,  # Extend for tail
                200,
            )

        cbf = self.get_cbf(params)
        residue_norm = self.residue_function(params, time_points)
        dt = float(time_points[1] - time_points[0])

        # CBV = CBF × ∫R_norm(t) dt (no /60 — relative units)
        cbv = cbf * float(np.trapezoid(residue_norm, dx=dt))

        return cbv

    def get_mtt(
        self,
        params: ParamsLike,
        time_points: npt.NDArray[np.floating] | None = None,
    ) -> float:
        """Compute MTT from central volume theorem.

        MTT = CBV / CBF = ∫R_norm(t) dt  (seconds)

        Args:
            params: Parameter values [cbf, r1, ..., r(N-1)].
            time_points: Optional time points for CBV integration.

        Returns:
            MTT value (seconds).
        """
        cbf = self.get_cbf(params)
        if cbf < EPSILON_CONCENTRATION:
            return 0.0

        if time_points is None:
            time_points = np.linspace(
                self._time_range[0],
                self._time_range[1] * 1.5,
                200,
            )

        residue_norm = self.residue_function(params, time_points)
        dt = float(time_points[1] - time_points[0])

        # MTT = CBV / CBF = ∫R_norm dt (seconds, no /60 conversion)
        mtt = float(np.trapezoid(residue_norm, dx=dt))

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
    noise_std: float | None = None,
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
    weights: npt.NDArray[np.floating] | None = None,
    sloppy: bool = False,
) -> DSCFitResult | None:
    """Least-squares fit of CPI model using L-BFGS-B.

    L-BFGS-B handles box bounds natively. Monotonicity is enforced via
    hard projection in enforce_constraints() (enabled by setting
    _hard_monotonicity=True for the duration of the fit).

    The CBF upper bound is set adaptively based on the data scale
    (10 × max tissue concentration) to ensure the optimizer can reach
    the correct solution regardless of input units.

    Args:
        model: CPIModel instance.
        concentration: Measured tissue concentration.
        aif: Arterial input function.
        time_points: Time points (seconds).
        weights: Optional per-frame motion weights (1.0 = full weight).
        sloppy: Use fast, low-quality settings for testing.

    Returns:
        DSCFitResult with fitted parameters, or None if fitting failed.
        fitted_residue contains the unnormalized residue (CBF × R_norm)
        so downstream code can extract CBF = max(residue) and
        CBV = trapezoid(residue, dx=dt).
    """
    from scipy import optimize  # type: ignore[import-untyped]

    n_params = model.n_parameters
    bounds = model.get_bounds()

    # Adaptive CBF upper bound: 10× max observed concentration
    max_conc = float(np.max(np.abs(concentration)))
    cbf_upper = max(10.0 * max_conc, 1.0)
    bounds[0] = (0.0, cbf_upper)

    # Data-driven CBF initial guess: max(concentration) / max(AIF) is a
    # rough estimate of CBF (ratio of tissue to arterial peak)
    max_aif = float(np.max(np.abs(aif)))
    if max_aif > EPSILON_CONCENTRATION:
        cbf_init = max_conc / max_aif
    else:
        cbf_init = 0.1

    x0 = model.get_initial_guess()
    x0[0] = cbf_init

    # Enable hard monotonicity for NLLS fitting
    model._hard_monotonicity = True

    def objective(params: npt.NDArray[np.floating]) -> float:
        """Weighted residual sum of squares objective."""
        predicted = model.tissue_curve(params, aif, time_points)
        residuals = concentration - predicted
        if weights is not None:
            return float(np.sum(weights * residuals**2))
        return float(np.sum(residuals**2))

    try:
        result = optimize.minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 50 if sloppy else 500, "ftol": 1e-6},
        )

        if result.success or result.fun < np.inf:
            # Store optimizer-space params
            opt_params = np.array(result.x, dtype=np.float64)
            # Clip fractions to [0,1]
            opt_params[1:] = np.clip(opt_params[1:], 0.0, 1.0)
            opt_params[0] = max(opt_params[0], 0.0)

            # Compute fitted tissue curve for R²
            fitted_tissue = model.tissue_curve(opt_params, aif, time_points)

            # Compute unnormalized residue (CBF × R_norm) for downstream
            # compatibility: CBF = max(residue), CBV = trapezoid(residue)
            cbf_fitted = float(opt_params[0])
            residue_norm = model.residue_function(opt_params, time_points)
            fitted_residue = cbf_fitted * residue_norm

            # Calculate R²
            ss_res = result.fun
            ss_tot = float(np.sum((concentration - np.mean(concentration))**2))
            r_squared = 1.0 - (ss_res / max(ss_tot, EPSILON_CONCENTRATION))

            return DSCFitResult(
                parameters=model.params_to_dict(opt_params),
                r_squared=r_squared,
                residual_sum_squares=float(result.fun),
                model_name=model.name,
                n_parameters=n_params,
                n_observations=len(concentration),
                converged=result.success,
                fitting_method="nlls",
                fitted_residue=fitted_residue,
                fitted_tissue_curve=fitted_tissue,
            )

    except (ValueError, RuntimeError):
        pass
    finally:
        model._hard_monotonicity = False

    return None
