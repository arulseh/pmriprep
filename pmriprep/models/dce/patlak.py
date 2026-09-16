"""Patlak pharmacokinetic model with graphical analysis.

Implements the Patlak model for DCE-MRI analysis.
This model assumes irreversible or negligible backflux of contrast agent
from the extravascular space and is appropriate for tissues with intact
blood-brain barrier or when only early time points are used.

The tissue concentration is given by:
    C_t(t) = vp × Cp(t) + Ktrans × ∫Cp(τ)dτ

where:
    Ktrans = volume transfer constant (min⁻¹)
    vp = plasma volume fraction (dimensionless)
    Cp = plasma concentration (AIF)
    ∫Cp(τ)dτ = cumulative integral of AIF from 0 to t

The graphical analysis form (Patlak plot):
    C_t(t) / Cp(t) = vp + Ktrans × [∫Cp(τ)dτ / Cp(t)]

This linear relationship allows fast fitting using linear regression.
"""

from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit, lsq_linear

from pmriprep.models.bayesian_utils import TransformType
from pmriprep.models.dce.base import FitResult, ModelParameter, ParamsType, PKModel
from pmriprep.utils.dce_constants import VP_BOUNDS


class PatlakModel(PKModel):
    """Patlak pharmacokinetic model assuming no backflux.

    The Patlak model describes unidirectional contrast agent transfer from
    the blood plasma to the extravascular space, assuming negligible backflux
    (kep ≈ 0). This assumption is valid when:

    - BBB is intact (kep << Ktrans)
    - Only early post-injection time points are used
    - ve is large relative to Ktrans × t_max

    Model equation::

        C_t(t) = vp × Cp(t) + Ktrans × ∫₀ᵗ Cp(τ)dτ

    Graphical analysis (Patlak plot)::

        Y = C_t(t) / Cp(t)
        X = ∫₀ᵗ Cp(τ)dτ / Cp(t)

        Y = vp + Ktrans × X

        where:
            - slope = Ktrans
            - intercept = vp

    Parameters
    ----------
        Ktrans: Volume transfer constant (min⁻¹), 0-3 min⁻¹
            - Reflects unidirectional influx rate
            - For intact BBB: typically < 0.05 min⁻¹
            - For disrupted BBB: can exceed 0.1 min⁻¹

        vp: Plasma volume fraction, 0-1
            - Fraction of tissue that is blood plasma
            - Typical brain values: 0.01-0.05
            - Well-vascularized tumors: can exceed 0.1

    Advantages:
        - Fast fitting via linear regression (graphical method)
        - Stable parameter estimates (only 2 parameters)
        - Appropriate for intact BBB or early enhancement

    Limitations:
        - Assumes no backflux (kep = 0)
        - Not valid when significant contrast egress occurs
        - Should use only early time points if backflux is present
    """

    @property
    def name(self) -> str:
        """Model name for identification."""
        return "patlak"

    @property
    def n_parameters(self) -> int:
        """Number of free parameters (Ktrans, vp)."""
        return 2

    def get_parameter_definitions(self) -> Sequence[ModelParameter]:
        """Get parameter definitions for Patlak model.

        Returns
        -------
        Sequence[ModelParameter]
            List of two ModelParameter objects:
                - Ktrans: volume transfer constant (influx)
                - vp: plasma volume fraction
        """
        return [
            ModelParameter(
                name="Ktrans",
                bounds=(0.0, 5.0),  # Standard Patlak bounds
                typical_range=(0.001, 0.1),
                transform=TransformType.LOG,
                description="Volume transfer constant (influx)",
                units="min^-1",
            ),
            ModelParameter(
                name="vp",
                bounds=(VP_BOUNDS.min, VP_BOUNDS.max),
                # Brain vp 0.005-0.04. Match Extended Tofts. Wider prior
                # (0.01, 0.2) centered ~3× too high vs physiology and
                # biased Patlak Ki via the Patlak-plot intercept.
                typical_range=(0.001, 0.04),
                transform=TransformType.LOG_LOGIT,
                description="Plasma volume fraction",
                units="",
            ),
        ]

    def estimate_initial_guess(
        self,
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> list[float]:
        """Estimate initial parameters from Patlak graphical analysis.

        Uses the fast graphical (linear regression) method to provide
        data-driven starting values for Bayesian MAP multi-start.

        Parameters
        ----------
        tissue_curve : np.ndarray
            Observed tissue concentration (mM).
        aif : np.ndarray
            Arterial input function (mM).
        time_points : np.ndarray
            Time points (minutes).

        Returns
        -------
        list of float
            List of [Ktrans, vp] estimates.
        """
        try:
            result = self.fit_graphical(tissue_curve, aif, time_points)
            ktrans = result.parameters.get("Ktrans", 0.0)
            vp_val = result.parameters.get("vp", 0.0)
            # Keep the uptake initialization within its typical range.
            ktrans = max(0.001, min(ktrans, 1.0))
            vp_val = float(np.clip(vp_val, VP_BOUNDS.min, VP_BOUNDS.max))
            return [ktrans, vp_val]
        except Exception:
            return self.get_initial_guess()

    def evaluate(
        self,
        params: ParamsType,
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Evaluate the Patlak model.

        Computes C_t(t) = vp × Cp(t) + Ktrans × ∫₀ᵗ Cp(τ)dτ

        The integral is computed using the cumulative trapezoidal rule.

        Parameters
        ----------
        params : ParamsType
            Parameter values [Ktrans, vp].
        aif : np.ndarray
            Arterial input function plasma concentration values Cp(t) in mM.
        time_points : np.ndarray
            Time points (minutes) at which to evaluate C_t.

        Returns
        -------
        np.ndarray
            Tissue concentration values C_t(t) at each time point in mM.
        """
        ktrans = params[0]
        vp = params[1]

        # Support non-uniform time steps
        n_times = len(time_points)
        if n_times < 2:
            return vp * aif if n_times == 1 else np.zeros(n_times, dtype=np.float64)

        # Compute cumulative integral of AIF using trapezoidal rule
        # ∫₀ᵗ Cp(τ)dτ at each time point
        dts = np.diff(time_points)

        # Use cumulative trapezoid (initial value is 0)
        aif_integral = np.zeros(n_times, dtype=np.float64)
        aif_integral[1:] = np.cumsum(0.5 * (aif[:-1] + aif[1:]) * dts)

        # Patlak model: C_t(t) = vp × Cp(t) + Ktrans × ∫Cp(τ)dτ
        c_tissue = vp * aif + ktrans * aif_integral

        # Sanitize: replace non-finite values with 0
        c_tissue = np.where(np.isfinite(c_tissue), c_tissue, 0.0)

        return c_tissue

    def compute_patlak_plot_variables(
        self,
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
        """Compute variables for Patlak graphical analysis.

        The Patlak plot is:
            Y = C_t(t) / Cp(t)
            X = ∫₀ᵗ Cp(τ)dτ / Cp(t)

        Where Y = vp + Ktrans × X (linear relationship).

        Parameters
        ----------
        tissue_curve : np.ndarray
            Observed tissue concentration values C_t(t) in mM.
        aif : np.ndarray
            Arterial input function plasma concentration values Cp(t) in mM.
        time_points : np.ndarray
            Time points (minutes) corresponding to measurements.

        Returns
        -------
        tuple
            Tuple of (X, Y, valid_mask) where:
                - X: normalized cumulative AIF (independent variable)
                - Y: normalized tissue concentration (dependent variable)
                - valid_mask: boolean mask for valid points (Cp > epsilon)
        """
        n_times = len(time_points)
        if n_times < 2:
            return (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.bool_),
            )

        dts = np.diff(time_points)

        # Compute cumulative integral of AIF
        aif_integral = np.zeros(n_times, dtype=np.float64)
        aif_integral[1:] = np.cumsum(0.5 * (aif[:-1] + aif[1:]) * dts)

        # Use relative AIF threshold (5% of peak, 1e-3 mM floor)
        # to exclude time points where Cp(t) is too small for stable division
        aif_peak = float(np.max(aif))
        aif_threshold = max(0.05 * aif_peak, 1e-3)
        valid_mask = aif > aif_threshold

        # Initialize output arrays
        x_values = np.zeros(n_times, dtype=np.float64)
        y_values = np.zeros(n_times, dtype=np.float64)

        # Compute Patlak plot variables for valid points
        # X = ∫Cp(τ)dτ / Cp(t)
        # Y = C_t(t) / Cp(t)
        x_values[valid_mask] = aif_integral[valid_mask] / aif[valid_mask]
        y_values[valid_mask] = tissue_curve[valid_mask] / aif[valid_mask]

        return x_values, y_values, valid_mask

    def fit_graphical(
        self,
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        start_index: Optional[int] = None,
    ) -> FitResult:
        """Fit the Patlak model using graphical (linear regression) analysis.

        Uses the Patlak plot linearization:
            C_t(t) / Cp(t) = vp + Ktrans × [∫Cp(τ)dτ / Cp(t)]

        This is computationally fast and provides robust parameter estimates
        when the no-backflux assumption is valid.

        Parameters
        ----------
        tissue_curve : np.ndarray
            Observed tissue concentration values C_t(t) in mM.
        aif : np.ndarray
            Arterial input function plasma concentration values Cp(t) in mM.
        time_points : np.ndarray
            Time points (minutes) corresponding to measurements.
        start_index : int, optional
            Optional index to start linear fit (skip early points).
            If None, uses points after peak AIF.

        Returns
        -------
        FitResult
            FitResult containing:
                - parameters: Fitted Ktrans and vp values
                - goodness-of-fit: R², AIC, BIC
                - fitted curve
        """
        n_obs = len(tissue_curve)

        # Compute Patlak plot variables
        x_values, y_values, valid_mask = self.compute_patlak_plot_variables(
            tissue_curve, aif, time_points
        )

        # Determine start index for linear fit
        # Typically start after AIF peak to ensure steady-state
        if start_index is None:
            # Find peak of AIF and start fitting after it
            peak_idx = int(np.argmax(aif))
            start_index = min(peak_idx + 1, n_obs - 2)

        # Create fit mask: valid points after start_index
        fit_mask = valid_mask.copy()
        fit_mask[:start_index] = False

        # Check if we have enough points for fitting
        n_fit_points = int(np.sum(fit_mask))
        if n_fit_points < 2:
            # Not enough points, return default values
            return FitResult(
                parameters={"Ktrans": 0.0, "vp": 0.01},
                r_squared=0.0,
                residual_sum_squares=float("inf"),
                aic=float("inf"),
                bic=float("inf"),
                model_name=self.name,
                n_parameters=self.n_parameters,
                n_observations=n_obs,
                converged=False,
                fitting_method="graphical",
                fitted_curve=self.evaluate([0.0, 0.01], aif, time_points),
            )

        # Extract points for linear regression
        x_fit = x_values[fit_mask]
        y_fit = y_values[fit_mask]

        # Bounded linear regression: Y = vp + Ktrans × X. Solving with
        # bounds directly is necessary because clipping an unconstrained
        # slope or intercept does not re-optimize the other coefficient.
        try:
            design = np.column_stack([x_fit, np.ones_like(x_fit)])
            bounds = self.get_bounds()
            result = lsq_linear(
                design,
                y_fit,
                bounds=(
                    [bounds[0][0], bounds[1][0]],
                    [bounds[0][1], bounds[1][1]],
                ),
                method="trf",
                lsmr_tol="auto",
            )
            ktrans = float(result.x[0])
            vp = float(result.x[1])
            converged = bool(result.success)
        except (np.linalg.LinAlgError, ValueError):
            # Linear regression failed
            ktrans = 0.0
            vp = 0.01
            converged = False

        # Clip to model bounds
        fitted_params = self.clip_params([ktrans, vp])
        ktrans, vp = fitted_params[0], fitted_params[1]

        # Compute fitted curve and goodness-of-fit
        fitted_curve = self.evaluate(fitted_params, aif, time_points)
        rss = float(np.sum((tissue_curve - fitted_curve) ** 2))
        r_squared = self.compute_r_squared(tissue_curve, fitted_curve)
        aic = self.compute_aic(rss, n_obs)
        bic = self.compute_bic(rss, n_obs)

        # Build result
        param_dict = self.params_to_dict(fitted_params)

        return FitResult(
            parameters=param_dict,
            r_squared=r_squared,
            residual_sum_squares=rss,
            aic=aic,
            bic=bic,
            model_name=self.name,
            n_parameters=self.n_parameters,
            n_observations=n_obs,
            converged=converged,
            fitting_method="graphical",
            fitted_curve=fitted_curve,
        )

    def fit(
        self,
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        initial_guess: Optional[Sequence[float]] = None,
        weights: Optional[npt.NDArray[np.floating]] = None,
        use_graphical: bool = False,
    ) -> FitResult:
        """Fit the Patlak model to observed tissue concentration data.

        By default, uses nonlinear least squares.
        Can optionally use graphical (linear regression) method.

        Parameters
        ----------
        tissue_curve : np.ndarray
            Observed tissue concentration values C_t(t) in mM.
        aif : np.ndarray
            Arterial input function plasma concentration values Cp(t) in mM.
        time_points : np.ndarray
            Time points (minutes) corresponding to measurements.
        initial_guess : Sequence[float], optional
            Optional initial parameter values [Ktrans, vp].
            Used only for nonlinear fitting (use_graphical=False).
        weights : np.ndarray, optional
            Optional per-frame weights for motion-weighted fitting.
        use_graphical : bool, optional
            If True, use fast graphical analysis. If False (default), use
            nonlinear least squares optimization.

        Returns
        -------
        FitResult
            FitResult containing:
                - parameters: Fitted Ktrans and vp values
                - goodness-of-fit: R², AIC, BIC
                - fitted curve
        """
        if use_graphical:
            return self.fit_graphical(tissue_curve, aif, time_points)

        # Nonlinear least squares fitting with curve_fit/TRF
        n_obs = len(tissue_curve)

        # Set initial guess
        if initial_guess is None:
            x0 = self.get_initial_guess()
        else:
            x0 = list(initial_guess)

        # Get bounds for optimization
        bounds = self.get_bounds()
        lower_bounds = [b[0] for b in bounds]
        upper_bounds = [b[1] for b in bounds]

        def _curve_func(
            t: npt.NDArray[np.floating], *params: float
        ) -> npt.NDArray[np.floating]:
            return self.evaluate(list(params), aif, t)

        converged = False
        fitted_params = x0
        uncertainties: dict[str, float] = {}

        # Convert weights to sigma for curve_fit: sigma = 1/sqrt(weight)
        sigma_arr = None
        if weights is not None:
            sigma_arr = 1.0 / np.sqrt(np.maximum(weights, 1e-10))

        # Primary: curve_fit with TRF
        try:
            popt, pcov = curve_fit(
                _curve_func,
                time_points,
                tissue_curve,
                p0=x0,
                bounds=(lower_bounds, upper_bounds),
                method="trf",
                maxfev=2000,
                sigma=sigma_arr,
                absolute_sigma=False,
            )
            converged = True
            fitted_params = popt.tolist()
            perr = np.sqrt(np.diag(pcov))
            param_names = self.get_parameter_names()
            uncertainties = dict(zip(param_names, perr.tolist()))
        except (RuntimeError, ValueError):
            converged = False
            fitted_params = [0.0] * self.n_parameters

        # Clip parameters to bounds
        fitted_params = self.clip_params(fitted_params)

        # Compute fitted curve and goodness-of-fit
        fitted_curve = self.evaluate(fitted_params, aif, time_points)
        rss = float(np.sum((tissue_curve - fitted_curve) ** 2))
        r_squared = self.compute_r_squared(tissue_curve, fitted_curve)
        aic = self.compute_aic(rss, n_obs)
        bic = self.compute_bic(rss, n_obs)

        # Build result
        param_dict = self.params_to_dict(fitted_params)

        return FitResult(
            parameters=param_dict,
            uncertainties=uncertainties,
            r_squared=r_squared,
            residual_sum_squares=rss,
            aic=aic,
            bic=bic,
            model_name=self.name,
            n_parameters=self.n_parameters,
            n_observations=n_obs,
            converged=converged,
            fitting_method="nlls",
            fitted_curve=fitted_curve,
        )

    def compute_derived_parameters(
        self, params: ParamsType
    ) -> dict[str, float]:
        """Compute derived parameters from fitted values.

        For the Patlak model, there are no derived parameters since
        kep is assumed to be zero (no backflux).

        Parameters
        ----------
        params : ParamsType
            Fitted parameter values [Ktrans, vp].

        Returns
        -------
        dict
            Empty dictionary (no derived parameters for Patlak).
        """
        # Patlak assumes kep = 0 (no backflux), so no derived parameters
        return {}
