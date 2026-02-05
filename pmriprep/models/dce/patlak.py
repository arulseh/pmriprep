"""Patlak pharmacokinetic model with graphical analysis.

Implements the Patlak model (Patlak et al. 1983) for DCE-MRI analysis.
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
from scipy import optimize
from scipy.optimize import curve_fit

from pmriprep.models.dce.base import FitResult, ModelParameter, ParamsType, PKModel
from pmriprep.utils.dce_constants import (
    EPSILON_KTRANS,
    KTRANS_BOUNDS,
    VP_BOUNDS,
)


# Epsilon for AIF values to prevent division by zero in Patlak plot
EPSILON_AIF: float = 1e-9


class PatlakModel(PKModel):
    """Patlak pharmacokinetic model assuming no backflux.

    The Patlak model describes unidirectional contrast agent transfer from
    the blood plasma to the extravascular space, assuming negligible backflux
    (kep ≈ 0). This assumption is valid when:
        - BBB is intact (kep << Ktrans)
        - Only early post-injection time points are used
        - ve is large relative to Ktrans × t_max

    Model equation:
        C_t(t) = vp × Cp(t) + Ktrans × ∫₀ᵗ Cp(τ)dτ

    Graphical analysis (Patlak plot):
        Y = C_t(t) / Cp(t)
        X = ∫₀ᵗ Cp(τ)dτ / Cp(t)

        Y = vp + Ktrans × X

        where:
            - slope = Ktrans
            - intercept = vp

    Parameters:
        Ktrans: Volume transfer constant (min⁻¹), 0-3 min⁻¹
            - Reflects unidirectional influx rate
            - For intact BBB: typically < 0.05 min⁻¹
            - For disrupted BBB: can exceed 0.1 min⁻¹

        vp: Plasma volume fraction, 0.001-0.3
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

    References:
        Patlak CS, Blasberg RG, Fenstermacher JD. Graphical evaluation of
        blood-to-brain transfer constants from multiple-time uptake data.
        J Cereb Blood Flow Metab. 1983;3(1):1-7.

        Tofts PS, Brix G, Buckley DL, et al. Estimating kinetic parameters
        from dynamic contrast-enhanced T1-weighted MRI of a diffusable
        tracer: standardized quantities and symbols. J Magn Reson Imaging.
        1999;10(3):223-232.
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

        Returns:
            List of two ModelParameter objects:
                - Ktrans: volume transfer constant (influx)
                - vp: plasma volume fraction
        """
        return [
            ModelParameter(
                name="Ktrans",
                bounds=(KTRANS_BOUNDS.min, KTRANS_BOUNDS.max),
                typical_range=(0.01, 0.5),
                log_transform=True,
                description="Volume transfer constant (influx)",
                units="min^-1",
            ),
            ModelParameter(
                name="vp",
                bounds=(VP_BOUNDS.min, VP_BOUNDS.max),
                typical_range=(0.01, 0.1),
                log_transform=False,
                description="Plasma volume fraction",
                units="",
            ),
        ]

    def evaluate(
        self,
        params: ParamsType,
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Evaluate the Patlak model.

        Computes C_t(t) = vp × Cp(t) + Ktrans × ∫₀ᵗ Cp(τ)dτ

        The integral is computed using the cumulative trapezoidal rule.

        Args:
            params: Parameter values [Ktrans, vp].
            aif: Arterial input function plasma concentration values Cp(t) in mM.
            time_points: Time points (minutes) at which to evaluate C_t.

        Returns:
            Tissue concentration values C_t(t) at each time point in mM.
        """
        ktrans = params[0]
        vp = params[1]

        # Get time step (assume uniform sampling)
        n_times = len(time_points)
        if n_times < 2:
            return vp * aif if n_times == 1 else np.zeros(n_times, dtype=np.float64)

        # Compute cumulative integral of AIF using trapezoidal rule
        # ∫₀ᵗ Cp(τ)dτ at each time point
        dt = float(time_points[1] - time_points[0])

        # Use cumulative trapezoid (initial value is 0)
        aif_integral = np.zeros(n_times, dtype=np.float64)
        aif_integral[1:] = np.cumsum(0.5 * (aif[:-1] + aif[1:]) * dt)

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

        Args:
            tissue_curve: Observed tissue concentration values C_t(t) in mM.
            aif: Arterial input function plasma concentration values Cp(t) in mM.
            time_points: Time points (minutes) corresponding to measurements.

        Returns:
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

        dt = float(time_points[1] - time_points[0])

        # Compute cumulative integral of AIF
        aif_integral = np.zeros(n_times, dtype=np.float64)
        aif_integral[1:] = np.cumsum(0.5 * (aif[:-1] + aif[1:]) * dt)

        # Identify valid points where Cp(t) is non-zero
        valid_mask = aif > EPSILON_AIF

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

        Args:
            tissue_curve: Observed tissue concentration values C_t(t) in mM.
            aif: Arterial input function plasma concentration values Cp(t) in mM.
            time_points: Time points (minutes) corresponding to measurements.
            start_index: Optional index to start linear fit (skip early points).
                If None, uses points after peak AIF.

        Returns:
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

        # Linear regression: Y = vp + Ktrans × X
        # Using numpy.polyfit (degree 1)
        try:
            # polyfit returns [slope, intercept] for degree 1
            coeffs = np.polyfit(x_fit, y_fit, 1)
            ktrans = float(coeffs[0])  # slope
            vp = float(coeffs[1])      # intercept
            converged = True
        except Exception:
            # Linear regression failed
            ktrans = 0.0
            vp = 0.01
            converged = False

        # Clip to bounds
        ktrans = float(np.clip(ktrans, KTRANS_BOUNDS.min, KTRANS_BOUNDS.max))
        vp = float(np.clip(vp, VP_BOUNDS.min, VP_BOUNDS.max))

        fitted_params = [ktrans, vp]

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
        use_graphical: bool = True,
    ) -> FitResult:
        """Fit the Patlak model to observed tissue concentration data.

        By default, uses the fast graphical (linear regression) method.
        Can optionally use nonlinear least squares for comparison.

        Args:
            tissue_curve: Observed tissue concentration values C_t(t) in mM.
            aif: Arterial input function plasma concentration values Cp(t) in mM.
            time_points: Time points (minutes) corresponding to measurements.
            initial_guess: Optional initial parameter values [Ktrans, vp].
                Used only for nonlinear fitting (use_graphical=False).
            use_graphical: If True (default), use fast graphical analysis.
                If False, use nonlinear least squares optimization.

        Returns:
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
            )
            converged = True
            fitted_params = popt.tolist()
            perr = np.sqrt(np.diag(pcov))
            param_names = self.get_parameter_names()
            uncertainties = dict(zip(param_names, perr.tolist()))
        except (RuntimeError, ValueError):
            # Fallback: L-BFGS-B
            try:
                def objective(params: npt.NDArray[np.floating]) -> float:
                    predicted = self.evaluate(params, aif, time_points)
                    return float(np.sum((tissue_curve - predicted) ** 2))

                result = optimize.minimize(
                    objective,
                    x0=x0,
                    method="L-BFGS-B",
                    bounds=list(zip(lower_bounds, upper_bounds)),
                    options={"maxiter": 1000, "ftol": 1e-10},
                )
                converged = result.success
                fitted_params = result.x.tolist()
            except Exception:
                converged = False
                fitted_params = x0

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

        # Compute quality flag
        quality_flag = "good"
        if not converged:
            quality_flag = "poor_convergence"
        elif uncertainties:
            for pname, unc in uncertainties.items():
                pval = param_dict.get(pname, 0.0)
                if abs(pval) > 1e-10 and unc / abs(pval) > 0.5:
                    quality_flag = "high_uncertainty"
                    break

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
            quality_flag=quality_flag,
        )

    def fit_constrained(
        self,
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        fixed_vp: float,
        initial_guess: Optional[Sequence[float]] = None,
    ) -> FitResult:
        """Fit Patlak model with vp fixed from DSC CBV.

        Only Ktrans is a free parameter. Uses graphical analysis:
            C_t(t) - vp * Cp(t) = Ktrans * integral(Cp)
        which simplifies to linear regression for Ktrans (slope).

        Args:
            tissue_curve: Observed tissue concentration values C_t(t) in mM.
            aif: Arterial input function plasma concentration values Cp(t) in mM.
            time_points: Time points (minutes) corresponding to measurements.
            fixed_vp: Fixed plasma volume fraction from DSC CBV.
            initial_guess: Not used (graphical method).

        Returns:
            FitResult with n_parameters=1 and fitting_method='graphical_constrained_vp'.
        """
        n_obs = len(time_points)

        # Clamp fixed_vp
        fixed_vp = float(np.clip(fixed_vp, VP_BOUNDS.min, VP_BOUNDS.max))

        if n_obs < 2:
            return FitResult(
                parameters={"Ktrans": 0.0, "vp": fixed_vp},
                r_squared=0.0,
                residual_sum_squares=float("inf"),
                aic=float("inf"),
                bic=float("inf"),
                model_name=self.name,
                n_parameters=1,
                n_observations=n_obs,
                converged=False,
                fitting_method="graphical_constrained_vp",
                fitted_curve=self.evaluate([0.0, fixed_vp], aif, time_points),
            )

        dt = float(time_points[1] - time_points[0])

        # Compute cumulative integral of AIF
        aif_integral = np.zeros(n_obs, dtype=np.float64)
        aif_integral[1:] = np.cumsum(0.5 * (aif[:-1] + aif[1:]) * dt)

        # Subtract vascular contribution: residual = C_t(t) - vp * Cp(t)
        residual = tissue_curve - fixed_vp * aif

        # Find start index after AIF peak
        peak_idx = int(np.argmax(aif))
        start_index = min(peak_idx + 1, n_obs - 2)

        # Valid points: AIF integral > 0, after peak
        valid = np.zeros(n_obs, dtype=bool)
        valid[start_index:] = True
        valid &= aif_integral > EPSILON_AIF

        n_valid = int(np.sum(valid))
        if n_valid < 2:
            return FitResult(
                parameters={"Ktrans": 0.0, "vp": fixed_vp},
                r_squared=0.0,
                residual_sum_squares=float("inf"),
                aic=float("inf"),
                bic=float("inf"),
                model_name=self.name,
                n_parameters=1,
                n_observations=n_obs,
                converged=False,
                fitting_method="graphical_constrained_vp",
                fitted_curve=self.evaluate([0.0, fixed_vp], aif, time_points),
            )

        # Linear regression: residual = Ktrans * aif_integral (no intercept)
        # Ktrans = sum(residual * aif_integral) / sum(aif_integral^2)
        x = aif_integral[valid]
        y = residual[valid]

        try:
            ktrans = float(np.sum(x * y) / np.sum(x * x))
            converged = True
        except Exception:
            ktrans = 0.0
            converged = False

        ktrans = float(np.clip(ktrans, KTRANS_BOUNDS.min, KTRANS_BOUNDS.max))

        fitted_params = [ktrans, fixed_vp]
        fitted_curve = self.evaluate(fitted_params, aif, time_points)
        rss = float(np.sum((tissue_curve - fitted_curve) ** 2))
        r_squared = self.compute_r_squared(tissue_curve, fitted_curve)

        n_free = 1
        aic = float(n_obs * np.log(rss / n_obs) + 2 * n_free) if rss > 0 and n_obs > 0 else float("inf")
        bic = float(n_obs * np.log(rss / n_obs) + n_free * np.log(n_obs)) if rss > 0 and n_obs > 0 else float("inf")

        return FitResult(
            parameters={"Ktrans": ktrans, "vp": fixed_vp},
            r_squared=r_squared,
            residual_sum_squares=rss,
            aic=aic,
            bic=bic,
            model_name=self.name,
            n_parameters=n_free,
            n_observations=n_obs,
            converged=converged,
            fitting_method="graphical_constrained_vp",
            fitted_curve=fitted_curve,
        )

    def compute_derived_parameters(
        self, params: ParamsType
    ) -> dict[str, float]:
        """Compute derived parameters from fitted values.

        For the Patlak model, there are no derived parameters since
        kep is assumed to be zero (no backflux).

        Args:
            params: Fitted parameter values [Ktrans, vp].

        Returns:
            Empty dictionary (no derived parameters for Patlak).
        """
        # Patlak assumes kep = 0 (no backflux), so no derived parameters
        return {}
