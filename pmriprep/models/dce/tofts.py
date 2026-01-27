"""Standard Tofts pharmacokinetic model.

Implements the Standard Tofts model (Tofts & Kermode 1991) for DCE-MRI
analysis. This model assumes negligible plasma volume (vp ≈ 0) and is
appropriate when vascular contribution is minimal.

The tissue concentration is given by:
    C_t(t) = Ktrans × [Cp ⊗ exp(-kep×t)]

where:
    Ktrans = volume transfer constant (min⁻¹)
    ve = extravascular extracellular volume fraction (dimensionless)
    kep = efflux rate constant = Ktrans/ve (min⁻¹)
    Cp = plasma concentration (AIF)
    ⊗ = convolution operator
"""

from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt
from scipy import optimize

from pmriprep.models.dce.base import FitResult, ModelParameter, PKModel
from pmriprep.utils.dce_constants import EPSILON_VE, KTRANS_BOUNDS, VE_BOUNDS


class ToftsModel(PKModel):
    """Standard Tofts pharmacokinetic model.

    The Standard Tofts model describes contrast agent exchange between
    the blood plasma and the extravascular extracellular space (EES),
    assuming negligible plasma volume contribution.

    Model equation:
        C_t(t) = Ktrans × ∫[Cp(τ) × exp(-kep×(t-τ))]dτ
               = Ktrans × [Cp ⊗ exp(-kep×t)]

    Parameters:
        Ktrans: Volume transfer constant (min⁻¹), 0-3 min⁻¹
            - Reflects permeability × surface area per unit tissue volume
            - Typical tumor values: 0.01-0.5 min⁻¹
            - High-grade glioma: can exceed 0.5 min⁻¹

        ve: Extravascular extracellular volume fraction, 0.01-1.0
            - Fraction of tissue that is interstitial space
            - Typical brain values: 0.1-0.3
            - Tumor values: often increased 0.3-0.6

    Derived parameter:
        kep = Ktrans / ve: Efflux rate constant (min⁻¹)
            - Rate of contrast efflux from EES back to plasma

    References:
        Tofts PS, Kermode AG. Measurement of the blood-brain barrier
        permeability and leakage space using dynamic MR imaging.
        1. Fundamental concepts. Magn Reson Med. 1991;17(2):357-367.
    """

    @property
    def name(self) -> str:
        """Model name for identification."""
        return "tofts"

    @property
    def n_parameters(self) -> int:
        """Number of free parameters (Ktrans, ve)."""
        return 2

    def get_parameter_definitions(self) -> Sequence[ModelParameter]:
        """Get parameter definitions for Standard Tofts model.

        Returns:
            List of two ModelParameter objects:
                - Ktrans: volume transfer constant
                - ve: extravascular extracellular volume fraction
        """
        return [
            ModelParameter(
                name="Ktrans",
                bounds=(KTRANS_BOUNDS.min, KTRANS_BOUNDS.max),
                typical_range=(0.01, 0.5),
                log_transform=True,
                description="Volume transfer constant",
                units="min^-1",
            ),
            ModelParameter(
                name="ve",
                bounds=(VE_BOUNDS.min, VE_BOUNDS.max),
                typical_range=(0.1, 0.5),
                log_transform=False,
                description="Extravascular extracellular volume fraction",
                units="",
            ),
        ]

    def evaluate(
        self,
        params: Sequence[float],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Evaluate the Standard Tofts model.

        Computes C_t(t) = Ktrans × [Cp ⊗ exp(-kep×t)]

        The convolution is implemented using the trapezoidal rule for
        numerical integration.

        Args:
            params: Parameter values [Ktrans, ve].
            aif: Arterial input function plasma concentration values Cp(t) in mM.
            time_points: Time points (minutes) at which to evaluate C_t.

        Returns:
            Tissue concentration values C_t(t) at each time point in mM.
        """
        ktrans = params[0]
        ve = max(params[1], EPSILON_VE)  # Guard against division by zero

        # Compute efflux rate constant
        kep = ktrans / ve

        # Get time step (assume uniform sampling)
        n_times = len(time_points)
        if n_times < 2:
            return np.zeros(n_times, dtype=np.float64)

        dt = float(time_points[1] - time_points[0])

        # Compute convolution using trapezoidal rule
        # C_t(t) = Ktrans × ∫₀ᵗ Cp(τ) × exp(-kep×(t-τ)) dτ
        c_tissue = np.zeros(n_times, dtype=np.float64)

        # Precompute exponential decay factors
        # exp_decay[i] = exp(-kep × i × dt)
        exp_decay = np.exp(-kep * dt * np.arange(n_times))

        for i in range(n_times):
            if i == 0:
                c_tissue[i] = 0.0
                continue

            # Convolution integral from 0 to t[i]
            # Use trapezoidal rule: sum of Cp(τ) × exp(-kep×(t-τ)) × dt
            # At time point i, we integrate over all previous time points
            # t - τ = (i - j) × dt, so exp(-kep×(t-τ)) = exp_decay[i-j]

            # Build the integrand: Cp[j] × exp_decay[i-j] for j = 0 to i
            integrand = aif[: i + 1] * exp_decay[i::-1][: i + 1]

            # Trapezoidal integration
            integral = np.trapezoid(integrand, dx=dt)

            c_tissue[i] = ktrans * integral

        return c_tissue

    def fit(
        self,
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        initial_guess: Optional[Sequence[float]] = None,
    ) -> FitResult:
        """Fit the Standard Tofts model to observed tissue concentration data.

        Uses nonlinear least squares optimization (L-BFGS-B) with bounds.

        Args:
            tissue_curve: Observed tissue concentration values C_t(t) in mM.
            aif: Arterial input function plasma concentration values Cp(t) in mM.
            time_points: Time points (minutes) corresponding to measurements.
            initial_guess: Optional initial parameter values [Ktrans, ve].
                If None, uses default initial guess from typical ranges.

        Returns:
            FitResult containing:
                - parameters: Fitted Ktrans and ve values
                - derived parameters: kep
                - goodness-of-fit: R², AIC, BIC
                - fitted curve
        """
        n_obs = len(tissue_curve)

        # Set initial guess
        if initial_guess is None:
            x0 = self.get_initial_guess()
        else:
            x0 = list(initial_guess)

        # Get bounds for optimization
        bounds = self.get_bounds()
        bounds_scipy = list(zip(*bounds))  # Convert to scipy format

        # Define objective function (sum of squared residuals)
        def objective(params: npt.NDArray[np.floating]) -> float:
            predicted = self.evaluate(params, aif, time_points)
            return float(np.sum((tissue_curve - predicted) ** 2))

        # Run optimization
        try:
            result = optimize.minimize(
                objective,
                x0=x0,
                method="L-BFGS-B",
                bounds=list(zip(bounds_scipy[0], bounds_scipy[1])),
                options={"maxiter": 1000, "ftol": 1e-10},
            )
            converged = result.success
            fitted_params = result.x.tolist()
        except Exception:
            # Optimization failed, return initial guess
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

        # Compute derived parameters
        derived = self.compute_derived_parameters(fitted_params)

        # Build result
        param_dict = self.params_to_dict(fitted_params)
        param_dict.update(derived)  # Add kep to parameters

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
            fitting_method="nlls",
            fitted_curve=fitted_curve,
        )

    def compute_derived_parameters(
        self, params: Sequence[float]
    ) -> dict[str, float]:
        """Compute kep from Ktrans and ve.

        kep = Ktrans / ve

        Args:
            params: Fitted parameter values [Ktrans, ve].

        Returns:
            Dictionary with 'kep' value.
        """
        ktrans = params[0]
        ve = max(params[1], EPSILON_VE)

        kep = ktrans / ve

        return {"kep": kep}
