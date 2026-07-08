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
from scipy.optimize import curve_fit

from pmriprep.models.bayesian_utils import TransformType
from pmriprep.models.dce.base import FitResult, ModelParameter, ParamsType, PKModel
from pmriprep.models.dce.initialization import (
    estimate_ktrans_uptake,
    estimate_ve_late_phase,
)
from pmriprep.utils.dce_constants import (
    EPSILON_VE,
    KEP_BOUNDS,
    KTRANS_BOUNDS,
    MAX_KEP,
    VE_BOUNDS,
)


class ToftsModel(PKModel):
    """Standard Tofts pharmacokinetic model.

    The Standard Tofts model describes contrast agent exchange between
    the blood plasma and the extravascular extracellular space (EES),
    assuming negligible plasma volume contribution.

    Model equation:
        C_t(t) = Ktrans × ∫[Cp(τ) × exp(-kep×(t-τ))]dτ
               = Ktrans × [Cp ⊗ exp(-kep×t)]

    Parameters
    ----------
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

        Returns
        -------
        Sequence[ModelParameter]
            List of two ModelParameter objects:
                - Ktrans: volume transfer constant
                - ve: extravascular extracellular volume fraction
        """
        return [
            ModelParameter(
                name="Ktrans",
                bounds=(KTRANS_BOUNDS.min, KTRANS_BOUNDS.max),
                typical_range=(0.001, 0.2),
                transform=TransformType.LOG,
                description="Volume transfer constant",
                units="min^-1",
            ),
            ModelParameter(
                name="ve",
                bounds=(VE_BOUNDS.min, VE_BOUNDS.max),
                typical_range=(0.01, 0.4),
                transform=TransformType.LOG_LOGIT,
                description="Extravascular extracellular volume fraction",
                units="",
            ),
        ]

    def estimate_initial_guess(
        self,
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> list[float]:
        """Estimate data-driven initial parameters from curve shape.

        Estimates Ktrans from early uptake slope relative to AIF, and ve
        from the washout decay rate.

        Parameters
        ----------
        tissue_curve : np.ndarray
            Observed tissue concentration C_t(t).
        aif : np.ndarray
            Arterial input function Cp(t).
        time_points : np.ndarray
            Time points in minutes.

        Returns
        -------
        list of float
            List of [Ktrans_init, ve_init].
        """
        try:
            peak_idx = int(np.argmax(aif))

            ktrans_init = estimate_ktrans_uptake(tissue_curve, aif, time_points, peak_idx)
            ve_init = estimate_ve_late_phase(tissue_curve, aif, peak_idx)

            ktrans_init = float(np.clip(ktrans_init, KTRANS_BOUNDS.min + 1e-6, KTRANS_BOUNDS.max))
            ve_init = float(np.clip(ve_init, VE_BOUNDS.min, VE_BOUNDS.max))
            return [ktrans_init, ve_init]
        except Exception:
            return self.get_initial_guess()

    def evaluate(
        self,
        params: ParamsType,
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Evaluate the Standard Tofts model.

        Computes C_t(t) = Ktrans × [Cp ⊗ exp(-kep×t)]

        The convolution is implemented using the rectangular rule
        (left Riemann sum via np.convolve).

        Parameters
        ----------
        params : ParamsType
            Parameter values [Ktrans, ve].
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
        ve = max(params[1], EPSILON_VE)  # Guard against division by zero

        # Compute efflux rate constant, clamped to physiological maximum
        kep = min(ktrans / ve, MAX_KEP)

        # Get time step (assume uniform sampling)
        n_times = len(time_points)
        if n_times < 2:
            return np.zeros(n_times, dtype=np.float64)

        dt = float(time_points[1] - time_points[0])

        # Compute convolution using np.convolve
        # C_t(t) = Ktrans × [Cp ⊗ exp(-kep×t)] × dt
        exp_kernel = np.exp(-kep * dt * np.arange(n_times))
        c_tissue = np.convolve(aif, exp_kernel, mode="full")[:n_times] * dt * ktrans

        # Sanitize: replace non-finite values with 0
        c_tissue = np.where(np.isfinite(c_tissue), c_tissue, 0.0)

        return c_tissue

    def fit(
        self,
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        initial_guess: Optional[Sequence[float]] = None,
        weights: Optional[npt.NDArray[np.floating]] = None,
        vp_bounds: Optional[tuple[float, float]] = None,
    ) -> FitResult:
        """Fit the Standard Tofts model to observed tissue concentration data.

        Uses scipy.optimize.curve_fit with TRF method (Trust Region Reflective).
        TRF natively returns parameter covariance for uncertainty estimation.
        On TRF failure, parameters are zeroed (no fallback optimizer).

        Parameters
        ----------
        tissue_curve : np.ndarray
            Observed tissue concentration values C_t(t) in mM.
        aif : np.ndarray
            Arterial input function plasma concentration values Cp(t) in mM.
        time_points : np.ndarray
            Time points (minutes) corresponding to measurements.
        initial_guess : Sequence[float], optional
            Optional initial parameter values [Ktrans, ve].
            If None, uses data-driven estimate or default.
        weights : np.ndarray, optional
            Optional per-frame weights for motion-weighted fitting.
        vp_bounds : tuple, optional
            Accepted for interface compatibility but ignored
            (standard Tofts has no vp parameter).

        Returns
        -------
        FitResult
            FitResult containing:
                - parameters: Fitted Ktrans and ve values
                - derived parameters: kep
                - uncertainties: from covariance matrix (curve_fit)
                - quality_flag: good/poor_convergence/high_uncertainty
                - goodness-of-fit: R², AIC, BIC
                - fitted curve
        """
        n_obs = len(tissue_curve)

        # Set initial guess: data-driven if no explicit guess
        if initial_guess is None:
            x0 = self.estimate_initial_guess(tissue_curve, aif, time_points)
        else:
            x0 = list(initial_guess)

        # Get bounds for optimization
        bounds = self.get_bounds()
        lower_bounds = [b[0] for b in bounds]
        upper_bounds = [b[1] for b in bounds]

        # curve_fit wrapper: must accept (time_points, *params)
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
            # Extract uncertainties from covariance diagonal
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

        # Compute derived parameters
        derived = self.compute_derived_parameters(fitted_params)

        # Build result
        param_dict = self.params_to_dict(fitted_params)
        param_dict.update(derived)  # Add kep to parameters

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

    def compute_derived_parameters(
        self, params: ParamsType
    ) -> dict[str, float]:
        """Compute kep from Ktrans and ve.

        kep = Ktrans / ve

        Parameters
        ----------
        params : ParamsType
            Fitted parameter values [Ktrans, ve].

        Returns
        -------
        dict
            Dictionary with 'kep' value.
        """
        ktrans = params[0]
        ve = params[1]

        # Reported kep is clamped to the physiological cap; above it kep is
        # intrinsically non-identifiable (see _compute_derived_params).
        if ve > EPSILON_VE:
            kep = min(ktrans / ve, KEP_BOUNDS.max)
        else:
            kep = 0.0

        return {"kep": kep}
