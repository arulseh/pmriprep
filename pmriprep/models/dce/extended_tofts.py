"""Extended Tofts pharmacokinetic model.

Implements the Extended Tofts model for DCE-MRI
analysis. This model includes a vascular plasma volume term (vp) in
addition to the Standard Tofts model parameters.

The tissue concentration is given by:
    C_t(t) = vp × Cp(t) + Ktrans × [Cp ⊗ exp(-kep×t)]

where:
    Ktrans = volume transfer constant (min⁻¹)
    ve = extravascular extracellular volume fraction (dimensionless)
    vp = plasma volume fraction (dimensionless)
    kep = efflux rate constant = Ktrans/ve (min⁻¹)
    Cp = plasma concentration (AIF)
    ⊗ = convolution operator

The constraint ve + vp ≤ 1.0 is enforced during fitting.
"""

from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

from pmriprep.models.bayesian_utils import TransformType
from pmriprep.models.dce._convolution import analytic_exp_convolution
from pmriprep.models.dce.base import FitResult, ModelParameter, ParamsType, PKModel
from pmriprep.models.dce.initialization import (
    estimate_ktrans_uptake,
    estimate_ve_late_phase,
    estimate_vp_peak_ratio,
)
from pmriprep.utils.dce_constants import (
    EPSILON_VE_FORWARD,
    KTRANS_BOUNDS,
    VE_BOUNDS,
    VP_BOUNDS,
    VP_VE_TOTAL_MAX,
)
from pmriprep.utils.dce_metrics import compute_kep


class ExtendedToftsModel(PKModel):
    """Extended Tofts pharmacokinetic model with plasma volume term.

    The Extended Tofts model describes contrast agent exchange between
    the blood plasma and the extravascular extracellular space (EES),
    including the contribution from plasma volume (vp).

    Model equation:
        C_t(t) = vp × Cp(t) + Ktrans × ∫[Cp(τ) × exp(-kep×(t-τ))]dτ
               = vp × Cp(t) + Ktrans × [Cp ⊗ exp(-kep×t)]

    Parameters
    ----------
        Ktrans: Volume transfer constant (min⁻¹), 0-5 min⁻¹
        - Describes transfer from plasma to the extravascular space

        ve: Extravascular extracellular volume fraction, 0-1
            - Fraction of tissue that is interstitial space

        vp: Plasma volume fraction, 0-1
            - Fraction of tissue that is blood plasma

    Constraint:
        ve + vp ≤ 1.0 (enforced during fitting)

    Derived parameter:
        kep = Ktrans / ve: Efflux rate constant (min⁻¹)
            - Rate of contrast efflux from EES back to plasma
    """

    @property
    def name(self) -> str:
        """Model name for identification."""
        return "extended_tofts"

    @property
    def n_parameters(self) -> int:
        """Number of free parameters (Ktrans, ve, vp)."""
        return 3

    def get_parameter_definitions(self) -> Sequence[ModelParameter]:
        """Get parameter definitions for Extended Tofts model.

        Returns
        -------
        Sequence[ModelParameter]
            List of three ModelParameter objects:
                - Ktrans: volume transfer constant
                - ve: extravascular extracellular volume fraction
                - vp: plasma volume fraction
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
            ModelParameter(
                name="vp",
                bounds=(VP_BOUNDS.min, VP_BOUNDS.max),
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
        """Estimate data-driven initial parameters from curve shape.

        Estimates Ktrans from early slope, ve from washout decay, and
        vp from the peak tissue-to-AIF ratio.

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
            List of [Ktrans_init, ve_init, vp_init].
        """
        try:
            peak_idx = int(np.argmax(aif))

            vp_init = estimate_vp_peak_ratio(tissue_curve, aif)
            ktrans_init = estimate_ktrans_uptake(tissue_curve, aif, time_points, peak_idx)
            ve_init = estimate_ve_late_phase(tissue_curve, aif, peak_idx)

            ktrans_init = float(np.clip(ktrans_init, KTRANS_BOUNDS.min + 1e-6, KTRANS_BOUNDS.max))
            ve_init = float(np.clip(ve_init, VE_BOUNDS.min, VE_BOUNDS.max))
            vp_init = float(np.clip(vp_init, VP_BOUNDS.min, VP_BOUNDS.max))
            # Project onto the ve + vp <= 1 feasible set so the optimizer starts
            # on the physical side of the volume constraint.
            ktrans_init, ve_init, vp_init = self._apply_volume_constraint(
                [ktrans_init, ve_init, vp_init]
            )
            return [ktrans_init, ve_init, vp_init]
        except Exception:
            return self.get_initial_guess()

    def evaluate(
        self,
        params: ParamsType,
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Evaluate the Extended Tofts model.

        Computes C_t(t) = vp × Cp(t) + Ktrans × [Cp ⊗ exp(-kep×t)]

        The convolution is evaluated exactly for a piecewise-linear Cp
        (analytic expconv).

        Parameters
        ----------
        params : ParamsType
            Parameter values [Ktrans, ve, vp].
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
        ve = max(params[1], EPSILON_VE_FORWARD)
        vp = params[2]

        if params[1] <= 0:
            return vp * np.asarray(aif, dtype=np.float64)
        kep = ktrans / ve

        # Get time step (assume uniform sampling)
        n_times = len(time_points)
        if n_times < 2:
            return np.zeros(n_times, dtype=np.float64)

        dt = float(time_points[1] - time_points[0])

        # C_t(t) = vp*Cp(t) + Ktrans * [Cp (*) exp(-kep*t)], the convolution via
        # exact analytic (piecewise-linear AIF) integration.
        conv_term = ktrans * analytic_exp_convolution(aif, kep, dt)
        c_tissue = vp * aif + conv_term

        # Sanitize: replace non-finite values with 0
        c_tissue = np.where(np.isfinite(c_tissue), c_tissue, 0.0)

        return c_tissue

    def _apply_volume_constraint(
        self, params: ParamsType
    ) -> tuple[float, float, float]:
        """Apply the ve + vp ≤ 1.0 constraint.

        If the sum exceeds 1.0, scales both ve and vp proportionally
        to satisfy the constraint while maintaining their ratio.

        Parameters
        ----------
        params : ParamsType
            Parameter values [Ktrans, ve, vp].

        Returns
        -------
        tuple
            Tuple of (Ktrans, ve, vp) with constraint satisfied.
        """
        ktrans = params[0]
        ve = params[1]
        vp = params[2]

        total_volume = ve + vp
        if total_volume > VP_VE_TOTAL_MAX:
            # Scale down proportionally
            scale_factor = VP_VE_TOTAL_MAX / total_volume
            ve = ve * scale_factor
            vp = vp * scale_factor

        return ktrans, ve, vp

    def fit(
        self,
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        initial_guess: Optional[Sequence[float]] = None,
        weights: Optional[npt.NDArray[np.floating]] = None,
    ) -> FitResult:
        """Fit the Extended Tofts model to observed tissue concentration data.

        Uses scipy.optimize.curve_fit with TRF method.
        If the box-constrained fit violates ve + vp <= 1.0, refits with
        vp = 1 - ve. Failure of the initial fit produces clipped zero
        parameters. Failure of the boundary refit retains projected
        parameters, clears uncertainties and marks the fit unconverged.

        Parameters
        ----------
        tissue_curve : np.ndarray
            Observed tissue concentration values C_t(t) in mM.
        aif : np.ndarray
            Arterial input function plasma concentration values Cp(t) in mM.
        time_points : np.ndarray
            Time points (minutes) corresponding to measurements.
        initial_guess : Sequence[float], optional
            Optional initial parameter values [Ktrans, ve, vp].
            If None, uses data-driven estimate or default.
        weights : np.ndarray, optional
            Optional per-frame likelihood weights.

        Returns
        -------
        FitResult
            FitResult containing fitted parameters, uncertainties, and quality flag.
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

        # Evaluate natural parameters directly. The joint-volume constraint
        # is handled by the boundary refit below.
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

        # If the box-constrained optimum violates the joint tissue-volume
        # constraint, refit on its active boundary with covariance in the
        # reduced parameter space.
        if converged and fitted_params[1] + fitted_params[2] > VP_VE_TOTAL_MAX:
            ve_lower = max(bounds[1][0], VP_VE_TOTAL_MAX - bounds[2][1])
            ve_upper = min(bounds[1][1], VP_VE_TOTAL_MAX - bounds[2][0])

            def _boundary_curve_func(
                t: npt.NDArray[np.floating],
                ktrans: float,
                ve: float,
            ) -> npt.NDArray[np.floating]:
                return self.evaluate(
                    [ktrans, ve, VP_VE_TOTAL_MAX - ve],
                    aif,
                    t,
                )

            boundary_x0 = [
                float(np.clip(fitted_params[0], bounds[0][0], bounds[0][1])),
                float(np.clip(fitted_params[1], ve_lower, ve_upper)),
            ]
            try:
                boundary_popt, boundary_pcov = curve_fit(
                    _boundary_curve_func,
                    time_points,
                    tissue_curve,
                    p0=boundary_x0,
                    bounds=(
                        [bounds[0][0], ve_lower],
                        [bounds[0][1], ve_upper],
                    ),
                    method="trf",
                    maxfev=2000,
                    sigma=sigma_arr,
                    absolute_sigma=False,
                )
                ktrans_boundary = float(boundary_popt[0])
                ve_boundary = float(boundary_popt[1])
                fitted_params = [
                    ktrans_boundary,
                    ve_boundary,
                    VP_VE_TOTAL_MAX - ve_boundary,
                ]
                boundary_perr = np.sqrt(np.diag(boundary_pcov))
                uncertainties = {
                    "Ktrans": float(boundary_perr[0]),
                    "ve": float(boundary_perr[1]),
                    "vp": float(boundary_perr[1]),
                }
            except (RuntimeError, ValueError):
                converged = False
                uncertainties = {}

        # Clip parameters and guard against numerical boundary overshoot.
        fitted_params = self.clip_params(fitted_params)
        ktrans, ve, vp = self._apply_volume_constraint(fitted_params)
        fitted_params = [ktrans, ve, vp]

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
        param_dict.update(derived)

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
        """Compute kep from Ktrans and ve.

        kep = Ktrans / ve

        Parameters
        ----------
        params : ParamsType
            Fitted parameter values [Ktrans, ve, vp].

        Returns
        -------
        dict
            Dictionary with 'kep' value.
        """
        ktrans = params[0]
        ve = params[1]

        return {"kep": compute_kep(ktrans, ve)}
