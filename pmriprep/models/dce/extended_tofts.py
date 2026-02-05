"""Extended Tofts pharmacokinetic model.

Implements the Extended Tofts model (Tofts et al. 1999) for DCE-MRI
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
from scipy import optimize
from scipy.optimize import curve_fit

from pmriprep.models.dce.base import FitResult, ModelParameter, ParamsType, PKModel
from pmriprep.utils.dce_constants import (
    EPSILON_VE,
    KTRANS_BOUNDS,
    MAX_KEP,
    VE_BOUNDS,
    VP_BOUNDS,
)


class ExtendedToftsModel(PKModel):
    """Extended Tofts pharmacokinetic model with plasma volume term.

    The Extended Tofts model describes contrast agent exchange between
    the blood plasma and the extravascular extracellular space (EES),
    including the contribution from plasma volume (vp).

    Model equation:
        C_t(t) = vp × Cp(t) + Ktrans × ∫[Cp(τ) × exp(-kep×(t-τ))]dτ
               = vp × Cp(t) + Ktrans × [Cp ⊗ exp(-kep×t)]

    Parameters:
        Ktrans: Volume transfer constant (min⁻¹), 0-3 min⁻¹
            - Reflects permeability × surface area per unit tissue volume
            - Typical tumor values: 0.01-0.5 min⁻¹
            - High-grade glioma: can exceed 0.5 min⁻¹

        ve: Extravascular extracellular volume fraction, 0.01-1.0
            - Fraction of tissue that is interstitial space
            - Typical brain values: 0.1-0.3
            - Tumor values: often increased 0.3-0.6

        vp: Plasma volume fraction, 0.001-0.3
            - Fraction of tissue that is blood plasma
            - Typical brain values: 0.01-0.05
            - Well-vascularized tumors: can exceed 0.1

    Constraint:
        ve + vp ≤ 1.0 (enforced during fitting)

    Derived parameter:
        kep = Ktrans / ve: Efflux rate constant (min⁻¹)
            - Rate of contrast efflux from EES back to plasma

    References:
        Tofts PS, Brix G, Buckley DL, et al. Estimating kinetic parameters
        from dynamic contrast-enhanced T1-weighted MRI of a diffusable
        tracer: standardized quantities and symbols. J Magn Reson Imaging.
        1999;10(3):223-232.
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

        Returns:
            List of three ModelParameter objects:
                - Ktrans: volume transfer constant
                - ve: extravascular extracellular volume fraction
                - vp: plasma volume fraction
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
            ModelParameter(
                name="vp",
                bounds=(VP_BOUNDS.min, VP_BOUNDS.max),
                typical_range=(0.01, 0.1),
                log_transform=False,
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

        Args:
            tissue_curve: Observed tissue concentration C_t(t).
            aif: Arterial input function Cp(t).
            time_points: Time points in minutes.

        Returns:
            List of [Ktrans_init, ve_init, vp_init].
        """
        try:
            peak_idx = int(np.argmax(aif))
            peak_time = time_points[peak_idx]

            # Estimate vp from peak ratio (tissue peak / AIF peak)
            aif_peak = float(np.max(aif))
            tissue_peak = float(np.max(tissue_curve))
            if aif_peak > 1e-6:
                vp_init = float(np.clip(tissue_peak / aif_peak, VP_BOUNDS.min, VP_BOUNDS.max))
            else:
                vp_init = 0.02

            # Estimate Ktrans from early slope (0.5-2.0 min post-peak)
            early_mask = (time_points > peak_time + 0.5) & (time_points < peak_time + 2.0)
            if np.sum(early_mask) >= 2:
                coeffs = np.polyfit(time_points[early_mask], tissue_curve[early_mask], 1)
                slope = coeffs[0]
                mean_aif_early = float(np.mean(aif[early_mask]))
                ktrans_init = max(slope / max(mean_aif_early, 1e-6), 0.001)
            else:
                ktrans_init = 0.1

            # Estimate ve from late-phase decay
            late_mask = time_points > time_points[-1] * 0.6
            if np.sum(late_mask) >= 3 and ktrans_init > 0.001:
                late_t = time_points[late_mask]
                late_c = tissue_curve[late_mask]
                late_c_safe = np.maximum(late_c, 1e-10)
                log_c = np.log(late_c_safe)
                coeffs_late = np.polyfit(late_t, log_c, 1)
                kep_est = max(-coeffs_late[0], 0.01)
                ve_init = float(np.clip(ktrans_init / kep_est, 0.01, 1.0))
            else:
                ve_init = 0.2

            ktrans_init = float(np.clip(ktrans_init, KTRANS_BOUNDS.min + 1e-6, KTRANS_BOUNDS.max))
            ve_init = float(np.clip(ve_init, VE_BOUNDS.min, VE_BOUNDS.max))
            vp_init = float(np.clip(vp_init, VP_BOUNDS.min, VP_BOUNDS.max))
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

        The convolution is implemented using the trapezoidal rule for
        numerical integration.

        Args:
            params: Parameter values [Ktrans, ve, vp].
            aif: Arterial input function plasma concentration values Cp(t) in mM.
            time_points: Time points (minutes) at which to evaluate C_t.

        Returns:
            Tissue concentration values C_t(t) at each time point in mM.
        """
        ktrans = params[0]
        ve = max(params[1], EPSILON_VE)  # Guard against division by zero
        vp = params[2]

        # Compute efflux rate constant, clamped to physiological maximum
        kep = min(ktrans / ve, MAX_KEP)

        # Get time step (assume uniform sampling)
        n_times = len(time_points)
        if n_times < 2:
            return np.zeros(n_times, dtype=np.float64)

        dt = float(time_points[1] - time_points[0])

        # Compute convolution using trapezoidal rule
        # C_t(t) = vp × Cp(t) + Ktrans × ∫₀ᵗ Cp(τ) × exp(-kep×(t-τ)) dτ
        c_tissue = np.zeros(n_times, dtype=np.float64)

        # Precompute exponential decay factors
        # exp_decay[i] = exp(-kep × i × dt)
        exp_decay = np.exp(-kep * dt * np.arange(n_times))

        for i in range(n_times):
            # Vascular contribution: vp × Cp(t)
            vascular_term = vp * aif[i]

            if i == 0:
                c_tissue[i] = vascular_term
                continue

            # Convolution integral from 0 to t[i]
            # Use trapezoidal rule: sum of Cp(τ) × exp(-kep×(t-τ)) × dt
            # At time point i, we integrate over all previous time points
            # t - τ = (i - j) × dt, so exp(-kep×(t-τ)) = exp_decay[i-j]

            # Build the integrand: Cp[j] × exp_decay[i-j] for j = 0 to i
            integrand = aif[: i + 1] * exp_decay[i::-1][: i + 1]

            # Trapezoidal integration
            integral = np.trapezoid(integrand, dx=dt)

            # Extravasation contribution: Ktrans × convolution
            extravasation_term = ktrans * integral

            c_tissue[i] = vascular_term + extravasation_term

        # Sanitize: replace non-finite values with 0
        c_tissue = np.where(np.isfinite(c_tissue), c_tissue, 0.0)

        return c_tissue

    def _apply_volume_constraint(
        self, params: ParamsType
    ) -> tuple[float, float, float]:
        """Apply the ve + vp ≤ 1.0 constraint.

        If the sum exceeds 1.0, scales both ve and vp proportionally
        to satisfy the constraint while maintaining their ratio.

        Args:
            params: Parameter values [Ktrans, ve, vp].

        Returns:
            Tuple of (Ktrans, ve, vp) with constraint satisfied.
        """
        ktrans = params[0]
        ve = params[1]
        vp = params[2]

        total_volume = ve + vp
        if total_volume > 1.0:
            # Scale down proportionally
            scale_factor = 1.0 / total_volume
            ve = ve * scale_factor
            vp = vp * scale_factor

        return ktrans, ve, vp

    def fit(
        self,
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        initial_guess: Optional[Sequence[float]] = None,
    ) -> FitResult:
        """Fit the Extended Tofts model to observed tissue concentration data.

        Uses scipy.optimize.curve_fit with TRF method, with L-BFGS-B fallback.
        Enforces the constraint ve + vp <= 1.0 after fitting.

        Args:
            tissue_curve: Observed tissue concentration values C_t(t) in mM.
            aif: Arterial input function plasma concentration values Cp(t) in mM.
            time_points: Time points (minutes) corresponding to measurements.
            initial_guess: Optional initial parameter values [Ktrans, ve, vp].
                If None, uses data-driven estimate or default.

        Returns:
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

        # curve_fit wrapper with volume constraint
        def _curve_func(
            t: npt.NDArray[np.floating], *params: float
        ) -> npt.NDArray[np.floating]:
            ktrans, ve, vp = self._apply_volume_constraint(list(params))
            return self.evaluate([ktrans, ve, vp], aif, t)

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
                    ktrans, ve, vp = self._apply_volume_constraint(params)
                    predicted = self.evaluate([ktrans, ve, vp], aif, time_points)
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

        # Clip parameters to bounds and apply constraint
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
        """Fit Extended Tofts with vp fixed from DSC CBV.

        Only Ktrans and ve are free parameters. The ve upper bound is
        adjusted so that ve + fixed_vp <= 1.0.

        Args:
            tissue_curve: Observed tissue concentration values C_t(t) in mM.
            aif: Arterial input function plasma concentration values Cp(t) in mM.
            time_points: Time points (minutes) corresponding to measurements.
            fixed_vp: Fixed plasma volume fraction from DSC CBV.
            initial_guess: Optional initial values [Ktrans, ve].
                If None, uses default initial guess.

        Returns:
            FitResult with n_parameters=2 and fitting_method='nlls_constrained_vp'.
        """
        n_obs = len(tissue_curve)

        # Clamp fixed_vp to valid range
        fixed_vp = float(np.clip(fixed_vp, VP_BOUNDS.min, VP_BOUNDS.max))

        # Set initial guess for [Ktrans, ve]
        if initial_guess is None:
            defs = self.get_parameter_definitions()
            x0 = [defs[0].get_prior_mean(), defs[1].get_prior_mean()]
        else:
            x0 = list(initial_guess[:2])

        # Bounds: ve upper bound clamped so ve + vp <= 1.0
        ve_upper = min(VE_BOUNDS.max, 1.0 - fixed_vp)
        ve_upper = max(ve_upper, EPSILON_VE)
        bounds_lo = [KTRANS_BOUNDS.min, VE_BOUNDS.min]
        bounds_hi = [KTRANS_BOUNDS.max, ve_upper]

        # Objective: evaluate with [Ktrans, ve, fixed_vp]
        def objective(params: npt.NDArray[np.floating]) -> float:
            full_params = [params[0], params[1], fixed_vp]
            predicted = self.evaluate(full_params, aif, time_points)
            return float(np.sum((tissue_curve - predicted) ** 2))

        try:
            result = optimize.minimize(
                objective,
                x0=x0,
                method="L-BFGS-B",
                bounds=list(zip(bounds_lo, bounds_hi)),
                options={"maxiter": 1000, "ftol": 1e-10},
            )
            converged = result.success
            fitted_2p = result.x.tolist()
        except Exception:
            converged = False
            fitted_2p = x0

        # Clip to bounds
        ktrans = float(np.clip(fitted_2p[0], KTRANS_BOUNDS.min, KTRANS_BOUNDS.max))
        ve = float(np.clip(fitted_2p[1], VE_BOUNDS.min, ve_upper))

        fitted_params = [ktrans, ve, fixed_vp]

        # Compute fitted curve and goodness-of-fit
        fitted_curve = self.evaluate(fitted_params, aif, time_points)
        rss = float(np.sum((tissue_curve - fitted_curve) ** 2))
        r_squared = self.compute_r_squared(tissue_curve, fitted_curve)
        # Use 2 free parameters for AIC/BIC
        n_free = 2
        aic = float(n_obs * np.log(rss / n_obs) + 2 * n_free) if rss > 0 and n_obs > 0 else float("inf")
        bic = float(n_obs * np.log(rss / n_obs) + n_free * np.log(n_obs)) if rss > 0 and n_obs > 0 else float("inf")

        # Derived parameters
        kep = ktrans / max(ve, EPSILON_VE)

        param_dict = {"Ktrans": ktrans, "ve": ve, "vp": fixed_vp, "kep": kep}

        return FitResult(
            parameters=param_dict,
            r_squared=r_squared,
            residual_sum_squares=rss,
            aic=aic,
            bic=bic,
            model_name=self.name,
            n_parameters=n_free,
            n_observations=n_obs,
            converged=converged,
            fitting_method="nlls_constrained_vp",
            fitted_curve=fitted_curve,
        )

    def compute_derived_parameters(
        self, params: ParamsType
    ) -> dict[str, float]:
        """Compute kep from Ktrans and ve.

        kep = Ktrans / ve

        Args:
            params: Fitted parameter values [Ktrans, ve, vp].

        Returns:
            Dictionary with 'kep' value.
        """
        ktrans = params[0]
        ve = max(params[1], EPSILON_VE)

        kep = ktrans / ve

        return {"kep": kep}
