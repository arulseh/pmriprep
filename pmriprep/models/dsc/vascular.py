"""Vascular (gamma-variate) model for DSC perfusion.

Implements a parametric model where the residue function is based on
a gamma-variate distribution of capillary transit times.

The model accounts for transit time heterogeneity (CTH) in the
microvasculature, which affects the relationship between CBF and CBV.
"""

import numpy as np
import numpy.typing as npt
from scipy import special  # type: ignore[import-untyped]

from pmriprep.models.dsc.base import (
    DSCFitResult,
    DSCModel,
    DSCModelParameter,
    ParamsLike,
)
from pmriprep.utils.dsc_constants import (
    CBF_BOUNDS,
    MTT_BOUNDS,
    VASCULAR_ALPHA_BOUNDS,
    VASCULAR_BETA_BOUNDS,
    VASCULAR_DELAY_BOUNDS,
)
from pmriprep.utils.dsc_deconvolution import build_volterra_matrix


_CBF_TRANSFORMED_PRIOR_SIGMA: float = 5.0


class VascularModel(DSCModel):
    """Gamma-variate vascular model for transit time heterogeneity.

    The residue function is derived from assuming a gamma distribution
    of capillary transit times::

        R(t) = CBF × [1 - gammainc(α, t / β)]

    where gammainc is the regularized lower incomplete gamma function. The
    residue is delay-agnostic; bolus-arrival delay is applied at the AIF level
    in :meth:`tissue_curve`.

    This model captures the physiology of capillary transit time
    heterogeneity (CTH), which affects oxygen extraction and is
    altered in various pathological conditions.

    Parameters
    ----------
        cbf: Relative flow amplitude (s⁻¹). This is R(0), the
             maximum height of the residue function.
        alpha: Shape parameter of the gamma distribution (dimensionless).
               Controls the sharpness of the transit time distribution.
               Range: 0.5 to 100.
        beta: Scale parameter of the gamma distribution (seconds).
              Controls the spread of transit times.
              Derived as MTT / alpha; marginal range: 0.005 to 60 s.
              MTT and alpha define the joint fitting support.
        delay: Bolus arrival delay relative to AIF (seconds). Applied as a
               time shift of the convolved curve in tissue_curve, not to the
               residue shape. Can be negative for tissue ahead of the AIF
               sampling site. Range: -3 to +10s.

    Derived physiological parameters::

        MTT: Mean transit time = α × β (seconds)
        CTH: Capillary transit time heterogeneity = β × √α (seconds)
             Standard deviation of the transit time distribution.
        RCTH: Relative CTH = 1/√α (dimensionless)
              Also known as coefficient of variation (CV) of transit times.
              RCTH = CTH / MTT = 1/√α

    Notes
    -----
    Two modelling choices are deliberate and documented:

    - Priors are fixed, weakly-informative population priors (centred on the
      typical range, wide in log space) so the data dominate the fit. Per-voxel
      priors seeded from an SVD pre-fit are deliberately avoided: they would
      re-import the SVD delay and low-flow bias this model exists to remove.
    - The shape domain includes narrow transit-time distributions. Shape
      recovery remains sensitive to sampling and noise; its limit can affect
      jointly fitted CBF and MTT. MTT is fitted within 0.5 to 30 s, and beta
      is derived from MTT and alpha rather than independently constrained.

    """

    @property
    def name(self) -> str:
        """Model name for identification."""
        return "vascular"

    @property
    def n_parameters(self) -> int:
        """Number of free parameters (CBF, alpha, beta, delay)."""
        return 4

    def get_parameter_definitions(self) -> list[DSCModelParameter]:
        """Get parameter definitions for the vascular model.

        These are marginal model-space ranges. Joint support is specified by
        :meth:`get_fitting_parameter_definitions` through MTT = alpha * beta.

        Returns
        -------
        list of DSCModelParameter
            List of DSCModelParameter objects for CBF, alpha, beta, delay.
        """
        from pmriprep.models.bayesian_utils import TransformType

        return [
            DSCModelParameter(
                name="cbf",
                bounds=(CBF_BOUNDS.min, CBF_BOUNDS.max),
                typical_range=(0.01, 0.15),  # Relative s⁻¹; 3T typical 0.04-0.08
                transform=TransformType.LOG,
                transformed_prior_sigma=_CBF_TRANSFORMED_PRIOR_SIGMA,
                description="Cerebral blood flow",
                units="s⁻¹",
            ),
            DSCModelParameter(
                name="alpha",
                bounds=(VASCULAR_ALPHA_BOUNDS.min, VASCULAR_ALPHA_BOUNDS.max),
                typical_range=(1.5, 6.0),  # prior_mean=3.75
                transform=TransformType.LOG,
                description="Gamma shape parameter",
                units="",
            ),
            DSCModelParameter(
                name="beta",
                bounds=(VASCULAR_BETA_BOUNDS.min, VASCULAR_BETA_BOUNDS.max),
                typical_range=(0.5, 5.0),  # prior_mean=2.75
                transform=TransformType.LOG,
                description="Gamma scale parameter",
                units="s",
            ),
            DSCModelParameter(
                name="delay",
                bounds=(VASCULAR_DELAY_BOUNDS.min, VASCULAR_DELAY_BOUNDS.max),
                typical_range=(-1.0, 5.0),  # prior_mean=2.0
                transform=TransformType.IDENTITY,  # Can be negative
                description="Bolus arrival delay",
                units="s",
            ),
        ]

    def get_parameter_names(self) -> list[str]:
        """Get list of parameter names.

        Returns
        -------
        list of str
            List: ['cbf', 'alpha', 'beta', 'delay']
        """
        return ["cbf", "alpha", "beta", "delay"]

    def get_fitting_parameter_definitions(self) -> list[DSCModelParameter]:
        """Get parameter definitions for fitting space (CBF, MTT, alpha, delay).

        Fitting uses MTT in place of beta to separate transit duration from
        distribution shape. This reduces alpha/beta coupling but does not
        guarantee parameter identifiability at a given sampling rate or noise
        level. Beta is derived as MTT / alpha.

        Returns
        -------
        list of DSCModelParameter
            List of DSCModelParameter objects for CBF, MTT, alpha, delay.
        """
        from pmriprep.models.bayesian_utils import TransformType

        return [
            DSCModelParameter(
                name="cbf",
                bounds=(CBF_BOUNDS.min, CBF_BOUNDS.max),
                typical_range=(0.01, 0.15),  # Relative s⁻¹; 3T typical 0.04-0.08
                transform=TransformType.LOG,
                transformed_prior_sigma=_CBF_TRANSFORMED_PRIOR_SIGMA,
                description="Cerebral blood flow",
                units="s⁻¹",
            ),
            DSCModelParameter(
                name="mtt",
                bounds=(MTT_BOUNDS.min, MTT_BOUNDS.max),
                typical_range=(2.0, 8.0),  # mean=5.0s
                transform=TransformType.LOG,
                transformed_prior_sigma=1.0,
                description="Mean transit time",
                units="s",
            ),
            DSCModelParameter(
                name="alpha",
                bounds=(VASCULAR_ALPHA_BOUNDS.min, VASCULAR_ALPHA_BOUNDS.max),
                typical_range=(1.5, 6.0),
                transform=TransformType.LOG,
                transformed_prior_sigma=1.0,  # FABBER convention: uniform width
                description="Gamma shape parameter",
                units="",
            ),
            DSCModelParameter(
                name="delay",
                bounds=(VASCULAR_DELAY_BOUNDS.min, VASCULAR_DELAY_BOUNDS.max),
                typical_range=(-1.0, 5.0),
                transform=TransformType.IDENTITY,
                description="Bolus arrival delay",
                units="s",
            ),
        ]

    def fitting_to_model_params(
        self, fitting_params: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.float64]:
        """Convert fitting-space (CBF, MTT, alpha, delay) to model-space (CBF, alpha, beta, delay).

        Parameters
        ----------
        fitting_params : npt.NDArray[np.floating]
            [cbf, mtt, alpha, delay] in fitting space.

        Returns
        -------
        npt.NDArray[np.float64]
            [cbf, alpha, beta, delay] in model space.
        """
        cbf, mtt, alpha, delay = fitting_params
        beta = mtt / max(float(alpha), 1e-10)
        return np.array([cbf, alpha, beta, delay], dtype=np.float64)

    def model_to_fitting_params(
        self, model_params: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.float64]:
        """Convert model-space (CBF, alpha, beta, delay) to fitting-space (CBF, MTT, alpha, delay).

        Parameters
        ----------
        model_params : npt.NDArray[np.floating]
            [cbf, alpha, beta, delay] in model space.

        Returns
        -------
        npt.NDArray[np.float64]
            [cbf, mtt, alpha, delay] in fitting space.
        """
        cbf, alpha, beta, delay = model_params
        mtt = float(alpha) * float(beta)
        return np.array([cbf, mtt, alpha, delay], dtype=np.float64)

    def get_initial_guess(self) -> list[float]:
        """Get initial parameter guess for optimization.

        CBF=0.08 is the midpoint of typical_range in relative s⁻¹ units.
        Alpha=3, beta=2 gives MTT=6s (physiologically reasonable).

        Returns
        -------
        list of float
            List of initial parameter values [CBF, alpha, beta, delay].
        """
        return [0.08, 3.0, 2.0, 0.0]

    def residue_function(
        self,
        params: ParamsLike,
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute the residue function R(t) from vascular model parameters.

        The residue function is the fraction of contrast still present at
        time t, the survival function of a gamma transit-time distribution
        (shape α, scale β):

            R(t) = CBF × [1 - gammainc(α, t / β)]

        where gammainc is the regularized lower incomplete gamma function, so
        R(0) = CBF (the maximum). The residue is delay-agnostic: bolus-arrival
        ``delay`` is a property of the arterial input and is applied as a time
        shift in :meth:`tissue_curve`, not of the residue shape. It is accepted
        in ``params`` for signature consistency but does not affect R(t).

        Parameters
        ----------
        params : ParamsLike
            [cbf, alpha, beta, delay] parameter values. ``delay`` is ignored
            here (the arrival shift is applied in :meth:`tissue_curve`).
        time_points : npt.NDArray[np.floating]
            Time points (seconds) at which to evaluate R(t).

        Returns
        -------
        npt.NDArray[np.floating]
            Residue function values R(t) at each time point.
        """
        cbf, alpha, beta, _ = params

        # Guard against degenerate shape/scale parameters
        alpha = max(float(alpha), 1e-10)
        beta = max(float(beta), 1e-10)

        # R(t) = CBF × [1 - P(α, t/β)]; P = regularized lower incomplete gamma
        # (scipy.special.gammainc). Times ≤ 0 evaluate to R = CBF (the t=0 max).
        x = np.maximum(np.asarray(time_points, dtype=np.float64), 0.0) / beta
        gamma_inc = special.gammainc(alpha, x)
        residue: npt.NDArray[np.floating] = float(cbf) * (1.0 - gamma_inc)

        return residue

    def tissue_curve(
        self,
        params: ParamsLike,
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute tissue concentration curve by convolving AIF with residue.

        C_tissue(t) = AIF(t) ⊗ R(t)

        The convolution uses the shared (1, 4, 1)/6 product-integration
        (Volterra) operator. The
        bolus-arrival delay is applied last, as a post-hoc shift of the curve.

        Parameters
        ----------
        params : ParamsLike
            [cbf, alpha, beta, delay] parameter values.
        aif : npt.NDArray[np.floating]
            Arterial input function concentration values.
        time_points : npt.NDArray[np.floating]
            Time points (seconds) at which to evaluate C(t).

        Returns
        -------
        npt.NDArray[np.floating]
            Tissue concentration values C(t) at each time point.
        """
        n_times = len(time_points)
        if len(aif) != n_times:
            raise ValueError(
                f"AIF length ({len(aif)}) must match time_points ({n_times})"
            )

        if n_times < 2:
            return np.zeros(n_times, dtype=np.float64)

        cbf, alpha, beta, delay = params
        dt = float(time_points[1] - time_points[0])

        # Forward-model convolution C(t) = ∫ AIF(τ) R(t−τ) dτ via the
        # shared (1, 4, 1)/6 product-integration (Volterra) operator.
        # The delay shift below is applied AFTER, so it stays differentiable for
        # the MAP/VB Jacobian and the delay-agnostic residue is unaffected.
        R = self.residue_function(params, time_points)
        conv = build_volterra_matrix(aif, dt) @ R

        # Apply delay as post-hoc shift with linear interpolation
        # (smooth, differentiable w.r.t. delay for MAP/VB Jacobian)
        delay_frac = delay / dt
        n_shift = int(np.floor(delay_frac))
        f_shift = delay_frac - n_shift  # fractional part in [0, 1)

        # Vectorized: build index arrays for the two interpolation sources
        src_indices = np.arange(n_times) - n_shift  # integer-shifted source
        C_t = np.zeros(n_times, dtype=np.float64)

        # Integer-shift contribution: (1 - f) * conv[i - n_shift]
        valid_lo = (src_indices >= 0) & (src_indices < n_times)
        C_t[valid_lo] = (1.0 - f_shift) * conv[src_indices[valid_lo]]

        # Fractional contribution: f * conv[i - n_shift - 1]
        src_hi = src_indices - 1
        valid_hi = valid_lo & (src_hi >= 0) & (src_hi < n_times)
        C_t[valid_hi] += f_shift * conv[src_hi[valid_hi]]

        return C_t

    def compute_derived_parameters(
        self, params: ParamsLike
    ) -> dict[str, float]:
        """Compute derived physiological parameters from model parameters.

        Derived parameters::

            mtt: Mean transit time = α × β (seconds)
            cbv: Cerebral blood volume = CBF × MTT (relative, dimensionless)
            cth: Capillary transit heterogeneity = β × √α (seconds)
                 Standard deviation of transit time distribution
            rcth: Relative CTH = 1/√α (dimensionless)
                  Also known as CV (coefficient of variation) of transit times

        Parameters
        ----------
        params : ParamsLike
            [cbf, alpha, beta, delay] parameter values.

        Returns
        -------
        dict of str to float
            Dictionary with derived parameter values.
        """
        cbf, alpha, beta, _ = params

        # Guard against degenerate parameters
        alpha = max(float(alpha), 1e-10)
        beta = max(float(beta), 1e-10)

        # Mean transit time from gamma distribution: MTT = α × β
        mtt = alpha * beta

        # CBV from central volume theorem: CBV = CBF × MTT
        # CBF is in relative s⁻¹ (ΔR2* deconvolution units), MTT in seconds
        cbv = cbf * mtt

        # Capillary transit heterogeneity: CTH = β × √α
        # This is the standard deviation of the gamma distribution
        cth = beta * np.sqrt(alpha)

        # Relative CTH (coefficient of variation): RCTH = std/mean = 1/√α
        rcth = 1.0 / np.sqrt(alpha)

        return {
            "mtt": float(mtt),
            "cbv": float(cbv),
            "cth": float(cth),
            "rcth": float(rcth),
        }

    def get_mtt(self, alpha: float, beta: float) -> float:
        """Calculate mean transit time from shape and scale parameters.

        MTT = α × β (seconds)

        This is a convenience method for direct calculation.

        Parameters
        ----------
        alpha : float
            Gamma shape parameter.
        beta : float
            Gamma scale parameter (seconds).

        Returns
        -------
        float
            Mean transit time in seconds.
        """
        return alpha * beta

    def get_cth(self, alpha: float, beta: float) -> float:
        """Calculate capillary transit heterogeneity.

        CTH = β × √α (seconds)

        This represents the standard deviation of the transit time
        distribution and is a marker of microvascular dysfunction.

        Parameters
        ----------
        alpha : float
            Gamma shape parameter.
        beta : float
            Gamma scale parameter (seconds).

        Returns
        -------
        float
            Capillary transit heterogeneity in seconds.
        """
        return float(beta * np.sqrt(alpha))

    def get_rcth(self, alpha: float) -> float:
        """Calculate relative capillary transit heterogeneity.

        RCTH = 1/√α (dimensionless)

        This is the coefficient of variation (CV) of the transit time
        distribution. Lower values indicate more homogeneous transit times.

        Parameters
        ----------
        alpha : float
            Gamma shape parameter.

        Returns
        -------
        float
            Relative CTH (dimensionless).
        """
        alpha_safe = max(float(alpha), 1e-10)
        return float(1.0 / np.sqrt(alpha_safe))


def _fit_vascular_nlls(
    model: VascularModel,
    concentration: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating] | None = None,
    sloppy: bool = False,
) -> DSCFitResult | None:
    """Fit the vascular model with bounded nonlinear least squares.

    Optimization uses the model's identifiable fitting space
    ``(CBF, MTT, alpha, delay)`` and returns model-space parameters
    ``(CBF, alpha, beta, delay)``. Log-transformed positive parameters and
    explicit transformed bounds constrain every fitted coordinate.

    Parameters
    ----------
    model : VascularModel
        Vascular model instance.
    concentration : npt.NDArray[np.floating]
        Measured tissue concentration curve.
    aif : npt.NDArray[np.floating]
        Arterial input function.
    time_points : npt.NDArray[np.floating]
        Acquisition times in seconds.
    weights : npt.NDArray[np.floating] or None, optional
        Per-frame motion weights.
    sloppy : bool, optional
        Use a smaller evaluation budget for tests.

    Returns
    -------
    DSCFitResult or None
        Fitted model result, or ``None`` if optimization cannot be run.
    """
    from scipy import optimize  # type: ignore[import-untyped]

    from pmriprep.models.bayesian_utils import (
        TransformType,
        from_transformed,
        to_transformed,
    )

    fitting_defs = list(model.get_fitting_parameter_definitions())
    n_params = len(fitting_defs)
    n_obs = len(concentration)
    if n_obs <= n_params:
        return None

    transforms = [definition.transform for definition in fitting_defs]
    upper_bounds: list[float | None] = [
        definition.bounds[1]
        if definition.transform == TransformType.LOG_LOGIT
        else None
        for definition in fitting_defs
    ]

    def _transform_bound(value: float, index: int) -> float:
        if transforms[index] == TransformType.LOG:
            value = max(value, float(np.finfo(np.float64).tiny))
        return float(
            to_transformed(value, transforms[index], upper_bounds[index])
        )

    fit_lower = np.array(
        [_transform_bound(definition.bounds[0], j)
         for j, definition in enumerate(fitting_defs)],
        dtype=np.float64,
    )
    fit_upper = np.array(
        [_transform_bound(definition.bounds[1], j)
         for j, definition in enumerate(fitting_defs)],
        dtype=np.float64,
    )

    initial_model = np.asarray(model.get_initial_guess(), dtype=np.float64)
    initial_fitting = model.model_to_fitting_params(initial_model)
    x0_fit = np.array(
        [
            float(
                to_transformed(
                    initial_fitting[j], transforms[j], upper_bounds[j]
                )
            )
            for j in range(n_params)
        ],
        dtype=np.float64,
    )
    x0_fit = np.clip(x0_fit, fit_lower, fit_upper)

    sqrt_w = (
        np.sqrt(np.asarray(weights, dtype=np.float64))
        if weights is not None
        else np.ones(n_obs, dtype=np.float64)
    )

    def _to_natural(
        fitting_transformed: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.float64]:
        return np.array(
            [
                float(
                    from_transformed(
                        fitting_transformed[j],
                        transforms[j],
                        upper_bounds[j],
                    )
                )
                for j in range(n_params)
            ],
            dtype=np.float64,
        )

    def _residual(
        fitting_transformed: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.float64]:
        fitting_natural = _to_natural(fitting_transformed)
        model_params = model.fitting_to_model_params(fitting_natural)
        predicted = model.tissue_curve(model_params, aif, time_points)
        return sqrt_w * (concentration - predicted)

    max_nfev = 50 if sloppy else 100
    try:
        ls_result = optimize.least_squares(
            _residual,
            x0_fit,
            method="trf",
            bounds=(fit_lower, fit_upper),
            max_nfev=max_nfev,
            ftol=1e-8,
            xtol=1e-8,
        )
    except (ValueError, RuntimeError):
        return None

    optimum_fitting = _to_natural(
        np.asarray(ls_result.x, dtype=np.float64)
    )
    optimum_model = model.fitting_to_model_params(optimum_fitting)
    fitted_tissue = model.tissue_curve(optimum_model, aif, time_points)
    fitted_residue = model.compute_fitted_residue(
        optimum_model, time_points
    )

    residuals = np.asarray(concentration, dtype=np.float64) - fitted_tissue
    ss_res = float(np.sum((sqrt_w * residuals) ** 2))
    centered = np.asarray(concentration, dtype=np.float64) - float(
        np.mean(concentration)
    )
    ss_tot = float(np.sum((sqrt_w * centered) ** 2))
    r_squared = 1.0 - ss_res / max(
        ss_tot, float(np.finfo(np.float64).eps),
    )

    uncertainties: dict[str, float] = {}
    confidence_intervals: dict[str, tuple[float, float]] = {}
    try:
        jacobian = np.asarray(ls_result.jac, dtype=np.float64)
        information = jacobian.T @ jacobian + 1e-6 * np.eye(n_params)
        covariance = np.linalg.inv(information) * (
            ss_res / max(n_obs - n_params, 1)
        )
        cbf_std_transformed = float(np.sqrt(max(covariance[0, 0], 0.0)))
        uncertainties["cbf"] = float(optimum_fitting[0]) * cbf_std_transformed
        cbf_lower = float(
            from_transformed(
                ls_result.x[0] - 1.96 * cbf_std_transformed,
                transforms[0],
                upper_bounds[0],
            )
        )
        cbf_upper = float(
            from_transformed(
                ls_result.x[0] + 1.96 * cbf_std_transformed,
                transforms[0],
                upper_bounds[0],
            )
        )
        confidence_intervals["cbf"] = (
            max(cbf_lower, fitting_defs[0].bounds[0]),
            min(cbf_upper, fitting_defs[0].bounds[1]),
        )
    except np.linalg.LinAlgError:
        pass

    return DSCFitResult(
        parameters=model.params_to_dict(optimum_model),
        uncertainties=uncertainties,
        credible_intervals=confidence_intervals,
        r_squared=float(r_squared),
        residual_sum_squares=ss_res,
        model_name=model.name,
        n_parameters=n_params,
        n_observations=n_obs,
        converged=bool(ls_result.success),
        fitting_method="nlls",
        optimizer_status=int(ls_result.status),
        n_function_evaluations=int(ls_result.nfev),
        evaluation_budget_exhausted=bool(
            ls_result.status == 0 and ls_result.nfev >= max_nfev
        ),
        fitted_residue=fitted_residue,
        fitted_tissue_curve=fitted_tissue,
    )
