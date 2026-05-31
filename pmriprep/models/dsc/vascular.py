"""Vascular (gamma-variate) model for DSC perfusion.

Implements a parametric model where the residue function is based on
a gamma-variate distribution of capillary transit times, following
Mouridsen et al. (2006) and Jespersen et al. (2012).

The model accounts for transit time heterogeneity (CTH) in the
microvasculature, which affects the relationship between CBF and CBV.
"""

import numpy as np
import numpy.typing as npt
from scipy import special  # type: ignore[import-untyped]
from scipy.signal import fftconvolve  # type: ignore[import-untyped]

from pmriprep.models.dsc.base import (
    DSCModel,
    DSCModelParameter,
    ParamsLike,
)
from pmriprep.utils.dsc_constants import (
    CBF_BOUNDS,
    VASCULAR_ALPHA_BOUNDS,
    VASCULAR_BETA_BOUNDS,
    VASCULAR_DELAY_BOUNDS,
)


class VascularModel(DSCModel):
    """Gamma-variate vascular model for transit time heterogeneity.

    The residue function is derived from assuming a gamma distribution
    of capillary transit times:

        R(t) = CBF × [1 - gammainc(α, (t - delay) / β)]

    where gammainc is the regularized lower incomplete gamma function.

    This model captures the physiology of capillary transit time
    heterogeneity (CTH), which affects oxygen extraction and is
    altered in various pathological conditions.

    Parameters
    ----------
        cbf: Cerebral blood flow (mL/100g/min). This is R(0), the
             maximum height of the residue function.
        alpha: Shape parameter of the gamma distribution (dimensionless).
               Controls the sharpness of the transit time distribution.
               Range: 0.5 to 10.
        beta: Scale parameter of the gamma distribution (seconds).
              Controls the spread of transit times.
              Range: 0.1 to 10s.
        delay: Bolus arrival delay relative to AIF (seconds).
               Can be negative for tissue ahead of AIF sampling site.
               Range: -3 to +10s.

    Derived physiological parameters:
        MTT: Mean transit time = α × β (seconds)
        CTH: Capillary transit time heterogeneity = β × √α (seconds)
             Standard deviation of the transit time distribution.
        RCTH: Relative CTH = 1/√α (dimensionless)
              Also known as coefficient of variation (CV) of transit times.
              RCTH = CTH / MTT = 1/√α

    References
    ----------
        Mouridsen K, Friston K, Hjort N, et al. (2006).
        "Bayesian estimation of cerebral perfusion using a physiological
        model of microvasculature." NeuroImage 33(2):570-579.

        Jespersen SN, Østergaard L (2012).
        "The roles of cerebral blood flow, capillary transit time
        heterogeneity, and oxygen tension in brain oxygenation and
        metabolism." J Cereb Blood Flow Metab 32(2):264-277.
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
                transformed_prior_sigma=1.0,  # FABBER convention: wide log-space prior
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

        The Bayesian fitter works in this reparameterized space where MTT
        replaces beta. This breaks the banana-shaped (alpha, beta) degeneracy
        since MTT = alpha * beta is directly identifiable from the residue
        function area, while alpha captures only the shape.

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
                transformed_prior_sigma=1.0,
                description="Cerebral blood flow",
                units="s⁻¹",
            ),
            DSCModelParameter(
                name="mtt",
                bounds=(0.5, 30.0),
                typical_range=(2.0, 8.0),  # mean=5.0s, Ostergaard 1996
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

        The residue function represents the fraction of contrast agent
        remaining at time t, based on a gamma distribution of transit times:

            R(t) = CBF × [1 - gammainc(α, (t - delay) / β)]

        For t < delay, R(t) = CBF (no contrast has left yet).

        Parameters
        ----------
        params : ParamsLike
            [cbf, alpha, beta, delay] parameter values.
        time_points : npt.NDArray[np.floating]
            Time points (seconds) at which to evaluate R(t).

        Returns
        -------
        npt.NDArray[np.floating]
            Residue function values R(t) at each time point.
        """
        cbf, alpha, beta, delay = params

        # Guard against degenerate shape/scale parameters
        alpha = max(float(alpha), 1e-10)
        beta = max(float(beta), 1e-10)

        # Shift time by delay
        t_shifted = time_points - delay

        # Initialize residue array
        residue = np.zeros_like(time_points, dtype=np.float64)

        # For t < delay (or t_shifted < 0): R(t) = CBF
        # No contrast has left the tissue yet
        early_mask = t_shifted <= 0
        residue[early_mask] = cbf

        # For t >= delay: R(t) = CBF × [1 - gammainc(α, t_shifted/β)]
        # gammainc is the regularized lower incomplete gamma function
        late_mask = t_shifted > 0
        if np.any(late_mask):
            x = t_shifted[late_mask] / beta
            # scipy.special.gammainc is the regularized lower incomplete gamma
            # P(a,x) = gammainc(a, x) = γ(a,x) / Γ(a)
            gamma_inc = special.gammainc(alpha, x)
            residue[late_mask] = cbf * (1.0 - gamma_inc)

        return residue

    def tissue_curve(
        self,
        params: ParamsLike,
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute tissue concentration curve by convolving AIF with residue.

        C_tissue(t) = AIF(t) ⊗ R(t)

        Uses FFT convolution with post-hoc delay shift.
        The residue is computed without delay, then the convolved result is
        shifted by delay_samples to apply the arrival time offset.

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

        # Compute residue WITHOUT delay
        params_no_delay = [cbf, alpha, beta, 0.0]
        R = self.residue_function(params_no_delay, time_points)

        # FFT convolution
        conv = fftconvolve(aif, R, mode='full')[:n_times] * dt

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

        Derived parameters:
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
