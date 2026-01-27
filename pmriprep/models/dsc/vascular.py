"""Vascular (gamma-variate) model for DSC perfusion.

Implements a parametric model where the residue function is based on
a gamma-variate distribution of capillary transit times, following
Mouridsen et al. (2006) and Jesperson et al. (2012).

The model accounts for transit time heterogeneity (CTH) in the
microvasculature, which affects the relationship between CBF and CBV.
"""

from typing import Sequence

import numpy as np
import numpy.typing as npt
from scipy import special  # type: ignore[import-untyped]

from pmriprep.models.dsc.base import (
    DSCModel,
    DSCModelParameter,
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

    Parameters:
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

    References:
        Mouridsen K, Friston K, Hajnal N, et al. (2006).
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

        Returns:
            List of DSCModelParameter objects for CBF, alpha, beta, delay.
        """
        return [
            DSCModelParameter(
                name="cbf",
                bounds=(CBF_BOUNDS.min, CBF_BOUNDS.max),
                typical_range=(10.0, 100.0),  # Typical gray/white matter range
                log_transform=False,
                description="Cerebral blood flow",
                units="mL/100g/min",
            ),
            DSCModelParameter(
                name="alpha",
                bounds=(VASCULAR_ALPHA_BOUNDS.min, VASCULAR_ALPHA_BOUNDS.max),
                typical_range=(1.0, 5.0),  # Most physiological values
                log_transform=True,  # Spans orders of magnitude
                description="Gamma shape parameter",
                units="",
            ),
            DSCModelParameter(
                name="beta",
                bounds=(VASCULAR_BETA_BOUNDS.min, VASCULAR_BETA_BOUNDS.max),
                typical_range=(0.5, 4.0),  # Typical range for brain tissue
                log_transform=True,  # Better fitting in log space
                description="Gamma scale parameter",
                units="s",
            ),
            DSCModelParameter(
                name="delay",
                bounds=(VASCULAR_DELAY_BOUNDS.min, VASCULAR_DELAY_BOUNDS.max),
                typical_range=(-1.0, 3.0),  # Most delays are positive and small
                log_transform=False,  # Can be negative
                description="Bolus arrival delay",
                units="s",
            ),
        ]

    def get_parameter_names(self) -> list[str]:
        """Get list of parameter names.

        Returns:
            List: ['cbf', 'alpha', 'beta', 'delay']
        """
        return ["cbf", "alpha", "beta", "delay"]

    def residue_function(
        self,
        params: Sequence[float],
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute the residue function R(t) from vascular model parameters.

        The residue function represents the fraction of contrast agent
        remaining at time t, based on a gamma distribution of transit times:

            R(t) = CBF × [1 - gammainc(α, (t - delay) / β)]

        For t < delay, R(t) = CBF (no contrast has left yet).

        Args:
            params: [cbf, alpha, beta, delay] parameter values.
            time_points: Time points (seconds) at which to evaluate R(t).

        Returns:
            Residue function values R(t) at each time point.
        """
        cbf, alpha, beta, delay = params

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
        params: Sequence[float],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute tissue concentration curve by convolving AIF with residue.

        C_tissue(t) = AIF(t) ⊗ R(t)

        Uses trapezoidal integration for the convolution.

        Args:
            params: [cbf, alpha, beta, delay] parameter values.
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

        # Get time step (assume uniform spacing)
        if n_times < 2:
            return np.zeros(n_times, dtype=np.float64)

        dt = float(time_points[1] - time_points[0])

        # Compute residue function
        residue = self.residue_function(params, time_points)

        # Convolution: C_tissue(t) = dt * sum_{tau=0}^{t} AIF(tau) * R(t - tau)
        # Using direct convolution (trapezoidal integration)
        tissue = np.zeros(n_times, dtype=np.float64)

        for i in range(n_times):
            # Integrate from 0 to t_i
            # C(t_i) = integral_0^{t_i} AIF(tau) * R(t_i - tau) d_tau
            # With discrete sum: sum_{j=0}^{i} AIF[j] * R[i-j] * dt
            if i == 0:
                tissue[i] = 0.0
            else:
                # Create arrays for convolution integral
                # R(t_i - t_j) for j = 0..i
                r_values = residue[:i+1][::-1]  # R[i], R[i-1], ..., R[0]
                aif_values = aif[:i+1]  # AIF[0], AIF[1], ..., AIF[i]

                # Trapezoidal rule
                integrand = aif_values * r_values
                tissue[i] = np.trapezoid(integrand, dx=dt)

        return tissue

    def compute_derived_parameters(
        self, params: Sequence[float]
    ) -> dict[str, float]:
        """Compute derived physiological parameters from model parameters.

        Derived parameters:
            mtt: Mean transit time = α × β (seconds)
            cbv: Cerebral blood volume = CBF × MTT / 60 (mL/100g)
                 Conversion factor of 60 from CBF in mL/100g/min to mL/100g
            cth: Capillary transit heterogeneity = β × √α (seconds)
                 Standard deviation of transit time distribution
            rcth: Relative CTH = 1/√α (dimensionless)
                  Also known as CV (coefficient of variation) of transit times

        Args:
            params: [cbf, alpha, beta, delay] parameter values.

        Returns:
            Dictionary with derived parameter values.
        """
        cbf, alpha, beta, _ = params

        # Mean transit time from gamma distribution: MTT = α × β
        mtt = alpha * beta

        # CBV from central volume theorem: CBV = CBF × MTT
        # CBF is in mL/100g/min, MTT is in seconds
        # CBV should be in mL/100g, so divide by 60 (seconds per minute)
        cbv = (cbf * mtt) / 60.0

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

        Args:
            alpha: Gamma shape parameter.
            beta: Gamma scale parameter (seconds).

        Returns:
            Mean transit time in seconds.
        """
        return alpha * beta

    def get_cth(self, alpha: float, beta: float) -> float:
        """Calculate capillary transit heterogeneity.

        CTH = β × √α (seconds)

        This represents the standard deviation of the transit time
        distribution and is a marker of microvascular dysfunction.

        Args:
            alpha: Gamma shape parameter.
            beta: Gamma scale parameter (seconds).

        Returns:
            Capillary transit heterogeneity in seconds.
        """
        return float(beta * np.sqrt(alpha))

    def get_rcth(self, alpha: float) -> float:
        """Calculate relative capillary transit heterogeneity.

        RCTH = 1/√α (dimensionless)

        This is the coefficient of variation (CV) of the transit time
        distribution. Lower values indicate more homogeneous transit times.

        Args:
            alpha: Gamma shape parameter.

        Returns:
            Relative CTH (dimensionless).
        """
        return float(1.0 / np.sqrt(alpha))
