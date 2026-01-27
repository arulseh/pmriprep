"""Two-Compartment Exchange pharmacokinetic model (2CXM).

Implements the Two-Compartment Exchange Model (Sourbron & Buckley 2011) for
DCE-MRI analysis. This model accounts for flow-limited contrast agent exchange
and is appropriate for tissues with high permeability relative to flow.

The model describes contrast agent kinetics in terms of:
    - Fp: Plasma flow rate (mL/min/100mL of tissue)
    - PS: Permeability-surface area product (mL/min/100mL)
    - ve: Extravascular extracellular volume fraction
    - vp: Plasma volume fraction

The tissue concentration is computed using an analytical solution based on
eigenvalue decomposition of the two-compartment system.

References:
    Sourbron SP, Buckley DL. On the scope and interpretation of the Tofts
    models for DCE-MRI. Magn Reson Med. 2011;66(3):735-745.

    Brix G, Kiessling F, Luecht R, et al. Microcirculation and microvasculature
    in breast tumors: pharmacokinetic analysis of dynamic MR image series.
    Magn Reson Med. 2004;52(2):420-429.
"""

from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt
from scipy import optimize

from pmriprep.models.dce.base import FitResult, ModelParameter, PKModel
from pmriprep.utils.dce_constants import (
    EPSILON_VE,
    EPSILON_VP,
    FP_BOUNDS,
    PS_BOUNDS,
    VE_BOUNDS,
    VP_BOUNDS,
)


# Small epsilon to prevent division by zero in eigenvalue calculations
EPSILON_FLOW: float = 1e-9
EPSILON_EIGENVALUE: float = 1e-10


class TwoCompartmentExchangeModel(PKModel):
    """Two-Compartment Exchange Model (2CXM) for flow-limited kinetics.

    The 2CXM describes contrast agent kinetics using a two-compartment model
    where the plasma compartment (vp) exchanges with the extravascular
    extracellular space (ve) through a permeable capillary membrane.

    System of differential equations:
        vp × dCp/dt = Fp × (Ca(t) - Cp) - PS × (Cp - Ce)
        ve × dCe/dt = PS × (Cp - Ce)

    where:
        Ca = arterial input function (AIF)
        Cp = plasma compartment concentration
        Ce = EES compartment concentration

    The tissue concentration is:
        C_t(t) = vp × Cp(t) + ve × Ce(t)

    Analytical solution uses eigenvalue decomposition:
        C_t(t) = Fp × [A1 × exp(-K1×t) + A2 × exp(-K2×t)] ⊗ Ca(t)

    where K1, K2 are eigenvalues and A1, A2 are coefficients derived from
    the model parameters.

    Parameters:
        Fp: Plasma flow rate (mL/min/100mL), 0-200
            - Represents blood plasma flow through the tissue
            - Higher values indicate more perfusion
            - Typical brain values: 40-60 mL/min/100mL
            - Tumor values: often elevated 60-120 mL/min/100mL

        PS: Permeability-surface area product (mL/min/100mL), 0-100
            - Represents capillary permeability × surface area
            - Higher values indicate more permeable vessels
            - Typical brain values: 0-5 mL/min/100mL (intact BBB)
            - Tumor values: can exceed 20 mL/min/100mL

        ve: Extravascular extracellular volume fraction, 0.01-0.8
            - Fraction of tissue that is interstitial space
            - Typical brain values: 0.1-0.3
            - Tumor values: often increased 0.3-0.6

        vp: Plasma volume fraction, 0.001-0.3
            - Fraction of tissue that is blood plasma
            - Typical brain values: 0.01-0.05
            - Well-vascularized tumors: can exceed 0.1

    Derived parameters:
        E: Extraction fraction = PS / (Fp + PS)
            - Fraction of contrast extracted in single capillary pass
            - Range: 0 (no extraction) to 1 (complete extraction)

        Ktrans: Volume transfer constant = E × Fp (min⁻¹)
            - Equivalent to Tofts Ktrans in exchange limit
            - Ktrans = PS when Fp >> PS (permeability limited)
            - Ktrans = Fp when PS >> Fp (flow limited)

        kep: Efflux rate constant = Ktrans / ve (min⁻¹)
            - Rate of contrast efflux from EES back to plasma

    Model regimes:
        - Flow-limited (PS >> Fp): Ktrans ≈ Fp
        - Permeability-limited (Fp >> PS): Ktrans ≈ PS
        - Mixed regime: Ktrans = E × Fp

    References:
        Sourbron SP, Buckley DL. On the scope and interpretation of the Tofts
        models for DCE-MRI. Magn Reson Med. 2011;66(3):735-745.

        Brix G, Kiessling F, Luecht R, et al. Microcirculation and
        microvasculature in breast tumors: pharmacokinetic analysis of
        dynamic MR image series. Magn Reson Med. 2004;52(2):420-429.
    """

    @property
    def name(self) -> str:
        """Model name for identification."""
        return "2cxm"

    @property
    def n_parameters(self) -> int:
        """Number of free parameters (Fp, PS, ve, vp)."""
        return 4

    def get_parameter_definitions(self) -> Sequence[ModelParameter]:
        """Get parameter definitions for 2CXM model.

        Returns:
            List of four ModelParameter objects:
                - Fp: plasma flow rate
                - PS: permeability-surface area product
                - ve: extravascular extracellular volume fraction
                - vp: plasma volume fraction
        """
        return [
            ModelParameter(
                name="Fp",
                bounds=(FP_BOUNDS.min, FP_BOUNDS.max),
                typical_range=(20.0, 100.0),
                log_transform=True,
                description="Plasma flow rate",
                units="mL/min/100mL",
            ),
            ModelParameter(
                name="PS",
                bounds=(PS_BOUNDS.min, PS_BOUNDS.max),
                typical_range=(5.0, 50.0),
                log_transform=True,
                description="Permeability-surface area product",
                units="mL/min/100mL",
            ),
            ModelParameter(
                name="ve",
                bounds=(VE_BOUNDS.min, 0.8),  # Restricted to 0.8 for 2CXM
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

    def _compute_eigenvalues_and_coefficients(
        self,
        fp: float,
        ps: float,
        ve: float,
        vp: float,
    ) -> tuple[float, float, float, float]:
        """Compute eigenvalues and coefficients for analytical solution.

        The two-compartment system can be written as:
            d/dt [Cp; Ce] = A × [Cp; Ce] + B × Ca(t)

        where A is the system matrix. The eigenvalues of A give the decay
        rates K1 and K2, and the eigenvectors give the coefficients A1 and A2.

        Args:
            fp: Plasma flow rate (mL/min/100mL)
            ps: Permeability-surface area product (mL/min/100mL)
            ve: Extravascular extracellular volume fraction
            vp: Plasma volume fraction

        Returns:
            Tuple of (K1, K2, A1, A2) where:
                K1, K2: Eigenvalues (decay rates, min⁻¹)
                A1, A2: Coefficients for exponential terms
        """
        # Guard against division by zero
        vp = max(vp, EPSILON_VP)
        ve = max(ve, EPSILON_VE)
        fp = max(fp, EPSILON_FLOW)

        # System matrix elements (after normalizing by volume fractions)
        # Matrix A = [-(Fp + PS)/vp, PS/vp; PS/ve, -PS/ve]

        # Rate constants
        k_pe = ps / vp  # Transfer rate from plasma to EES
        k_ep = ps / ve  # Transfer rate from EES to plasma
        k_out = fp / vp  # Outflow rate from plasma

        # Trace and determinant of the system matrix
        # A = [-k_out - k_pe, k_pe; k_ep, -k_ep]
        trace = -(k_out + k_pe + k_ep)
        det = k_out * k_ep

        # Eigenvalues from characteristic polynomial: λ² - trace×λ + det = 0
        discriminant = trace * trace - 4 * det
        discriminant = max(discriminant, 0.0)  # Ensure non-negative
        sqrt_disc = np.sqrt(discriminant)

        # K1 and K2 are the magnitudes of the eigenvalues (positive)
        k1 = (-trace + sqrt_disc) / 2
        k2 = (-trace - sqrt_disc) / 2

        # Ensure K1 > K2 (K1 is the larger eigenvalue)
        if k2 > k1:
            k1, k2 = k2, k1

        # Guard against zero eigenvalues
        k1 = max(k1, EPSILON_EIGENVALUE)
        k2 = max(k2, EPSILON_EIGENVALUE)

        # Compute coefficients A1 and A2 for the impulse response:
        # h(t) = A1 × exp(-K1×t) + A2 × exp(-K2×t)
        #
        # From the analytical solution (Sourbron & Buckley 2011):
        # The tissue concentration impulse response is:
        # h(t) = Fp/vp × [(K1 - k_ep) × exp(-K1×t) + (k_ep - K2) × exp(-K2×t)] / (K1 - K2)
        #
        # This can be rewritten as:
        # h(t) = A1 × exp(-K1×t) + A2 × exp(-K2×t)
        #
        # where the total integral ∫h(t)dt = (vp + ve), giving:
        # A1 + A2 = (vp + ve) × Fp / vp = Fp × (1 + ve/vp)

        # Avoid division by zero in degenerate case
        if abs(k1 - k2) < EPSILON_EIGENVALUE:
            # Degenerate case: single exponential
            a1 = fp * (1 + ve / vp)
            a2 = 0.0
            k2 = k1
        else:
            # Standard two-exponential case
            # Coefficients derived from partial fraction expansion
            a1 = fp * (k1 - k_ep) / (k1 - k2)
            a2 = fp * (k_ep - k2) / (k1 - k2)

        return k1, k2, a1, a2

    def evaluate(
        self,
        params: Sequence[float],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Evaluate the Two-Compartment Exchange model.

        Computes tissue concentration using the analytical solution:
            C_t(t) = [A1 × exp(-K1×t) + A2 × exp(-K2×t)] ⊗ Ca(t)

        where ⊗ denotes convolution, and K1, K2, A1, A2 are computed from
        the model parameters using eigenvalue decomposition.

        Args:
            params: Parameter values [Fp, PS, ve, vp].
            aif: Arterial input function plasma concentration values Ca(t) in mM.
            time_points: Time points (minutes) at which to evaluate C_t.

        Returns:
            Tissue concentration values C_t(t) at each time point in mM.
        """
        fp = params[0]
        ps = params[1]
        ve = max(params[2], EPSILON_VE)
        vp = max(params[3], EPSILON_VP)

        # Get number of time points
        n_times = len(time_points)
        if n_times < 2:
            return np.zeros(n_times, dtype=np.float64)

        # Get time step (assume uniform sampling)
        dt = float(time_points[1] - time_points[0])

        # Compute eigenvalues and coefficients
        k1, k2, a1, a2 = self._compute_eigenvalues_and_coefficients(fp, ps, ve, vp)

        # Build the impulse response function h(t)
        # h(t) = A1 × exp(-K1×t) + A2 × exp(-K2×t)
        time_array = np.arange(n_times) * dt
        impulse_response = a1 * np.exp(-k1 * time_array) + a2 * np.exp(-k2 * time_array)

        # Tissue concentration is convolution of impulse response with AIF
        # C_t(t) = h(t) ⊗ Ca(t) = ∫₀ᵗ h(τ) × Ca(t-τ) dτ
        c_tissue = np.zeros(n_times, dtype=np.float64)

        for i in range(n_times):
            if i == 0:
                c_tissue[i] = 0.0
                continue

            # Convolution integral using trapezoidal rule
            # ∫₀ᵗⁱ h(τ) × Ca(t_i - τ) dτ
            # At time t_i, τ goes from 0 to t_i, so Ca(t_i - τ) uses AIF reversed

            # h(τ) for τ = 0 to t_i
            h_values = impulse_response[: i + 1]

            # Ca(t_i - τ) = Ca reversed from index i to 0
            aif_reversed = aif[i::-1][: i + 1]

            # Trapezoidal integration
            integrand = h_values * aif_reversed
            c_tissue[i] = np.trapezoid(integrand, dx=dt)

        return c_tissue

    def _apply_volume_constraint(
        self, params: Sequence[float]
    ) -> tuple[float, float, float, float]:
        """Apply the ve + vp <= 1.0 constraint.

        If the sum exceeds 1.0, scales both ve and vp proportionally
        to satisfy the constraint while maintaining their ratio.

        Args:
            params: Parameter values [Fp, PS, ve, vp].

        Returns:
            Tuple of (Fp, PS, ve, vp) with constraint satisfied.
        """
        fp = params[0]
        ps = params[1]
        ve = params[2]
        vp = params[3]

        total_volume = ve + vp
        if total_volume > 1.0:
            # Scale down proportionally
            scale_factor = 1.0 / total_volume
            ve = ve * scale_factor
            vp = vp * scale_factor

        return fp, ps, ve, vp

    def fit(
        self,
        tissue_curve: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        initial_guess: Optional[Sequence[float]] = None,
    ) -> FitResult:
        """Fit the 2CXM model to observed tissue concentration data.

        Uses nonlinear least squares optimization (L-BFGS-B) with bounds.
        Enforces the constraint ve + vp <= 1.0 during fitting.

        Args:
            tissue_curve: Observed tissue concentration values C_t(t) in mM.
            aif: Arterial input function plasma concentration values Ca(t) in mM.
            time_points: Time points (minutes) corresponding to measurements.
            initial_guess: Optional initial parameter values [Fp, PS, ve, vp].
                If None, uses default initial guess from typical ranges.

        Returns:
            FitResult containing:
                - parameters: Fitted Fp, PS, ve, and vp values
                - derived parameters: E, Ktrans, kep
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

        # Define objective function with constraint penalty
        def objective(params: npt.NDArray[np.floating]) -> float:
            # Apply volume constraint
            fp, ps, ve, vp = self._apply_volume_constraint(params)
            constrained_params = [fp, ps, ve, vp]

            predicted = self.evaluate(constrained_params, aif, time_points)
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

        # Clip parameters to bounds and apply constraint
        fitted_params = self.clip_params(fitted_params)
        fp, ps, ve, vp = self._apply_volume_constraint(fitted_params)
        fitted_params = [fp, ps, ve, vp]

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
        param_dict.update(derived)  # Add E, Ktrans, kep to parameters

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
        """Compute derived parameters from fitted values.

        Derived parameters:
            E = PS / (Fp + PS): Extraction fraction
            Ktrans = E × Fp: Volume transfer constant (min⁻¹)
            kep = Ktrans / ve: Efflux rate constant (min⁻¹)

        Args:
            params: Fitted parameter values [Fp, PS, ve, vp].

        Returns:
            Dictionary with 'E', 'Ktrans', and 'kep' values.
        """
        fp = params[0]
        ps = params[1]
        ve = max(params[2], EPSILON_VE)

        # Guard against division by zero
        fp_plus_ps = max(fp + ps, EPSILON_FLOW)

        # Extraction fraction
        e = ps / fp_plus_ps

        # Volume transfer constant (equivalent to Tofts Ktrans)
        ktrans = e * fp

        # Efflux rate constant
        kep = ktrans / ve

        return {"E": e, "Ktrans": ktrans, "kep": kep}
