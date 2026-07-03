"""Control Point Interpolation (CPI) model for DSC perfusion.

Implements a non-parametric model where the residue function is represented
as a PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) spline through
N control points. This approach allows flexible residue shapes while
maintaining physiological constraints.

The CPI model is particularly useful when the underlying transit time
distribution is unknown or doesn't follow a standard parametric form.

The parameterization follows the Quantiphyse convention:
- R(t) is **dimensionless** (0-1), with R(0) = 1.0 fixed
- CBF is a **separate** scaling parameter
- C_tissue(t) = CBF × [A @ R_norm]
"""

from typing import Sequence

import numpy as np
import numpy.typing as npt
from scipy import interpolate  # type: ignore[import-untyped]

from pmriprep.models.bayesian_utils import TransformType
from pmriprep.models.dsc.base import (
    DSCFitResult,
    DSCModel,
    DSCModelParameter,
    ParamsLike,
)
from pmriprep.utils.dsc_constants import (
    CBF_BOUNDS,
    EPSILON_CONCENTRATION,
    VASCULAR_DELAY_BOUNDS,
)


# Default bounds for CPI control points
CPI_N_MIN: int = 4
CPI_N_MAX: int = 24
CPI_DEFAULT_N: int = 6


class CPIModel(DSCModel):
    """Control Point Interpolation model for non-parametric residue estimation.

    The residue function is represented as a PCHIP spline through N control
    points. Control points are evenly spaced in time (or optimally placed),
    and the spline interpolation ensures a smooth residue curve.

    Following FABBER's multiplicative fraction parameterization:
        - R(t) is dimensionless (0-1), with R(0) = 1.0 fixed
        - CBF is a separate scaling parameter (relative units, s⁻¹)
        - C(t) = CBF × [A @ R_norm]

    Monotonicity is guaranteed by construction: each parameter f_i ∈ (0,1)
    is a multiplicative fraction, so R(t_i) = f_1 × f_2 × ... × f_i is
    automatically non-increasing. No external penalty or projection needed.

    At low SNR, the N(0,1) logit prior biases late control points toward
    zero (median R(t_5) = 0.5^5 ≈ 0.03 for N=6). This matches FABBER and
    is directionally correct for tumor perfusion (short MTT).

    Attributes
    ----------
        n_control_points: Number of control points in the spline (4-24).
        n_acquisition_frames: Number of frames in the DSC time series.
        dt: Frame period in seconds.

    Parameters
    ----------
        - cbf: CBF scaling parameter (relative units, s⁻¹)
        - f1, f2, ..., f(N-1): Multiplicative fractions (0-1).
          R(t_i) = f_1 × f_2 × ... × f_i (cumulative product).

    Examples
    --------
        For N=6 control points:
        - Parameters: [cbf, f1, f2, f3, f4, f5]
        - cbf = flow scaling (max of unnormalized residue)
        - R(0) = 1.0 (fixed, not a parameter)
        - R(t_1) = f1, R(t_2) = f1*f2, ..., R(t_5) = f1*f2*f3*f4*f5
        - Control point times: uniformly spaced from 0 to t_max
    """

    def __init__(
        self,
        n_control_points: int = CPI_DEFAULT_N,
        n_acquisition_frames: int | None = None,
        dt: float | None = None,
        infer_cpt: bool = False,
        infer_delay: bool = False,
    ) -> None:
        """Initialize the CPI model.

        Control point times follow FABBER's convention:
        ``tpos[i] = (i+1) * n_acquisition_frames / num_cps`` (frame-uniform).
        Terminal zero anchor sits one frame beyond the data:
        ``t_terminal = (n_acquisition_frames + 1) * dt``.

        Parameters
        ----------
        n_control_points : int, optional
            Number of control points (4-24, default 6).
            The FABBER implementation used 5; production default is 6
            with ``select_n_control_points`` available for auto-selection.
        n_acquisition_frames : int, optional
            Number of frames in the DSC acquisition
            (length of the time series). Used to lay out CPs uniformly
            across the acquisition.
        dt : float, optional
            Frame period in seconds.
        infer_cpt : bool, optional
            If True, optimize control point positions (log-spacing
            parameterization). Doubles parameter count to 2N-1.
        infer_delay : bool, optional
            If True, fit a per-voxel bolus arrival delay applied as a
            post-hoc shift of the convolved curve (off by default).
        """
        self._n_control_points = max(CPI_N_MIN, min(CPI_N_MAX, n_control_points))
        self._infer_cpt = infer_cpt
        self._infer_delay = bool(infer_delay)

        if n_acquisition_frames is None or dt is None:
            raise ValueError(
                "CPIModel requires both n_acquisition_frames and dt "
                "(FABBER-parity CP layout)."
            )
        if n_acquisition_frames < self._n_control_points:
            raise ValueError(
                f"n_acquisition_frames ({n_acquisition_frames}) must be "
                f">= n_control_points ({self._n_control_points})"
            )

        self._n_acquisition_frames = int(n_acquisition_frames)
        self._dt = float(dt)

        # FABBER tpos = (i+1) * nt / N for i in 0..N-1 (in frame indices)
        i = np.arange(self._n_control_points)
        tpos_frames = (i + 1) * self._n_acquisition_frames / self._n_control_points
        self._control_point_times = tpos_frames.astype(np.float64) * self._dt

        # FABBER terminal anchor: one acquisition frame beyond the data
        self._terminal_time = (self._n_acquisition_frames + 1) * self._dt

        # Cache for the convolution matrix (built on first use)
        self._conv_cache: npt.NDArray[np.floating] | None = None
        self._conv_aif_id: int | None = None


    @property
    def name(self) -> str:
        """Model name for identification."""
        return f"cpi_{self._n_control_points}"

    @property
    def n_parameters(self) -> int:
        """Number of free parameters: CBF + N fractions [+ N log-spacings] [+ delay].

        FABBER convention: each CP has its own free amplitude fraction. The
        boundary condition R(0)=1 is enforced by prepending a (t=0, amp=1)
        anchor at PCHIP-construction time, not by forcing the first
        amplitude parameter. The optional bolus delay (infer_delay) appends one
        trailing parameter.
        """
        n = 1 + self._n_control_points
        if self._infer_cpt:
            n += self._n_control_points
        if self._infer_delay:
            n += 1
        return n

    @property
    def infer_cpt(self) -> bool:
        """Whether control point times are optimized."""
        return self._infer_cpt

    @property
    def infer_delay(self) -> bool:
        """Whether a per-voxel bolus arrival delay is fitted."""
        return self._infer_delay

    @property
    def terminal_time(self) -> float:
        """Terminal zero-anchor time (seconds)."""
        return self._terminal_time

    @property
    def n_control_points(self) -> int:
        """Number of control points in the spline."""
        return self._n_control_points

    @property
    def control_point_times(self) -> npt.NDArray[np.floating]:
        """Times at which control points are placed (seconds)."""
        return self._control_point_times

    @property
    def n_acquisition_frames(self) -> int:
        """Number of acquisition frames the CP grid was built against."""
        return self._n_acquisition_frames

    @property
    def dt(self) -> float:
        """Frame period in seconds."""
        return self._dt

    def get_parameter_definitions(self) -> list[DSCModelParameter]:
        """Get parameter definitions for CPI model (FABBER-parity priors).

        Priors mirror ``fabber_models_dsc/fwdmodel_dsc_cpi.cc``:

        - CBF: identity transform, mean=1.0, var=1e6 (essentially flat).
        - Fractions f_i: SIGMOID transform (FABBER ``TRANSFORM_FRACTIONAL``),
          fitting-space prior N(0.5, sqrt(1e7)).
        - Log-spacings (infer_cpt): identity, mean=log(uniform_spacing),
          var=10 (FABBER ``cpt_i`` variance, units anchored to seconds).
        - Delay (infer_delay): identity, mean=0, var=25 (FABBER ``inferdelay``).

        Returns
        -------
        list[DSCModelParameter]
            List of DSCModelParameter objects.
        """
        params = []

        # CBF: identity-space N(1.0, sqrt(1e6)) — FABBER non-informative.
        cbf_sigma = float(np.sqrt(1e6))
        params.append(
            DSCModelParameter(
                name="cbf",
                bounds=(0.0, CBF_BOUNDS.max),
                typical_range=(1.0, 1.0),  # mean=1.0
                transform=TransformType.IDENTITY,
                transformed_prior_sigma=cbf_sigma,
                description="CBF scaling (FABBER identity-space)",
                units="s⁻¹",
            )
        )

        # Fractions: SIGMOID, fitting-space N(0.5, sqrt(1e7)) per FABBER.
        # natural-space prior mean = sigmoid(-0.5) ≈ 0.378.
        # One fraction per CP — first CP amplitude no longer forced to 1.0;
        # the (t=0, amp=1) anchor is enforced via PCHIP knot prepend.
        amp_nat_mean = float(1.0 / (1.0 + np.exp(0.5)))
        amp_fit_sigma = float(np.sqrt(1e7))
        for i in range(self._n_control_points):
            t = self._control_point_times[i]
            params.append(
                DSCModelParameter(
                    name=f"f{i + 1}",
                    bounds=(0.0, 1.0),
                    typical_range=(amp_nat_mean, amp_nat_mean),
                    transform=TransformType.SIGMOID,
                    transformed_prior_sigma=amp_fit_sigma,
                    description=f"Multiplicative fraction at t={t:.1f}s",
                    units="fraction",
                )
            )

        if self._infer_cpt:
            uniform_spacing = float(
                self._control_point_times[1] - self._control_point_times[0]
            ) if self._n_control_points >= 2 else 1.0
            log_spacing = float(np.log(max(uniform_spacing, 1e-6)))
            spacing_sigma = float(np.sqrt(10.0))
            for i in range(self._n_control_points):
                params.append(
                    DSCModelParameter(
                        name=f"s{i + 1}",
                        bounds=(-5.0, 5.0),
                        typical_range=(log_spacing, log_spacing),
                        transform=TransformType.IDENTITY,
                        transformed_prior_sigma=spacing_sigma,
                        description=f"Log-spacing for CP {i + 1}",
                        units="log(s)",
                    )
                )

        if self._infer_delay:
            params.append(
                DSCModelParameter(
                    name="delay",
                    bounds=(VASCULAR_DELAY_BOUNDS.min, VASCULAR_DELAY_BOUNDS.max),
                    typical_range=(0.0, 0.0),  # mean=0
                    transform=TransformType.IDENTITY,
                    transformed_prior_sigma=float(np.sqrt(25.0)),
                    description="Bolus arrival delay (FABBER inferdelay)",
                    units="s",
                )
            )

        return params

    def get_initial_guess(self) -> list[float]:
        """Get FABBER-parity initial guess in NATURAL space.

        Mirrors FABBER's ``GetInitialPosterior`` for ``dsc_cpi`` — no
        data-driven warm-start.

        Returns
        -------
        list[float]
            List: [cbf=1.0, f1=0.378, ..., fN=0.378] (+ N log-spacings
            anchored at log(uniform_spacing) when infer_cpt is True).
        """
        cbf_init = 1.0
        amp_init = float(1.0 / (1.0 + np.exp(0.5)))  # ≈0.378

        x0: list[float] = [cbf_init]
        for _i in range(self._n_control_points):
            x0.append(amp_init)

        if self._infer_cpt:
            uniform_spacing = float(
                self._control_point_times[1] - self._control_point_times[0]
            ) if self._n_control_points >= 2 else 1.0
            log_spacing = float(np.log(max(uniform_spacing, 1e-6)))
            for _i in range(self._n_control_points):
                x0.append(log_spacing)

        if self._infer_delay:
            x0.append(0.0)

        return x0

    def get_parameter_names(self) -> list[str]:
        """Get list of parameter names.

        Returns
        -------
        list[str]
            List: ['cbf', 'f1', ..., 'fN'] (+ 's1'...'sN' when infer_cpt,
            + 'delay' when infer_delay).
        """
        names = ["cbf"] + [f"f{i + 1}" for i in range(self._n_control_points)]
        if self._infer_cpt:
            names += [f"s{i + 1}" for i in range(self._n_control_points)]
        if self._infer_delay:
            names.append("delay")
        return names

    def enforce_constraints(
        self, params: ParamsLike
    ) -> npt.NDArray[np.floating]:
        """Convert multiplicative fractions to dimensionless R(t) values.

        Each f_i is clipped to [0,1] and the cumulative product gives
        R(t_i) = f_1 × f_2 × ... × f_i (FABBER convention), which is
        monotonically non-increasing by construction. The (t=0, amp=1)
        anchor is added downstream by ``residue_function`` / the batched
        forward, so the first-CP amplitude is a free parameter rather
        than a forced 1.0.

        Parameters
        ----------
        params : ParamsLike
            [cbf, f1, f2, ..., fN] where cbf is the scaling
            parameter and f_i are multiplicative fractions.

        Returns
        -------
        np.ndarray
            Dimensionless control point values (cumprod of fractions),
            length = n_control_points.
        """
        n = self._n_control_points
        fracs = np.array(params[1 : 1 + n], dtype=np.float64)
        fracs = np.clip(fracs, 0.0, 1.0)
        control_values = np.cumprod(fracs)
        return control_values

    def _compute_cp_times_from_params(
        self, params: ParamsLike
    ) -> npt.NDArray[np.floating]:
        """Compute control point times, optionally from log-spacing params.

        Layout under ``infer_cpt``: params = [cbf, f_1..f_N, s_1..s_N].
        ``s_i`` is the log-spacing between the (i-1)-th anchor and the
        i-th CP, so cp_times[i] = sum_{k<=i} exp(s_k).

        Parameters
        ----------
        params : ParamsLike
            Full parameter vector.

        Returns
        -------
        np.ndarray
            Control point times array, shape (N,).
        """
        if not self._infer_cpt:
            return self._control_point_times

        n = self._n_control_points
        log_spacings = np.array(
            [float(params[1 + n + i]) for i in range(n)], dtype=np.float64
        )
        spacings = np.exp(log_spacings)
        cp_times: npt.NDArray[np.floating] = np.cumsum(spacings).astype(np.float64)
        return cp_times

    def residue_function(
        self,
        params: ParamsLike,
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute the dimensionless residue function R(t) using PCHIP.

        Uses the FABBER terminal anchor at ``(n_acquisition_frames+1)*dt``
        when ``infer_cpt`` is False; under ``infer_cpt`` the terminal sits
        one inferred-spacing past the last CP.

        Parameters
        ----------
        params : ParamsLike
            Parameter values [cbf, f1, ..., f(N-1), [s1, ..., s(N-1)]].
        time_points : np.ndarray
            Time points (seconds) at which to evaluate R(t).

        Returns
        -------
        np.ndarray
            Dimensionless residue function values R(t) at each time point.
        """
        control_values = self.enforce_constraints(params)
        cp_times = self._compute_cp_times_from_params(params)

        # Append terminal zero-anchor (FABBER: t = (nt+1)*dt for fixed CPs)
        if self._infer_cpt:
            last_spacing = float(cp_times[-1] - cp_times[-2]) if len(cp_times) >= 2 else 1.0
            t_terminal = float(cp_times[-1]) + last_spacing
        else:
            t_terminal = self._terminal_time

        # FABBER BC: prepend (t=0, amp=1) anchor when first CP is at t > 0
        if float(cp_times[0]) > 0.0:
            cp_times_ext = np.concatenate([[0.0], cp_times, [t_terminal]])
            cp_vals_ext = np.concatenate([[1.0], control_values, [0.0]])
        else:
            cp_times_ext = np.append(cp_times, t_terminal)
            cp_vals_ext = np.append(control_values, 0.0)

        try:
            pchip = interpolate.PchipInterpolator(
                cp_times_ext,
                cp_vals_ext,
                extrapolate=True,
            )
            residue: npt.NDArray[np.floating] = np.asarray(
                pchip(time_points), dtype=np.float64
            )
        except ValueError:
            residue = np.interp(time_points, cp_times_ext, cp_vals_ext)

        residue = np.clip(residue, 0.0, 1.0)
        return residue


    def compute_fitted_residue(
        self,
        params: ParamsLike,
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute CBF-scaled residue for Bayesian fitted_residue output.

        CPI's residue_function() returns dimensionless R(t) in [0,1],
        but downstream code expects fitted_residue = CBF × R(t) so that
        CBF = max(residue) and CBV = trapezoid(residue, dx=dt).

        Parameters
        ----------
        params : ParamsLike
            Parameter values [cbf, f1, ..., f(N-1)].
        time_points : np.ndarray
            Time points (seconds).

        Returns
        -------
        np.ndarray
            CBF-scaled residue function (CBF × R_norm).
        """
        cbf = max(float(params[0]), 0.0)
        return cbf * self.residue_function(params, time_points)

    def tissue_curve(
        self,
        params: ParamsLike,
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute tissue concentration curve by convolving AIF with residue.

        C_tissue(t) = CBF × [A @ R_norm]

        Uses the Volterra (1, 4, 1)/6 convolution matrix.

        Parameters
        ----------
        params : ParamsLike
            Parameter values [cbf, f1, ..., f(N-1)].
        aif : np.ndarray
            Arterial input function concentration values.
        time_points : np.ndarray
            Time points (seconds) at which to evaluate C(t).

        Returns
        -------
        np.ndarray
            Tissue concentration values C(t) at each time point.
        """
        n_times = len(time_points)
        if len(aif) != n_times:
            raise ValueError(
                f"AIF length ({len(aif)}) must match time_points ({n_times})"
            )

        if n_times < 2:
            return np.zeros(n_times, dtype=np.float64)

        # Extract CBF scaling
        cbf = max(float(params[0]), 0.0)

        # Compute dimensionless residue function R(t) with R(0)=1.0
        residue = self.residue_function(params, time_points)

        # Build or reuse the convolution matrix (includes dt in its elements)
        aif_id = id(aif)
        if self._conv_cache is None or self._conv_aif_id != aif_id:
            from pmriprep.utils.dsc_deconvolution import build_volterra_matrix
            dt = float(time_points[1] - time_points[0])
            self._conv_cache = build_volterra_matrix(aif, dt)
            self._conv_aif_id = aif_id

        # C(t) = CBF × (A @ R_norm)
        # matrix already includes dt, so no additional multiplication
        tissue = cbf * (self._conv_cache @ residue)

        # Optional bolus delay: shift the convolved curve by delay via linear
        # interpolation. For a linear convolution, shifting the result equals
        # shifting the AIF before convolving, so the precomputed matrix is
        # reused. The residue (hence CBV and MTT) is unaffected by delay.
        if self._infer_delay:
            delay = float(params[-1])
            dt = float(time_points[1] - time_points[0])
            delay_frac = delay / dt
            n_shift = int(np.floor(delay_frac))
            f_shift = delay_frac - n_shift
            src = np.arange(n_times) - n_shift
            shifted = np.zeros(n_times, dtype=np.float64)
            lo = (src >= 0) & (src < n_times)
            shifted[lo] = (1.0 - f_shift) * tissue[src[lo]]
            hi_src = src - 1
            hi = lo & (hi_src >= 0) & (hi_src < n_times)
            shifted[hi] += f_shift * tissue[hi_src[hi]]
            tissue = shifted

        return tissue

    def get_cbf(self, params: ParamsLike) -> float:
        """Get CBF from the scaling parameter.

        Parameters
        ----------
        params : ParamsLike
            Parameter values [cbf, f1, ..., f(N-1)].

        Returns
        -------
        float
            CBF value (relative, s⁻¹).
        """
        return float(params[0]) if len(params) > 0 else 0.0

    def get_cbv(
        self,
        params: ParamsLike,
        time_points: npt.NDArray[np.floating] | None = None,
    ) -> float:
        """Compute CBV by integrating the unnormalized residue function.

        CBV = ∫₀^∞ CBF × R_norm(t) dt = CBF × ∫R_norm(t) dt

        Since CBF is in relative s⁻¹ units, CBV = CBF × ∫R_norm dt
        has units of (dimensionless), matching the SVD path where
        CBV = trapezoid(residue, dx=dt).

        Parameters
        ----------
        params : ParamsLike
            Parameter values [cbf, f1, ..., f(N-1)].
        time_points : np.ndarray, optional
            Optional time points for integration.
            If None, uses control point times extended.

        Returns
        -------
        float
            CBV value (relative, dimensionless).
        """
        if time_points is None:
            time_points = np.linspace(
                0.0,
                self._terminal_time,
                200,
            )

        cbf = self.get_cbf(params)
        residue_norm = self.residue_function(params, time_points)
        dt = float(time_points[1] - time_points[0])

        cbv = cbf * float(np.trapezoid(residue_norm, dx=dt))

        return cbv

    def get_mtt(
        self,
        params: ParamsLike,
        time_points: npt.NDArray[np.floating] | None = None,
    ) -> float:
        """Compute MTT from central volume theorem.

        MTT = CBV / CBF = ∫R_norm(t) dt  (seconds)

        Parameters
        ----------
        params : ParamsLike
            Parameter values [cbf, f1, ..., f(N-1)].
        time_points : np.ndarray, optional
            Optional time points for CBV integration.

        Returns
        -------
        float
            MTT value (seconds).
        """
        cbf = self.get_cbf(params)
        if cbf < EPSILON_CONCENTRATION:
            return 0.0

        if time_points is None:
            time_points = np.linspace(
                0.0,
                self._terminal_time,
                200,
            )

        residue_norm = self.residue_function(params, time_points)
        dt = float(time_points[1] - time_points[0])

        # MTT = CBV / CBF = ∫R_norm dt (seconds, no /60 conversion)
        mtt = float(np.trapezoid(residue_norm, dx=dt))

        return mtt


_INFER_CPT_DECAY_DIVISOR_SEC: float = 3.0
"""Empirically-calibrated divisor for the tail-decay → N rule under
``infer_cpt=True``. Calibration anchor:
``tests/benchmarks/run_cpi_n_sweep_infer_cpt.py`` showed a sharp cliff at
N=7→8 for Long MTT=12s tissue (CBF bias +9.5%→+0.6%); divisor=3.0 gives
N=8 for the gamma-variate DRO mean tissue (decay≈22.5s), landing exactly
at the cliff with no headroom.

This is intentional: the cliff is sharp, so N=8 is genuinely sufficient
(not borderline). Adding margin (e.g. divisor=2.5 → N=9 on the DRO)
would erase the variance gain on short-MTT data without measurable
accuracy improvement. The defense for unusual cohorts is the adaptive
*scaling*: stroke-like data with longer tail decay scales N upward
naturally (τ=20s → N=15), so users with pathologically long MTT are
covered automatically.

If real clinical data shows the cliff position is noise- or
SNR-dependent, recalibrate via the N-sweep benchmark before changing
this constant."""


def select_n_control_points(
    bolus_duration_frames: int,
    data_duration_sec: float = 30.0,
    target_spacing_sec: float = 10.0,
    min_n: int = CPI_N_MIN,
    max_n: int = CPI_N_MAX,
    *,
    infer_cpt: bool = False,
    mean_tissue: npt.NDArray[np.floating] | None = None,
    dt: float | None = None,
) -> int:
    """Automatically select number of control points.

    Two distinct paths based on ``infer_cpt``:

    - **Fixed-CP layout (``infer_cpt=False``)**: legacy heuristic
      ``max(n_bolus, n_spacing)`` — sufficient bolus density for
      uniformly-placed CPs.
    - **Migrating CPs (``infer_cpt=True``) with ``mean_tissue`` + ``dt``**:
      pure tail-decay-driven N. ``n_bolus`` is dropped here because CPs
      migrate to optimal locations and don't need a-priori dense bolus
      sampling. The empirical N-sweep shows N=4 fits short-MTT tissue
      perfectly under migration.

    If ``infer_cpt=True`` but ``mean_tissue`` or ``dt`` are missing, falls
    back to the legacy formula (callers without mean_tissue can't use the
    adaptive path).

    Parameters
    ----------
    bolus_duration_frames : int
        Duration of bolus passage in frames.
    data_duration_sec : float, optional
        Total data duration in seconds.
    target_spacing_sec : float, optional
        Target CP spacing in seconds.
    min_n : int, optional
        Minimum number of control points.
    max_n : int, optional
        Maximum number of control points.
    infer_cpt : bool, optional
        When True and ``mean_tissue``/``dt`` are provided, use
        the tail-decay-adaptive formula instead of the legacy
        heuristic.
    mean_tissue : np.ndarray, optional
        Mean tissue concentration across in-mask voxels,
        shape (T,). Required for the adaptive path.
    dt : float, optional
        Frame period in seconds. Required for the adaptive path.

    Returns
    -------
    int
        Recommended number of control points, clamped to ``[min_n, max_n]``.
    """
    if infer_cpt and mean_tissue is not None and dt is not None:
        decay_sec = estimate_tail_decay_time_seconds(mean_tissue, float(dt))
        n_recommended = int(np.ceil(decay_sec / _INFER_CPT_DECAY_DIVISOR_SEC))
    else:
        n_bolus = int(bolus_duration_frames / 2.0) + 2
        n_spacing = int(data_duration_sec / target_spacing_sec) + 1
        n_recommended = max(n_bolus, n_spacing)

    return max(min_n, min(max_n, n_recommended))


def estimate_bolus_duration(
    concentration: npt.NDArray[np.floating],
    threshold_fraction: float = 0.1,
) -> int:
    """Estimate bolus passage duration from concentration curve.

    Finds the number of frames where concentration exceeds a threshold
    fraction of the peak value.

    Parameters
    ----------
    concentration : np.ndarray
        Concentration time course.
    threshold_fraction : float, optional
        Fraction of peak defining bolus region (default 0.1).

    Returns
    -------
    int
        Estimated bolus duration in frames.
    """
    if len(concentration) == 0:
        return 10  # Default fallback

    peak = float(np.max(concentration))
    if peak < EPSILON_CONCENTRATION:
        return 10  # Default fallback

    threshold = peak * threshold_fraction

    # Find frames above threshold
    above_threshold = concentration > threshold
    n_frames_above = int(np.sum(above_threshold))

    # Return at least 5 frames
    return max(5, n_frames_above)


def estimate_tail_decay_time_seconds(
    mean_tissue: npt.NDArray[np.floating],
    dt: float,
    threshold_fraction: float = 0.1,
    fallback_sec: float = 30.0,
    max_decay_fraction: float = 0.5,
) -> float:
    """Time from post-peak frame to first frame at <``threshold_fraction`` of peak.

    Returns a property of the mean tissue concentration curve. For
    exponential residues this scales roughly with mean MTT, but it
    conflates AIF tail and tissue MTT, so callers treat it as an opaque
    proxy and apply an empirically-calibrated divisor to derive an N
    floor. See :func:`select_n_control_points` for the calibration.

    Parameters
    ----------
    mean_tissue : np.ndarray
        Mean tissue concentration across in-mask voxels,
        shape (T,). Pre-computed by the caller.
    dt : float
        Frame period in seconds.
    threshold_fraction : float, optional
        Fraction of peak defining the tail end.
        Matches ``estimate_bolus_duration``'s convention (default 0.1).
    fallback_sec : float, optional
        Returned when peak detection fails (zero curve,
        peak at last frame, etc.).
    max_decay_fraction : float, optional
        Cap returned decay time at
        ``max_decay_fraction × T·dt``. Prevents runaway N on
        preloaded studies where the tail never drops below threshold.

    Returns
    -------
    float
        Decay time in seconds, in [0, max_decay_fraction × T·dt].
    """
    n_frames = len(mean_tissue)
    if n_frames < 2:
        return fallback_sec
    if not np.all(np.isfinite(mean_tissue)):
        return fallback_sec
    peak_frame = int(np.argmax(mean_tissue))
    peak_val = float(mean_tissue[peak_frame])
    if peak_val < EPSILON_CONCENTRATION:
        return fallback_sec
    if peak_frame >= n_frames - 1:
        return fallback_sec
    threshold = peak_val * threshold_fraction
    post_peak = mean_tissue[peak_frame:]
    below = np.where(post_peak < threshold)[0]
    if len(below) == 0:
        decay_frames = len(post_peak)
    else:
        decay_frames = int(below[0])
    decay_sec = float(decay_frames * dt)
    cap_sec = float(max_decay_fraction * n_frames * dt)
    return min(decay_sec, cap_sec)


def select_n_by_bic(
    concentration: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    n_candidates: Sequence[int] = (4, 5, 6, 7, 8, 9, 10),
    noise_std: float | None = None,
) -> tuple[int, list[tuple[int, float, float]]]:
    """Select optimal number of control points using BIC.

    Bayesian Information Criterion balances goodness-of-fit against model
    complexity. Lower BIC indicates a better model.

    BIC = n × ln(RSS/n) + k × ln(n)

    where:
        n = number of observations (time points)
        RSS = residual sum of squares
        k = number of parameters (control points)

    Parameters
    ----------
    concentration : np.ndarray
        Measured tissue concentration curve.
    aif : np.ndarray
        Arterial input function.
    time_points : np.ndarray
        Time points (seconds).
    n_candidates : Sequence[int], optional
        Candidate numbers of control points to evaluate.
    noise_std : float, optional
        Optional noise standard deviation for weighting.

    Returns
    -------
    tuple[int, list[tuple[int, float, float]]]
        Tuple of:
            - Optimal number of control points
            - List of (n, BIC, RSS) for each candidate
    """
    n_obs = len(concentration)
    results: list[tuple[int, float, float]] = []

    dt_local = float(time_points[1] - time_points[0]) if len(time_points) >= 2 else 1.0
    for n in n_candidates:
        if n < CPI_N_MIN or n > CPI_N_MAX:
            continue

        # Create model with this number of control points
        model = CPIModel(
            n_control_points=n,
            n_acquisition_frames=len(time_points),
            dt=dt_local,
        )

        # Fit model (simple least squares for speed)
        fit_result = _fit_cpi_simple(model, concentration, aif, time_points)

        if fit_result is None:
            continue

        # Calculate BIC
        rss = fit_result.residual_sum_squares
        k = n  # Number of parameters

        if rss > 0:
            bic = n_obs * np.log(rss / n_obs) + k * np.log(n_obs)
        else:
            bic = float("-inf")  # Perfect fit (unlikely)

        results.append((n, float(bic), float(rss)))

    if not results:
        # Fallback to default
        return CPI_DEFAULT_N, []

    # Find minimum BIC
    best_n = min(results, key=lambda x: x[1])[0]

    return best_n, results


def _fit_cpi_simple(
    model: CPIModel,
    concentration: npt.NDArray[np.floating],
    aif: npt.NDArray[np.floating],
    time_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating] | None = None,
    sloppy: bool = False,
) -> DSCFitResult | None:
    """FABBER-parity NLLS fit of the CPI model (priors-off least squares).

    Mirrors FABBER's ``inference_nlls.cc`` for ``--model=dsc_cpi``:
    Levenberg-Marquardt-style trust-region reflective optimisation in
    *fitting space* (sigmoid-transformed fractions, identity CBF). The
    sigmoid transform implicitly bounds fractions to (0, 1), so no box
    constraints are imposed beyond the identity-CBF ≥ 0 floor.

    Parameter covariance is reported via ``(JᵀJ + 1e-6·I)⁻¹ · MSE``
    using the Jacobian returned by ``scipy.optimize.least_squares``.
    Natural-space std on CBF and CBV is recovered via the delta method.

    Parameters
    ----------
    model : CPIModel
        CPIModel instance.
    concentration : np.ndarray
        Measured tissue concentration.
    aif : np.ndarray
        Arterial input function.
    time_points : np.ndarray
        Time points (seconds).
    weights : np.ndarray, optional
        Optional per-frame motion weights (1.0 = full weight).
    sloppy : bool, optional
        Use fewer iterations for benchmarks/tests.

    Returns
    -------
    DSCFitResult or None
        DSCFitResult with fitted parameters in NATURAL space, or None on
        failure. ``fitted_residue`` contains CBF × R_norm so downstream
        code can extract CBF = max(residue) and CBV = trapezoid(residue, dx=dt).
    """
    from scipy import optimize  # type: ignore[import-untyped]

    from pmriprep.models.bayesian_utils import (
        TransformType,
        from_transformed,
        to_transformed,
    )

    fitting_defs = list(model.get_fitting_parameter_definitions())
    n_params = len(fitting_defs)
    transforms = [d.transform for d in fitting_defs]
    upper_bounds: list[float | None] = [
        d.bounds[1] if d.transform == TransformType.LOG_LOGIT else None
        for d in fitting_defs
    ]
    n_obs = len(concentration)

    if n_obs <= n_params:
        return None

    # Initial guess in NATURAL space → fitting space
    x0_nat = np.asarray(model.get_initial_guess(), dtype=np.float64)
    x0_fit = np.array([
        float(to_transformed(x0_nat[j], transforms[j], upper_bounds[j]))
        for j in range(n_params)
    ], dtype=np.float64)

    sqrt_w = (
        np.sqrt(np.asarray(weights, dtype=np.float64))
        if weights is not None
        else np.ones(n_obs, dtype=np.float64)
    )

    def _to_natural(x_fit: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        x_nat = np.empty(n_params, dtype=np.float64)
        for j in range(n_params):
            x_nat[j] = float(
                from_transformed(float(x_fit[j]), transforms[j], upper_bounds[j])
            )
        return x_nat

    def _residual(x_fit: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        x_nat = _to_natural(x_fit)
        predicted = model.tissue_curve(x_nat, aif, time_points)
        return sqrt_w * (concentration - predicted)

    try:
        max_nfev = 200 if sloppy else 2000
        ls_result = optimize.least_squares(
            _residual,
            x0_fit,
            method="trf",
            max_nfev=max_nfev,
            ftol=1e-8,
            xtol=1e-8,
        )
    except (ValueError, RuntimeError):
        return None

    opt_fit = np.array(ls_result.x, dtype=np.float64)
    opt_nat = _to_natural(opt_fit)
    opt_nat[0] = max(opt_nat[0], 0.0)  # CBF ≥ 0

    fitted_tissue = model.tissue_curve(opt_nat, aif, time_points)
    residue_norm = model.residue_function(opt_nat, time_points)
    fitted_residue = float(opt_nat[0]) * residue_norm

    residuals = concentration - fitted_tissue
    ss_res = float(np.sum((sqrt_w * residuals) ** 2))
    ss_tot = float(np.sum((concentration - np.mean(concentration)) ** 2))
    r_squared = 1.0 - (ss_res / max(ss_tot, EPSILON_CONCENTRATION))

    # Parameter covariance in fitting space: (J^T J + 1e-6 I)^-1 · MSE
    # FABBER inference_nlls.cc adds a 1e-6 diagonal floor for stability.
    sidecar_std: dict[str, float] = {}
    try:
        J = np.asarray(ls_result.jac, dtype=np.float64)
        JtJ = J.T @ J + 1e-6 * np.eye(n_params)
        cov_fit = np.linalg.inv(JtJ) * (ss_res / max(n_obs - n_params, 1))
        # Delta-method natural-space sigma per parameter
        for j in range(n_params):
            tf = transforms[j]
            sigma_fit_j = float(np.sqrt(max(cov_fit[j, j], 0.0)))
            if tf == TransformType.LOG:
                # |dnat/dfit| = exp(fit) = nat
                jac = float(opt_nat[j])
            elif tf == TransformType.SIGMOID:
                nat = float(opt_nat[j])
                jac = nat * (1.0 - nat)  # |dnat/dfit| at the optimum
            elif tf == TransformType.LOG_LOGIT:
                ub = float(upper_bounds[j] or 1.0)
                nat = float(opt_nat[j])
                jac = (nat * (ub - nat)) / max(ub, 1e-30)
            else:  # IDENTITY
                jac = 1.0
            sidecar_std[fitting_defs[j].name] = jac * sigma_fit_j
    except np.linalg.LinAlgError:
        pass

    return DSCFitResult(
        parameters=model.params_to_dict(opt_nat),
        uncertainties=sidecar_std,
        r_squared=r_squared,
        residual_sum_squares=ss_res,
        model_name=model.name,
        n_parameters=n_params,
        n_observations=n_obs,
        converged=bool(ls_result.success),
        fitting_method="nlls",
        fitted_residue=fitted_residue,
        fitted_tissue_curve=fitted_tissue,
    )
