"""Continuous DSC reference curves independent of fitted forward operators.

Times are in seconds. Residues have unit initial amplitude; multiplying the
convolution by flow introduces no implicit unit conversion. These helpers
generate point samples, not averages over a repetition interval. Signal
formation, acquisition noise and leakage are separate operations.
"""

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.signal import fftconvolve
from scipy.special import gammaincc
from scipy.stats import gamma as gamma_distribution
from scipy.stats import lognorm

Curve = Callable[[npt.ArrayLike], npt.NDArray[np.float64]]
ResidueFamily = Literal["exponential", "gamma", "box", "triangle"]


def _positive(value: float, name: str) -> None:
    """Require a finite positive scalar."""
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _integrate(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    breakpoints: Sequence[float],
    epsabs: float,
    epsrel: float,
) -> tuple[float, float]:
    """Integrate known smooth segments, rejecting numerical failures."""
    points = sorted({float(p) for p in breakpoints if lower < p < upper})
    result = quad(
        function, lower, upper, points=points, epsabs=epsabs, epsrel=epsrel,
        limit=max(100, len(points) + 2), full_output=1,
    )
    if len(result) != 3:
        raise RuntimeError(f"Reference quadrature failed: {result[3]}")
    value, error = float(result[0]), float(result[1])
    if not np.isfinite(value) or not np.isfinite(error):
        raise RuntimeError("Reference quadrature returned non-finite values")
    return value, error


@dataclass(frozen=True)
class ContinuousResidue:
    """A causal unit-height residue and its full integral.

    Parameters
    ----------
    family : str
        Shape label, not a claim of in-vivo prevalence.
    mtt_seconds : float
        Integral over the full residue support, independent of scan length.
    evaluate : callable
        Evaluate the residue at scalar or array-valued lag times in seconds.
    breakpoints : tuple of float
        Nonnegative lag times at which integration segments should be split.
    """

    family: str
    mtt_seconds: float
    evaluate: Curve
    breakpoints: tuple[float, ...] = ()

    def area_until(self, duration_seconds: float) -> float:
        """Return a scan-limited integral without redefining full MTT.

        Parameters
        ----------
        duration_seconds : float
            Nonnegative residue-lag horizon, not absolute scan end time.

        Returns
        -------
        float
            Integral from zero to the horizon, in seconds.
        """
        if not np.isfinite(duration_seconds) or duration_seconds < 0:
            raise ValueError("duration_seconds must be finite and nonnegative")
        if duration_seconds == 0:
            return 0.0
        value, _ = _integrate(
            lambda t: float(self.evaluate(t)), 0.0, duration_seconds,
            self.breakpoints, 1e-10, 1e-10,
        )
        return value


@dataclass(frozen=True)
class ContinuousTissue:
    """Point-sampled tissue and numerical quadrature error estimates.

    Parameters
    ----------
    values : ndarray
        Tissue values in the units of AIF multiplied by flow and seconds.
    quadrature_error : ndarray
        Absolute integration error estimates in the same units. These do not
        quantify acquisition noise, interpolation error or model uncertainty.
    """

    values: npt.NDArray[np.float64]
    quadrature_error: npt.NDArray[np.float64]


@dataclass(frozen=True)
class ContinuousTransport:
    """Unit-area transport density normalized over infinite support.

    Parameters
    ----------
    family : str
        Distribution name, separate from the tissue residue family.
    mean_seconds : float
        Mean transport time in seconds.
    density, cumulative, survival : callable
        Density, cumulative probability and remaining mass at lag times.
        No finite-window renormalization is applied.
    """

    family: str
    mean_seconds: float
    density: Curve
    cumulative: Curve
    survival: Curve


@dataclass(frozen=True)
class DispersedTissue:
    """Verified dispersed tissue and separately labelled response diagnostics.

    Parameters
    ----------
    values : ndarray
        Tissue at the requested point-sample times.
    response_lag_seconds, effective_response : ndarray
        Fine lag grid and flow-scaled dispersed response on that grid.
    intrinsic_flow, intrinsic_mtt_seconds : float
        Primary flow and full tissue MTT truths, unchanged by transport.
    effective_peak : float
        Grid-estimated response peak, not the physiological flow truth.
    step_seconds, refinement_error : float
        Accepted numerical grid step and maximum relative refinement check.
    kernel_mass_in_scan, residue_area_in_scan_seconds : float
        Captured mass/area over the maximum post-arrival lag available in scan.
    """

    values: npt.NDArray[np.float64]
    response_lag_seconds: npt.NDArray[np.float64]
    effective_response: npt.NDArray[np.float64]
    intrinsic_flow: float
    intrinsic_mtt_seconds: float
    effective_peak: float
    step_seconds: float
    refinement_error: float
    kernel_mass_in_scan: float
    residue_area_in_scan_seconds: float


def transport_kernel(
    family: Literal["gamma", "lognormal"],
    mean_seconds: float,
    *,
    gamma_shape: float = 2.0,
    log_sd: float = 0.75,
) -> ContinuousTransport:
    """Construct continuous in-family and out-of-family transport controls.

    Parameters
    ----------
    family : {"gamma", "lognormal"}
        Gamma transport is a model-family control; lognormal is a different
        kernel family. Both are independent of fitted model implementations.
    mean_seconds : float
        Positive mean delay introduced by transport, in addition to arrival shift.
    gamma_shape : float
        Shape at least one, giving a finite density at zero on a numerical grid.
    log_sd : float
        Positive standard deviation of log transport time for lognormal kernels.

    Returns
    -------
    ContinuousTransport
        Analytically normalized density with its true CDF and survival function.
    """
    _positive(mean_seconds, "mean_seconds")
    if family == "gamma":
        if not np.isfinite(gamma_shape) or gamma_shape < 1:
            raise ValueError("gamma_shape must be finite and at least one")
        distribution = gamma_distribution(a=gamma_shape, scale=mean_seconds / gamma_shape)
    elif family == "lognormal":
        _positive(log_sd, "log_sd")
        distribution = lognorm(s=log_sd, scale=mean_seconds * np.exp(-0.5 * log_sd**2))
    else:
        raise ValueError(f"Unknown transport family: {family}")
    return ContinuousTransport(
        family, mean_seconds,
        lambda t: np.asarray(distribution.pdf(t), dtype=np.float64),
        lambda t: np.asarray(distribution.cdf(t), dtype=np.float64),
        lambda t: np.asarray(distribution.sf(t), dtype=np.float64),
    )


def _causal_trapezoid(
    left: npt.NDArray[np.float64], right: npt.NDArray[np.float64], step: float,
) -> npt.NDArray[np.float64]:
    """Integrate the causal convolution including both half-weight endpoints."""
    result = step * (
        fftconvolve(left, right)[:len(left)] - 0.5 * left[0] * right - 0.5 * right[0] * left
    )
    result[0] = 0.0
    return np.asarray(result, dtype=np.float64)


def dispersed_tissue(
    time_seconds: npt.ArrayLike,
    aif: Curve,
    residue: ContinuousResidue,
    transport: ContinuousTransport,
    *,
    flow: float,
    aif_onset_seconds: float,
    delay_seconds: float = 0.0,
    initial_step_seconds: float = 0.025,
    relative_tolerance: float = 1e-5,
    max_refinements: int = 9,
) -> DispersedTissue:
    """Generate dispersed observations by refined continuous-grid convolution.

    Parameters
    ----------
    time_seconds : array_like
        Strictly increasing finite point-sample times, not necessarily uniform.
    aif : callable
        Continuous measured-site input, zero before aif_onset_seconds.
    residue : ContinuousResidue
        Causal non-increasing intrinsic residue with full MTT known.
    transport : ContinuousTransport
        Unit-area kernel; never normalize it by the captured scan area.
    flow : float
        Nonnegative primary tissue flow in inverse seconds.
    aif_onset_seconds, delay_seconds : float
        Continuous AIF onset and signed tissue arrival offset, in seconds.
    initial_step_seconds : float
        Initial numerical step, halved until all precision checks pass.
    relative_tolerance : float
        Required tissue/peak refinement and absolute kernel-mass accuracy.
    max_refinements : int
        Positive limit on step halvings; failure to meet precision raises.

    Returns
    -------
    DispersedTissue
        Shared tissue samples and diagnostic dispersed response. The peak is
        estimated on the accepted fine grid. A monotonic-residue tail bound
        ensures that an unobserved later response cannot exceed this peak.

    Notes
    -----
    Tissue uses A * D * R, while the measured-site AIF remains A. Both A and R
    are evaluated continuously on each finer grid, never reconstructed from
    acquired frames. Numerical accuracy is checked before any model is fitted.
    """
    time = np.asarray(time_seconds, dtype=np.float64)
    if (time.ndim != 1 or time.size == 0 or not np.all(np.isfinite(time))
            or np.any(np.diff(time) <= 0)):
        raise ValueError("time_seconds must be a finite, increasing nonempty vector")
    if not np.isfinite(flow) or flow < 0:
        raise ValueError("flow must be finite and nonnegative")
    if not np.all(np.isfinite([aif_onset_seconds, delay_seconds])):
        raise ValueError("onset and delay must be finite")
    _positive(initial_step_seconds, "initial_step_seconds")
    _positive(relative_tolerance, "relative_tolerance")
    if (isinstance(max_refinements, bool) or not isinstance(max_refinements, int)
            or max_refinements < 1):
        raise ValueError("max_refinements must be a positive integer")
    sample_lags = time - aif_onset_seconds - delay_seconds
    scan_lag = max(0.0, float(sample_lags[-1]))
    horizon = max(scan_lag, 2 * residue.mtt_seconds, 2 * transport.mean_seconds,
                  2 * initial_step_seconds)
    previous: tuple[npt.NDArray[np.float64], float] | None = None
    for refinement in range(max_refinements + 1):
        step = initial_step_seconds / 2**refinement
        for _ in range(8):
            lag = np.arange(int(np.ceil(horizon / step)) + 1, dtype=np.float64) * step
            density, r = transport.density(lag), residue.evaluate(lag)
            if (not np.all(np.isfinite(density)) or np.any(density < 0)
                    or not np.all(np.isfinite(r)) or np.any(r < 0) or r[0] != 1
                    or np.any(np.diff(r) > 1e-12)):
                raise ValueError("Reference transport/residue must be finite, causal and physical")
            mass_error = abs(float(np.trapezoid(density, lag) - transport.cumulative(lag[-1])))
            if mass_error > relative_tolerance:
                break
            response = _causal_trapezoid(density, r, step)
            peak = float(np.max(response))
            # For t >= H, (D*R)(t) <= R(H/2) + P(D > H/2).
            tail_bound = float(residue.evaluate(lag[-1] / 2) + transport.survival(lag[-1] / 2))
            if peak > 0 and tail_bound < peak * (1 - relative_tolerance):
                break
            horizon *= 2
        else:
            raise RuntimeError("Could not bound the unobserved dispersed-response peak")
        if mass_error > relative_tolerance:
            previous = None
            continue
        arterial = aif(aif_onset_seconds + lag)
        if not np.all(np.isfinite(arterial)) or np.any(arterial < 0):
            raise ValueError("Continuous truth AIF must be finite and nonnegative")
        curve = _causal_trapezoid(arterial, response, step)
        sampled = np.asarray(np.interp(sample_lags, lag, curve, left=0.0), dtype=np.float64)
        sampled[sample_lags <= 0] = 0.0
        if previous is not None:
            scale = max(float(np.max(np.abs(sampled))), np.finfo(np.float64).tiny)
            difference = float(np.max(np.abs(sampled - previous[0]))) / scale
            peak_difference = abs(peak - previous[1]) / peak
            error = max(difference, peak_difference, mass_error)
            if error <= relative_tolerance:
                return DispersedTissue(
                    flow * sampled, lag, flow * response, flow, residue.mtt_seconds,
                    flow * peak, step, error, float(transport.cumulative(scan_lag)),
                    residue.area_until(scan_lag),
                )
        previous = sampled, peak
    raise RuntimeError("Dispersed reference did not meet the requested refinement tolerance")


def gamma_variate_aif(
    time_seconds: npt.ArrayLike,
    *,
    onset_seconds: float,
    peak: float,
    alpha: float,
    beta_seconds: float,
) -> npt.NDArray[np.float64]:
    """Evaluate a causal bolus with a sampling-independent analytic peak.

    Parameters
    ----------
    time_seconds : array_like
        Absolute sample times; scalar evaluation is also supported.
    onset_seconds : float
        Bolus onset, not restricted to a frame boundary.
    peak : float
        Positive continuous peak, in the caller's AIF units.
    alpha : float
        Positive power of time since arrival.
    beta_seconds : float
        Positive decay scale in seconds.

    Returns
    -------
    ndarray
        AIF values, zero at and before onset. The continuous peak occurs at
        onset_seconds + alpha * beta_seconds, whether or not it is sampled.
    """
    _positive(peak, "peak")
    _positive(alpha, "alpha")
    _positive(beta_seconds, "beta_seconds")
    time = np.asarray(time_seconds, dtype=np.float64)
    if not np.isfinite(onset_seconds) or not np.all(np.isfinite(time)):
        raise ValueError("AIF times and onset must be finite")
    u = (time - onset_seconds) / beta_seconds
    safe = np.maximum(u, np.finfo(np.float64).tiny)
    value = peak * np.exp(alpha * (np.log(safe) - np.log(alpha)) + alpha - safe)
    return np.asarray(np.where(u > 0, value, 0.0), dtype=np.float64)


def recirculating_gamma_aif(
    time_seconds: npt.ArrayLike,
    *,
    onset_seconds: float,
    peak: float,
    alpha: float,
    beta_seconds: float,
    recirculation_fraction: float,
    recirculation_delay_seconds: float,
    broadening: float,
) -> npt.NDArray[np.float64]:
    """Evaluate a first bolus plus a delayed, broadened second bolus.

    Parameters
    ----------
    time_seconds : array_like
        Absolute point-sample times, or a scalar integration time.
    onset_seconds : float
        First-pass onset in seconds.
    peak : float
        Analytic peak of the first pass, not of the combined curve.
    alpha : float
        Positive power of time since onset, shared by both passes.
    beta_seconds : float
        First-pass decay scale in seconds.
    recirculation_fraction : float
        Nonnegative ratio of the full second-pass area to first-pass area.
        Zero returns the first pass exactly. This is not a leakage coefficient.
    recirculation_delay_seconds : float
        Positive second-pass onset offset relative to the first pass.
    broadening : float
        Second-pass scale divided by first-pass scale, at least one. The second
        peak is scaled by recirculation_fraction / broadening to preserve the
        specified area ratio without renormalizing over a finite scan window.

    Returns
    -------
    ndarray
        Continuous two-pass AIF values. This is a synthetic input-shape control,
        not a whole-circulation model or an exact reproduction of a different
        recirculation-kernel construction. Pass this same function to tissue
        generation and evaluate it at the acquired times for the measured AIF.
    """
    if not np.isfinite(recirculation_fraction) or recirculation_fraction < 0:
        raise ValueError("recirculation_fraction must be finite and nonnegative")
    _positive(recirculation_delay_seconds, "recirculation_delay_seconds")
    if not np.isfinite(broadening) or broadening < 1:
        raise ValueError("broadening must be finite and at least one")
    first = gamma_variate_aif(
        time_seconds, onset_seconds=onset_seconds, peak=peak,
        alpha=alpha, beta_seconds=beta_seconds,
    )
    if recirculation_fraction == 0:
        return first
    second = gamma_variate_aif(
        time_seconds, onset_seconds=onset_seconds + recirculation_delay_seconds,
        peak=peak, alpha=alpha, beta_seconds=beta_seconds * broadening,
    )
    return np.asarray(
        first + (recirculation_fraction / broadening) * second, dtype=np.float64,
    )


def analytic_residue(
    family: ResidueFamily,
    mtt_seconds: float,
    *,
    gamma_shape: float = 3.0,
) -> ContinuousResidue:
    """Construct an analytic residue without a perfusion-model dependency.

    Parameters
    ----------
    family : {"exponential", "gamma", "box", "triangle"}
        Gamma uses the transit-time survival function; triangle decreases
        linearly to zero at twice MTT; box drops to zero at MTT.
    mtt_seconds : float
        Positive full residue integral in seconds.
    gamma_shape : float
        Positive gamma shape, used only for the gamma family.

    Returns
    -------
    ContinuousResidue
        Causal residue with R(0)=1, known full MTT, and integration breakpoints.
    """
    _positive(mtt_seconds, "mtt_seconds")
    if family not in ("exponential", "gamma", "box", "triangle"):
        raise ValueError(f"Unknown residue family: {family}")
    if family == "gamma":
        _positive(gamma_shape, "gamma_shape")

    def evaluate(time_seconds: npt.ArrayLike) -> npt.NDArray[np.float64]:
        time = np.asarray(time_seconds, dtype=np.float64)
        u = np.maximum(time, 0.0) / mtt_seconds
        if family == "exponential":
            values = np.exp(-u)
        elif family == "gamma":
            values = gammaincc(gamma_shape, u * gamma_shape)
        elif family == "box":
            values = np.asarray(u < 1.0, dtype=np.float64)
        else:
            values = np.maximum(1.0 - u / 2.0, 0.0)
        return np.asarray(np.where(time < 0, 0.0, values), dtype=np.float64)

    breaks = ((mtt_seconds,) if family == "box" else
              (2.0 * mtt_seconds,) if family == "triangle" else ())
    return ContinuousResidue(family, mtt_seconds, evaluate, breaks)


def spline_residue(
    knot_seconds: npt.ArrayLike,
    amplitudes: npt.ArrayLike,
) -> ContinuousResidue:
    """Construct a finite, non-increasing PCHIP residue directly from knots.

    Parameters
    ----------
    knot_seconds : array_like
        Strictly increasing finite lag times starting at zero.
    amplitudes : array_like
        Matching non-increasing amplitudes starting at one and ending at zero.

    Returns
    -------
    ContinuousResidue
        Piecewise-cubic residue with its exact polynomial integral. Membership
        in the PCHIP family does not imply representability under every fitted
        model's knot, endpoint, slope or parameter-support constraints.
    """
    knots = np.array(knot_seconds, dtype=np.float64, copy=True)
    values = np.array(amplitudes, dtype=np.float64, copy=True)
    if (knots.ndim != 1 or len(knots) < 2 or values.shape != knots.shape
            or not np.all(np.isfinite(knots)) or not np.all(np.isfinite(values))
            or knots[0] != 0 or np.any(np.diff(knots) <= 0)
            or values[0] != 1 or values[-1] != 0 or np.any(np.diff(values) > 0)):
        raise ValueError("Spline needs increasing causal knots and amplitudes from 1 to 0")
    interpolator = PchipInterpolator(knots, values, extrapolate=False)

    def evaluate(time_seconds: npt.ArrayLike) -> npt.NDArray[np.float64]:
        time = np.asarray(time_seconds, dtype=np.float64)
        bounded = np.clip(time, 0.0, knots[-1])
        return np.asarray(np.where(
            (time >= 0) & (time <= knots[-1]), interpolator(bounded), 0.0,
        ), dtype=np.float64)

    return ContinuousResidue(
        "pchip", float(interpolator.integrate(0.0, knots[-1])), evaluate,
        tuple(float(t) for t in knots[1:]),
    )


def continuous_tissue(
    time_seconds: npt.ArrayLike,
    aif: Curve,
    residue: ContinuousResidue,
    *,
    flow: float,
    aif_onset_seconds: float,
    delay_seconds: float = 0.0,
    aif_breakpoints: Sequence[float] = (),
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
) -> ContinuousTissue:
    """Integrate F * A(u) * R(t-delay-u) before sampling the tissue curve.

    Parameters
    ----------
    time_seconds : array_like
        Finite, strictly increasing point-sample times; need not be uniform.
    aif : callable
        Continuous input, zero before aif_onset_seconds. No interpolation from
        frame samples is performed; sample this same function for fitting.
    residue : ContinuousResidue
        Continuous causal unit-height residue with known integration segments.
    flow : float
        Nonnegative flow multiplier in inverse seconds, e.g. mL/mL/s for
        concentration inputs. Convert physical display units explicitly outside
        this function; there is no hidden factor of 60 or 100.
    aif_onset_seconds : float
        Earliest nonzero AIF support, possibly before the acquired time window.
    delay_seconds : float
        Tissue arrival relative to the AIF, positive for later arrival.
        Negative shifts are evaluated continuously, not wrapped or clipped to
        the first acquired AIF sample.
    aif_breakpoints : sequence of float
        Absolute AIF times at which the integrator should split the interval.
    epsabs, epsrel : float
        Positive absolute and relative tolerances before multiplication by flow.

    Returns
    -------
    ContinuousTissue
        Sampled tissue and quadrature error estimates. Numerical failures raise
        rather than quietly emitting reference data of unknown accuracy.
    """
    time = np.asarray(time_seconds, dtype=np.float64)
    if (time.ndim != 1 or time.size == 0 or not np.all(np.isfinite(time))
            or np.any(np.diff(time) <= 0)):
        raise ValueError("time_seconds must be a finite, increasing nonempty vector")
    if not np.isfinite(flow) or flow < 0:
        raise ValueError("flow must be finite and nonnegative")
    if not np.all(np.isfinite([aif_onset_seconds, delay_seconds, *aif_breakpoints])):
        raise ValueError("AIF onset, delay and breakpoints must be finite")
    _positive(epsabs, "epsabs")
    _positive(epsrel, "epsrel")
    values = np.zeros_like(time)
    errors = np.zeros_like(time)
    if flow == 0:
        return ContinuousTissue(values, errors)
    for index, sample in enumerate(time):
        upper = float(sample - delay_seconds)
        if upper <= aif_onset_seconds:
            continue
        breaks = (*aif_breakpoints, *(upper - p for p in residue.breakpoints))
        value, error = _integrate(
            lambda u: float(aif(u)) * float(residue.evaluate(upper - u)),
            aif_onset_seconds, upper, breaks, epsabs, epsrel,
        )
        values[index], errors[index] = flow * value, flow * error
    return ContinuousTissue(values, errors)
