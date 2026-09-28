"""Independent exchange kinetics and an effective-compartment DSC signal model.

Concentrations are mM, times seconds, volumes fractions of tissue volume, and
flow/permeability tissue-volume fractions per second. Signal formation is a
simplified effective-relaxivity calculation, not a microstructure simulation.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.signal import fftconvolve

from .dsc_reference import ContinuousTransport, Curve


@dataclass(frozen=True)
class ExchangeParameters:
    """Physical exchange parameters with explicit whole-blood/plasma conversion.

    Parameters
    ----------
    blood_flow, blood_volume, ees_volume, permeability : float
        Whole-blood flow in s^-1, blood and EES volume fractions, and plasma/EES
        permeability-surface product in s^-1. Permeability may be exactly zero.
    hematocrit : float
        Shared artery/tissue hematocrit for this controlled simulation.
    """

    blood_flow: float
    blood_volume: float
    ees_volume: float
    permeability: float
    hematocrit: float

    def __post_init__(self) -> None:
        """Reject invalid physical domains before integration."""
        values = (self.blood_flow, self.blood_volume, self.ees_volume,
                  self.permeability, self.hematocrit)
        if not np.all(np.isfinite(values)):
            raise ValueError("Exchange parameters must be finite")
        if (self.blood_flow <= 0 or self.blood_volume <= 0 or self.ees_volume <= 0
                or self.blood_volume + self.ees_volume > 1 or self.permeability < 0
                or not 0 <= self.hematocrit < 1):
            raise ValueError("Invalid exchange parameter domain")

    @property
    def plasma_volume(self) -> float:
        """Plasma fraction per tissue volume."""
        return self.blood_volume * (1 - self.hematocrit)

    @property
    def plasma_flow(self) -> float:
        """Plasma flow per tissue volume per second."""
        return self.blood_flow * (1 - self.hematocrit)

    @property
    def vascular_mtt(self) -> float:
        """Blood residence time, not whole-exchanging-system residence time."""
        return self.blood_volume / self.blood_flow

    @property
    def ktrans(self) -> float:
        """Well-mixed exchange transfer constant in s^-1, not a BSW K2."""
        return self.plasma_flow * self.permeability / (self.plasma_flow + self.permeability)

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        """Linear compartment-concentration mass-balance operator."""
        ps, vp, ve = self.permeability, self.plasma_volume, self.ees_volume
        return np.array([[-(self.plasma_flow + ps) / vp, ps / vp], [ps / ve, -ps / ve]])


@dataclass(frozen=True)
class ExchangeCurves:
    """Point samples and numerical refinement evidence.

    Parameters
    ----------
    plasma, ees : ndarray
        Concentration per compartment volume in mM.
    local_blood_aif : ndarray
        Whole-blood input after tissue-only delay and transport, not the measured
        upstream AIF. It drives both compartments through the same plasma input.
    step_seconds, refinement_error : float
        Accepted forcing-grid/maximum ODE step and maximum normalized change
        in plasma, EES or local input between successive refinements. Each
        curve uses its own peak scale; an absent curve has zero change.
    """

    plasma: npt.NDArray[np.float64]
    ees: npt.NDArray[np.float64]
    local_blood_aif: npt.NDArray[np.float64]
    step_seconds: float
    refinement_error: float


def exchange_curves(
    time_seconds: npt.ArrayLike,
    blood_aif: Curve,
    parameters: ExchangeParameters,
    *,
    aif_onset_seconds: float,
    delay_seconds: float = 0.0,
    transport: ContinuousTransport | None = None,
    initial_step_seconds: float = 0.2,
    refinement_tolerance: float = 1e-5,
    max_refinements: int = 10,
) -> ExchangeCurves:
    """Integrate two-way exchange with delayed/dispersed continuous forcing.

    Parameters
    ----------
    time_seconds : array_like
        Finite increasing point-sample times, including optional baseline.
    blood_aif : callable
        Nonnegative continuous whole-blood concentration, zero before its onset.
    parameters : ExchangeParameters
        Physical compartment parameters. Initial tracer concentrations are zero.
    aif_onset_seconds, delay_seconds : float
        Upstream bolus onset and tissue-only shift, in seconds. Negative delay
        is allowed if the declared baseline still precedes tissue arrival.
    transport : ContinuousTransport or None
        Unit-mass upstream transport; neither truncated nor renormalized to the
        scan. Sampled density must be finite at zero.
    initial_step_seconds, refinement_tolerance : float
        Initial maximum ODE/forcing step and relative numerical tolerance.
    max_refinements : int
        Maximum grid halvings; two consecutive acceptable changes are required.

    Returns
    -------
    ExchangeCurves
        Accepted concentrations; integration/refinement failures raise explicitly.
    """
    time = np.asarray(time_seconds, dtype=np.float64)
    if (time.ndim != 1 or time.size < 2 or not np.all(np.isfinite(time))
            or np.any(np.diff(time) <= 0)):
        raise ValueError("time_seconds must be finite and strictly increasing")
    if (not np.all(np.isfinite([aif_onset_seconds, delay_seconds, initial_step_seconds,
                               refinement_tolerance])) or initial_step_seconds <= 0
            or not 0 < refinement_tolerance < 1 or max_refinements < 2):
        raise ValueError("Invalid timing or numerical refinement settings")
    onset = aif_onset_seconds + delay_seconds
    active = time > onset
    if not np.any(active):
        zeros = np.zeros_like(time)
        return ExchangeCurves(zeros.copy(), zeros.copy(), zeros, initial_step_seconds, 0.0)
    matrix = parameters.matrix
    forcing_scale = parameters.plasma_flow / parameters.plasma_volume / (1 - parameters.hematocrit)
    previous = None
    accepted = 0
    step = initial_step_seconds
    for _ in range(max_refinements + 1):
        count = int(np.ceil((time[-1] - onset) / step))
        if count > 500_000:
            raise RuntimeError("Exchange reference grid exceeds numerical budget")
        lag = np.arange(count + 1) * step
        input_values = np.asarray(blood_aif(aif_onset_seconds + lag), dtype=float)
        if (input_values.shape != lag.shape or not np.all(np.isfinite(input_values))
                or np.any(input_values < 0)):
            raise ValueError("blood_aif must return finite nonnegative concentrations")
        if transport is not None:
            kernel = np.asarray(transport.density(lag), dtype=float)
            if not np.all(np.isfinite(kernel)) or np.any(kernel < 0):
                raise ValueError("Transport density must be finite and nonnegative on the grid")
            mass_error = abs(float(np.trapezoid(kernel, dx=step))
                             - float(transport.cumulative(lag[-1])))
            if mass_error > refinement_tolerance:
                accepted, previous = 0, None
                step *= 0.5
                continue
            local = step * fftconvolve(input_values, kernel)[:count + 1]
            local -= 0.5 * step * (input_values[0] * kernel + kernel[0] * input_values)
            if np.min(local) < -1e-12 * max(1.0, float(np.max(input_values))):
                raise RuntimeError("Transport convolution produced negative concentrations")
            input_values = np.maximum(local, 0.0)
            input_values[0] = 0.0
        interpolated = PchipInterpolator(onset + lag, input_values, extrapolate=False)

        def derivative(t: float, state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            """Mass balance with one common arterial forcing."""
            value = float(interpolated(t))
            return np.asarray(matrix @ state + [forcing_scale * value, 0.0], dtype=np.float64)

        solution = solve_ivp(
            derivative, (onset, float(time[-1])), [0.0, 0.0], method="DOP853",
            t_eval=time[active], rtol=1e-10, atol=1e-12, max_step=step,
        )
        if not solution.success or not np.all(np.isfinite(solution.y)):
            raise RuntimeError(f"Exchange integration failed: {solution.message}")
        current = np.zeros((3, time.size))
        current[:2, active] = solution.y
        current[2, active] = interpolated(time[active])
        if np.min(current) < -1e-10:
            raise RuntimeError("Exchange integration produced negative concentrations")
        error = float("inf")
        if previous is not None:
            scale = np.maximum(np.max(np.abs(current), axis=1), 1e-12)
            error = float(np.max(np.max(np.abs(current - previous), axis=1) / scale))
            accepted = accepted + 1 if error <= refinement_tolerance else 0
            if accepted >= 2:
                return ExchangeCurves(current[0], current[1], current[2], step, error)
        previous = current
        step *= 0.5
    raise RuntimeError("Exchange generator did not meet refinement tolerance")


def apparent_delta_r2(
    intravascular_concentration: npt.ArrayLike,
    extravascular_concentration: npt.ArrayLike,
    *,
    tr_seconds: float,
    te_seconds: float,
    flip_angle_degrees: float,
    native_t1_seconds: float,
    r1_ees: float,
    r2_intravascular: float,
    r2_ees: float,
) -> npt.NDArray[np.float64]:
    """Map tissue-volume-weighted concentrations to apparent DSC delta-R2*.

    Parameters
    ----------
    intravascular_concentration, extravascular_concentration : array_like
        Nonnegative mM contributions per tissue volume, vp*Cp and ve*Ce, with
        the same shape. Zero pre-bolus concentrations correspond to S0.
    tr_seconds, te_seconds, native_t1_seconds : float
        Positive sequence repetition/echo times and baseline tissue T1.
    flip_angle_degrees : float
        Excitation angle in (0, 180) degrees.
    r1_ees, r2_intravascular, r2_ees : float
        Nonnegative effective relaxivities in L/mmol/s. Longitudinal enhancement
        is assigned to EES tracer only. Transverse susceptibility depends on
        the absolute difference between the two concentration contributions.

    Returns
    -------
    ndarray
        -log(S/S0)/TE from the normalized steady-state spoiled GRE signal.
        Negative values (enhancement above baseline) are retained, not clipped.
        Noise is added to the reconstructed signal, not to these clean values.
    """
    civ = np.asarray(intravascular_concentration, dtype=float)
    cev = np.asarray(extravascular_concentration, dtype=float)
    if (civ.shape != cev.shape or not np.all(np.isfinite(civ))
            or not np.all(np.isfinite(cev)) or np.any(civ < 0) or np.any(cev < 0)):
        raise ValueError("Compartment contributions must have equal shapes and be nonnegative")
    if (not np.all(np.isfinite([tr_seconds, te_seconds, native_t1_seconds,
                               flip_angle_degrees, r1_ees, r2_intravascular, r2_ees]))
            or min(tr_seconds, te_seconds, native_t1_seconds) <= 0
            or not 0 < flip_angle_degrees < 180 or min(r1_ees, r2_intravascular, r2_ees) < 0):
        raise ValueError("Invalid GRE signal parameters")
    cosine = np.cos(np.deg2rad(flip_angle_degrees))
    decay = tr_seconds * (1 / native_t1_seconds + r1_ees * cev)
    baseline_decay = tr_seconds * (1 / native_t1_seconds)
    log_e = np.log(-np.expm1(-decay)) - np.log1p(-cosine * np.exp(-decay))
    log_baseline = np.log(-np.expm1(-baseline_decay)) - np.log1p(-cosine * np.exp(-baseline_decay))
    transverse = r2_ees * cev + r2_intravascular * np.abs(civ - cev)
    return np.asarray(transverse - (log_e - log_baseline) / te_seconds, dtype=np.float64)
