"""Paired DSC measurements with explicit noise units and measured baselines.

These helpers do not fit models or add leakage. The continuous generators
supply clean delta-R2* curves, including baseline samples. Each observation
contains both the noisy signal and the conversion of that same signal so
signal- and concentration-based methods receive equivalent measured inputs.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

NoiseModel = Literal["rician_signal", "gaussian_delta_r2"]


@dataclass(frozen=True)
class SignalObservation:
    """A measured curve and its explicitly defined noise/baseline quantities.

    Parameters
    ----------
    clean_signal, signal, delta_r2 : ndarray
        Clean signal, noisy measured magnitude signal, and delta-R2* calculated
        from that measurement. Time is the final axis; leading axes are curves.
    baseline_signal : ndarray
        Mean measured pre-bolus signal for each curve, not the true S0.
    noise_model : str
        Whether noise was applied to complex signal or delta-R2*.
    noise_sd : float
        Standard deviation per complex channel in signal units for Rician
        noise; standard deviation in s^-1 for Gaussian delta-R2* noise.
    baseline_snr : float or None
        True S0 / signal-noise SD for Rician data; None for the continuity
        convention, which does not have constant signal-domain noise.
    """

    clean_signal: npt.NDArray[np.float64]
    signal: npt.NDArray[np.float64]
    delta_r2: npt.NDArray[np.float64]
    baseline_signal: npt.NDArray[np.float64]
    noise_model: NoiseModel
    noise_sd: float
    baseline_snr: float | None


@dataclass(frozen=True)
class DSCObservations:
    """Common measurements for every deconvolution method in a paired comparison.

    Parameters
    ----------
    time_seconds, baseline_mask : ndarray
        Shared point-sample times and selected pre-bolus frames.
    te_seconds : float
        Echo time used for both signal formation and measured conversion.
    tissue, aif, references : SignalObservation
        Independently noised measurements. Reference curves occupy the leading
        axis; the measured-site AIF is not replaced by the tissue-dispersed AIF.
    """

    time_seconds: npt.NDArray[np.float64]
    baseline_mask: npt.NDArray[np.bool_]
    te_seconds: float
    tissue: SignalObservation
    aif: SignalObservation
    references: SignalObservation


def signal_observation(
    delta_r2: npt.ArrayLike,
    baseline_mask: npt.ArrayLike,
    *,
    te_seconds: float,
    s0: float,
    noise_model: NoiseModel,
    noise_sd: float,
    seed: int | np.random.SeedSequence,
) -> SignalObservation:
    """Generate a measured signal and convert it using its measured baseline.

    Parameters
    ----------
    delta_r2 : array_like
        Finite clean curve(s), including zero pre-bolus samples on the last axis.
        Negative post-bolus values are allowed for separately generated leakage.
    baseline_mask : array_like
        Boolean vector selecting known pre-bolus frames, at least two frames.
    te_seconds, s0 : float
        Positive echo time and true baseline signal used only to generate data.
    noise_model : {"rician_signal", "gaussian_delta_r2"}
        Primary complex-Gaussian/magnitude noise or labelled continuity noise.
    noise_sd : float
        Nonnegative absolute SD in the selected domain. Percentages are not
        inferred: callers must explicitly convert SNR or a peak-noise fraction.
    seed : int or numpy.random.SeedSequence
        Reproducible random-stream identity, independent of method or execution order.

    Returns
    -------
    SignalObservation
        The same measured curve in signal and delta-R2* form. No clipping of
        negative measured delta-R2* or replacement with clean data is performed.
    """
    clean = np.asarray(delta_r2, dtype=np.float64)
    mask = np.asarray(baseline_mask)
    if (clean.ndim not in (1, 2) or clean.size == 0 or not np.all(np.isfinite(clean))):
        raise ValueError("delta_r2 must contain finite nonempty curve(s)")
    if (mask.dtype != np.bool_ or mask.shape != (clean.shape[-1],)
            or np.count_nonzero(mask) < 2):
        raise ValueError("baseline_mask must be boolean with at least two pre-bolus frames")
    if not np.all(np.abs(clean[..., mask]) <= 1e-12):
        raise ValueError("Selected baseline frames must precede the simulated bolus")
    if not np.isfinite(te_seconds) or te_seconds <= 0 or not np.isfinite(s0) or s0 <= 0:
        raise ValueError("TE and S0 must be finite and positive")
    if not np.isfinite(noise_sd) or noise_sd < 0:
        raise ValueError("noise_sd must be finite and nonnegative")
    if noise_model not in ("rician_signal", "gaussian_delta_r2"):
        raise ValueError(f"Unknown noise model: {noise_model}")
    rng = np.random.default_rng(seed)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        clean_signal = np.asarray(s0 * np.exp(-te_seconds * clean), dtype=np.float64)
        if noise_model == "rician_signal":
            real = clean_signal + noise_sd * rng.standard_normal(clean.shape)
            imaginary = noise_sd * rng.standard_normal(clean.shape)
            signal = np.asarray(np.hypot(real, imaginary), dtype=np.float64)
            snr = s0 / noise_sd if noise_sd else float("inf")
        else:
            noisy_delta_r2 = clean + noise_sd * rng.standard_normal(clean.shape)
            signal = np.asarray(s0 * np.exp(-te_seconds * noisy_delta_r2), dtype=np.float64)
            snr = None
    if (not np.all(np.isfinite(clean_signal)) or np.any(clean_signal <= 0)
            or not np.all(np.isfinite(signal)) or np.any(signal <= 0)):
        raise ValueError("Simulated signals must be finite and positive; no floor is substituted")
    baseline = np.asarray(np.mean(signal[..., mask], axis=-1), dtype=np.float64)
    converted = np.asarray(-np.log(signal / baseline[..., None]) / te_seconds, dtype=np.float64)
    return SignalObservation(clean_signal, signal, converted, baseline, noise_model, noise_sd, snr)


def paired_observations(
    time_seconds: npt.ArrayLike,
    tissue_delta_r2: npt.ArrayLike,
    aif_delta_r2: npt.ArrayLike,
    reference_delta_r2: npt.ArrayLike,
    baseline_mask: npt.ArrayLike,
    *,
    te_seconds: float,
    s0: float,
    noise_model: NoiseModel,
    tissue_noise_sd: float,
    aif_noise_sd: float,
    reference_noise_sd: float,
    seed: int | np.random.SeedSequence,
) -> DSCObservations:
    """Assemble independent tissue/AIF/reference noise once for all methods.

    Parameters
    ----------
    time_seconds : array_like
        Shared finite, increasing point-sample times in seconds.
    tissue_delta_r2, aif_delta_r2 : array_like
        Clean one-dimensional curves at those times. AIF is the measured-site
        input, including any recirculation but excluding tissue-only transport.
    reference_delta_r2 : array_like
        Clean reference pool of shape (n_reference, n_time), including baseline.
    baseline_mask : array_like
        Boolean vector selecting pre-bolus frames common to all curves.
    te_seconds, s0 : float
        Declared signal-formation parameters.
    noise_model : {"rician_signal", "gaussian_delta_r2"}
        Explicitly labelled primary or continuity noise convention.
    tissue_noise_sd, aif_noise_sd, reference_noise_sd : float
        Absolute SDs in the specified noise domain, not percentages.
    seed : int or numpy.random.SeedSequence
        Stable observation identity. Deriving child streams does not mutate
        a supplied SeedSequence, so replaying an observation is deterministic.

    Returns
    -------
    DSCObservations
        Common measured arrays; no method-specific input regeneration is needed.
    """
    time = np.array(time_seconds, dtype=np.float64, copy=True)
    tissue, aif = np.asarray(tissue_delta_r2), np.asarray(aif_delta_r2)
    references = np.asarray(reference_delta_r2)
    if (time.ndim != 1 or len(time) < 2 or not np.all(np.isfinite(time))
            or np.any(np.diff(time) <= 0)):
        raise ValueError("time_seconds must be finite and strictly increasing")
    if tissue.shape != time.shape or aif.shape != time.shape:
        raise ValueError("Tissue and AIF must share the time vector")
    if references.ndim != 2 or references.shape[-1] != len(time) or references.shape[0] == 0:
        raise ValueError("References must be a nonempty pool on the shared time vector")
    root = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    streams = [np.random.SeedSequence(
        root.entropy, spawn_key=(*root.spawn_key, index), pool_size=root.pool_size,
    ) for index in range(3)]
    observed = [signal_observation(
                    curve, baseline_mask, noise_sd=sd, seed=stream,
                    te_seconds=te_seconds, s0=s0, noise_model=noise_model,
                )
                for curve, sd, stream in zip(
                    (tissue, aif, references),
                    (tissue_noise_sd, aif_noise_sd, reference_noise_sd), streams,
                )]
    return DSCObservations(
        time, np.array(baseline_mask, dtype=np.bool_, copy=True), te_seconds,
        observed[0], observed[1], observed[2],
    )
