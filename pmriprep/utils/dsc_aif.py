"""Gamma-variate AIF fitting for DSC-MRI.

Fits a gamma-variate model to a DSC arterial input function (recirculation
removal with AUC preservation), decides when the fit should be applied, and
computes the peak FWHM the fit uses. AIF candidate detection and quality
scoring are handled in the ``AIFDetection`` interface.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import optimize

from pmriprep.utils.dsc_constants import (
    AIF_SMOOTHING_TARGET_SECONDS,
    EPSILON_CONCENTRATION,
)


def get_aif_smoothing_window(n_timepoints: int, tr: float) -> int:
    """Return a TR-adaptive odd Savitzky-Golay smoothing window.

    Parameters
    ----------
    n_timepoints : int
        Length of the AIF curve.
    tr : float
        Frame period in seconds. Invalid values retain the historical
        five-frame fallback.

    Returns
    -------
    int
        Odd window length no larger than ``n_timepoints``. Curves shorter
        than five frames return their largest odd length; callers decide
        whether smoothing is appropriate.
    """
    if n_timepoints < 1:
        raise ValueError("n_timepoints must be positive")

    if not np.isfinite(tr) or tr <= 0.0:
        window = 5
    else:
        target_samples = int(round(AIF_SMOOTHING_TARGET_SECONDS / tr))
        # Quadratic Savitzky-Golay requires at least three samples. A five-frame
        # floor would span 15 s at TR=3 s and flatten the arterial peak.
        window = max(3, target_samples | 1)

    largest_odd = n_timepoints if n_timepoints % 2 else n_timepoints - 1
    return min(window, largest_odd)


def compute_aif_fwhm_frames(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
) -> int:
    """Compute full width at half maximum (FWHM) of AIF peak in frames.

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course.
    baseline_frames : int
        Number of pre-bolus baseline frames (default: 5).

    Returns
    -------
    int
        FWHM in number of frames. Arterial curves typically 5-15 frames.
        Returns 0 when the curve is too short or the global maximum lies within
        the baseline window (a spike/artifact with no measurable first pass).
    """
    if len(aif_curve) < baseline_frames + 1:
        return 0

    baseline_mean = float(np.mean(aif_curve[:baseline_frames]))
    peak = float(np.max(aif_curve))
    peak_idx = int(np.argmax(aif_curve))
    if peak_idx <= baseline_frames:
        # Global max inside the baseline window: the rising-edge search would be
        # empty and yield a spurious near-full-curve width, so report 0.
        return 0

    half_max = baseline_mean + (peak - baseline_mean) / 2.0

    # Find first crossing above half max (rising edge)
    rising_idx = 0
    for i in range(baseline_frames, peak_idx):
        if aif_curve[i] >= half_max:
            rising_idx = i
            break

    # Find first crossing below half max after peak (falling edge)
    falling_idx = len(aif_curve) - 1
    for i in range(peak_idx + 1, len(aif_curve)):
        if aif_curve[i] <= half_max:
            falling_idx = i
            break

    fwhm = falling_idx - rising_idx
    return max(0, fwhm)


# =============================================================================
# Gamma-Variate AIF Fitting
# =============================================================================


@dataclass
class GammaVariateFitResult:
    """Result of gamma-variate fitting to an AIF curve.

    The gamma-variate model is:
        C(t) = A × (t - t0)^α × exp(-(t - t0) / β)  for t > t0
        C(t) = 0                                     for t <= t0

    Where:
        A: Amplitude scaling factor
        t0: Bolus arrival time
        α (alpha): Shape parameter controlling the rise steepness
        β (beta): Scale parameter controlling the decay rate

    Derived parameters:
        Time to peak (TTP) = t0 + α × β
        AUC = A × β^(α+1) × Γ(α+1)  (total dose)

    Attributes
    ----------
    fitted_curve : npt.NDArray[Any]
        The fitted gamma-variate curve (same length as input).
    amplitude : float
        Amplitude parameter A.
    t0 : float
        Bolus arrival time in frames.
    alpha : float
        Shape parameter α (controls rise steepness).
    beta : float
        Scale parameter β (controls decay rate, in frames).
    auc_original : float
        First-pass-window AUC of the input curve when ``remove_recirculation=True``;
        full-curve AUC otherwise. Window matches ``auc_fitted`` so ``auc_ratio``
        is apples-to-apples.
    auc_fitted : float
        Area under curve of fitted result, integrated over the same window
        as ``auc_original``. Note: when ``preserve_auc=True`` (production
        default), this is post-scaling and equals ``auc_original`` by
        construction. ``auc_ratio`` is computed BEFORE scaling, so it
        retains shape-fit-quality semantics.
    auc_ratio : float
        Ratio of fitted/original AUC computed BEFORE preservation scaling
        — measures gamma-fit shape mismatch against the empirical
        first-pass area independent of dose-preservation. Should be ~1.0
        for a good fit. Reported in QC for provenance; not a hard gate
        (see ``should_apply_gamma_fit``).
    residual_rms : float
        RMS residual between original and fitted.
    r_squared : float
        Coefficient of determination (R²).
    fit_converged : bool
        Whether the optimization converged successfully.
    recirculation_removed : bool
        Whether recirculation was removed from the fit.
    first_pass_n : int
        Number of samples in the first-pass window
        ``[baseline_frames : peak_idx + max(10, 2*FWHM)]`` when
        ``remove_recirculation=True``; equal to ``n_frames - baseline_frames``
        otherwise. Gamma fitting is ill-posed below ~8 samples.
    failure_reason : str or None
        Stable reason code when optimization does not converge or raises.
    """

    fitted_curve: npt.NDArray[Any]
    amplitude: float
    t0: float
    alpha: float
    beta: float
    auc_original: float
    auc_fitted: float
    auc_ratio: float
    residual_rms: float
    r_squared: float
    fit_converged: bool
    recirculation_removed: bool = True
    first_pass_n: int = 0
    failure_reason: Optional[str] = None


def _gamma_variate_model(
    t: npt.NDArray[Any],
    amplitude: float,
    t0: float,
    alpha: float,
    beta: float,
) -> npt.NDArray[Any]:
    """Compute gamma-variate function.

    C(t) = A × (t - t0)^α × exp(-(t - t0) / β)  for t > t0
    C(t) = 0                                     for t <= t0

    Parameters
    ----------
    t : np.ndarray
        Time points (frame indices).
    amplitude : float
        Amplitude scaling factor A.
    t0 : float
        Bolus arrival time.
    alpha : float
        Shape parameter (>0).
    beta : float
        Scale parameter (>0).

    Returns
    -------
    np.ndarray
        Gamma-variate function values.
    """
    result = np.zeros_like(t, dtype=np.float64)
    mask = t > t0

    if np.any(mask):
        dt = t[mask] - t0
        # Avoid numerical issues with very small alpha
        alpha_safe = max(alpha, 0.01)
        beta_safe = max(beta, 0.01)

        # Compute gamma-variate: A × (t-t0)^α × exp(-(t-t0)/β)
        result[mask] = amplitude * np.power(dt, alpha_safe) * np.exp(-dt / beta_safe)

    return result


def _gamma_variate_residual(
    params: Tuple[float, float, float, float],
    t: npt.NDArray[Any],
    y: npt.NDArray[Any],
    weights: Optional[npt.NDArray[Any]] = None,
) -> npt.NDArray[Any]:
    """Compute weighted residuals for gamma-variate fitting.

    Parameters
    ----------
    params : Tuple[float, float, float, float]
        (amplitude, t0, alpha, beta)
    t : np.ndarray
        Time points.
    y : np.ndarray
        Observed values.
    weights : np.ndarray, optional
        Fitting weights.

    Returns
    -------
    np.ndarray
        Weighted residuals.
    """
    amplitude, t0, alpha, beta = params
    y_pred = _gamma_variate_model(t, amplitude, t0, alpha, beta)
    residuals = y - y_pred

    if weights is not None:
        residuals = residuals * weights

    return residuals  # type: ignore[no-any-return]


def _estimate_initial_params(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
) -> Tuple[float, float, float, float]:
    """Estimate initial parameters for gamma-variate fitting.

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course.
    baseline_frames : int
        Number of pre-bolus baseline frames.

    Returns
    -------
    Tuple[float, float, float, float]
        Initial estimates for (amplitude, t0, alpha, beta).
    """
    # Find peak
    baseline = np.mean(aif_curve[:baseline_frames])
    curve_shifted = aif_curve - baseline
    peak_idx = int(np.argmax(curve_shifted))
    peak_value = float(np.max(curve_shifted))

    if peak_value < EPSILON_CONCENTRATION:
        # Flat curve, return defaults
        return (1.0, float(baseline_frames), 2.0, 3.0)

    # Estimate t0 from the first post-baseline frame exceeding 10% of peak.
    threshold = baseline + 0.1 * peak_value
    rising_indices = np.where(aif_curve[baseline_frames:] > threshold)[0]
    if len(rising_indices) > 0:
        t0_estimate = float(baseline_frames + rising_indices[0]) - 0.5
    else:
        t0_estimate = float(baseline_frames)

    # Ensure t0 is not negative
    t0_estimate = max(0.0, t0_estimate)

    # Time to peak from t0
    ttp_from_t0 = max(1.0, peak_idx - t0_estimate)

    # For gamma-variate: TTP = alpha × beta
    # Estimate alpha from rise steepness (typical range 1-5)
    # Steeper rise → smaller alpha
    # Estimate based on FWHM
    fwhm = compute_aif_fwhm_frames(aif_curve, baseline_frames)
    if fwhm > 0:
        # Approximate relationship: FWHM ≈ 2.35 × beta for Gaussian-like shapes
        # For gamma-variate, FWHM/TTP gives info about alpha
        fwhm_ttp_ratio = fwhm / ttp_from_t0 if ttp_from_t0 > 0 else 2.0
        # Lower ratio → sharper peak → lower alpha
        alpha_estimate = max(0.5, min(5.0, 2.0 * fwhm_ttp_ratio))
    else:
        alpha_estimate = 2.0

    # Compute beta from TTP and alpha: TTP = alpha × beta → beta = TTP/alpha
    beta_estimate = ttp_from_t0 / alpha_estimate if alpha_estimate > 0 else 3.0
    beta_estimate = max(0.5, beta_estimate)

    # Estimate amplitude: at peak, A × (TTP/beta)^alpha × exp(-alpha)
    # Solving for A
    peak_term = np.power(ttp_from_t0, alpha_estimate) * np.exp(-alpha_estimate)
    if peak_term > EPSILON_CONCENTRATION:
        amplitude_estimate = peak_value / peak_term
    else:
        amplitude_estimate = peak_value

    return (amplitude_estimate, t0_estimate, alpha_estimate, beta_estimate)


def _compute_auc(
    curve: npt.NDArray[Any],
    tr: float = 1.0,
) -> float:
    """Compute area under curve using trapezoidal integration.

    Parameters
    ----------
    curve : np.ndarray
        1D time course.
    tr : float
        Time step (frame duration) for scaling.

    Returns
    -------
    float
        Area under curve.
    """
    # Use scipy's trapezoid for compatibility
    from scipy import integrate

    auc = float(integrate.trapezoid(curve)) * tr
    return auc


def _find_recirculation_start(
    aif_curve: npt.NDArray[Any],
    fitted_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
) -> int:
    """Find the frame where recirculation begins.

    Recirculation is detected where the measured curve starts to
    deviate above the fitted gamma-variate (first-pass only).

    Parameters
    ----------
    aif_curve : np.ndarray
        Original AIF curve.
    fitted_curve : np.ndarray
        Fitted gamma-variate curve.
    baseline_frames : int
        Number of baseline frames.

    Returns
    -------
    int
        Frame index where recirculation starts, or -1 if not detected.
    """
    n_frames = len(aif_curve)

    # Find peak of fitted curve
    peak_idx = int(np.argmax(fitted_curve))

    # Look for recirculation after the washout phase begins
    # Check for sustained deviation above fitted curve
    min_frames_after_peak = 5  # Wait for initial washout
    start_search = min(peak_idx + min_frames_after_peak, n_frames - 5)

    if start_search >= n_frames - 2:
        return -1

    # Compute residual
    residual = aif_curve - fitted_curve

    # Find where residual becomes consistently positive
    # (measured > fitted, indicating recirculation contribution)
    window = 3  # Require sustained deviation
    for i in range(start_search, n_frames - window):
        # Check if next few frames are all above fitted
        if np.all(residual[i : i + window] > 0):
            return i

    return -1


def fit_gamma_variate(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
    remove_recirculation: bool = True,
    preserve_auc: bool = True,
    tr: float = 1.0,
) -> GammaVariateFitResult:
    """Fit a gamma-variate model to an AIF curve.

    This function fits the model:
        C(t) = A × (t - t0)^α × exp(-(t - t0) / β)

    The gamma-variate model represents the first-pass bolus only,
    allowing removal of recirculation effects for preload correction.

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course.
    baseline_frames : int
        Number of pre-bolus baseline frames for baseline subtraction.
    remove_recirculation : bool
        If True, fit only the first-pass portion and remove recirculation.
        If False, fit the entire curve.
    preserve_auc : bool
        If True, scale the fitted curve to match the first-pass AUC (when
        remove_recirculation=True) or the full AUC (when False).
    tr : float
        Repetition time in seconds (for AUC calculation).

    Returns
    -------
    GammaVariateFitResult
        Fitted parameters and curve with quality metrics.

    Notes
    -----
    The gamma-variate model is widely used for AIF characterization because:
    - It naturally models the first-pass bolus shape
    - It has analytical AUC: AUC = A × β^(α+1) × Γ(α+1)
    - Recirculation appears as positive residual after washout

    For preload correction:
    - Fit gamma-variate to measured AIF
    - The fitted curve represents first-pass only
    - Residual (measured - fitted) contains recirculation
    - Using fitted AIF for deconvolution removes recirculation effects

    Clinical interpretation:
    - Alpha (α): Rise steepness; typical 1.5-3.5
    - Beta (β): Decay rate; typical 1.5-4.0 frames
    - TTP = α × β: Time to peak from arrival
    - MTT_AIF = (α + 1) × β: Mean transit time of bolus

    Examples
    --------
    >>> import numpy as np
    >>> # Create synthetic AIF with recirculation
    >>> t = np.arange(60)
    >>> aif = np.zeros(60)
    >>> # First pass: gamma-variate
    >>> first_pass = 10 * np.power(np.maximum(0, t - 5), 2.0) * np.exp(-(t - 5) / 3)
    >>> # Recirculation: delayed, smaller peak
    >>> recirc = 3 * np.power(np.maximum(0, t - 20), 1.5) * np.exp(-(t - 20) / 5)
    >>> aif = first_pass + recirc
    >>> result = fit_gamma_variate(aif, baseline_frames=5, remove_recirculation=True)
    >>> print(f"Fit R²: {result.r_squared:.3f}, AUC ratio: {result.auc_ratio:.3f}")
    Fit R²: 0.8..., AUC ratio: 0.9...
    """
    n_frames = len(aif_curve)
    t = np.arange(n_frames, dtype=np.float64)

    # Subtract baseline
    baseline = float(np.mean(aif_curve[:baseline_frames]))
    curve_shifted = aif_curve - baseline

    # Get initial parameter estimates
    p0 = _estimate_initial_params(aif_curve, baseline_frames)

    # Define bounds for parameters
    # (amplitude, t0, alpha, beta)
    bounds_lower = (0.0, 0.0, 0.1, 0.1)
    bounds_upper = (np.inf, float(n_frames) / 2.0, 10.0, 20.0)

    # Initial estimates are heuristic and can be driven outside the solver
    # domain by a late noisy maximum. Project finite estimates onto the same
    # declared bounds before optimization.
    p0_array = np.asarray(p0, dtype=np.float64)
    failure_reason: Optional[str] = None
    if not np.all(np.isfinite(p0_array)):
        # Keep initialization failures on the same non-converged result
        # contract as optimizer failures. Candidate-quality callers can then
        # reject this optional fit without aborting AIF detection.
        failure_reason = "nonfinite_initialization"
        p0_array = np.asarray(
            (1.0, float(baseline_frames), 2.0, 3.0),
            dtype=np.float64,
        )
    p0_projected = np.clip(
        p0_array,
        np.asarray(bounds_lower, dtype=np.float64),
        np.asarray(bounds_upper, dtype=np.float64),
    )
    p0 = (
        float(p0_projected[0]),
        float(p0_projected[1]),
        float(p0_projected[2]),
        float(p0_projected[3]),
    )

    # Create fitting weights
    # Higher weight around peak, lower in baseline and tail
    peak_idx = int(np.argmax(curve_shifted))
    weights = np.ones(n_frames, dtype=np.float64)

    # Reduce weight of baseline
    weights[:baseline_frames] = 0.1

    if remove_recirculation:
        # For recirculation removal, focus fitting on first pass
        # Use data up to ~2× FWHM after peak
        fwhm = compute_aif_fwhm_frames(aif_curve, baseline_frames)
        cutoff = min(peak_idx + max(10, 2 * fwhm), n_frames)
        weights[cutoff:] = 0.01  # Minimal weight on tail/recirculation

    # Perform nonlinear least squares fit
    if failure_reason is not None:
        fit_converged = False
        amplitude, t0, alpha, beta = p0
    else:
        try:
            result = optimize.least_squares(
                _gamma_variate_residual,
                p0,
                args=(t, curve_shifted, weights),
                bounds=(bounds_lower, bounds_upper),
                method="trf",
                max_nfev=1000,
            )
            fit_converged = result.success
            if not fit_converged:
                failure_reason = "optimizer_not_converged"
            amplitude, t0, alpha, beta = result.x
        except Exception:
            # Fit failed, return initial estimates
            fit_converged = False
            failure_reason = "optimizer_error"
            amplitude, t0, alpha, beta = p0

    # Generate fitted curve
    fitted_curve = _gamma_variate_model(t, amplitude, t0, alpha, beta)

    # Compute fit quality metrics on UNSCALED curve (before AUC scaling)
    # R² and residual_rms measure shape fit quality and must not be
    # corrupted by dose-preservation scaling
    if remove_recirculation:
        eval_end = min(
            peak_idx + max(10, 2 * compute_aif_fwhm_frames(aif_curve, baseline_frames)),
            n_frames,
        )
        eval_start = baseline_frames
    else:
        eval_start = baseline_frames
        eval_end = n_frames

    ss_res = float(
        np.sum((curve_shifted[eval_start:eval_end] - fitted_curve[eval_start:eval_end]) ** 2)
    )
    ss_tot = float(
        np.sum(
            (curve_shifted[eval_start:eval_end] - np.mean(curve_shifted[eval_start:eval_end])) ** 2
        )
    )

    if ss_tot > EPSILON_CONCENTRATION:
        r_squared = 1.0 - ss_res / ss_tot
    else:
        r_squared = 0.0

    # Clip R² to valid range (can be negative for poor fits)
    r_squared = max(-1.0, min(1.0, r_squared))

    # Compute RMS residual on unscaled curve
    residuals = curve_shifted - fitted_curve
    residual_rms = float(np.sqrt(np.mean(residuals**2)))

    # Detect recirculation if requested
    recirculation_removed = False
    if remove_recirculation:
        recirc_start = _find_recirculation_start(
            curve_shifted, fitted_curve, baseline_frames
        )
        if recirc_start > 0:
            recirculation_removed = True
            # The fitted gamma-variate naturally excludes recirculation
            # since it only models first-pass bolus

    # Symmetric AUC window: when removing recirculation, integrate BOTH the
    # original and the fitted curve over the same first-pass window so the
    # ratio is apples-to-apples (else the asymmetric ratio misfires on any
    # AIF with non-trivial post-bolus tail — preloaded protocols always,
    # and any clean recirculation pattern).
    if remove_recirculation:
        cutoff_for_auc = min(
            peak_idx + max(10, 2 * compute_aif_fwhm_frames(aif_curve, baseline_frames)),
            n_frames,
        )
        auc_original = _compute_auc(
            curve_shifted[baseline_frames:cutoff_for_auc], tr,
        )
        auc_target = auc_original
        auc_fitted = _compute_auc(
            fitted_curve[baseline_frames:cutoff_for_auc], tr,
        )
        first_pass_n = max(0, cutoff_for_auc - baseline_frames)
    else:
        auc_original = _compute_auc(curve_shifted, tr)
        auc_target = auc_original
        auc_fitted = _compute_auc(fitted_curve, tr)
        first_pass_n = max(0, n_frames - baseline_frames)

    # Compute AUC ratio BEFORE preservation scaling so it reflects the
    # actual shape mismatch between the gamma fit and the empirical
    # first-pass area. With preserve_auc=True (production default), scaling
    # forces auc_fitted == auc_target by construction, which would make a
    # post-scaling ratio mechanically 1.0 and useless as a QC metric.
    if auc_original > EPSILON_CONCENTRATION:
        auc_ratio = auc_fitted / auc_original
    else:
        auc_ratio = 1.0

    # Optionally scale to preserve AUC (first-pass or full, depending on mode)
    if preserve_auc and auc_fitted > EPSILON_CONCENTRATION:
        scale_factor = auc_target / auc_fitted
        fitted_curve = fitted_curve * scale_factor
        amplitude = amplitude * scale_factor
        auc_fitted = auc_target  # by construction

    return GammaVariateFitResult(
        fitted_curve=fitted_curve,
        amplitude=amplitude,
        t0=t0,
        alpha=alpha,
        beta=beta,
        auc_original=auc_original,
        auc_fitted=auc_fitted,
        auc_ratio=auc_ratio,
        residual_rms=residual_rms,
        r_squared=r_squared,
        fit_converged=fit_converged,
        failure_reason=failure_reason,
        recirculation_removed=recirculation_removed,
        first_pass_n=first_pass_n,
    )


# =============================================================================
# Auto Gamma-Variate Decision Logic
# =============================================================================

# Hard gate thresholds — fit must satisfy ALL to be eligible
GAMMA_R2_THRESHOLD: float = 0.85
GAMMA_PEAK_RATIO_MIN: float = 0.90
GAMMA_PEAK_RATIO_MAX: float = 1.10
# AUC ratio bounds are QC provenance only, not a hard gate.
# Symmetric-window auc_ratio (see ``fit_gamma_variate``) makes this metric
# meaningful, but R² + peak_ratio + defect detection already cover the
# pathology this band would catch.
GAMMA_AUC_RATIO_MIN: float = 0.85
GAMMA_AUC_RATIO_MAX: float = 1.15
# Soft warning band: auc_ratio outside this is logged in QC but does not
# trigger fallback.
GAMMA_AUC_RATIO_SOFT_MIN: float = 0.70
GAMMA_AUC_RATIO_SOFT_MAX: float = 1.30
# Minimum first-pass samples for gamma fitting to be well-posed.
GAMMA_MIN_FIRST_PASS_SAMPLES: int = 8

# Defect detection thresholds
GAMMA_SPR_THRESHOLD: float = 0.25
GAMMA_NOISE_RATIO_THRESHOLD: float = 2.5
GAMMA_NEG_TAIL_MIN_FRAMES: int = 3
GAMMA_NEG_TAIL_FRACTION: float = 0.05


@dataclass(frozen=True)
class GammaAutoDecision:
    """Result of the auto gamma-variate decision.

    Attributes
    ----------
    apply : bool
        Whether to apply the gamma-variate fit.
    reason : str
        Human-readable reason for the decision.
    defect_detected : str or None
        Type of defect detected: "recirculation",
        "noise_and_negative_tail", or None.
    """

    apply: bool
    reason: str
    defect_detected: Optional[str] = None


def should_apply_gamma_fit(
    original_aif: npt.NDArray[Any],
    fit_result: GammaVariateFitResult,
    baseline_frames: int = 5,
) -> tuple["GammaAutoDecision", dict[str, Any]]:
    """Decide whether to apply gamma-variate fitting to the AIF.

    Uses hard gates (fit quality) and defect triggers (recirculation,
    noise + negative tail) to decide. The gamma-variate is applied only
    when the fit is good AND a defect is detected that gamma fitting
    would correct.

    Hard gates (must all pass for eligibility):
        - ``fit_converged``
        - ``first_pass_n >= GAMMA_MIN_FIRST_PASS_SAMPLES`` (8 samples).
          Below this, gamma fitting is ill-posed regardless of metrics.
        - ``r_squared >= GAMMA_R2_THRESHOLD``
        - ``peak_ratio in [GAMMA_PEAK_RATIO_MIN, GAMMA_PEAK_RATIO_MAX]``

    AUC ratio is NOT a hard gate; it is reported in ``metrics["auc_ratio"]``
    for QC provenance, with ``metrics["auc_ratio_warning"]`` set when the
    symmetric-window ratio falls outside ``[GAMMA_AUC_RATIO_SOFT_MIN,
    GAMMA_AUC_RATIO_SOFT_MAX]``. R² + peak_ratio + defect detection already
    cover the pathology a hard AUC gate would catch.

    Parameters
    ----------
    original_aif : np.ndarray
        1D original AIF concentration curve (before gamma fitting).
    fit_result : GammaVariateFitResult
        Result from ``fit_gamma_variate()``.
    baseline_frames : int
        Number of pre-bolus baseline frames.

    Returns
    -------
    tuple[GammaAutoDecision, dict[str, Any]]
        Decision and detailed metrics for QC logging.
    """
    metrics: dict[str, Any] = {}

    # --- Hard gates ---
    if not fit_result.fit_converged:
        return (
            GammaAutoDecision(apply=False, reason="Fit did not converge"),
            metrics,
        )

    # Eligibility prerequisite: gamma fitting is ill-posed below ~8 first-pass
    # samples.
    metrics["first_pass_n"] = fit_result.first_pass_n
    if fit_result.first_pass_n < GAMMA_MIN_FIRST_PASS_SAMPLES:
        return (
            GammaAutoDecision(
                apply=False,
                reason=(
                    f"insufficient first-pass samples for gamma fit "
                    f"({fit_result.first_pass_n} < {GAMMA_MIN_FIRST_PASS_SAMPLES})"
                ),
            ),
            metrics,
        )

    if fit_result.r_squared < GAMMA_R2_THRESHOLD:
        metrics["r_squared"] = fit_result.r_squared
        return (
            GammaAutoDecision(
                apply=False,
                reason=f"R2={fit_result.r_squared:.3f} < {GAMMA_R2_THRESHOLD}",
            ),
            metrics,
        )

    peak_orig = float(np.max(original_aif))
    peak_fit = float(np.max(fit_result.fitted_curve))
    if peak_orig > EPSILON_CONCENTRATION:
        peak_ratio = peak_fit / peak_orig
    else:
        peak_ratio = 1.0
    metrics["peak_ratio"] = peak_ratio

    if not (GAMMA_PEAK_RATIO_MIN <= peak_ratio <= GAMMA_PEAK_RATIO_MAX):
        return (
            GammaAutoDecision(
                apply=False,
                reason=f"Peak ratio {peak_ratio:.3f} outside [{GAMMA_PEAK_RATIO_MIN}, {GAMMA_PEAK_RATIO_MAX}]",
            ),
            metrics,
        )

    # AUC ratio: reported for QC provenance with a soft warning band, but
    # NOT a hard gate. R² + peak_ratio + defect detection already cover the
    # pathology this gate would catch.
    auc_ratio = fit_result.auc_ratio
    metrics["auc_ratio"] = auc_ratio
    metrics["auc_ratio_warning"] = not (
        GAMMA_AUC_RATIO_SOFT_MIN <= auc_ratio <= GAMMA_AUC_RATIO_SOFT_MAX
    )

    # --- Defect detection ---
    n = len(original_aif)
    peak_idx = int(np.argmax(original_aif))
    baseline_mean = float(np.mean(original_aif[:baseline_frames])) if baseline_frames > 0 else 0.0
    peak_val = peak_orig - baseline_mean  # baseline-corrected peak

    # 1) Recirculation defect: secondary peak ratio > threshold AND is local max
    # Look in post-first-pass window
    fwhm = compute_aif_fwhm_frames(original_aif, baseline_frames)
    recirc_start = peak_idx + max(fwhm, 5)
    spr = 0.0
    spr_is_local_max = False

    if recirc_start < n - 1 and peak_val > EPSILON_CONCENTRATION:
        tail = original_aif[recirc_start:] - baseline_mean
        if len(tail) > 2:
            secondary_peak_idx_local = int(np.argmax(tail))
            secondary_peak_val = float(tail[secondary_peak_idx_local])
            spr = secondary_peak_val / peak_val if peak_val > EPSILON_CONCENTRATION else 0.0

            # Check if it's a true local maximum (derivative sign change)
            if 0 < secondary_peak_idx_local < len(tail) - 1:
                left_slope = tail[secondary_peak_idx_local] - tail[secondary_peak_idx_local - 1]
                right_slope = tail[secondary_peak_idx_local + 1] - tail[secondary_peak_idx_local]
                spr_is_local_max = left_slope > 0 and right_slope < 0

    metrics["spr"] = spr
    metrics["spr_is_local_max"] = spr_is_local_max
    recirculation_defect = spr > GAMMA_SPR_THRESHOLD and spr_is_local_max

    # 2) Noise defect: 2nd derivative L2 norm ratio (fitted vs original) > threshold
    if n > 4:
        d2_orig = np.diff(original_aif, n=2)
        d2_fit = np.diff(fit_result.fitted_curve, n=2)
        l2_orig = float(np.sqrt(np.sum(d2_orig**2)))
        l2_fit = float(np.sqrt(np.sum(d2_fit**2)))
        noise_ratio = l2_orig / l2_fit if l2_fit > EPSILON_CONCENTRATION else 0.0
    else:
        noise_ratio = 0.0
    metrics["noise_ratio"] = noise_ratio
    noise_defect = noise_ratio > GAMMA_NOISE_RATIO_THRESHOLD

    # 3) Negative tail defect: >N frames below zero AND min < -fraction of peak
    if peak_idx < n - 1 and peak_val > EPSILON_CONCENTRATION:
        tail_full = original_aif[peak_idx:] - baseline_mean
        neg_frames = int(np.sum(tail_full < 0))
        neg_min = float(np.min(tail_full))
        neg_min_fraction = abs(neg_min) / peak_val
    else:
        neg_frames = 0
        neg_min_fraction = 0.0
    metrics["neg_tail_frames"] = neg_frames
    metrics["neg_tail_min_fraction"] = neg_min_fraction
    negative_tail_defect = (
        neg_frames >= GAMMA_NEG_TAIL_MIN_FRAMES
        and neg_min_fraction > GAMMA_NEG_TAIL_FRACTION
    )

    # --- Decision ---
    if recirculation_defect:
        return (
            GammaAutoDecision(
                apply=True,
                reason=f"Recirculation detected (SPR={spr:.3f}, local max)",
                defect_detected="recirculation",
            ),
            metrics,
        )

    if noise_defect and negative_tail_defect:
        return (
            GammaAutoDecision(
                apply=True,
                reason=(
                    f"Noise (ratio={noise_ratio:.2f}) and negative tail "
                    f"({neg_frames} frames, min={neg_min_fraction:.3f}*peak)"
                ),
                defect_detected="noise_and_negative_tail",
            ),
            metrics,
        )

    return (
        GammaAutoDecision(
            apply=False,
            reason="No defect detected -- AIF is clean",
        ),
        metrics,
    )
