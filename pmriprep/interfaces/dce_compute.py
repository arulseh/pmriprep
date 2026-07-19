"""DCE map computation interface for pharmacokinetic modeling.

This module implements the main Nipype interface for computing DCE
pharmacokinetic parameter maps (Ktrans, ve, vp, kep) from concentration
curves and AIF using various PK models.
"""

import logging
from functools import partial
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional

import numpy as np
import numpy.typing as npt

from nipype.interfaces.base import (  # type: ignore[import-untyped]
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    isdefined,
    traits,
)

from pmriprep.utils.dce_constants import (
    ACQUISITION_LIMITED_OVERSHOOT_THRESHOLD,
    EPSILON_RATIO,
    EPSILON_VE,
    DEFAULT_TSNR_THRESHOLD,
    FP_BOUNDS,
    KEP_BOUNDS,
    KTRANS_BOUNDS,
    MIN_ENHANCEMENT_MM,
    PS_BOUNDS,
    VE_BOUNDS,
    VP_BOUNDS,
)
from pmriprep.utils.dce_voxel_prep import prepare_voxel_for_fitting

logger = logging.getLogger(__name__)


def _is_degenerate_dce_fit(
    model_name: str,
    params: dict[str, float],
    is_constrained: bool,
) -> bool:
    """Return whether a fitted DCE parameter set rails physiological bounds."""
    ktrans = params.get("Ktrans", 0.0)
    ve = params.get("ve", 0.0)
    vp = params.get("vp", 0.0)
    ktrans_low = ktrans < 0 if is_constrained else ktrans <= 0
    if model_name == "2cxm":
        fp = params.get("Fp", 0.0)
        ps = params.get("PS", 0.0)
        return (
            fp <= 0
            or fp >= FP_BOUNDS.max
            or ps < 0
            or ps >= PS_BOUNDS.max
            or ve <= 0
            or ve >= VE_BOUNDS.max
            or vp <= 0
            or vp >= VP_BOUNDS.max
        )
    if model_name == "patlak":
        return (
            ktrans < 0
            or ktrans >= KTRANS_BOUNDS.max
            or (not is_constrained and (vp <= 0 or vp >= VP_BOUNDS.max))
        )
    if model_name == "extended_tofts":
        return (
            ktrans_low
            or ktrans >= KTRANS_BOUNDS.max
            or ve <= 0
            or ve >= VE_BOUNDS.max
            or (not is_constrained and (vp <= 0 or vp >= VP_BOUNDS.max))
        )
    return (
        ktrans <= 0
        or ktrans >= KTRANS_BOUNDS.max
        or ve <= 0
        or ve >= VE_BOUNDS.max
    )


def _scatter_dce_uncertainties(
    maps: "_DCEParameterMaps",
    coords: tuple[int, int, int],
    model_name: str,
    uncertainties: dict[str, float],
) -> None:
    """Scatter model-applicable posterior uncertainties with AIF inflation."""
    x, y, z = coords
    maps.ktrans_std[x, y, z] = uncertainties.get("Ktrans", 0.0) * (
        maps.sigma_factor.get("Ktrans", 1.0)
    )
    if model_name in ("tofts", "extended_tofts", "2cxm"):
        maps.ve_std[x, y, z] = uncertainties.get("ve", 0.0) * (
            maps.sigma_factor.get("ve", 1.0)
        )
    if model_name in ("extended_tofts", "patlak", "2cxm"):
        maps.vp_std[x, y, z] = uncertainties.get("vp", 0.0) * (
            maps.sigma_factor.get("vp", 1.0)
        )
    if model_name == "2cxm":
        maps.fp_std[x, y, z] = uncertainties.get("Fp", 0.0) * (
            maps.sigma_factor.get("Fp", 1.0)
        )
        maps.ps_std[x, y, z] = uncertainties.get("PS", 0.0) * (
            maps.sigma_factor.get("PS", 1.0)
        )


def _scatter_dce_fit_artifacts(
    maps: "_DCEParameterMaps",
    coords: tuple[int, int, int],
    result: dict[str, Any],
) -> None:
    """Scatter fitted-curve, bound-indicator, and quality-label artifacts."""
    x, y, z = coords
    fitted_curve = result.get("fitted_curve")
    if maps.fitted_curves is not None and fitted_curve is not None:
        maps.fitted_curves[x, y, z, :] = fitted_curve.astype(np.float32)
    maps.vp_at_bound[x, y, z] = float(result.get("vp_at_bound", 0.0))
    quality_flag = result.get("quality_flag", "good")
    maps.quality_flag[x, y, z] = maps.quality_flag_codes.get(quality_flag, 1)


def _scatter_dce_parameters(
    maps: "_DCEParameterMaps",
    coords: tuple[int, int, int],
    model_name: str,
    params: dict[str, float],
    r_squared: float,
) -> None:
    """Scatter a non-degenerate model's applicable fitted parameters."""
    x, y, z = coords
    maps.ktrans[x, y, z] = params.get("Ktrans", 0.0)
    if model_name in ("tofts", "extended_tofts", "2cxm"):
        maps.ve[x, y, z] = params.get("ve", 0.0)
        maps.kep[x, y, z] = params.get("kep", 0.0)
    maps.r_squared[x, y, z] = r_squared
    if model_name in ("extended_tofts", "patlak", "2cxm"):
        maps.vp[x, y, z] = params.get("vp", 0.0)
    if model_name == "2cxm":
        maps.fp[x, y, z] = params.get("Fp", 0.0)
        maps.ps[x, y, z] = params.get("PS", 0.0)
        maps.extraction_fraction[x, y, z] = params.get("E", 0.0)


def _scatter_degenerate_dce_fit(
    maps: "_DCEParameterMaps",
    coords: tuple[int, int, int],
    model_name: str,
    is_auto: bool,
    r_squared: float,
) -> None:
    """Scatter the model-applicable sentinels for a rejected fit."""
    x, y, z = coords
    fill_value = np.nan if is_auto else 0.0
    maps.ktrans[x, y, z] = fill_value
    if model_name in ("tofts", "extended_tofts", "2cxm"):
        maps.ve[x, y, z] = fill_value
        maps.kep[x, y, z] = fill_value
    if model_name in ("extended_tofts", "patlak", "2cxm"):
        maps.vp[x, y, z] = fill_value
    maps.r_squared[x, y, z] = r_squared


def _compute_vp_prior_departure(
    constraint_mode: Any,
    inference: str,
    model_name: str,
    voxel_prior: Optional[float],
    fitted_vp: float,
    relative_uncertainty: float,
) -> Optional[float]:
    """Return the standardized Bayesian departure from a SOFT DSC vp prior."""
    from pmriprep.utils.dsc_cbv_constraint import ConstraintMode

    if (
        constraint_mode != ConstraintMode.SOFT
        or inference == "nlls"
        or model_name not in ("extended_tofts", "patlak", "2cxm")
        or voxel_prior is None
        or voxel_prior <= 0
    ):
        return None
    prior_sigma = max(voxel_prior * relative_uncertainty, 0.001)
    return abs(fitted_vp - voxel_prior) / prior_sigma


def _accumulate_mcmc_diagnostics(
    result: dict[str, Any],
    rhat_lists: dict[str, list[float]],
    ess_lists: dict[str, list[float]],
) -> int:
    """Accumulate one voxel's MCMC diagnostics and return its count increment."""
    voxel_rhat = result.get("rhat", {})
    if not voxel_rhat:
        return 0
    for parameter, value in voxel_rhat.items():
        rhat_lists.setdefault(parameter, []).append(value)
    for parameter, value in result.get("ess", {}).items():
        ess_lists.setdefault(parameter, []).append(value)
    return 1


def _account_retained_dce_fit(
    voxel: "VoxelInput",
    result: dict[str, Any],
    constraint_mode: Any,
) -> "_DCEFitAccountingUpdate":
    """Return convergence and constraint counter increments for one retained fit."""
    from pmriprep.utils.dsc_cbv_constraint import ConstraintMode

    constrained = bool(result.get("constrained", False))
    unconstrained_fallback = bool(
        not constrained
        and voxel.voxel_vp is not None
        and constraint_mode != ConstraintMode.SKIP
    )
    converged = bool(result["converged"])
    return _DCEFitAccountingUpdate(
        int(converged),
        int(not converged),
        int(constrained),
        int(unconstrained_fallback),
        float(result["r_squared"]) if converged else None,
    )


def _summarize_mcmc_diagnostics(
    n_mcmc_voxels: int,
    rhat_lists: dict[str, list[float]],
    ess_lists: dict[str, list[float]],
) -> Optional[dict[str, Any]]:
    """Summarize accumulated MCMC convergence diagnostics."""
    if n_mcmc_voxels <= 0:
        return None
    from pmriprep.models.dce.bayesian import QC_ESS_MIN, QC_RHAT_MAX

    summary: dict[str, Any] = {"n_mcmc_voxels": n_mcmc_voxels}
    for parameter, values in rhat_lists.items():
        array = np.array(values)
        summary[f"rhat_median_{parameter}"] = float(np.median(array))
        summary[f"rhat_frac_poor_{parameter}"] = float(
            np.mean(array >= QC_RHAT_MAX)
        )
    for parameter, values in ess_lists.items():
        array = np.array(values)
        summary[f"ess_median_{parameter}"] = float(np.median(array))
        summary[f"ess_frac_low_{parameter}"] = float(np.mean(array < QC_ESS_MIN))
    return summary


def _summarize_vp_prior_departures(values: list[float]) -> dict[str, float]:
    """Summarize standardized departures from SOFT DSC vp priors."""
    if not values:
        return {}
    array = np.array(values)
    return {
        "z_departure_median": float(np.median(array)),
        "z_departure_p90": float(np.percentile(array, 90)),
        "frac_prior_dominated": float(np.mean(array < 0.5)),
        "frac_strong_departure": float(np.mean(array > 2.0)),
    }


def _summarize_dce_categorical_qc(
    is_auto: bool,
    candidate_names: list[str],
    model_selection: npt.NDArray[Any],
    model_selection_codes: dict[str, int],
    quality_flag: npt.NDArray[Any],
    fitting_mask: npt.NDArray[Any],
) -> dict[str, Any]:
    """Summarize model-selection and quality-label categorical maps."""
    summary: dict[str, Any] = {
        "model_selection": "auto" if is_auto else "fixed",
    }
    if is_auto:
        summary["candidate_models"] = candidate_names
        n_total = int(np.sum(model_selection > 0))
        for model_name in candidate_names:
            count = int(np.sum(model_selection == model_selection_codes[model_name]))
            summary[f"n_{model_name}_selected"] = count
            summary[f"{model_name}_fraction"] = count / max(1, n_total)

    quality_values = quality_flag[fitting_mask]
    summary["quality_n_good"] = int(np.sum(quality_values == 1))
    summary["quality_n_poor_convergence"] = int(np.sum(quality_values == 2))
    summary["quality_n_high_uncertainty"] = int(np.sum(quality_values == 3))
    summary["quality_n_unreliable_high_kep"] = int(np.sum(quality_values == 4))
    return summary


def _summarize_dce_fitted_parameters(
    fitting_mask: npt.NDArray[Any],
    ktrans: npt.NDArray[Any],
    ve: npt.NDArray[Any],
    vp: npt.NDArray[Any],
    fp: npt.NDArray[Any],
    ps: npt.NDArray[Any],
    save_ve: bool,
    save_vp: bool,
    save_2cxm: bool,
) -> dict[str, Any]:
    """Summarize parameter maps over voxels with a retained DCE fit."""
    fitted_mask = fitting_mask & np.isfinite(ktrans) & (ktrans > 0)
    if not np.any(fitted_mask):
        return {}

    ktrans_values = ktrans[fitted_mask]
    summary: dict[str, Any] = {
        "ktrans_mean": float(np.nanmean(ktrans_values)),
        "ktrans_median": float(np.nanmedian(ktrans_values)),
    }
    hist_counts, hist_edges = np.histogram(
        ktrans_values[np.isfinite(ktrans_values)], bins=50
    )
    summary["ktrans_histogram"] = hist_counts.tolist()
    summary["ktrans_histogram_edges"] = hist_edges.tolist()

    if save_ve:
        ve_values = ve[fitted_mask]
        finite_ve = np.isfinite(ve_values)
        if np.any(finite_ve):
            summary["ve_mean"] = float(np.nanmean(ve_values[finite_ve]))
            summary["ve_median"] = float(np.nanmedian(ve_values[finite_ve]))
            ve_hist_counts, ve_hist_edges = np.histogram(ve_values[finite_ve], bins=50)
            summary["ve_histogram"] = ve_hist_counts.tolist()
            summary["ve_histogram_edges"] = ve_hist_edges.tolist()

    if save_vp:
        vp_values = vp[fitted_mask]
        finite_vp = np.isfinite(vp_values)
        if np.any(finite_vp):
            summary["vp_mean"] = float(np.nanmean(vp_values[finite_vp]))
            summary["vp_median"] = float(np.nanmedian(vp_values[finite_vp]))

    if save_2cxm:
        fp_values = fp[fitted_mask]
        finite_fp = np.isfinite(fp_values) & (fp_values > 0)
        if np.any(finite_fp):
            summary["fp_median"] = float(np.nanmedian(fp_values[finite_fp]))
        ps_values = ps[fitted_mask]
        finite_ps = np.isfinite(ps_values) & (ps_values > 0)
        if np.any(finite_ps):
            summary["ps_median"] = float(np.nanmedian(ps_values[finite_ps]))

    return summary


def _fraction_vp_at_bound(
    mask: npt.NDArray[Any],
    vp_at_bound: npt.NDArray[Any],
) -> float:
    """Return the finite in-mask fraction of fits ending at the vp bound."""
    values = vp_at_bound[mask]
    finite = np.isfinite(values)
    if not np.any(finite):
        return 0.0
    return float(np.mean(values[finite] > 0.5))


def _fraction_fxl_risk(
    concentration: npt.NDArray[Any],
    fitting_mask: npt.NDArray[Any],
) -> float:
    """Return the fitted-voxel fraction exceeding the FXL concentration limit."""
    peak_concentration = np.nanmax(concentration, axis=-1)
    valid = fitting_mask & np.isfinite(peak_concentration)
    if not np.any(valid):
        return 0.0
    return float(np.mean(peak_concentration[valid] > 0.8))


def _summarize_dce_r2(values: list[float]) -> dict[str, float]:
    """Summarize retained-fit coefficients of determination."""
    if not values:
        return {}
    return {
        "r2_mean": float(np.mean(values)),
        "r2_std": float(np.std(values)),
        "r2_median": float(np.median(values)),
    }


def _summarize_dce_fit_outcomes(
    n_converged: int,
    n_failed: int,
    n_voxels: int,
    n_constrained: int,
    n_unconstrained_fallback: int,
    n_rejected_postfit: int,
    failure_reasons: dict[str, int],
) -> dict[str, Any]:
    """Assemble aggregate DCE fit outcome counters."""
    return {
        "n_converged": n_converged,
        "n_failed": n_failed,
        "convergence_rate": n_converged / max(1, n_voxels),
        "n_constrained": n_constrained,
        "n_unconstrained_fallback": n_unconstrained_fallback,
        "n_rejected_postfit": n_rejected_postfit,
        "failure_reasons": failure_reasons,
    }


def _zero_dce_background(
    values: npt.NDArray[Any],
    mask: npt.NDArray[np.bool_],
) -> npt.NDArray[Any]:
    """Zero spatial background while preserving all in-mask values."""
    spatial_mask = mask
    for _ in range(values.ndim - mask.ndim):
        spatial_mask = spatial_mask[..., np.newaxis]
    return np.where(spatial_mask, values, 0)


def _build_dce_qc_mask(
    fitting_mask: npt.NDArray[np.bool_],
    tsnr: npt.NDArray[Any],
    peak_enhancement: Optional[npt.NDArray[Any]],
    ktrans: npt.NDArray[Any],
    r_squared: npt.NDArray[Any],
    ve: npt.NDArray[Any],
    vp: npt.NDArray[Any],
    kep: npt.NDArray[Any],
    model_selection: npt.NDArray[Any],
    model_selection_codes: dict[str, int],
    is_auto: bool,
    model_name: str,
) -> npt.NDArray[np.uint8]:
    """Build the physiological and fit-quality acceptance mask."""
    qc_mask = fitting_mask.copy()
    qc_mask &= tsnr >= 10
    if peak_enhancement is not None:
        qc_mask &= peak_enhancement >= 10
    qc_mask &= (ktrans > 0) & (ktrans < 5.0)
    qc_mask &= r_squared >= 0.5

    is_patlak_voxel = model_selection == model_selection_codes.get("patlak", -1)
    if is_auto or model_name in ["extended_tofts", "2cxm"]:
        ve_safe = np.where(np.isfinite(ve), ve, 0.0)
        vp_safe = np.where(np.isfinite(vp), vp, 0.0)
        volume_ok = (ve_safe + vp_safe) < 0.95
        qc_mask &= volume_ok | is_patlak_voxel
    if is_auto or model_name != "patlak":
        kep_ok = np.where(np.isfinite(kep), kep > 0.001, True)
        qc_mask &= kep_ok
    result: npt.NDArray[np.uint8] = qc_mask.astype(np.uint8)
    return result


def _compute_dce_coefficient_of_variation(
    values: npt.NDArray[Any],
    standard_deviation: npt.NDArray[Any],
    mask: npt.NDArray[np.bool_],
    is_auto: bool,
) -> npt.NDArray[np.float32]:
    """Compute a masked DCE coefficient-of-variation map."""
    fill_value = np.nan if is_auto else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        coefficient = np.where(
            values > EPSILON_VE,
            standard_deviation / values,
            fill_value,
        ).astype(np.float32)
    result: npt.NDArray[np.float32] = _zero_dce_background(coefficient, mask)
    return result


def _compute_dce_credible_interval(
    values: npt.NDArray[Any],
    standard_deviation: npt.NDArray[Any],
    upper_bound: Optional[float] = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Compute normal-approximation 95% credible interval maps."""
    lower = np.maximum(values - 1.96 * standard_deviation, 0.0)
    upper = values + 1.96 * standard_deviation
    if upper_bound is not None:
        lower = np.minimum(lower, upper_bound)
        upper = np.clip(upper, 0.0, upper_bound)
    lower_result: npt.NDArray[np.float32] = lower.astype(np.float32)
    upper_result: npt.NDArray[np.float32] = upper.astype(np.float32)
    return lower_result, upper_result


def _write_dce_map(
    data: npt.NDArray[Any],
    path: str,
    *,
    affine: npt.NDArray[Any],
    header: Any,
    dtype: Any = np.float32,
) -> None:
    """Write a DCE-derived NIfTI map with an explicit data dtype."""
    import nibabel as nib  # type: ignore[import-untyped]

    cast_data = data.astype(dtype)
    local_header = header.copy()
    local_header.set_data_dtype(dtype)
    image = nib.Nifti1Image(cast_data, affine, local_header)
    nib.save(image, path)


def _write_dce_fitted_curves(
    data: npt.NDArray[Any],
    path: str,
    affine: npt.NDArray[Any],
) -> None:
    """Write fitted DCE curves with their existing affine-only header."""
    import nibabel as nib  # type: ignore[import-untyped]

    image = nib.Nifti1Image(data, affine)
    nib.save(image, path)


def _write_dce_model_free_maps(
    maps: "_DCEModelFreeMaps",
    generate_path: Callable[[str], str],
    write_map: Callable[..., None],
) -> dict[str, str]:
    """Write model-free DCE maps and return their interface result paths."""
    outputs = {
        "out_iauc60": generate_path("_IAUC60"),
        "out_iauc90": generate_path("_IAUC90"),
        "out_iauc120": generate_path("_IAUC120"),
        "out_peak_enhancement": generate_path("_peak_enhancement"),
        "out_time_to_peak": generate_path("_time_to_peak"),
        "out_ser": generate_path("_SER"),
        "out_wash_in_slope": generate_path("_wash_in_slope"),
        "out_wash_out_slope": generate_path("_wash_out_slope"),
        "out_bat": generate_path("_BAT"),
        "out_curve_type": generate_path("_curve_type"),
    }
    write_map(maps.iauc60, outputs["out_iauc60"])
    write_map(maps.iauc90, outputs["out_iauc90"])
    write_map(maps.iauc120, outputs["out_iauc120"])
    if maps.peak_enhancement is not None:
        write_map(maps.peak_enhancement, outputs["out_peak_enhancement"])
    write_map(maps.time_to_peak, outputs["out_time_to_peak"])
    write_map(maps.signal_enhancement_ratio, outputs["out_ser"])
    write_map(maps.wash_in_slope, outputs["out_wash_in_slope"])
    write_map(maps.wash_out_slope, outputs["out_wash_out_slope"])
    write_map(maps.bolus_arrival_time, outputs["out_bat"])
    write_map(maps.curve_type, outputs["out_curve_type"], dtype=np.int8)
    return outputs


def _write_dce_parameter_maps(
    maps: "_DCEParameterMaps",
    mask: npt.NDArray[np.bool_],
    save_ve: bool,
    save_vp: bool,
    generate_path: Callable[[str], str],
    write_map: Callable[..., None],
) -> dict[str, str]:
    """Write fitted DCE parameter maps and return applicable result paths."""
    outputs = {
        "out_ktrans": generate_path("_Ktrans"),
        "out_r2": generate_path("_R2"),
        "out_quality_flag": generate_path("_quality_flag"),
        "out_vp_at_bound_map": generate_path("_vpAtBound"),
    }
    write_map(maps.ktrans, outputs["out_ktrans"])
    write_map(maps.r_squared, outputs["out_r2"])
    write_map(maps.quality_flag, outputs["out_quality_flag"], dtype=np.uint8)
    write_map(
        _zero_dce_background(maps.vp_at_bound, mask),
        outputs["out_vp_at_bound_map"],
    )
    if save_ve:
        outputs["out_ve"] = generate_path("_ve")
        outputs["out_kep"] = generate_path("_kep")
        write_map(maps.ve, outputs["out_ve"])
        write_map(maps.kep, outputs["out_kep"])
    if save_vp:
        outputs["out_vp"] = generate_path("_vp")
        write_map(maps.vp, outputs["out_vp"])
    if maps.save_2cxm:
        outputs["out_fp"] = generate_path("_Fp")
        outputs["out_ps"] = generate_path("_PS")
        outputs["out_e"] = generate_path("_E")
        write_map(maps.fp, outputs["out_fp"])
        write_map(maps.ps, outputs["out_ps"])
        write_map(maps.extraction_fraction, outputs["out_e"])
    return outputs


def _validate_dce_time_axis(
    time_points: npt.ArrayLike,
    n_dce_frames: int,
    n_aif_samples: int,
) -> npt.NDArray[np.float64]:
    """Validate the uniformly sampled time axis required by DCE fitting.

    Parameters
    ----------
    time_points : array-like
        Acquisition times in minutes.
    n_dce_frames : int
        Number of frames in the DCE concentration series.
    n_aif_samples : int
        Number of samples in the arterial input function.

    Returns
    -------
    numpy.ndarray
        Validated one-dimensional time axis as ``float64``.

    Raises
    ------
    ValueError
        If the time axis is not one-dimensional, finite, length-matched,
        strictly increasing, or uniformly spaced.

    Notes
    -----
    The Tofts-family and 2CXM analytic convolution implementations use one
    constant sample interval. Nonuniform sampling is therefore rejected rather
    than silently evaluated using only the first interval.
    """
    axis = np.asarray(time_points, dtype=np.float64)
    if axis.ndim != 1:
        raise ValueError(
            f"DCE time points must be one-dimensional; got shape {axis.shape}."
        )
    if axis.size < 2:
        raise ValueError(
            f"DCE time points must contain at least two samples; got {axis.size}."
        )
    if axis.size != n_dce_frames or axis.size != n_aif_samples:
        raise ValueError(
            "DCE time-point, concentration-frame, and AIF lengths must match; "
            f"got {axis.size}, {n_dce_frames}, and {n_aif_samples}."
        )
    if not np.all(np.isfinite(axis)):
        raise ValueError("DCE time points must contain only finite values.")

    intervals = np.diff(axis)
    if np.any(intervals <= 0.0):
        raise ValueError("DCE time points must be strictly increasing.")
    if not np.allclose(intervals, intervals[0], rtol=1e-5, atol=1e-8):
        raise ValueError(
            "Nonuniform DCE time points are unsupported by the current "
            "analytic convolution models."
        )
    return axis


class VoxelInput(NamedTuple):
    """Typed container for per-voxel fitting input data."""

    x: int
    y: int
    z: int
    tissue_curve: npt.NDArray[Any]
    voxel_vp: Optional[float]


class BatchedChunkArgs(NamedTuple):
    """Typed container for batched NLLS chunk-worker arguments.

    Pickle-safe across ``ProcessPoolExecutor.map`` because it is defined
    at module top-level (a nested NamedTuple definition would raise
    PicklingError at runtime). Named fields replace a prior positional
    tuple so that adding fields (e.g., ``vp_uncertainty``) is
    compile-time-checked instead of positionally-dropped silently.
    """

    model_name: str
    tissue_curves: npt.NDArray[Any]
    aif: npt.NDArray[Any]
    time_points: npt.NDArray[Any]
    fixed_vp: Optional[npt.NDArray[Any]]
    vp_uncertainty: Optional[float]
    weights: Optional[npt.NDArray[Any]]
    save_fitted_curves: bool
    compute_uncertainties: bool


class _DCEFitPreparation(NamedTuple):
    """Validated DCE inputs and pre-BAT fitting-domain state."""

    concentration: npt.NDArray[Any]
    affine: npt.NDArray[Any]
    header: Any
    mask: npt.NDArray[Any]
    aif: npt.NDArray[Any]
    time_points: npt.NDArray[np.float64]
    n_timepoints: int
    selection_tr_seconds: float
    model_name: str
    is_auto: bool
    candidate_names: list[str]
    baseline_frames: int
    tsnr_threshold: float
    tsnr_map: npt.NDArray[Any]
    peak_enhancement: Optional[npt.NDArray[Any]]
    fitting_mask: npt.NDArray[np.bool_]
    inference: str
    mcmc_samples: int
    save_fitted_curves: bool


class _DCEBATAlignment(NamedTuple):
    """AIF curve and QC state after optional tissue-arrival alignment."""

    aif: npt.NDArray[Any]
    correction: dict[str, Any]


class _DCEVPConstraint(NamedTuple):
    """DSC-derived vp prior, diagnostics, and resulting model-selection state."""

    vp_map: Optional[npt.NDArray[Any]]
    mode: Any
    vp_uncertainty: float
    qc: dict[str, Any]
    model_name: str
    candidate_names: list[str]


class _DCEModelFreeMaps(NamedTuple):
    """Vectorized DCE semi-quantitative maps computed before PK fitting."""

    iauc60: npt.NDArray[Any]
    iauc90: npt.NDArray[Any]
    iauc120: npt.NDArray[Any]
    peak_enhancement: Optional[npt.NDArray[Any]]
    time_to_peak: npt.NDArray[Any]
    signal_enhancement_ratio: npt.NDArray[Any]
    wash_in_slope: npt.NDArray[Any]
    wash_out_slope: npt.NDArray[Any]
    bolus_arrival_time: npt.NDArray[Any]
    curve_type: npt.NDArray[Any]


class _DCEMotionWeighting(NamedTuple):
    """Optional phase-adaptive motion weights and their QC record."""

    weights: Optional[npt.NDArray[np.floating]]
    qc: dict[str, Any]


class _DCEParameterMaps(NamedTuple):
    """Allocated mutable PK, uncertainty, categorical, and fit-curve maps."""

    ktrans: npt.NDArray[Any]
    ve: npt.NDArray[Any]
    vp: npt.NDArray[Any]
    kep: npt.NDArray[Any]
    r_squared: npt.NDArray[Any]
    fp: npt.NDArray[Any]
    ps: npt.NDArray[Any]
    extraction_fraction: npt.NDArray[Any]
    quality_flag: npt.NDArray[Any]
    quality_flag_codes: dict[str, int]
    model_selection: npt.NDArray[Any]
    model_selection_codes: dict[str, int]
    delta_bic: npt.NDArray[Any]
    ktrans_std: npt.NDArray[Any]
    ve_std: npt.NDArray[Any]
    vp_std: npt.NDArray[Any]
    fp_std: npt.NDArray[Any]
    ps_std: npt.NDArray[Any]
    sigma_factor: dict[str, float]
    vp_at_bound: npt.NDArray[Any]
    fitted_curves: Optional[npt.NDArray[Any]]
    save_2cxm: bool


class _DCEFitDispatch(NamedTuple):
    """Voxel ordering and worker state shared by all DCE fitting paths."""

    mask_coords: tuple[npt.NDArray[Any], ...]
    voxel_data: list[VoxelInput]
    init_args: tuple[Any, ...]
    n_procs: int
    sloppy: bool
    random_seed: Optional[int]


class _DCEFitAccountingUpdate(NamedTuple):
    """Counter increments contributed by one retained DCE fit result."""

    n_converged: int
    n_failed: int
    n_constrained: int
    n_unconstrained_fallback: int
    r_squared: Optional[float]


# Import-time pickle-safety assertion (S1 from code review round 3):
# BatchedChunkArgs MUST be defined at module top-level for pickling across
# ProcessPoolExecutor boundaries. A nested definition (e.g. inside a
# function) would break multiprocessing silently. Enforce by pinning the
# qualified name.
assert BatchedChunkArgs.__module__ == __name__, (
    "BatchedChunkArgs must be defined at module top-level for pickle safety"
)


# =========================================================================
# Module-level worker functions for multiprocessing (must be picklable)
# =========================================================================
_worker_ctx: dict[str, Any] = {}


def _compute_motion_weights(
    fd_values: list[float],
    bolus_frame: int,
    n_frames: int,
    min_voxel_dim: float,
) -> npt.NDArray[np.floating]:
    """Compute per-frame motion weights using phase-adaptive FD thresholds.

    Weight function: w_i = (threshold / FD_i)^2 for FD_i > threshold, else 1.0
    Wash-in frames (bolus ± 4) use 2x stricter threshold.
    Threshold = min(voxel_dims) * 0.25 (scales with acquisition resolution).

    Parameters
    ----------
    fd_values : list of float
        Framewise displacement per volume (mm).
    bolus_frame : int
        Bolus arrival frame index.
    n_frames : int
        Number of time frames.
    min_voxel_dim : float
        Minimum voxel dimension (mm) for threshold scaling.

    Returns
    -------
    numpy.ndarray
        Per-frame weight array (1.0 = full weight, <1.0 = downweighted).

    """
    weights = np.ones(n_frames, dtype=np.float64)
    fd_arr = np.array(fd_values[:n_frames], dtype=np.float64)
    base_threshold = max(min_voxel_dim * 0.25, 0.1)  # floor at 0.1mm
    washin_end = min(bolus_frame + 4, n_frames - 1)

    for i in range(n_frames):
        if bolus_frame <= i <= washin_end:
            eff_thresh = base_threshold * 0.5  # 2x stricter during wash-in
        else:
            eff_thresh = base_threshold

        if fd_arr[i] > eff_thresh:
            weights[i] = (eff_thresh / fd_arr[i]) ** 2

    return weights


def _init_worker(
    aif: "np.ndarray[Any, Any]",
    time_points: "np.ndarray[Any, Any]",
    model_name: str,
    inference: str,
    mcmc_samples: int,
    baseline_frames: int,
    constraint_mode_str: str,
    constraint_vp_uncertainty: float,
    save_fitted_curves: bool = False,
    motion_weights: Optional[npt.NDArray[np.floating]] = None,
    sloppy: bool = False,
    random_seed: Optional[int] = None,
    candidate_names: Optional[list[str]] = None,
) -> None:
    """Initialize per-worker model and fitter objects (called once per process)."""
    from pmriprep.utils.parallel import limit_worker_threads
    limit_worker_threads()

    import warnings

    from pmriprep.models.dce import BayesianPKFitter
    from pmriprep.models.dce import (
        ToftsModel,
        ExtendedToftsModel,
        PatlakModel,
        TwoCompartmentExchangeModel,
    )
    from pmriprep.models.dce.bayesian import create_vp_prior_from_dsc
    from pmriprep.utils.dsc_cbv_constraint import ConstraintMode

    # Suppress Nelder-Mead RuntimeWarnings in worker processes
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Seed RNG in worker process for reproducibility — offset by PID
    # so parallel workers explore different random sequences
    if random_seed is not None:
        import os
        np.random.seed(random_seed + os.getpid() % (2**31))

    model_map: dict[str, type] = {
        "tofts": ToftsModel,
        "extended_tofts": ExtendedToftsModel,
        "patlak": PatlakModel,
        "2cxm": TwoCompartmentExchangeModel,
    }
    model = model_map[model_name]()

    fitter = None
    if inference != "nlls":
        fitter = BayesianPKFitter(model)

    # BIC model selection: instantiate all candidate models
    bic_candidates: dict[str, Any] = {}
    is_auto = bool(candidate_names)
    for cname in (candidate_names or []):
        if cname != model_name:
            bic_candidates[cname] = model_map[cname]()

    _worker_ctx["aif"] = aif
    _worker_ctx["time_points"] = time_points
    _worker_ctx["model"] = model
    _worker_ctx["model_name"] = model_name
    _worker_ctx["fitter"] = fitter
    _worker_ctx["inference"] = inference
    _worker_ctx["mcmc_samples"] = mcmc_samples
    _worker_ctx["baseline_frames"] = baseline_frames
    _worker_ctx["constraint_mode_str"] = constraint_mode_str
    _worker_ctx["constraint_vp_uncertainty"] = constraint_vp_uncertainty
    _worker_ctx["save_fitted_curves"] = save_fitted_curves
    _worker_ctx["motion_weights"] = motion_weights
    _worker_ctx["sloppy"] = sloppy
    _worker_ctx["bic_candidates"] = bic_candidates
    _worker_ctx["is_auto"] = is_auto
    _worker_ctx["model_map"] = model_map
    _worker_ctx["BayesianPKFitter"] = BayesianPKFitter
    _worker_ctx["create_vp_prior_from_dsc"] = create_vp_prior_from_dsc
    _worker_ctx["ConstraintMode"] = ConstraintMode


def _fit_single_voxel(
    voxel_data: VoxelInput,
) -> dict[str, Any]:
    """Fit a single voxel using the worker-local model/fitter.

    Parameters
    ----------
    voxel_data : VoxelInput
        VoxelInput with (x, y, z, tissue_curve, voxel_vp).

    Returns
    -------
    dict
        Dict with keys: coords, params, r_squared, converged, uncertainties,
        fitted_curve, quality_flag, failed, failure_reason, constrained.
    """
    BayesianPKFitter = _worker_ctx["BayesianPKFitter"]
    create_vp_prior_from_dsc = _worker_ctx["create_vp_prior_from_dsc"]
    ConstraintMode = _worker_ctx["ConstraintMode"]

    x, y, z, tissue_curve, voxel_vp = voxel_data

    model = _worker_ctx["model"]
    fitter = _worker_ctx["fitter"]
    aif = _worker_ctx["aif"]
    time_points = _worker_ctx["time_points"]
    inference = _worker_ctx["inference"]
    mcmc_samples = _worker_ctx["mcmc_samples"]
    baseline_frames = _worker_ctx["baseline_frames"]
    constraint_mode = ConstraintMode(_worker_ctx["constraint_mode_str"])
    constraint_vp_uncertainty = _worker_ctx["constraint_vp_uncertainty"]
    motion_weights = _worker_ctx["motion_weights"]
    sloppy = _worker_ctx.get("sloppy", False)
    is_auto = _worker_ctx.get("is_auto", False)

    # Validate voxel (skip concentration-based variation check — raw signal
    # variation is filtered upstream in ComputeDCEMaps)
    prep = prepare_voxel_for_fitting(
        tissue_curve, baseline_frames, skip_variation_check=True
    )
    if not prep.is_valid:
        return {
            "coords": (x, y, z),
            "failed": True,
            "failure_reason": prep.failure_reason,
        }

    # Use baseline-subtracted, clipped concentration for fitting
    prepared_curve = prep.concentration

    try:
        result = None
        constrained = False
        fixed_vp: Optional[float] = None
        vp_prior = None  # Hoisted for BIC winner MAP refinement
        delta_bic = np.nan

        # Determine constraint state
        if voxel_vp is not None and constraint_mode != ConstraintMode.SKIP:
            fixed_vp = voxel_vp

        # SOFT constraint: Bayesian with informative vp prior
        if (
            fixed_vp is not None
            and constraint_mode == ConstraintMode.SOFT
            and inference != "nlls"
        ):
            vp_prior = create_vp_prior_from_dsc(
                fixed_vp, relative_uncertainty=constraint_vp_uncertainty
            )
            if is_auto:
                # BIC mode: use NLLS constrained for fair comparison
                # MAP refinement happens post-BIC-selection
                if hasattr(model, "fit_constrained"):
                    result = model.fit_constrained(
                        prepared_curve, aif, time_points, fixed_vp=fixed_vp,
                        weights=motion_weights,
                    )
                # Flag constrained so all BIC candidates get constrained vp
                constrained = True
            elif inference == "vi":
                soft_fitter = BayesianPKFitter(
                    model, vp_prior_override=vp_prior
                )
                result = soft_fitter.fit_vi(
                    prepared_curve, aif, time_points,
                    weights=motion_weights,
                    sloppy=sloppy,
                )
                constrained = True
            elif inference == "map":
                soft_fitter = BayesianPKFitter(
                    model, vp_prior_override=vp_prior
                )
                result = soft_fitter.fit_map(
                    prepared_curve, aif, time_points,
                    compute_uncertainties=not sloppy,
                    weights=motion_weights,
                    sloppy=sloppy,
                )
                constrained = True
            elif inference == "mcmc":
                soft_fitter = BayesianPKFitter(
                    model, vp_prior_override=vp_prior
                )
                result = soft_fitter.fit_mcmc(
                    prepared_curve, aif, time_points, n_samples=mcmc_samples,
                    sloppy=sloppy, weights=motion_weights,
                )
                constrained = True

        # SOFT constraint with NLLS: tighten vp bounds around DSC value.
        # Caller owns constraint_vp_uncertainty: must be positive here (upstream
        # validation at the batched-API boundary raises ValueError on σ = 0.0).
        elif (
            fixed_vp is not None
            and constraint_mode == ConstraintMode.SOFT
            and inference == "nlls"
        ):
            sigma = constraint_vp_uncertainty
            vp_lo = max(fixed_vp * (1.0 - sigma), VP_BOUNDS.min)
            vp_hi = min(fixed_vp * (1.0 + sigma), VP_BOUNDS.max)
            result = model.fit(
                prepared_curve, aif, time_points,
                weights=motion_weights,
                vp_bounds=(vp_lo, vp_hi),
            )
            constrained = True

        # Unconstrained fallback
        if result is None:
            if is_auto or inference == "nlls":
                # is_auto: NLLS for BIC comparison; MAP applied to winner later
                result = model.fit(
                    prepared_curve, aif, time_points,
                    weights=motion_weights,
                )
            elif inference == "vi" and fitter is not None:
                result = fitter.fit_vi(
                    prepared_curve, aif, time_points,
                    weights=motion_weights,
                    sloppy=sloppy,
                )
            elif inference == "map" and fitter is not None:
                result = fitter.fit_map(
                    prepared_curve, aif, time_points,
                    compute_uncertainties=not sloppy,
                    weights=motion_weights,
                    sloppy=sloppy,
                )
            elif inference == "mcmc" and fitter is not None:
                result = fitter.fit_mcmc(
                    prepared_curve, aif, time_points,
                    n_samples=mcmc_samples,
                    sloppy=sloppy, weights=motion_weights,
                )
            else:
                result = model.fit(
                    prepared_curve, aif, time_points,
                    weights=motion_weights,
                )

        # N-model BIC selection (auto mode)
        bic_candidates = _worker_ctx.get("bic_candidates", {})

        if is_auto and bic_candidates:
            all_results = [result]  # primary model already fitted

            for cname, cmodel in bic_candidates.items():
                try:
                    if constrained and fixed_vp is not None and hasattr(cmodel, "fit_constrained"):
                        cand_result = cmodel.fit_constrained(
                            prepared_curve, aif, time_points, fixed_vp=fixed_vp,
                            weights=motion_weights,
                        )
                    elif constrained and not hasattr(cmodel, "fit_constrained"):
                        # Model has no vp to constrain — fit unconstrained
                        cand_result = cmodel.fit(
                            prepared_curve, aif, time_points,
                            weights=motion_weights,
                        )
                    elif not constrained:
                        cand_result = cmodel.fit(
                            prepared_curve, aif, time_points,
                            weights=motion_weights,
                        )
                    else:
                        continue
                    if cand_result.converged and np.isfinite(cand_result.bic):
                        all_results.append(cand_result)
                except Exception:
                    logger.debug(
                        "BIC candidate %s failed at (%d,%d,%d)",
                        cname, x, y, z, exc_info=True,
                    )
                    continue

            # Select winner = lowest BIC
            best = min(
                all_results,
                key=lambda r: r.bic if np.isfinite(r.bic) else np.inf,
            )

            # Delta-BIC: positive = winner's margin over runner-up
            sorted_bics = sorted(
                r.bic for r in all_results if np.isfinite(r.bic)
            )
            delta_bic = (
                sorted_bics[1] - sorted_bics[0]
                if len(sorted_bics) > 1
                else np.nan
            )

            # Require strong BIC evidence to upgrade from simplest model.
            # ΔBIC ≥ 6 = "strong evidence".
            # Prevents noise-driven model selection in low-enhancement
            # voxels where complex models have unidentifiable parameters.
            _MIN_DELTA_BIC_UPGRADE = 6.0
            simplest = min(all_results, key=lambda r: r.n_parameters)
            if (best.model_name != simplest.model_name
                    and simplest.converged
                    and delta_bic < _MIN_DELTA_BIC_UPGRADE):
                logger.debug(
                    "delta-BIC guard: %s -> %s (ΔBIC=%.2f < %.1f)"
                    " at (%d,%d,%d)",
                    best.model_name, simplest.model_name, delta_bic,
                    _MIN_DELTA_BIC_UPGRADE, x, y, z,
                )
                best = simplest

            # Bayesian refinement of BIC winner
            if inference == "vi":
                model_map = _worker_ctx.get("model_map", {})
                winner_model_cls = model_map.get(best.model_name)
                if winner_model_cls is not None:
                    winner_model = winner_model_cls()
                    vp_prior_kw = (
                        {"vp_prior_override": vp_prior}
                        if vp_prior is not None else {}
                    )
                    winner_fitter = BayesianPKFitter(winner_model, **vp_prior_kw)
                    try:
                        best = winner_fitter.fit_vi(
                            prepared_curve, aif, time_points,
                            weights=motion_weights, sloppy=sloppy,
                        )
                    except Exception:
                        logger.debug(
                            "VI refinement failed at (%d,%d,%d), keeping NLLS",
                            x, y, z, exc_info=True,
                        )

            elif inference == "map":
                model_map = _worker_ctx.get("model_map", {})
                winner_model_cls = model_map.get(best.model_name)
                if winner_model_cls is not None:
                    winner_model = winner_model_cls()
                    vp_prior_kw = (
                        {"vp_prior_override": vp_prior}
                        if vp_prior is not None else {}
                    )
                    winner_fitter = BayesianPKFitter(winner_model, **vp_prior_kw)
                    try:
                        best = winner_fitter.fit_map(
                            prepared_curve, aif, time_points,
                            compute_uncertainties=not sloppy,
                            weights=motion_weights, sloppy=sloppy,
                        )
                    except Exception:
                        logger.debug(
                            "MAP refinement failed at (%d,%d,%d), keeping NLLS",
                            x, y, z, exc_info=True,
                        )

            elif inference == "mcmc":
                model_map = _worker_ctx.get("model_map", {})
                winner_model_cls = model_map.get(best.model_name)
                if winner_model_cls is not None:
                    winner_model = winner_model_cls()
                    vp_prior_kw = (
                        {"vp_prior_override": vp_prior}
                        if vp_prior is not None else {}
                    )
                    winner_fitter = BayesianPKFitter(winner_model, **vp_prior_kw)
                    try:
                        best = winner_fitter.fit_mcmc(
                            prepared_curve, aif, time_points,
                            n_samples=mcmc_samples, sloppy=sloppy,
                            weights=motion_weights,
                        )
                    except Exception:
                        logger.debug(
                            "MCMC refinement failed at (%d,%d,%d), keeping NLLS",
                            x, y, z, exc_info=True,
                        )

            result = best

        # Determine effective model name for this voxel
        selected_model_name = result.model_name

        # Build output dict
        effective_model = selected_model_name
        has_vp = effective_model in ("extended_tofts", "patlak", "2cxm")
        uncertainties = {}
        if result.has_uncertainty():
            uncertainties["Ktrans"] = result.get_uncertainty("Ktrans", 0.0)
            if effective_model not in ("patlak",):
                uncertainties["ve"] = result.get_uncertainty("ve", 0.0)
            if has_vp:
                uncertainties["vp"] = result.get_uncertainty("vp", 0.0)
            # 2CXM-specific: Fp and PS uncertainties
            if effective_model == "2cxm":
                uncertainties["Fp"] = result.get_uncertainty("Fp", 0.0)
                uncertainties["PS"] = result.get_uncertainty("PS", 0.0)

        fitted_curve = None
        if _worker_ctx.get("save_fitted_curves") and result.fitted_curve is not None:
            fitted_curve = result.fitted_curve

        # Clamp the reported kep and flag the non-identifiable high-kep corner,
        # matching the batched-assembly path so the QC flag has full parity.
        params = dict(result.parameters)
        kep_railed = _compute_derived_params(params, effective_model)
        base_quality = getattr(result, "quality_flag", "good")
        quality_flag = (
            "unreliable_high_kep"
            if kep_railed and base_quality == "good"
            else base_quality
        )

        return {
            "coords": (x, y, z),
            "params": params,
            "r_squared": result.r_squared,
            "converged": result.converged,
            "uncertainties": uncertainties,
            "fitted_curve": fitted_curve,
            "quality_flag": quality_flag,
            "rhat": getattr(result, "rhat", {}),
            "ess": getattr(result, "ess", {}),
            "failed": False,
            "failure_reason": None,
            "constrained": constrained,
            "selected_model_name": selected_model_name,
            "delta_bic": float(delta_bic) if is_auto else np.nan,
        }

    except Exception:
        return {
            "coords": (x, y, z),
            "failed": True,
            "failure_reason": "exception",
        }


_MODEL_HAS_VP = {"extended_tofts", "patlak", "2cxm"}

#: Minimum enhancing-tumor voxels (after intersecting with the PK fitting mask)
#: needed to estimate a tissue arrival time for AIF BAT pre-alignment. Below
#: this the reference is too small to be reliable and BAT no-ops.
BAT_TUMOR_REFERENCE_MIN_VOXELS = 10


def _resolve_bat_tissue_reference(
    tumor_mask_file: str,
    fitting_mask: npt.NDArray[Any],
    conc_data: npt.NDArray[Any],
) -> tuple[Optional[npt.NDArray[Any]], dict[str, Any]]:
    """Select the enhancing-tumor ROI used as the BAT tissue-arrival reference.

    Parameters
    ----------
    tumor_mask_file : str
        Path to the enhancing-tumor mask (HD-GLIO label 2) resampled to the DCE
        grid, or an empty string when tumor segmentation is unavailable.
    fitting_mask : numpy.ndarray
        Boolean PK fitting mask (brain & tSNR & enhancement) on the DCE grid.
    conc_data : numpy.ndarray
        4-D concentration-time data; only its spatial shape is used here.

    Returns
    -------
    tuple of (numpy.ndarray or None, dict)
        ``(roi_mask, info)``. ``roi_mask`` is the boolean reference ROI
        (enhancing tumor intersected with the fitting mask) when a reliable
        reference is available, else ``None``. When ``None``, ``info`` carries
        the no-op ``reason``; otherwise ``info`` carries the reference voxel
        count.

    Notes
    -----
    The enhancing tumor is a tissue-arrival *proxy* (leaky, delayed tissue),
    not arterial ground truth; the BAT correction targets the AIF-to-tissue
    transit interval. The mask is intersected with the fitting mask so the
    reference is enhancing tumor that also passed the tSNR/enhancement gate,
    dropping necrotic and partial-volume voxels a nearest-neighbour resample
    clips in. Without a tumor segmentation there is no reliable reference
    (whole-brain arrival is too noisy and white matter's arrival is not the
    tumor's), so the caller no-ops.
    """
    import nibabel as nib  # type: ignore[import-untyped]

    if not tumor_mask_file:
        return None, {"applied": False, "reason": "no_tumor_reference"}
    enhancing = nib.load(tumor_mask_file).get_fdata() > 0  # type: ignore[attr-defined]
    if enhancing.shape != conc_data.shape[:3]:
        return None, {"applied": False, "reason": "tumor_reference_grid_mismatch"}
    roi_mask = enhancing & fitting_mask
    n_roi = int(roi_mask.sum())
    if n_roi < BAT_TUMOR_REFERENCE_MIN_VOXELS:
        return None, {
            "applied": False,
            "reason": "tumor_reference_too_small",
            "n_reference_voxels": n_roi,
        }
    return roi_mask, {"n_reference_voxels": n_roi}


#: K2 leakage correction metric keys to extract from preload QC dict.
#:
#: These keys are part of the producer-shape ``preload_qc`` dict that the
#: ``--skip-dsc`` resume checkpoint (``work/<wf>/dsc_resume.json``)
#: must round-trip verbatim. If you add a new key here, also add it to
#: the oracle in ``tests/unit/test_resume_qc_round_trip_consumer_keys.py``
#: so the round-trip parity guard catches future schema drift.
_K2_QC_KEYS = {
    "k2_mean", "k2_std", "k2_max", "k2_median",
    "percent_leaky_voxels", "percent_severe_leakage",
    "percent_above_clinical_max",
    "n_brain_voxels", "auc_mean",
    "cbv_positive_tail_fraction",
}


def _normalize_param_name(pname: str) -> str:
    """Normalize batched result param name to output format.

    Parameters
    ----------
    pname : str
        Raw parameter name from batched result (e.g. 'ktrans', 'fp', 'ps').

    Returns
    -------
    str
        Normalized name matching output format.
    """
    if pname == "ktrans":
        return "Ktrans"
    if pname == "fp":
        return "Fp"
    if pname == "ps":
        return "PS"
    return pname


def _compute_derived_params(
    params_out: dict[str, float],
    model_name: str,
) -> bool:
    """Compute derived parameters (kep, E, Ktrans) in place.

    The reported kep is clamped to the physiological cap ``KEP_BOUNDS.max``.
    A raw kep = Ktrans/ve above that cap marks the kep->inf / ve->0 corner
    where the reversible (ve*Cp) and vascular (vp*Cp) forms coincide and kep
    is intrinsically non-identifiable; the exact value there is not
    trustworthy, so it is capped and the voxel is reported as low-confidence.

    Parameters
    ----------
    params_out : dict
        Parameter dict to augment with derived values.
    model_name : str
        Model name for determining which derivations to apply.

    Returns
    -------
    bool
        True if the raw kep exceeded ``KEP_BOUNDS.max`` (the reported kep was
        clamped), flagging the voxel as low-confidence.
    """
    kep_railed = False
    if model_name in ("tofts", "extended_tofts"):
        ve_val = params_out.get("ve", EPSILON_VE)
        raw_kep = params_out.get("Ktrans", 0.0) / max(ve_val, EPSILON_VE)
        kep_railed = raw_kep > KEP_BOUNDS.max
        params_out["kep"] = min(raw_kep, KEP_BOUNDS.max)
    elif model_name == "2cxm":
        fp_val = params_out.get("Fp", 0.0)
        ps_val = params_out.get("PS", 0.0)
        E = ps_val / (fp_val + ps_val + EPSILON_RATIO)
        params_out["E"] = E
        params_out["Ktrans"] = E * fp_val / 100.0
        ve_val = params_out.get("ve", EPSILON_VE)
        raw_kep = params_out["Ktrans"] / max(ve_val, EPSILON_VE)
        kep_railed = raw_kep > KEP_BOUNDS.max
        params_out["kep"] = min(raw_kep, KEP_BOUNDS.max)
    return kep_railed


def _merge_fit_results(
    r_const: Any,
    const_idx: npt.NDArray[np.intp],
    r_free: Any,
    free_idx: npt.NDArray[np.intp],
    n_voxels: int,
    model_name: str,
) -> Any:
    """Merge two FitResult objects at original indices.

    Scatters constrained and free results back into full (N,) arrays.

    Parameters
    ----------
    r_const : Any
        FitResult for constrained voxels.
    const_idx : numpy.ndarray
        Global indices of constrained voxels.
    r_free : Any
        FitResult for free (unconstrained) voxels.
    free_idx : numpy.ndarray
        Global indices of free voxels.
    n_voxels : int
        Total number of voxels.
    model_name : str
        Model name for the merged result.

    Returns
    -------
    Any
        Merged FitResult with all voxels at their original positions.
    """
    from pmriprep.models.dce.batched_fitting import FitResult

    # Merge params
    params: dict[str, npt.NDArray[np.floating]] = {}
    all_keys = set(r_const.params.keys()) | set(r_free.params.keys())
    for key in all_keys:
        arr = np.zeros(n_voxels, dtype=np.float64)
        if key in r_const.params:
            arr[const_idx] = r_const.params[key]
        if key in r_free.params:
            arr[free_idx] = r_free.params[key]
        params[key] = arr

    # Merge r_squared and converged
    r_squared = np.zeros(n_voxels, dtype=np.float64)
    r_squared[const_idx] = r_const.r_squared
    r_squared[free_idx] = r_free.r_squared

    converged = np.zeros(n_voxels, dtype=bool)
    converged[const_idx] = r_const.converged
    converged[free_idx] = r_free.converged

    # Merge fitted_curves
    fitted_curves: Optional[npt.NDArray[np.floating]] = None
    if r_const.fitted_curves is not None and r_free.fitted_curves is not None:
        n_time = r_const.fitted_curves.shape[1]
        fitted_curves = np.zeros((n_voxels, n_time), dtype=np.float64)
        fitted_curves[const_idx] = r_const.fitted_curves
        fitted_curves[free_idx] = r_free.fitted_curves
    elif r_const.fitted_curves is not None:
        n_time = r_const.fitted_curves.shape[1]
        fitted_curves = np.zeros((n_voxels, n_time), dtype=np.float64)
        fitted_curves[const_idx] = r_const.fitted_curves
    elif r_free.fitted_curves is not None:
        n_time = r_free.fitted_curves.shape[1]
        fitted_curves = np.zeros((n_voxels, n_time), dtype=np.float64)
        fitted_curves[free_idx] = r_free.fitted_curves

    # Merge uncertainties
    uncertainties: dict[str, npt.NDArray[np.floating]] = {}
    unc_keys = set(r_const.uncertainties.keys()) | set(r_free.uncertainties.keys())
    for key in unc_keys:
        arr = np.full(n_voxels, np.nan)
        if key in r_const.uncertainties:
            arr[const_idx] = r_const.uncertainties[key]
        if key in r_free.uncertainties:
            arr[free_idx] = r_free.uncertainties[key]
        uncertainties[key] = arr

    n_iterations = max(r_const.n_iterations, r_free.n_iterations)

    # Preserve vp_at_bound from the constrained branch; unconstrained voxels
    # (fixed_vp=0) cannot hit a bound by definition, so they're 0 in the map.
    vp_at_bound: Optional[npt.NDArray[np.floating]] = None
    if r_const.vp_at_bound is not None:
        vp_at_bound = np.zeros(n_voxels, dtype=np.float32)
        vp_at_bound[const_idx] = r_const.vp_at_bound

    return FitResult(
        params=params,
        r_squared=r_squared,
        converged=converged,
        fitted_curves=fitted_curves,
        n_iterations=n_iterations,
        model_name=model_name,
        uncertainties=uncertainties,
        vp_at_bound=vp_at_bound,
    )


def _fit_with_split_constraint(
    fit_fn: Any,
    tissue_curves: npt.NDArray[np.floating],
    fixed_vp: Optional[npt.NDArray[np.floating]],
    model_name: str,
    n_voxels: int,
    vp_uncertainty: Optional[float] = None,
    **fit_kwargs: Any,
) -> Any:
    """Handle mixed constrained/unconstrained voxels by splitting batches.

    When fixed_vp contains a mix of zeros (unconstrained) and positive values
    (constrained), splits into two batches, fits separately, and merges.
    ``vp_uncertainty`` is forwarded to the constrained batch only; the
    unconstrained batch receives ``vp_uncertainty=None`` since there is no
    prior to apply to those voxels.

    Parameters
    ----------
    fit_fn : Any
        Fitting function (batched_lm_fit or batched_patlak_lstsq).
    tissue_curves : numpy.ndarray
        Observed tissue curves, shape (N, T).
    fixed_vp : numpy.ndarray or None
        Fixed vp values or None. Zeros = unconstrained.
    model_name : str
        Model name.
    n_voxels : int
        Total number of voxels.
    vp_uncertainty : float, optional
        Relative σ for SOFT box constraint (None = HARD).
    **fit_kwargs : Any
        Additional kwargs passed to fit_fn.

    Returns
    -------
    Any
        FitResult with all voxels correctly handled.
    """
    if fixed_vp is None or model_name not in _MODEL_HAS_VP:
        return fit_fn(
            tissue_curves, fixed_vp=None, vp_uncertainty=None, **fit_kwargs,
        )

    has_constraint = fixed_vp > 0
    if np.all(has_constraint):
        return fit_fn(
            tissue_curves, fixed_vp=fixed_vp,
            vp_uncertainty=vp_uncertainty, **fit_kwargs,
        )
    if not np.any(has_constraint):
        return fit_fn(
            tissue_curves, fixed_vp=None, vp_uncertainty=None, **fit_kwargs,
        )

    # Mixed: split, fit separately, merge.
    const_idx = np.where(has_constraint)[0]
    free_idx = np.where(~has_constraint)[0]
    r_const = fit_fn(
        tissue_curves[const_idx], fixed_vp=fixed_vp[const_idx],
        vp_uncertainty=vp_uncertainty, **fit_kwargs,
    )
    r_free = fit_fn(
        tissue_curves[free_idx], fixed_vp=None,
        vp_uncertainty=None, **fit_kwargs,
    )
    return _merge_fit_results(
        r_const, const_idx, r_free, free_idx, n_voxels, model_name,
    )


def _init_batched_worker() -> None:
    """Minimal ProcessPool initializer for batched LM chunk workers.

    Caps BLAS threads to 1 per worker so ``n_procs × OpenBLAS_threads``
    does not oversubscribe the CPU. Does NOT build models / fitters /
    priors — those belong to the per-voxel VI/MCMC ``_init_worker``.
    """
    from pmriprep.utils.parallel import limit_worker_threads

    limit_worker_threads()
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)


def _fit_chunk_nlls(args: BatchedChunkArgs) -> Any:
    """Fit one voxel chunk via batched LM / lstsq with mixed-constraint split.

    Picklable entry point for ``ProcessPoolExecutor``. Deterministic:
    ``batched_lm_fit`` carries no RNG, so no seed plumbing is needed
    unless that changes in the future.

    Accepts a ``BatchedChunkArgs`` NamedTuple (see module docstring) — this
    replaces a prior positional tuple to prevent silent parameter drops
    (e.g. the historical `vp_uncertainty` drop that caused SOFT to behave
    as HARD in the batched NLLS path).

    Returns
    -------
    Any
        Per-chunk FitResult.
    """
    from pmriprep.models.dce.batched_fitting import (
        batched_lm_fit,
        batched_patlak_lstsq,
    )

    model_name = args.model_name
    tissue_curves = args.tissue_curves
    aif = args.aif
    time_points = args.time_points
    fixed_vp = args.fixed_vp
    vp_uncertainty = args.vp_uncertainty
    weights = args.weights
    save_fitted_curves = args.save_fitted_curves
    compute_uncertainties = args.compute_uncertainties

    # Closures are cheap to build inside the worker (not pickled across
    # processes); aif and time_points have already been transferred.
    if model_name == "patlak":
        def fit_fn(tc: npt.NDArray[np.floating], **kw: Any) -> Any:
            return batched_patlak_lstsq(tc, aif, time_points, **kw)
    else:
        def fit_fn(tc: npt.NDArray[np.floating], **kw: Any) -> Any:
            return batched_lm_fit(model_name, tc, aif, time_points, **kw)

    return _fit_with_split_constraint(
        fit_fn,
        tissue_curves,
        fixed_vp,
        model_name,
        len(tissue_curves),
        vp_uncertainty=vp_uncertainty,
        weights=weights,
        save_fitted_curves=save_fitted_curves,
        compute_uncertainties=compute_uncertainties,
    )


class ComputeDCEMapsInputSpec(BaseInterfaceInputSpec):
    """Input specification for ComputeDCEMaps interface."""

    concentration_file = File(
        exists=True,
        mandatory=True,
        desc="4D NIfTI concentration curves (output from signal_to_concentration)",
    )
    dce_signal_file = File(
        exists=True,
        desc="4D motion-corrected raw DCE signal for tSNR calculation",
    )
    aif_file = File(
        exists=True,
        mandatory=True,
        desc="1D text file with AIF plasma concentration curve (mM)",
    )
    brain_mask_file = File(
        exists=True,
        mandatory=True,
        desc="3D brain mask NIfTI",
    )
    # Model selection
    model = traits.Enum(
        "auto",
        "tofts",
        "extended_tofts",
        "patlak",
        "2cxm",
        usedefault=True,
        desc="Pharmacokinetic model: auto, tofts, extended_tofts, patlak, or 2cxm",
    )

    # Time parameters
    volume_tr = traits.Float(
        desc="Volume TR in seconds for model auto-selection (RepetitionTime, temporal resolution)",
    )
    time_points_file = File(
        exists=True,
        desc=(
            "Optional 1D text file with uniformly spaced time points (minutes); "
            "if not provided, uses volume TR"
        ),
    )

    # Inference engine options
    inference = traits.Enum(
        "nlls",
        "vi",
        "map",
        "mcmc",
        usedefault=True,
        desc="Inference engine: nlls (default, scipy least_squares), "
        "vi (variational Bayes), map (Laplace MAP), mcmc (emcee).",
    )
    mcmc_samples = traits.Int(
        500,
        usedefault=True,
        desc="Number of MCMC samples for Bayesian inference",
    )

    # tSNR filtering
    tsnr_threshold = traits.Float(
        DEFAULT_TSNR_THRESHOLD,
        usedefault=True,
        desc="tSNR threshold for voxel inclusion (0 = no threshold)",
    )
    baseline_frames = traits.Int(
        0,
        usedefault=True,
        desc="Number of baseline frames for tSNR calculation (0 = auto-detect)",
    )

    # DSC CBV constraint options
    dsc_ncbv_file = traits.Str(
        desc="DSC NAWM-normalized CBV (nCBV, dimensionless ratio) used for "
             "the vp prior via vp = NAWM_VP_REFERENCE × nCBV. Empty string "
             "when DSC has no NAWM reference → constraint cleanly SKIPs.",
    )
    tumor_mask_file = traits.Str(
        desc="Enhancing-tumor mask (HD-GLIO label 2) on the DCE grid, used as "
             "the tissue-arrival reference for AIF BAT pre-alignment. Empty "
             "string when tumor segmentation is unavailable → BAT no-ops.",
    )
    dsc_aif_qc = traits.Dict(
        mandatory=False,
        desc="DSC AIF QC metrics dict for CBV quality assessment (optional)",
    )
    dsc_preload_qc = traits.Dict(
        mandatory=False,
        desc="DSC preload + K2 QC metrics dict from preload_correct node",
    )
    dsc_preprocessing_stats = traits.Dict(
        mandatory=False,
        desc="DSC preprocessing QC stats (fraction_cnr_gt_4, mean_cnr)",
    )
    constraint_mode = traits.Enum(
        "auto",
        "skip",
        usedefault=True,
        desc="DSC CBV constraint mode: auto or skip",
    )
    dsc_dose = traits.Float(
        1.0,
        usedefault=True,
        desc=(
            "DSC bolus dose multiplier relative to the agent's standard "
            "single dose (1.0 = full, 0.5 = half). Scales CNR coverage "
            "thresholds in the vp constraint quality gate."
        ),
    )
    dsc_agent_standard_dose = traits.Float(
        0.1,
        usedefault=True,
        desc=(
            "DSC agent's standard clinical dose in mmol Gd/kg (0.1 = Gd-DOTA "
            "class, 0.05 = gadopiclenol/Vueway, 0.04 = gadoquatrane). "
            "Together with dsc_dose sets the effective Gd fraction used to "
            "scale CNR coverage thresholds."
        ),
    )

    # Processing options
    n_procs = traits.Int(
        1,
        usedefault=True,
        desc="Number of parallel processes for voxel fitting (1 = serial)",
    )
    save_fitted_curves = traits.Bool(
        False,
        usedefault=True,
        desc="Save fitted concentration curves (large output)",
    )

    # Motion weighting inputs (from dce_preprocess)
    fd_values = traits.List(
        traits.Float,
        desc="Framewise displacement per volume (mm) from motion correction",
    )
    bolus_arrival_frame = traits.Int(
        desc="Bolus arrival frame index for phase-adaptive weighting",
    )

    # AIF detection method (gates AIF->tissue BAT pre-alignment)
    aif_detection_method = traits.Str(
        "population",
        usedefault=True,
        desc="AIF detection method: 'population', 'whole_brain', 'atlas', "
             "'external'. Gates BAT pre-alignment, which applies only to "
             "measured 'whole_brain'/'atlas' AIFs (see aif_bat_alignment).",
    )

    aif_bat_alignment = traits.Enum(
        "auto",
        "off",
        usedefault=True,
        desc="AIF-to-tissue bolus-arrival pre-alignment. 'auto' shifts a "
             "measured (whole_brain/atlas) AIF in time to correct the "
             "arterial-to-tissue delay (guarded: no-op below temporal "
             "resolution; benefits fine temporal resolution). 'off' = no shift.",
    )

    aif_uncertainty_inflation_factor = traits.Float(
        1.0,
        usedefault=True,
        desc="Base multiplier on PK posterior sigma from upstream AIF "
             "confidence (1.0 = standard, 2.0 = G6-admitted low-scale, "
             "3.0 = G6-admitted high-scale or population AIF). Applied "
             "per-parameter via DCE_AIF_G6_SIGMA_INFLATION map; ve is "
             "uninflated (decoupled from AIF peak amplitude per "
             "Buckley 2002).",
    )

    # Sloppy / reproducibility
    sloppy = traits.Bool(
        False,
        usedefault=True,
        desc="Use fast, low-quality settings for testing (NOT for production)",
    )
    random_seed = traits.Int(
        desc="Random seed for reproducibility in worker processes",
    )

    # Output files
    out_ktrans = File(
        genfile=True,
        desc="Output Ktrans map NIfTI (min^-1)",
    )
    out_ve = File(
        genfile=True,
        desc="Output ve map NIfTI (dimensionless)",
    )
    out_vp = File(
        genfile=True,
        desc="Output vp map NIfTI (dimensionless, if applicable)",
    )
    out_kep = File(
        genfile=True,
        desc="Output kep map NIfTI (min^-1)",
    )
    out_vp_at_bound_map = File(
        genfile=True,
        desc=(
            "Output per-voxel binary map (0/1) indicating whether vp ended at "
            "a bound in the active-set SOFT NLLS path. 0 elsewhere."
        ),
    )


class ComputeDCEMapsOutputSpec(TraitedSpec):
    """Output specification for ComputeDCEMaps interface."""

    out_ktrans = File(exists=True, desc="Ktrans map (volume transfer constant, min^-1)")
    out_ve = File(desc="ve map (extravascular extracellular volume fraction, if applicable)")
    out_vp = File(desc="vp map (plasma volume fraction, if applicable)")
    out_kep = File(desc="kep map (efflux rate constant, if applicable)")

    # 2CXM-specific outputs
    out_fp = File(desc="Fp map (plasma flow rate, mL/min/100mL, 2cxm only)")
    out_ps = File(desc="PS map (permeability-surface area product, mL/min/100mL, 2cxm only)")
    out_e = File(desc="E map (extraction fraction, 2cxm only)")

    # Uncertainty maps (Bayesian only)
    out_ktrans_std = File(desc="Ktrans standard deviation map (Bayesian)")
    out_ve_std = File(desc="ve standard deviation map (Bayesian)")
    out_vp_std = File(desc="vp standard deviation map (Bayesian)")
    out_fp_std = File(desc="Fp standard deviation map (Bayesian, 2CXM only)")
    out_ps_std = File(desc="PS standard deviation map (Bayesian, 2CXM only)")

    # Coefficient of Variation maps (Bayesian only)
    out_ktrans_cv = File(desc="Ktrans coefficient of variation (std/mean, Bayesian)")
    out_ve_cv = File(desc="ve coefficient of variation (std/mean, Bayesian)")
    out_vp_cv = File(desc="vp coefficient of variation (std/mean, Bayesian)")

    # 95% Credible Interval bounds (Bayesian only)
    out_ktrans_ci95_lower = File(desc="Ktrans 95% CI lower bound (Bayesian)")
    out_ktrans_ci95_upper = File(desc="Ktrans 95% CI upper bound (Bayesian)")
    out_ve_ci95_lower = File(desc="ve 95% CI lower bound (Bayesian)")
    out_ve_ci95_upper = File(desc="ve 95% CI upper bound (Bayesian)")
    out_vp_ci95_lower = File(desc="vp 95% CI lower bound (Bayesian)")
    out_vp_ci95_upper = File(desc="vp 95% CI upper bound (Bayesian)")

    # Model-free maps
    out_iauc60 = File(desc="IAUC at 60s (mM·min)")
    out_iauc90 = File(desc="IAUC at 90s (mM·min)")
    out_iauc120 = File(desc="IAUC at 120s (mM·min)")
    out_peak_enhancement = File(desc="Peak enhancement percentage")
    out_time_to_peak = File(desc="Time to peak (seconds)")
    out_ser = File(desc="Signal Enhancement Ratio (washout characterization)")
    out_wash_in_slope = File(desc="Wash-in slope (maximum upslope, mM/min)")
    out_wash_out_slope = File(desc="Wash-out slope (linear fit, mM/min, signed)")
    out_bat = File(desc="Bolus arrival time (seconds)")
    out_curve_type = File(desc="Curve type classification (0=unclassified, 1=persistent, 2=plateau, 3=washout)")

    # Model selection map (auto mode)
    out_model_selection = File(desc="Model selection map (int8: 0=unfitted, 1=patlak, 2=tofts, 3=extended_tofts, 4=2cxm)")

    # Delta-BIC confidence map (auto mode)
    out_delta_bic = File(desc="Delta-BIC confidence map (runner-up BIC minus winner BIC; >2 = positive evidence)")

    # QC outputs
    out_quality_flag = File(desc="Quality flag map (0=unfitted, 1=good, 2=poor_convergence, 3=high_uncertainty, 4=unreliable_high_kep)")
    out_r2 = File(desc="R² (goodness of fit) map")
    out_tsnr = File(desc="tSNR map (always computed)")
    out_qc_mask = File(desc="QC mask (1=good fit, 0=unfitted/poor)")
    out_fitted_curves = File(desc="Fitted concentration curves (if save_fitted_curves=True)")
    out_vp_at_bound_map = File(
        desc=(
            "Per-voxel binary indicator (0/1) of whether vp ended at a bound "
            "in the active-set SOFT NLLS path. 0 for interior / HARD / SKIP / "
            "Bayesian fits. Produced only when constraint_mode=SOFT + "
            "inference=nlls."
        ),
    )
    qc_metrics = traits.Dict(desc="QC metrics dictionary")
    constraint_qc = traits.Dict(desc="DSC CBV constraint QC report")


class ComputeDCEMaps(BaseInterface):
    """Compute DCE pharmacokinetic parameter maps.

    This interface computes pharmacokinetic parameter maps from DCE-MRI
    concentration curves and an arterial input function (AIF) using various
    PK models including Standard Tofts, Extended Tofts, Patlak, and
    Two-Compartment Exchange models.

    Supported models:
    - tofts: Standard Tofts model (Ktrans, ve)
    - extended_tofts: Extended Tofts model (Ktrans, ve, vp)
    - patlak: Patlak model with graphical analysis (Ktrans, vp)
    - 2cxm: Two-Compartment Exchange Model (Fp, PS, ve, vp)

    Auto-selection logic (when model='auto'):
    - Patlak is always included
    - Extended Tofts included if TR <= 10s
    - 2CXM included if TR <= 3s
    - Winner selected by BIC (Standard Tofts excluded; vp=0 invalid in brain)

    Output maps:
    - Ktrans: Volume transfer constant (min⁻¹)
    - ve: Extravascular extracellular volume fraction
    - vp: Plasma volume fraction (extended_tofts, patlak, 2cxm)
    - kep: Efflux rate constant (Ktrans/ve, min⁻¹)

    For 2CXM additionally:
    - Fp: Plasma flow rate (mL/min/100mL)
    - PS: Permeability-surface area product (mL/min/100mL)
    - E: Extraction fraction (PS / (Fp + PS))

    Parameters
    ----------
    concentration_file : str
        4D NIfTI with tissue concentration time courses (mM).
    aif_file : str
        1D text file with AIF plasma concentration values (mM).
    brain_mask_file : str
        3D brain mask NIfTI.
    model : str
        PK model: auto, tofts, extended_tofts, patlak, or 2cxm. Default: 'auto'.
    volume_tr : float, optional
        Temporal spacing between DCE volumes in seconds.
    time_points_file : str, optional
        Text file with time points in minutes; if not provided, uses volume TR.
    inference : str
        Inference engine: nlls (default, scipy least_squares),
        vi (variational Bayes), map (Laplace MAP), or mcmc (emcee).
    mcmc_samples : int
        MCMC samples for Bayesian. Default: 500.
    tsnr_threshold : float
        tSNR threshold for voxel inclusion (0 = no threshold). Default: 0.
    baseline_frames : int
        Baseline frames for tSNR calculation (0 = auto-detect). Default: 0.
    save_fitted_curves : bool
        Save fitted curves. Default: False.

    Returns
    -------
    out_ktrans : str
        Path to Ktrans map NIfTI.
    out_ve : str
        Path to ve map NIfTI.
    out_vp : str
        Path to vp map NIfTI (if model has vp).
    out_kep : str
        Path to kep map NIfTI.
    out_fp : str, optional
        Path to Fp map (2cxm only).
    out_ps : str, optional
        Path to PS map (2cxm only).
    out_e : str, optional
        Path to E map (2cxm only).
    out_ktrans_std : str, optional
        Ktrans uncertainty (Bayesian only).
    out_ve_std : str, optional
        ve uncertainty (Bayesian only).
    out_vp_std : str, optional
        vp uncertainty (Bayesian only).
    out_r2 : str
        R² goodness-of-fit map.
    out_tsnr : str
        tSNR map (if threshold > 0).
    out_fitted_curves : str, optional
        Fitted curves (if save_fitted_curves=True).
    qc_metrics : dict
        QC metrics including fitting statistics.

    Examples
    --------
    >>> compute = ComputeDCEMaps()
    >>> compute.inputs.concentration_file = 'concentration_4d.nii.gz'
    >>> compute.inputs.aif_file = 'aif_curve.txt'
    >>> compute.inputs.brain_mask_file = 'brain_mask.nii.gz'
    >>> compute.inputs.model = 'extended_tofts'
    >>> compute.inputs.inference = 'map'
    >>> result = compute.run()  # doctest: +SKIP

    Notes
    -----
    Time units:
    - Time points are in minutes for model fitting
    - Ktrans and kep are in min⁻¹
    - Concentration is in mM (millimolar)

    For Bayesian methods (map, mcmc), uncertainty maps are automatically
    generated.

    The tSNR threshold can exclude noisy voxels from fitting. tSNR is
    computed as mean(baseline) / std(baseline). A typical threshold is 4.0.

    """

    input_spec = ComputeDCEMapsInputSpec
    output_spec = ComputeDCEMapsOutputSpec

    def _gen_outfilename(self, suffix: str) -> str:
        """Generate output filename based on input."""
        in_file = Path(self.inputs.concentration_file)
        if in_file.name.endswith(".nii.gz"):
            base = in_file.name[:-7]
        elif in_file.name.endswith(".nii"):
            base = in_file.stem
        else:
            base = in_file.stem
        return str(Path.cwd() / f"{base}{suffix}.nii.gz")

    def _compute_tsnr(
        self,
        data: npt.NDArray[Any],
        mask: npt.NDArray[Any],
        baseline_frames: int,
    ) -> npt.NDArray[Any]:
        """Compute temporal SNR from baseline frames.

        tSNR = mean(baseline) / std(baseline)

        Parameters
        ----------
        data : numpy.ndarray
            4D concentration data (x, y, z, t).
        mask : numpy.ndarray
            3D brain mask.
        baseline_frames : int
            Number of baseline frames.

        Returns
        -------
        numpy.ndarray
            3D tSNR map.
        """
        spatial_shape = data.shape[:3]
        tsnr = np.zeros(spatial_shape, dtype=np.float64)

        if baseline_frames < 2:
            # Not enough frames for tSNR calculation
            tsnr[mask > 0] = np.inf  # All voxels pass
            return tsnr

        # Extract baseline data
        baseline_data = data[..., :baseline_frames]

        # Compute mean and std along time axis
        with np.errstate(divide="ignore", invalid="ignore"):
            baseline_mean = np.mean(baseline_data, axis=3)
            baseline_std = np.std(baseline_data, axis=3)

            # tSNR = mean / std (handle division by zero)
            tsnr = np.where(
                baseline_std > EPSILON_VE,
                np.abs(baseline_mean) / baseline_std,
                0.0,
            )

        return tsnr

    def _detect_baseline_frames(
        self,
        data: npt.NDArray[Any],
        mask: npt.NDArray[Any],
    ) -> int:
        """Auto-detect number of baseline frames from global signal.

        For DCE, signal INCREASES at bolus arrival (T1 shortening).

        Parameters
        ----------
        data : numpy.ndarray
            4D concentration data.
        mask : numpy.ndarray
            3D brain mask.

        Returns
        -------
        int
            Number of baseline frames.
        """
        n_volumes = data.shape[3]

        # Compute global mean concentration time course
        mask_bool = mask > 0
        global_signal = np.array([
            np.mean(data[..., t][mask_bool])
            for t in range(n_volumes)
        ])

        # Use first few frames to estimate baseline
        init_window = min(5, n_volumes // 4)
        baseline_mean = np.mean(global_signal[:init_window])
        baseline_std = np.std(global_signal[:init_window])

        # Threshold for detecting contrast arrival
        threshold = max(
            np.abs(baseline_mean) * 0.05,  # 5% increase
            baseline_std * 2.0,  # 2 std above baseline
        )

        # Find first frame where signal exceeds threshold
        for t in range(init_window, n_volumes):
            if global_signal[t] > baseline_mean + threshold:
                return t

        # Default if no bolus detected
        return int(min(5, n_volumes - 1))

    def _has_vp(self, model_name: str) -> bool:
        """Check if model has vp parameter.

        vp is estimated by extended_tofts, patlak, and 2cxm.
        Standard Tofts assumes vp=0 and does not estimate it.

        Parameters
        ----------
        model_name : str
            Model name.

        Returns
        -------
        bool
            True if model outputs vp.
        """
        return model_name in ("extended_tofts", "patlak", "2cxm")

    def _has_ve(self, model_name: str) -> bool:
        """Check if model has ve parameter.

        ve (and derived kep = Ktrans/ve) is estimated by tofts,
        extended_tofts, and 2cxm. Patlak does not estimate ve.

        Parameters
        ----------
        model_name : str
            Model name.

        Returns
        -------
        bool
            True if model outputs ve.
        """
        return model_name in ("tofts", "extended_tofts", "2cxm")

    # Temporal resolution gates for BIC candidate set
    _MAX_VOLUME_TR_2CXM = 3.0       # 2CXM temporal-resolution limit
    _MAX_VOLUME_TR_ETOFTS = 10.0    # Conservative clinical limit

    def _build_candidate_set(self, volume_tr: float) -> list[str]:
        """Build BIC candidate model set gated by temporal resolution.

        Standard Tofts is excluded from auto selection because its vp=0
        assumption is never valid in brain tissue. The intravascular
        contribution vp×Cp(t) is absorbed into Ktrans, causing systematic
        overestimation. Patlak (2 params,
        has vp) and Extended Tofts (3 params, has vp) form a complete
        nested hierarchy for brain DCE. Standard Tofts remains available
        via explicit ``model=tofts`` selection.

        Parameters
        ----------
        volume_tr : float
            Volume TR in seconds.

        Returns
        -------
        list
            List of candidate model names, ordered by complexity.
        """
        candidates = ["patlak"]
        if volume_tr <= self._MAX_VOLUME_TR_ETOFTS:
            candidates.append("extended_tofts")
        if volume_tr <= self._MAX_VOLUME_TR_2CXM:
            candidates.append("2cxm")
        return candidates

    def _prepare_fit_inputs(self) -> _DCEFitPreparation:
        """Load and validate DCE inputs and construct the pre-BAT fit domain.

        Returns
        -------
        _DCEFitPreparation
            Image geometry, validated time axis, model-selection state, signal
            QC maps, and the voxel mask admitted to pharmacokinetic fitting.
        """
        import nibabel as nib  # type: ignore[import-untyped]

        conc_img = nib.load(self.inputs.concentration_file)
        conc_data: npt.NDArray[Any] = conc_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]
        affine = conc_img.affine  # type: ignore[attr-defined]
        header = conc_img.header  # type: ignore[attr-defined]

        mask_img = nib.load(self.inputs.brain_mask_file)
        mask: npt.NDArray[Any] = mask_img.get_fdata()  # type: ignore[attr-defined]
        aif = np.loadtxt(self.inputs.aif_file)
        n_timepoints = conc_data.shape[3]

        if self.inputs.time_points_file:
            time_points = np.loadtxt(self.inputs.time_points_file)
        elif isdefined(self.inputs.volume_tr) and self.inputs.volume_tr is not None:
            vol_tr_minutes = float(self.inputs.volume_tr) / 60.0
            time_points = np.arange(n_timepoints, dtype=np.float64) * vol_tr_minutes
        else:
            raise ValueError(
                "DCE time vector requires either time_points_file or volume_tr "
                "(BIDS RepetitionTime). Pulse TR (RepetitionTimeExcitation) "
                "must not be used as a substitute for 3D acquisitions."
            )

        time_points = _validate_dce_time_axis(
            time_points,
            n_dce_frames=n_timepoints,
            n_aif_samples=int(np.size(aif)),
        )
        # Model eligibility depends on the volume sampling interval, never the
        # pulse TR. Recover it from an explicit time axis when necessary.
        if isdefined(self.inputs.volume_tr) and self.inputs.volume_tr is not None:
            selection_tr_seconds = float(self.inputs.volume_tr)
        else:
            selection_tr_seconds = float(time_points[1] - time_points[0]) * 60.0

        model_name = str(self.inputs.model)
        is_auto = model_name == "auto"
        candidate_names: list[str] = []
        if is_auto:
            candidate_names = self._build_candidate_set(selection_tr_seconds)
            model_name = candidate_names[-1]

        baseline_frames = int(self.inputs.baseline_frames)
        if baseline_frames == 0:
            baseline_frames = self._detect_baseline_frames(conc_data, mask)

        # Raw signal is the valid domain for tSNR and peak-enhancement QC;
        # concentration baseline is approximately zero by construction.
        tsnr_threshold = float(self.inputs.tsnr_threshold)
        signal_std_map: Optional[npt.NDArray[Any]] = None
        peak_enhancement: Optional[npt.NDArray[Any]]
        if isdefined(self.inputs.dce_signal_file):
            signal_img = nib.load(self.inputs.dce_signal_file)
            signal_data = signal_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]
            tsnr_map = self._compute_tsnr(signal_data, mask, baseline_frames)
            signal_std_map = np.std(signal_data, axis=-1)
            from pmriprep.utils.dce_model_free import compute_peak_enhancement

            peak_enhancement = compute_peak_enhancement(signal_data, baseline_frames)
            del signal_data
        else:
            logger.warning(
                "dce_signal_file not provided; computing tSNR from concentration "
                "data (values will be low — consider wiring dce_preprocess.out_file)"
            )
            tsnr_map = self._compute_tsnr(conc_data, mask, baseline_frames)
            peak_enhancement = None

        # Admit only brain voxels meeting configured signal quality, raw-signal
        # variation, and a minimal measurable concentration enhancement.
        mask_bool = mask > 0
        if tsnr_threshold > 0:
            fitting_mask = mask_bool & (tsnr_map >= tsnr_threshold)
        else:
            fitting_mask = mask_bool
        if signal_std_map is not None:
            fitting_mask = fitting_mask & (signal_std_map >= 1.0)
        enhancement_mask = np.max(conc_data, axis=-1) > MIN_ENHANCEMENT_MM
        fitting_mask = fitting_mask & enhancement_mask

        inference = str(self.inputs.inference)
        mcmc_samples = int(self.inputs.mcmc_samples)
        save_fitted_curves = bool(self.inputs.save_fitted_curves)
        # Preserve the identifiability warning for every 2CXM fitting method.
        if model_name == "2cxm":
            logger.warning(
                "2CXM individual parameter estimates (especially PS) may be "
                "unreliable due to PS/ve coupling degeneracy (Sourbron & "
                "Buckley, MRM 2011). Consider Extended Tofts if Ktrans is "
                "the primary parameter of interest."
            )

        return _DCEFitPreparation(
            conc_data,
            affine,
            header,
            mask,
            aif,
            time_points,
            n_timepoints,
            selection_tr_seconds,
            model_name,
            is_auto,
            candidate_names,
            baseline_frames,
            tsnr_threshold,
            tsnr_map,
            peak_enhancement,
            fitting_mask,
            inference,
            mcmc_samples,
            save_fitted_curves,
        )

    def _align_aif_to_tissue(
        self,
        prepared: _DCEFitPreparation,
    ) -> _DCEBATAlignment:
        """Apply optional measured-AIF arrival alignment to enhancing tissue.

        Parameters
        ----------
        prepared : _DCEFitPreparation
            Validated pre-BAT inputs and fitting domain.

        Returns
        -------
        _DCEBATAlignment
            Possibly shifted AIF and the corresponding correction QC record.

        Notes
        -----
        Population AIFs are pre-aligned at generation and external AIF timing
        is unknown, so neither is shifted. Measured AIFs use the median
        enhancing-tissue curve; sub-resolution or unreliable delays remain
        no-ops in the existing delay estimator.
        """
        aif_detection_method = str(self.inputs.aif_detection_method)
        bat_alignment_mode = str(self.inputs.aif_bat_alignment)
        if bat_alignment_mode == "off":
            return _DCEBATAlignment(
                prepared.aif,
                {"applied": False, "reason": "alignment_off"},
            )
        if aif_detection_method in ("population", "external"):
            return _DCEBATAlignment(
                prepared.aif,
                {
                    "applied": False,
                    "reason": f"{aif_detection_method}_aif_not_aligned",
                },
            )
        if not np.any(prepared.fitting_mask):
            return _DCEBATAlignment(
                prepared.aif,
                {"applied": False, "reason": "bat_detection_failed"},
            )

        from pmriprep.utils.dce_bat import compute_aif_tissue_delay, shift_aif

        tumor_mask_file = (
            self.inputs.tumor_mask_file
            if isdefined(self.inputs.tumor_mask_file)
            else ""
        )
        roi_mask, ref_info = _resolve_bat_tissue_reference(
            tumor_mask_file,
            prepared.fitting_mask,
            prepared.concentration,
        )
        if roi_mask is None:
            return _DCEBATAlignment(prepared.aif, ref_info)

        roi_curve = np.median(prepared.concentration[roi_mask], axis=0)
        delay_s, correction = compute_aif_tissue_delay(
            aif_curve=prepared.aif,
            tissue_roi_curve=roi_curve,
            time_points_min=prepared.time_points,
            tr_seconds=prepared.selection_tr_seconds,
            baseline_frames=prepared.baseline_frames,
        )
        correction["granularity"] = "enhancing_tumor"
        correction["n_reference_voxels"] = ref_info["n_reference_voxels"]
        aligned_aif = prepared.aif
        if delay_s != 0.0:
            aligned_aif = shift_aif(prepared.aif, prepared.time_points, delay_s)
        return _DCEBATAlignment(aligned_aif, correction)

    def _prepare_vp_constraint(
        self,
        prepared: _DCEFitPreparation,
        aif: npt.NDArray[Any],
    ) -> _DCEVPConstraint:
        """Prepare the DSC-derived vp prior and acquisition-limited QC state.

        Parameters
        ----------
        prepared : _DCEFitPreparation
            Validated DCE inputs, fitting domain, and initial model selection.
        aif : numpy.ndarray
            Post-BAT-alignment plasma AIF used by pharmacokinetic fitting.

        Returns
        -------
        _DCEVPConstraint
            Prior map, constraint configuration, diagnostic QC, and any
            constraint-driven adjustment to automatic model selection.

        Notes
        -----
        The prior remains NAWM-anchored as ``vp = NAWM_VP_REFERENCE * nCBV``.
        The AIF-to-tissue overshoot is informational and never changes the
        constraint mode or uncertainty.
        """
        from pmriprep.utils.dsc_cbv_constraint import (
            NO_DSC_RUN_REASON_STRING,
            SKIP_REASON_NO_DSC_RUN,
            SKIP_REASON_SYNTHETIC_AIF,
            ConstraintMode,
            prepare_vp_constraint_from_data,
        )

        vp_constraint_map: Optional[npt.NDArray[Any]] = None
        constraint_mode = ConstraintMode.SKIP
        constraint_vp_uncertainty = 0.2
        constraint_qc: dict[str, Any] = {
            "mode": "skip",
            "reason": NO_DSC_RUN_REASON_STRING,
            "skip_reason": SKIP_REASON_NO_DSC_RUN,
        }
        model_name = prepared.model_name
        candidate_names = list(prepared.candidate_names)
        ncbv_available = isdefined(self.inputs.dsc_ncbv_file) and bool(
            self.inputs.dsc_ncbv_file
        )
        needs_vp_constraint = model_name in ("extended_tofts", "2cxm") or prepared.is_auto
        dsc_inputs_present = (
            (isdefined(self.inputs.dsc_ncbv_file) and bool(self.inputs.dsc_ncbv_file))
            or (isdefined(self.inputs.dsc_aif_qc) and bool(self.inputs.dsc_aif_qc))
            or (isdefined(self.inputs.dsc_preload_qc) and bool(self.inputs.dsc_preload_qc))
        )

        if needs_vp_constraint and dsc_inputs_present:
            try:
                aif_qc_data = (
                    dict(self.inputs.dsc_aif_qc)
                    if isdefined(self.inputs.dsc_aif_qc)
                    else {}
                )
                if isdefined(self.inputs.dsc_preprocessing_stats):
                    prep_stats = self.inputs.dsc_preprocessing_stats
                    if prep_stats and "fraction_cnr_gt_4" in prep_stats:
                        aif_qc_data["cnr_whole_brain"] = {
                            "fraction_cnr_gt_4": prep_stats["fraction_cnr_gt_4"],
                            "mean_cnr": prep_stats.get("mean_cnr"),
                        }

                preload_qc = (
                    dict(self.inputs.dsc_preload_qc)
                    if isdefined(self.inputs.dsc_preload_qc)
                    else {}
                )
                k2_metrics = (
                    {key: value for key, value in preload_qc.items() if key in _K2_QC_KEYS}
                    or None
                )
                vp_result = prepare_vp_constraint_from_data(
                    ncbv_file=self.inputs.dsc_ncbv_file if ncbv_available else None,
                    aif_qc_data=aif_qc_data,
                    brain_mask_file=self.inputs.brain_mask_file,
                    mode_override=str(self.inputs.constraint_mode),
                    preload_classification=preload_qc.get("preload_classification"),
                    k2_metrics=k2_metrics,
                    dose=float(self.inputs.dsc_dose),
                    agent_standard_dose=float(self.inputs.dsc_agent_standard_dose),
                )
                vp_constraint_map = vp_result.vp_map
                constraint_mode = vp_result.config.mode
                constraint_vp_uncertainty = vp_result.config.vp_uncertainty
                constraint_qc = vp_result.qc_report

                if (
                    constraint_mode == ConstraintMode.SOFT
                    and str(self.inputs.constraint_mode) == "auto"
                    and vp_constraint_map is not None
                ):
                    aif_peak = float(np.max(aif))
                    if aif_peak > 0:
                        valid_constrained = (
                            prepared.fitting_mask
                            & np.isfinite(vp_constraint_map)
                            & (vp_constraint_map > 0)
                        )
                        if int(np.sum(valid_constrained)) > 100:
                            vp_cp_peak = vp_constraint_map[valid_constrained] * aif_peak
                            ct_peak_arr = np.max(
                                prepared.concentration[valid_constrained], axis=-1
                            )
                            overshoot_frac = float(np.mean(vp_cp_peak > ct_peak_arr))
                            constraint_qc["aif_tissue_overshoot_fraction"] = overshoot_frac
                            if (
                                isdefined(self.inputs.volume_tr)
                                and self.inputs.volume_tr is not None
                                and float(self.inputs.volume_tr) > 0
                            ):
                                volume_tr_s = float(self.inputs.volume_tr)
                                aif_peak_frame = int(np.argmax(aif))
                                tissue_peak_frames = np.argmax(
                                    prepared.concentration[valid_constrained], axis=-1
                                )
                                median_tissue_peak_frame = float(
                                    np.median(tissue_peak_frames)
                                )
                                constraint_qc["peak_alignment_lag_s"] = (
                                    median_tissue_peak_frame - aif_peak_frame
                                ) * volume_tr_s
                                constraint_qc["volume_tr_s"] = volume_tr_s
                            else:
                                constraint_qc["peak_alignment_lag_s"] = None
                                constraint_qc["peak_alignment_lag_reason"] = (
                                    "volume_tr_undefined"
                                )
                            if overshoot_frac > ACQUISITION_LIMITED_OVERSHOOT_THRESHOLD:
                                constraint_qc["acquisition_limited_note"] = (
                                    "High AIF-to-tissue overshoot (>70%): "
                                    "observed tissue peak under-samples the "
                                    "first pass (TR ≲ 2 s recommended for "
                                    "vp identifiability; Buckley MRM 2002). "
                                    "Posterior vp is prior-dominated by "
                                    "design at this acquisition; "
                                    "informational only — no action taken."
                                )

                if (
                    aif_qc_data.get("detection_method") == "synthetic"
                    or aif_qc_data.get("cbv_auc_denominator_is_synthetic")
                ):
                    constraint_qc.setdefault("aif_tissue_overshoot_fraction", None)
                    constraint_qc["overshoot_applicable"] = False
                    constraint_qc["overshoot_reason"] = SKIP_REASON_SYNTHETIC_AIF
            except Exception as exc:
                constraint_qc = {
                    "mode": "skip",
                    "reason": f"Failed to load DSC CBV constraint: {exc}",
                }

            if prepared.is_auto and constraint_mode != ConstraintMode.SKIP:
                if "extended_tofts" not in candidate_names:
                    candidate_names.append("extended_tofts")
                model_name = candidate_names[-1]

        return _DCEVPConstraint(
            vp_constraint_map,
            constraint_mode,
            constraint_vp_uncertainty,
            constraint_qc,
            model_name,
            candidate_names,
        )

    def _compute_model_free_maps(
        self,
        prepared: _DCEFitPreparation,
    ) -> _DCEModelFreeMaps:
        """Compute vectorized semi-quantitative maps before PK fitting."""
        from pmriprep.utils.dce_model_free import (
            compute_bolus_arrival_time,
            compute_curve_type,
            compute_iauc,
            compute_signal_enhancement_ratio,
            compute_time_to_peak,
            compute_wash_in_slope,
            compute_wash_out_slope,
        )

        concentration = prepared.concentration
        time_points = prepared.time_points
        volume_tr_s: Optional[float] = (
            float(self.inputs.volume_tr)
            if isdefined(self.inputs.volume_tr) and self.inputs.volume_tr is not None
            else None
        )
        ser = compute_signal_enhancement_ratio(
            concentration,
            time_points,
            bolus_arrival_frame=prepared.baseline_frames,
        )
        return _DCEModelFreeMaps(
            compute_iauc(concentration, time_points, cutoff_s=60.0),
            compute_iauc(concentration, time_points, cutoff_s=90.0),
            compute_iauc(concentration, time_points, cutoff_s=120.0),
            prepared.peak_enhancement,
            compute_time_to_peak(concentration, time_points),
            ser,
            compute_wash_in_slope(
                concentration,
                time_points,
                bolus_arrival_frame=prepared.baseline_frames,
                volume_tr_s=volume_tr_s,
            ),
            compute_wash_out_slope(
                concentration,
                time_points,
                volume_tr_s=volume_tr_s,
            ),
            compute_bolus_arrival_time(
                concentration,
                time_points,
                baseline_frames=prepared.baseline_frames,
                brain_mask=prepared.mask,
            ),
            compute_curve_type(ser),
        )

    def _prepare_motion_weighting(
        self,
        header: Any,
        n_timepoints: int,
    ) -> _DCEMotionWeighting:
        """Build phase-adaptive motion weights from framewise displacement."""
        if not (
            isdefined(self.inputs.fd_values)
            and self.inputs.fd_values
            and isdefined(self.inputs.bolus_arrival_frame)
        ):
            return _DCEMotionWeighting(None, {"applied": False})

        min_voxel_dim = float(min(header.get_zooms()[:3]))
        bolus_frame = int(self.inputs.bolus_arrival_frame)
        weights = _compute_motion_weights(
            fd_values=self.inputs.fd_values,
            bolus_frame=bolus_frame,
            n_frames=n_timepoints,
            min_voxel_dim=min_voxel_dim,
        )
        n_downweighted = int(np.sum(weights < 0.95))
        washin_end = min(bolus_frame + 4, n_timepoints - 1)
        qc = {
            "applied": True,
            "n_downweighted_frames": n_downweighted,
            "min_weight": float(np.min(weights)),
            "washin_weights": weights[bolus_frame:washin_end + 1].tolist(),
        }
        logger.info(
            "Motion weighting: %d/%d frames downweighted (min weight=%.3f)",
            n_downweighted,
            n_timepoints,
            float(np.min(weights)),
        )
        return _DCEMotionWeighting(weights, qc)

    def _initialize_parameter_maps(
        self,
        prepared: _DCEFitPreparation,
        model_name: str,
        candidate_names: list[str],
    ) -> _DCEParameterMaps:
        """Allocate mutable output state before pharmacokinetic fitting."""
        from pmriprep.utils.dce_constants import DCE_AIF_G6_SIGMA_INFLATION

        spatial_shape = prepared.concentration.shape[:3]
        fill_value = np.nan if prepared.is_auto else 0.0
        quality_flag_codes = {
            "good": 1,
            "poor_convergence": 2,
            "high_uncertainty": 3,
            "unreliable_high_kep": 4,
        }
        model_selection_codes = {
            "patlak": 1,
            "tofts": 2,
            "extended_tofts": 3,
            "2cxm": 4,
        }
        base_inflation = float(self.inputs.aif_uncertainty_inflation_factor)
        if base_inflation <= 1.0 + 1e-9:
            sigma_factor = {
                parameter: 1.0 for parameter in DCE_AIF_G6_SIGMA_INFLATION
            }
        else:
            sigma_factor = {
                parameter: 1.0 + (factor - 1.0) * (base_inflation - 1.0)
                for parameter, factor in DCE_AIF_G6_SIGMA_INFLATION.items()
            }
        fitted_curves = (
            np.zeros(prepared.concentration.shape, dtype=np.float32)
            if prepared.save_fitted_curves
            else None
        )
        return _DCEParameterMaps(
            np.full(spatial_shape, fill_value, dtype=np.float64),
            np.full(spatial_shape, fill_value, dtype=np.float64),
            np.full(spatial_shape, fill_value, dtype=np.float64),
            np.full(spatial_shape, fill_value, dtype=np.float64),
            np.zeros(spatial_shape, dtype=np.float64),
            np.zeros(spatial_shape, dtype=np.float64),
            np.zeros(spatial_shape, dtype=np.float64),
            np.zeros(spatial_shape, dtype=np.float64),
            np.zeros(spatial_shape, dtype=np.int8),
            quality_flag_codes,
            np.zeros(spatial_shape, dtype=np.int8),
            model_selection_codes,
            np.full(spatial_shape, np.nan, dtype=np.float32),
            np.zeros(spatial_shape, dtype=np.float64),
            np.zeros(spatial_shape, dtype=np.float64),
            np.zeros(spatial_shape, dtype=np.float64),
            np.zeros(spatial_shape, dtype=np.float64),
            np.zeros(spatial_shape, dtype=np.float64),
            sigma_factor,
            np.zeros(spatial_shape, dtype=np.float32),
            fitted_curves,
            (prepared.is_auto and "2cxm" in candidate_names) or model_name == "2cxm",
        )

    def _prepare_fit_dispatch(
        self,
        prepared: _DCEFitPreparation,
        aif: npt.NDArray[Any],
        vp_constraint: _DCEVPConstraint,
        motion_weights: Optional[npt.NDArray[np.floating]],
    ) -> _DCEFitDispatch:
        """Build ordered voxel inputs and shared worker initialization state."""
        from pmriprep.utils.dsc_cbv_constraint import ConstraintMode

        mask_coords = np.where(prepared.fitting_mask)
        voxel_data: list[VoxelInput] = []
        for x_raw, y_raw, z_raw in zip(*mask_coords):
            x, y, z = int(x_raw), int(y_raw), int(z_raw)
            voxel_vp: Optional[float] = None
            if (
                vp_constraint.vp_map is not None
                and vp_constraint.mode != ConstraintMode.SKIP
            ):
                vp_value = float(vp_constraint.vp_map[x, y, z])
                if np.isfinite(vp_value) and vp_value > 0:
                    voxel_vp = vp_value
            voxel_data.append(
                VoxelInput(x, y, z, prepared.concentration[x, y, z, :], voxel_vp)
            )

        sloppy = bool(self.inputs.sloppy)
        random_seed = (
            int(self.inputs.random_seed) if isdefined(self.inputs.random_seed) else None
        )
        init_args = (
            aif,
            prepared.time_points,
            vp_constraint.model_name,
            prepared.inference,
            prepared.mcmc_samples,
            prepared.baseline_frames,
            vp_constraint.mode.value,
            vp_constraint.vp_uncertainty,
            prepared.save_fitted_curves,
            motion_weights,
            sloppy,
            random_seed,
            vp_constraint.candidate_names if prepared.is_auto else None,
        )
        return _DCEFitDispatch(
            mask_coords,
            voxel_data,
            init_args,
            max(1, int(self.inputs.n_procs)),
            sloppy,
            random_seed,
        )

    def _run_interface(self, runtime: Any) -> Any:
        """Execute DCE map computation."""
        prepared = self._prepare_fit_inputs()
        conc_data = prepared.concentration
        affine = prepared.affine
        header = prepared.header
        mask = prepared.mask
        aif = prepared.aif
        time_points = prepared.time_points
        n_timepoints = prepared.n_timepoints
        selection_tr_seconds = prepared.selection_tr_seconds
        model_name = prepared.model_name
        is_auto = prepared.is_auto
        candidate_names = prepared.candidate_names
        baseline_frames = prepared.baseline_frames
        tsnr_threshold = prepared.tsnr_threshold
        tsnr_map = prepared.tsnr_map
        peak_enh = prepared.peak_enhancement
        fitting_mask = prepared.fitting_mask
        mask_bool = mask > 0
        inference = prepared.inference
        mcmc_samples = prepared.mcmc_samples
        save_fitted_curves = prepared.save_fitted_curves

        bat_alignment = self._align_aif_to_tissue(prepared)
        aif = bat_alignment.aif
        bat_correction = bat_alignment.correction

        vp_constraint = self._prepare_vp_constraint(prepared, aif)
        vp_constraint_map = vp_constraint.vp_map
        constraint_mode = vp_constraint.mode
        constraint_vp_uncertainty = vp_constraint.vp_uncertainty
        constraint_qc = vp_constraint.qc
        model_name = vp_constraint.model_name
        candidate_names = vp_constraint.candidate_names
        model_free = self._compute_model_free_maps(prepared)
        iauc60 = model_free.iauc60
        iauc90 = model_free.iauc90
        iauc120 = model_free.iauc120
        peak_enh = model_free.peak_enhancement
        ttp = model_free.time_to_peak
        ser = model_free.signal_enhancement_ratio
        wash_in_slope = model_free.wash_in_slope
        wash_out_slope = model_free.wash_out_slope
        bat = model_free.bolus_arrival_time
        curve_type = model_free.curve_type
        parameter_maps = self._initialize_parameter_maps(
            prepared, model_name, candidate_names,
        )
        ktrans = parameter_maps.ktrans
        ve = parameter_maps.ve
        vp = parameter_maps.vp
        kep = parameter_maps.kep
        r2_map = parameter_maps.r_squared
        fp_map = parameter_maps.fp
        ps_map = parameter_maps.ps
        e_map = parameter_maps.extraction_fraction
        quality_flag_map = parameter_maps.quality_flag
        model_selection_map = parameter_maps.model_selection
        _MODEL_SELECTION_CODES = parameter_maps.model_selection_codes
        delta_bic_map = parameter_maps.delta_bic
        ktrans_std = parameter_maps.ktrans_std
        ve_std = parameter_maps.ve_std
        vp_std = parameter_maps.vp_std
        fp_std = parameter_maps.fp_std
        ps_std = parameter_maps.ps_std
        vp_at_bound_map = parameter_maps.vp_at_bound
        fitted_curves = parameter_maps.fitted_curves
        save_2cxm = parameter_maps.save_2cxm

        # Get voxel coordinates to fit
        mask_coords = np.where(fitting_mask)
        n_voxels = len(mask_coords[0])

        # QC tracking
        n_brain = int(np.sum(mask_bool))
        qc_metrics: dict[str, Any] = {
            "model": "auto" if is_auto else model_name,
            "inference": inference,
            "n_voxels_brain": n_brain,
            "n_voxels_fitted": n_voxels,
            "n_voxels_signal_filtered": n_brain - n_voxels,
            "tsnr_threshold": tsnr_threshold,
            "baseline_frames": baseline_frames,
            "selection_tr_seconds": selection_tr_seconds,  # TR used for model selection
            "dsc_cbv_constraint": constraint_qc,
            "bat_correction": bat_correction,
        }

        n_converged = 0
        n_failed = 0
        n_constrained = 0
        n_unconstrained_fallback = 0
        n_rejected_postfit = 0
        r2_list: list[float] = []
        failure_reason_counts: dict[str, int] = {}
        mcmc_rhat_lists: dict[str, list[float]] = {}
        mcmc_ess_lists: dict[str, list[float]] = {}
        n_mcmc_voxels: int = 0
        z_departure_list: list[float] = []  # soft constraint prior departure

        motion = self._prepare_motion_weighting(header, n_timepoints)
        motion_weights = motion.weights
        qc_metrics["motion_weighting"] = motion.qc

        dispatch = self._prepare_fit_dispatch(
            prepared, aif, vp_constraint, motion_weights,
        )
        mask_coords = dispatch.mask_coords
        voxel_data_list = dispatch.voxel_data
        init_args = dispatch.init_args
        n_procs = dispatch.n_procs
        sloppy = dispatch.sloppy
        random_seed = dispatch.random_seed

        # Always use batched fitting (vectorized operations)
        results = self._fit_batched(
            conc_data=conc_data,
            mask_coords=mask_coords,
            aif=aif,
            time_points=time_points,
            model_name=model_name,
            inference=inference,
            is_auto=is_auto,
            candidate_names=candidate_names,
            vp_constraint_map=vp_constraint_map,
            constraint_mode=constraint_mode,
            constraint_vp_uncertainty=constraint_vp_uncertainty,
            motion_weights=motion_weights,
            save_fitted_curves=save_fitted_curves,
            sloppy=sloppy,
            mcmc_samples=mcmc_samples,
            baseline_frames=baseline_frames,
            n_procs=n_procs,
            random_seed=random_seed,
            init_args=init_args,
            voxel_data_list=voxel_data_list,
        )

        # Clear worker state to avoid stale references across runs
        _worker_ctx.clear()

        # Per-voxel model selection tracking
        from collections import Counter
        model_counts: Counter[str] = Counter()

        # Scatter results into output arrays
        # For auto mode, we always save ve/vp/2cxm maps (NaN where not applicable)
        save_ve = is_auto or self._has_ve(model_name)
        save_vp = is_auto or self._has_vp(model_name)
        for vd, res in zip(voxel_data_list, results):
            if res["failed"]:
                n_failed += 1
                reason = res.get("failure_reason", "unknown")
                failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1
                continue

            x, y, z = res["coords"]
            params = res["params"]

            # Per-voxel effective model name (may differ from global model_name
            # when BIC model selection is active)
            effective_model_name = res.get("selected_model_name", model_name)

            # Record model selection
            model_selection_map[x, y, z] = _MODEL_SELECTION_CODES.get(
                effective_model_name, 0
            )
            model_counts[effective_model_name] += 1

            # Record delta-BIC
            delta_bic_map[x, y, z] = float(res.get("delta_bic", np.nan))

            # Post-fit validation: reject bound-railing fits only.
            # Parameters at exact optimizer bounds are degenerate (optimizer
            # wanted to exceed the bound). R² is NOT used for rejection —
            # low-R² voxels with valid params have uncertain but informative
            # estimates. Zeroing them biases distributions upward and
            # destroys information. R² quality gating belongs in the QC mask
            # (line 1354: r2_map >= 0.5), separate from parameter estimation.
            is_constrained = res.get("constrained", False)
            degenerate = _is_degenerate_dce_fit(
                effective_model_name,
                params,
                is_constrained,
            )
            if degenerate:
                _scatter_degenerate_dce_fit(
                    parameter_maps,
                    (x, y, z),
                    effective_model_name,
                    is_auto,
                    res["r_squared"],
                )
                n_rejected_postfit += 1
                continue

            _scatter_dce_parameters(
                parameter_maps,
                (x, y, z),
                effective_model_name,
                params,
                res["r_squared"],
            )

            # Uncertainty (Bayesian only). Per-parameter inflation
            # propagates AIF reduced-confidence to PK posterior sigma;
            # ve uninflated.
            if inference != "nlls" and res["uncertainties"]:
                _scatter_dce_uncertainties(
                    parameter_maps,
                    (x, y, z),
                    effective_model_name,
                    res["uncertainties"],
                )

            _scatter_dce_fit_artifacts(parameter_maps, (x, y, z), res)

            z_departure = _compute_vp_prior_departure(
                constraint_mode,
                inference,
                effective_model_name,
                vd.voxel_vp,
                params.get("vp", 0.0),
                constraint_vp_uncertainty,
            )
            if z_departure is not None:
                z_departure_list.append(z_departure)

            accounting = _account_retained_dce_fit(vd, res, constraint_mode)
            n_converged += accounting.n_converged
            n_failed += accounting.n_failed
            n_constrained += accounting.n_constrained
            n_unconstrained_fallback += accounting.n_unconstrained_fallback
            if accounting.r_squared is not None:
                r2_list.append(accounting.r_squared)

            n_mcmc_voxels += _accumulate_mcmc_diagnostics(
                res,
                mcmc_rhat_lists,
                mcmc_ess_lists,
            )

        # Compile QC metrics
        qc_metrics.update(
            _summarize_dce_fit_outcomes(
                n_converged,
                n_failed,
                n_voxels,
                n_constrained,
                n_unconstrained_fallback,
                n_rejected_postfit,
                failure_reason_counts,
            )
        )

        # FXL-suspect whole-brain flag. Tissue Ct peaks above ~0.8 mM
        # violate the Tofts fast-exchange-limit assumption. conc_data is
        # already baseline-subtracted
        # (C_t = (R1(t) - R1(0))/r1). Tumor-mask-stratified fraction
        # emitted in _generate_dce_pk_fit_qc. Shutter-speed model is not
        # viable at this pipeline's clinical TR_vol — report only.
        qc_metrics["frac_fxl_risk_whole_brain"] = _fraction_fxl_risk(
            conc_data,
            fitting_mask,
        )

        qc_metrics.update(
            _summarize_dce_categorical_qc(
                is_auto,
                candidate_names,
                model_selection_map,
                _MODEL_SELECTION_CODES,
                quality_flag_map,
                fitting_mask,
            )
        )

        qc_metrics.update(_summarize_dce_r2(r2_list))

        mcmc_diag = _summarize_mcmc_diagnostics(
            n_mcmc_voxels,
            mcmc_rhat_lists,
            mcmc_ess_lists,
        )
        if mcmc_diag is not None:
            qc_metrics["mcmc_diagnostics"] = mcmc_diag

        qc_metrics.update(_summarize_vp_prior_departures(z_departure_list))

        qc_metrics.update(
            _summarize_dce_fitted_parameters(
                fitting_mask,
                ktrans,
                ve,
                vp,
                fp_map,
                ps_map,
                save_ve,
                save_vp,
                save_2cxm,
            )
        )

        # Compute whole-brain fraction of SOFT-NLLS voxels that ended
        # at a bound (active-set Pass 2 fired). Reported in
        # constraint_qc; downstream dce_pk_maps_qc stratifies over ROI
        # masks. Zero everywhere for HARD / SKIP / Bayesian fits.
        constraint_qc["fraction_vp_at_bound_whole_brain"] = _fraction_vp_at_bound(
            mask_bool,
            vp_at_bound_map,
        )

        # Zero background (outside brain mask) on every float map before
        # save. Preserves in-brain NaN so auto-mode "parameter not applicable
        # for the winning model" semantics still reach downstream consumers.
        ktrans = _zero_dce_background(ktrans, mask_bool)
        ve = _zero_dce_background(ve, mask_bool)
        vp = _zero_dce_background(vp, mask_bool)
        kep = _zero_dce_background(kep, mask_bool)
        r2_map = _zero_dce_background(r2_map, mask_bool)
        fp_map = _zero_dce_background(fp_map, mask_bool)
        ps_map = _zero_dce_background(ps_map, mask_bool)
        e_map = _zero_dce_background(e_map, mask_bool)
        delta_bic_map = _zero_dce_background(delta_bic_map, mask_bool)
        ktrans_std = _zero_dce_background(ktrans_std, mask_bool)
        ve_std = _zero_dce_background(ve_std, mask_bool)
        vp_std = _zero_dce_background(vp_std, mask_bool)
        fp_std = _zero_dce_background(fp_std, mask_bool)
        ps_std = _zero_dce_background(ps_std, mask_bool)
        tsnr_map = _zero_dce_background(tsnr_map, mask_bool)
        iauc60 = _zero_dce_background(iauc60, mask_bool)
        iauc90 = _zero_dce_background(iauc90, mask_bool)
        iauc120 = _zero_dce_background(iauc120, mask_bool)
        ttp = _zero_dce_background(ttp, mask_bool)
        ser = _zero_dce_background(ser, mask_bool)
        wash_in_slope = _zero_dce_background(wash_in_slope, mask_bool)
        wash_out_slope = _zero_dce_background(wash_out_slope, mask_bool)
        bat = _zero_dce_background(bat, mask_bool)
        curve_type = _zero_dce_background(curve_type, mask_bool)
        quality_flag_map = _zero_dce_background(quality_flag_map, mask_bool)
        model_selection_map = _zero_dce_background(model_selection_map, mask_bool)
        if peak_enh is not None:
            peak_enh = _zero_dce_background(peak_enh, mask_bool)
        if fitted_curves is not None:
            fitted_curves = _zero_dce_background(fitted_curves, mask_bool)

        _save_dce_map = partial(_write_dce_map, affine=affine, header=header)
        parameter_outputs = _write_dce_parameter_maps(
            parameter_maps._replace(
                ktrans=ktrans,
                ve=ve,
                vp=vp,
                kep=kep,
                r_squared=r2_map,
                fp=fp_map,
                ps=ps_map,
                extraction_fraction=e_map,
                quality_flag=quality_flag_map,
                vp_at_bound=vp_at_bound_map,
            ),
            mask_bool,
            save_ve,
            save_vp,
            self._gen_outfilename,
            _save_dce_map,
        )

        model_free_outputs = _write_dce_model_free_maps(
            _DCEModelFreeMaps(
                iauc60,
                iauc90,
                iauc120,
                peak_enh,
                ttp,
                ser,
                wash_in_slope,
                wash_out_slope,
                bat,
                curve_type,
            ),
            self._gen_outfilename,
            _save_dce_map,
        )

        # Initialize results
        self._results: dict[str, Any] = {
            "qc_metrics": qc_metrics,
            "constraint_qc": constraint_qc,
        }
        self._results.update(parameter_outputs)
        self._results.update(model_free_outputs)

        # Save uncertainty maps
        if inference != "nlls":
            out_ktrans_std = self._gen_outfilename("_Ktrans_std")
            _save_dce_map(ktrans_std, out_ktrans_std)
            self._results["out_ktrans_std"] = out_ktrans_std

            if save_ve:
                out_ve_std = self._gen_outfilename("_ve_std")
                _save_dce_map(ve_std, out_ve_std)
                self._results["out_ve_std"] = out_ve_std

            if save_vp:
                out_vp_std = self._gen_outfilename("_vp_std")
                _save_dce_map(vp_std, out_vp_std)
                self._results["out_vp_std"] = out_vp_std

            # 2CXM-specific: save Fp and PS uncertainty maps
            if save_2cxm:
                out_fp_std_file = self._gen_outfilename("_Fp_std")
                _save_dce_map(fp_std, out_fp_std_file)
                self._results["out_fp_std"] = out_fp_std_file

                out_ps_std_file = self._gen_outfilename("_PS_std")
                _save_dce_map(ps_std, out_ps_std_file)
                self._results["out_ps_std"] = out_ps_std_file

            # Compute and save Coefficient of Variation (CV = std / mean)
            # Use safe division to avoid divide-by-zero
            ktrans_cv = _compute_dce_coefficient_of_variation(
                ktrans,
                ktrans_std,
                mask_bool,
                is_auto,
            )

            out_ktrans_cv = self._gen_outfilename("_Ktrans_cv")
            _save_dce_map(ktrans_cv, out_ktrans_cv)
            self._results["out_ktrans_cv"] = out_ktrans_cv

            if save_ve:
                ve_cv = _compute_dce_coefficient_of_variation(
                    ve,
                    ve_std,
                    mask_bool,
                    is_auto,
                )
                out_ve_cv = self._gen_outfilename("_ve_cv")
                _save_dce_map(ve_cv, out_ve_cv)
                self._results["out_ve_cv"] = out_ve_cv

            if save_vp:
                vp_cv = _compute_dce_coefficient_of_variation(
                    vp,
                    vp_std,
                    mask_bool,
                    is_auto,
                )
                out_vp_cv = self._gen_outfilename("_vp_cv")
                _save_dce_map(vp_cv, out_vp_cv)
                self._results["out_vp_cv"] = out_vp_cv

            # Compute and save 95% Credible Intervals (mean ± 1.96 * std)
            # Assuming approximately normal posterior (valid for MAP/MLE)
            ktrans_ci95_lower, ktrans_ci95_upper = _compute_dce_credible_interval(
                ktrans,
                ktrans_std,
            )

            out_ktrans_ci95_lower = self._gen_outfilename("_Ktrans_ci95lower")
            out_ktrans_ci95_upper = self._gen_outfilename("_Ktrans_ci95upper")

            _save_dce_map(ktrans_ci95_lower, out_ktrans_ci95_lower)
            _save_dce_map(ktrans_ci95_upper, out_ktrans_ci95_upper)

            self._results["out_ktrans_ci95_lower"] = out_ktrans_ci95_lower
            self._results["out_ktrans_ci95_upper"] = out_ktrans_ci95_upper

            if save_ve:
                ve_ci95_lower, ve_ci95_upper = _compute_dce_credible_interval(
                    ve,
                    ve_std,
                    upper_bound=1.0,
                )

                out_ve_ci95_lower = self._gen_outfilename("_ve_ci95lower")
                out_ve_ci95_upper = self._gen_outfilename("_ve_ci95upper")

                _save_dce_map(ve_ci95_lower, out_ve_ci95_lower)
                _save_dce_map(ve_ci95_upper, out_ve_ci95_upper)

                self._results["out_ve_ci95_lower"] = out_ve_ci95_lower
                self._results["out_ve_ci95_upper"] = out_ve_ci95_upper

            if save_vp:
                vp_ci95_lower, vp_ci95_upper = _compute_dce_credible_interval(
                    vp,
                    vp_std,
                    upper_bound=1.0,
                )

                out_vp_ci95_lower = self._gen_outfilename("_vp_ci95lower")
                out_vp_ci95_upper = self._gen_outfilename("_vp_ci95upper")

                _save_dce_map(vp_ci95_lower, out_vp_ci95_lower)
                _save_dce_map(vp_ci95_upper, out_vp_ci95_upper)

                self._results["out_vp_ci95_lower"] = out_vp_ci95_lower
                self._results["out_vp_ci95_upper"] = out_vp_ci95_upper

        # Save tSNR map (always computed)
        out_tsnr = self._gen_outfilename("_tSNR")
        _save_dce_map(tsnr_map, out_tsnr)
        self._results["out_tsnr"] = out_tsnr

        # Create QC mask using physiological bounds
        # Include voxels with: good R², physiological parameter values, adequate SNR
        qc_mask = _build_dce_qc_mask(
            fitting_mask,
            tsnr_map,
            peak_enh,
            ktrans,
            r2_map,
            ve,
            vp,
            kep,
            model_selection_map,
            _MODEL_SELECTION_CODES,
            is_auto,
            model_name,
        )
        out_qc_mask = self._gen_outfilename("_qc_mask")
        _save_dce_map(qc_mask, out_qc_mask, dtype=np.uint8)
        self._results["out_qc_mask"] = out_qc_mask

        # Save model selection map (auto/BIC mode)
        if is_auto:
            out_model_sel = self._gen_outfilename("_modelSelection")
            _save_dce_map(model_selection_map, out_model_sel, dtype=np.int8)
            self._results["out_model_selection"] = out_model_sel

            # Save delta-BIC confidence map
            out_delta_bic = self._gen_outfilename("_deltaBIC")
            _save_dce_map(delta_bic_map, out_delta_bic)
            self._results["out_delta_bic"] = out_delta_bic

        # Save fitted curves
        if save_fitted_curves and fitted_curves is not None:
            out_fitted = self._gen_outfilename("_fitted_curves")
            _write_dce_fitted_curves(fitted_curves, out_fitted, affine)
            self._results["out_fitted_curves"] = out_fitted

        return runtime

    def _dispatch_batched_chunks(
        self,
        model_name: str,
        tissue_curves: npt.NDArray[np.floating],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        fixed_vp: Optional[npt.NDArray[np.floating]],
        weights: Optional[npt.NDArray[np.floating]],
        save_fitted_curves: bool,
        compute_uncertainties: bool,
        n_procs: int,
        n_voxels: int,
        vp_uncertainty: Optional[float] = None,
    ) -> Any:
        """Split voxels into n_procs chunks and fit each chunk in parallel.

        Each worker runs batched LM / lstsq on its slice (preserving the
        NumPy vectorization win) while n_procs workers run in parallel
        (giving multi-core utilization). The parent concatenates the
        per-chunk FitResults in voxel order.

        Falls back to a single in-process call when ``n_procs <= 1`` or
        the voxel count is below ``max(1000, 4 * n_procs)`` — below that
        threshold, process-spawn overhead (~100 ms) dominates the batched
        LM runtime and multiproc is a net loss.

        ``vp_uncertainty`` is the relative σ for the SOFT box constraint;
        passed through to ``_fit_chunk_nlls`` via ``BatchedChunkArgs``.
        Pass None for HARD / unconstrained.
        """
        from pmriprep.models.dce.batched_fitting import concat_fit_results

        compute_unc = bool(compute_uncertainties)
        save_curves = bool(save_fitted_curves)

        chunk_floor = max(1000, 4 * n_procs)
        if n_procs <= 1 or n_voxels < chunk_floor:
            return _fit_chunk_nlls(
                BatchedChunkArgs(
                    model_name=model_name,
                    tissue_curves=tissue_curves,
                    aif=aif,
                    time_points=time_points,
                    fixed_vp=fixed_vp,
                    vp_uncertainty=vp_uncertainty,
                    weights=weights,
                    save_fitted_curves=save_curves,
                    compute_uncertainties=compute_unc,
                )
            )

        # Contiguous index chunks preserve voxel order across concat.
        chunk_index_arrays = np.array_split(np.arange(n_voxels), n_procs)
        chunk_args: list[BatchedChunkArgs] = []
        for idx in chunk_index_arrays:
            tc_chunk = tissue_curves[idx]
            fvp_chunk = fixed_vp[idx] if fixed_vp is not None else None
            chunk_args.append(
                BatchedChunkArgs(
                    model_name=model_name,
                    tissue_curves=tc_chunk,
                    aif=aif,
                    time_points=time_points,
                    fixed_vp=fvp_chunk,
                    vp_uncertainty=vp_uncertainty,
                    weights=weights,
                    save_fitted_curves=save_curves,
                    compute_uncertainties=compute_unc,
                )
            )

        from concurrent.futures import ProcessPoolExecutor

        executor = ProcessPoolExecutor(
            max_workers=n_procs,
            initializer=_init_batched_worker,
        )
        try:
            chunk_results = list(executor.map(_fit_chunk_nlls, chunk_args))
        except (KeyboardInterrupt, SystemExit):
            logger.warning(
                "Interrupted during parallel DCE batched fitting, "
                "cancelling workers"
            )
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        return concat_fit_results(chunk_results, n_voxels)

    def _fit_batched(
        self,
        conc_data: npt.NDArray[Any],
        mask_coords: tuple[npt.NDArray[Any], ...],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        model_name: str,
        inference: str,
        is_auto: bool,
        candidate_names: list[str],
        vp_constraint_map: Optional[npt.NDArray[Any]],
        constraint_mode: Any,  # ConstraintMode enum
        constraint_vp_uncertainty: float,
        motion_weights: Optional[npt.NDArray[np.floating]],
        save_fitted_curves: bool,
        sloppy: bool,
        mcmc_samples: int,
        baseline_frames: int,
        n_procs: int,
        random_seed: Optional[int],
        init_args: tuple[Any, ...],
        voxel_data_list: list[VoxelInput],
    ) -> list[dict[str, Any]]:
        """Batched fitting path.

        Fits all voxels simultaneously using batched LM/lstsq,
        then optionally refines with CPU-based MAP/MCMC.

        Parameters
        ----------
        conc_data : numpy.ndarray
            Concentration data, shape (X, Y, Z, T).
        mask_coords : tuple of numpy.ndarray
            Tuple of coordinate arrays from np.where(fitting_mask).
        aif : numpy.ndarray
            AIF, shape (T,).
        time_points : numpy.ndarray
            Time points, shape (T,).
        model_name : str
            Primary model name.
        inference : str
            Inference engine ('nlls', 'map', 'vi', 'mcmc').
        is_auto : bool
            Whether auto-mode BIC selection is active.
        candidate_names : list
            List of candidate model names for auto mode.
        vp_constraint_map : numpy.ndarray or None
            vp constraint map or None.
        constraint_mode : ConstraintMode
            Constraint mode (SKIP, SOFT).
        constraint_vp_uncertainty : float
            Uncertainty for soft constraint.
        motion_weights : numpy.ndarray or None
            Per-frame motion weights or None.
        save_fitted_curves : bool
            Whether to save fitted curves.
        sloppy : bool
            Whether to use sloppy mode.
        mcmc_samples : int
            Number of MCMC samples.
        baseline_frames : int
            Number of baseline frames.
        n_procs : int
            Number of CPU processes for MAP/MCMC refinement.
        random_seed : int or None
            Random seed or None.
        init_args : tuple
            Worker init args for CPU fallback.
        voxel_data_list : list of VoxelInput
            List of VoxelInput for CPU fallback paths.

        Returns
        -------
        list
            List of result dicts matching _fit_single_voxel output format.
        """
        from pmriprep.models.dce.batched_fitting import FitResult
        from pmriprep.utils.dsc_cbv_constraint import ConstraintMode

        n_voxels = len(mask_coords[0])
        n_timepoints = len(aif)

        # VI/MCMC: fall through to CPU multiprocessing (unchanged)
        if inference in ("vi", "mcmc"):
            logger.info("Batched mode with %s: falling through to CPU multiprocessing", inference)
            _init_worker(*init_args)
            if n_procs > 1:
                from concurrent.futures import ProcessPoolExecutor
                chunk_size = max(1, n_voxels // (n_procs * 10))
                executor = ProcessPoolExecutor(
                    max_workers=n_procs,
                    initializer=_init_worker,  # type: ignore[arg-type]
                    initargs=init_args,
                )
                try:
                    return list(executor.map(
                        _fit_single_voxel, voxel_data_list, chunksize=chunk_size,
                    ))
                except (KeyboardInterrupt, SystemExit):
                    logger.warning("Interrupted during parallel DCE fitting, cancelling workers")
                    raise
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)
            return [_fit_single_voxel(vd) for vd in voxel_data_list]

        # Build batched arrays
        tissue_curves = conc_data[mask_coords[0], mask_coords[1], mask_coords[2], :]
        tissue_curves = np.nan_to_num(tissue_curves, nan=0.0, posinf=0.0, neginf=0.0)
        coords = np.stack(mask_coords, axis=1)  # (N, 3)

        # Build fixed_vp array
        fixed_vp: Optional[npt.NDArray[np.floating]] = None
        if vp_constraint_map is not None and constraint_mode != ConstraintMode.SKIP:
            vp_vals = vp_constraint_map[mask_coords[0], mask_coords[1], mask_coords[2]]
            valid_vp = np.isfinite(vp_vals) & (vp_vals > 0)
            fixed_vp = np.where(valid_vp, vp_vals, 0.0)
            # Zero out invalid entries so they're treated as unconstrained
            fixed_vp[~valid_vp] = 0.0

        # vp_uncertainty is honored on the batched path only when the
        # user explicitly selected inference=nlls + SOFT. For inference=map + SOFT,
        # the batched Pass acts as HARD warm-start and the CPU MAP refinement
        # downstream applies the Gaussian prior via create_vp_prior_from_dsc.
        # For HARD/SKIP, vp_uncertainty is moot (None).
        vp_uncertainty_batched: Optional[float] = None
        if (
            constraint_mode == ConstraintMode.SOFT
            and inference == "nlls"
            and fixed_vp is not None
        ):
            vp_uncertainty_batched = float(constraint_vp_uncertainty)

        results: list[dict[str, Any]] = []

        if not is_auto:
            # Single-model batched fit with split constraint handling,
            # parallelized across n_procs voxel chunks.
            fit_result = self._dispatch_batched_chunks(
                model_name=model_name,
                tissue_curves=tissue_curves,
                aif=aif,
                time_points=time_points,
                fixed_vp=fixed_vp,
                vp_uncertainty=vp_uncertainty_batched,
                weights=motion_weights,
                save_fitted_curves=(
                    save_fitted_curves or inference == "map"
                ) if model_name != "patlak" else save_fitted_curves,
                compute_uncertainties=inference == "nlls",
                n_procs=n_procs,
                n_voxels=n_voxels,
            )

            results = self._fit_result_to_dicts(
                fit_result, coords, model_name,
                fixed_vp=fixed_vp,
            )

            # MAP refinement: use batched fit as warm-start for CPU MAP
            if inference == "map":
                results = self._refine_map_from_batched(
                    results, voxel_data_list, init_args, n_procs,
                )

            return results

        # Auto mode: fit ALL candidate models using batched fitting, select via BIC
        from pmriprep.models.dce.base import EPSILON_RSS

        fit_results: dict[str, FitResult] = {}
        for cname in candidate_names:
            try:
                fit_results[cname] = self._dispatch_batched_chunks(
                    model_name=cname,
                    tissue_curves=tissue_curves,
                    aif=aif,
                    time_points=time_points,
                    fixed_vp=fixed_vp,
                    vp_uncertainty=vp_uncertainty_batched,
                    weights=motion_weights,
                    save_fitted_curves=True,
                    compute_uncertainties=False,
                    n_procs=n_procs,
                    n_voxels=n_voxels,
                )
            except Exception:
                logger.warning("Batched auto-mode: model %s failed", cname, exc_info=True)

        if not fit_results:
            logger.warning("All batched auto-mode models failed, falling to CPU")
            _init_worker(*init_args)
            return [_fit_single_voxel(vd) for vd in voxel_data_list]

        # BIC per model per voxel
        T = n_timepoints
        _N_PARAMS: dict[str, int] = {
            "tofts": 2, "extended_tofts": 3, "patlak": 2, "2cxm": 4,
        }

        bic_arrays: dict[str, npt.NDArray[np.floating]] = {}
        for cname, result in fit_results.items():
            if result.fitted_curves is None:
                bic_arrays[cname] = np.full(n_voxels, np.inf)
                continue
            rss = np.sum((tissue_curves - result.fitted_curves) ** 2, axis=1)
            rss_safe = np.maximum(rss, EPSILON_RSS)
            # Per-voxel effective param count:
            # SOFT box constraint: k = k_base constant. Self-Liang (JASA
            # 1987) shows boundary-parameter effective d.o.f. lies between
            # k_base-1 and k_base; k_base with the full data likelihood is
            # the standard pragmatic choice and the only one that keeps
            # Patlak / Extended Tofts BIC comparison stable. Unconstrained
            # path also uses k_base.
            k_base = _N_PARAMS[cname]
            k_arr = np.full(n_voxels, k_base, dtype=float)
            bic_arrays[cname] = T * np.log(rss_safe / T) + k_arr * np.log(T)

        # Filter non-converged → BIC = inf
        for cname, result in fit_results.items():
            if cname in bic_arrays:
                bic_arrays[cname][~result.converged] = np.inf

        # Select min-BIC model per voxel
        cand_list = [c for c in candidate_names if c in fit_results]
        bic_stack = np.stack([bic_arrays[c] for c in cand_list], axis=0)
        winner_idx = np.argmin(bic_stack, axis=0)

        # Compute delta-BIC (margin over runner-up)
        # Non-converged fits have BIC=inf. When computing delta-BIC,
        # inf - finite = inf, which breaks percentile calculations.
        # Convert to nan for cleaner downstream handling.
        sorted_bics = np.sort(bic_stack, axis=0)
        if sorted_bics.shape[0] > 1:
            # Both winner and runner-up must be finite for meaningful delta
            both_finite = np.isfinite(sorted_bics[0]) & np.isfinite(sorted_bics[1])
            delta_bic = np.where(
                both_finite,
                sorted_bics[1] - sorted_bics[0],
                np.nan,
            )
        else:
            delta_bic = np.full(n_voxels, np.nan)

        # Kass & Raftery guard: require strong BIC evidence to upgrade
        # from simplest model. ΔBIC >= 6 = "strong evidence".
        _MIN_DELTA_BIC_UPGRADE = 6.0

        # Find simplest model (minimum parameters) using existing _N_PARAMS dict
        n_params_arr = np.array([_N_PARAMS[c] for c in cand_list])
        simplest_idx = int(np.argmin(n_params_arr))

        # Apply guard: downgrade to simplest if evidence is weak
        # Only downgrade if simplest model actually converged
        weak_evidence = (delta_bic < _MIN_DELTA_BIC_UPGRADE) | ~np.isfinite(delta_bic)
        not_simplest = winner_idx != simplest_idx
        simplest_converged = fit_results[cand_list[simplest_idx]].converged
        downgrade_mask = weak_evidence & not_simplest & simplest_converged

        n_downgraded = int(downgrade_mask.sum())
        if n_downgraded > 0:
            logger.debug(
                "Kass-Raftery guard: downgraded %d voxels to %s (delta-BIC < %.1f)",
                n_downgraded, cand_list[simplest_idx], _MIN_DELTA_BIC_UPGRADE,
            )
            winner_idx[downgrade_mask] = simplest_idx

        # Downgraded voxels have no meaningful delta-BIC
        delta_bic[downgrade_mask] = np.nan

        # Assemble per-voxel results from winners
        results = []
        for i in range(n_voxels):
            winner_name = cand_list[winner_idx[i]]
            winner_result = fit_results[winner_name]

            x, y, z = int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2])

            # Determine constraint state for this voxel
            is_constrained = (
                fixed_vp is not None
                and winner_name in _MODEL_HAS_VP
                and fixed_vp[i] > 0
            )

            # Normalize param names to match CPU output format
            params_out: dict[str, float] = {}
            for pname, parr in winner_result.params.items():
                params_out[_normalize_param_name(pname)] = float(parr[i])

            kep_railed = _compute_derived_params(params_out, winner_name)

            # Uncertainties
            uncertainties: dict[str, float] = {}
            if winner_result.uncertainties:
                for pname, uarr in winner_result.uncertainties.items():
                    uncertainties[_normalize_param_name(pname)] = float(uarr[i])

            fitted_curve = None
            if save_fitted_curves and winner_result.fitted_curves is not None:
                fitted_curve = winner_result.fitted_curves[i]

            # vp_at_bound propagation: read from the winner's per-voxel
            # indicator (non-None only for the SOFT NLLS path).
            vp_at_bound_val = 0.0
            if winner_result.vp_at_bound is not None:
                vp_at_bound_val = float(winner_result.vp_at_bound[i])

            results.append({
                "coords": (x, y, z),
                "params": params_out,
                "r_squared": float(winner_result.r_squared[i]),
                "converged": bool(winner_result.converged[i]),
                "uncertainties": uncertainties,
                "fitted_curve": fitted_curve,
                "quality_flag": (
                    "poor_convergence" if not winner_result.converged[i]
                    else "unreliable_high_kep" if kep_railed
                    else "good"
                ),
                "rhat": {},
                "ess": {},
                "failed": not bool(winner_result.converged[i]) and float(winner_result.r_squared[i]) < -0.5,
                "failure_reason": None,
                "constrained": is_constrained,
                "selected_model_name": winner_name,
                "delta_bic": float(delta_bic[i]),
                "vp_at_bound": vp_at_bound_val,
            })

        # MAP refinement of BIC winners
        if inference == "map":
            results = self._refine_map_from_batched(
                results, voxel_data_list, init_args, n_procs,
            )

        return results

    def _fit_result_to_dicts(
        self,
        fit_result: Any,
        coords: npt.NDArray[Any],
        model_name: str,
        fixed_vp: Optional[npt.NDArray[np.floating]] = None,
    ) -> list[dict[str, Any]]:
        """Convert FitResult to list of per-voxel dicts.

        Computes derived parameters (kep, E, Ktrans_derived) and builds
        dicts matching _fit_single_voxel output format.

        Parameters
        ----------
        fit_result : Any
            FitResult from batched fitting.
        coords : numpy.ndarray
            Voxel coordinates, shape (N, 3).
        model_name : str
            Model name.
        fixed_vp : numpy.ndarray, optional
            Fixed vp values or None.

        Returns
        -------
        list
            List of result dicts.
        """
        n_voxels = len(coords)
        results: list[dict[str, Any]] = []

        for i in range(n_voxels):
            x, y, z = int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2])

            is_constrained = (
                fixed_vp is not None
                and model_name in _MODEL_HAS_VP
                and fixed_vp[i] > 0
            )

            # Build params dict with normalized names
            params_out: dict[str, float] = {}
            for pname, parr in fit_result.params.items():
                params_out[_normalize_param_name(pname)] = float(parr[i])

            kep_railed = _compute_derived_params(params_out, model_name)

            # Uncertainties
            uncertainties: dict[str, float] = {}
            if fit_result.uncertainties:
                for pname, uarr in fit_result.uncertainties.items():
                    uncertainties[_normalize_param_name(pname)] = float(uarr[i])

            fitted_curve = None
            if fit_result.fitted_curves is not None:
                fitted_curve = fit_result.fitted_curves[i]

            # vp_at_bound propagation: only populated for the SOFT NLLS path.
            vp_at_bound_val = 0.0
            if fit_result.vp_at_bound is not None:
                vp_at_bound_val = float(fit_result.vp_at_bound[i])

            results.append({
                "coords": (x, y, z),
                "params": params_out,
                "r_squared": float(fit_result.r_squared[i]),
                "converged": bool(fit_result.converged[i]),
                "uncertainties": uncertainties,
                "fitted_curve": fitted_curve,
                "quality_flag": (
                    "poor_convergence" if not fit_result.converged[i]
                    else "unreliable_high_kep" if kep_railed
                    else "good"
                ),
                "rhat": {},
                "ess": {},
                "failed": not bool(fit_result.converged[i]) and float(fit_result.r_squared[i]) < -0.5,
                "failure_reason": None,
                "constrained": is_constrained,
                "selected_model_name": model_name,
                "delta_bic": np.nan,
                "vp_at_bound": vp_at_bound_val,
            })

        return results

    def _refine_map_from_batched(
        self,
        batched_results: list[dict[str, Any]],
        voxel_data_list: list[VoxelInput],
        init_args: tuple[Any, ...],
        n_procs: int,
    ) -> list[dict[str, Any]]:
        """Run CPU MAP fitting on all voxels.

        Uses CPU MAP where it converges, falling back to batched NLLS otherwise.

        Parameters
        ----------
        batched_results : list
            Per-voxel result dicts from batched fitting.
        voxel_data_list : list of VoxelInput
            VoxelInput list for CPU fitting.
        init_args : tuple
            Worker initialization args.
        n_procs : int
            Number of CPU processes.

        Returns
        -------
        list
            Refined result dicts (CPU MAP where converged, batched NLLS otherwise).
        """
        # CPU MAP refinement using multiprocessing
        _init_worker(*init_args)
        n_voxels = len(voxel_data_list)

        if n_procs > 1:
            from concurrent.futures import ProcessPoolExecutor
            chunk_size = max(1, n_voxels // (n_procs * 10))
            executor = ProcessPoolExecutor(
                max_workers=n_procs,
                initializer=_init_worker,  # type: ignore[arg-type]
                initargs=init_args,
            )
            try:
                cpu_results = list(executor.map(
                    _fit_single_voxel, voxel_data_list, chunksize=chunk_size,
                ))
            except (KeyboardInterrupt, SystemExit):
                logger.warning("Interrupted during hybrid DCE refinement, cancelling workers")
                raise
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        else:
            cpu_results = [_fit_single_voxel(vd) for vd in voxel_data_list]

        # Use CPU MAP results where they converged, batched NLLS otherwise
        final_results = []
        for fit_res, cpu_res in zip(batched_results, cpu_results):
            if cpu_res.get("converged", False) and not cpu_res.get("failed", True):
                final_results.append(cpu_res)
            else:
                final_results.append(fit_res)

        return final_results

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List interface outputs.

        Returns only explicitly set results from _run_interface, avoiding
        stale defaults from output_spec().get() (which returns '.' for
        optional File traits without exists=True).
        """
        if hasattr(self, "_results"):
            return dict(self._results)
        return self.output_spec().get()  # type: ignore[misc,no-any-return]
