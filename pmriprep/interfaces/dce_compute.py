"""DCE map computation interface for pharmacokinetic modeling.

This module implements the main Nipype interface for computing DCE
pharmacokinetic parameter maps (Ktrans, ve, vp, kep) from concentration
curves and AIF using various PK models.
"""

import logging
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional, Literal, cast

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
    EPSILON_VE,
    DEFAULT_TSNR_THRESHOLD,
    KEP_BOUNDS,
    KTRANS_BOUNDS,
    MIN_ENHANCEMENT_MM,
    VE_BOUNDS,
    VP_BOUNDS,
    VP_VE_TOTAL_MAX,
)
from pmriprep.models.dce.retention import is_degenerate_fit as _is_degenerate_dce_fit

logger = logging.getLogger("nipype.interface")

# Absolute fraction tolerance used only to classify optimizer contact with the
# physical volume endpoints in the secondary quantitative QC mask.
_VOLUME_QC_ENDPOINT_ATOL: float = 1e-6

_K2_QC_KEYS = {
    "k2_mean", "k2_std", "k2_min", "k2_max", "k2_median", "k2_p05", "k2_p95",
    "percent_negative_k2", "percent_k2_undefined",
    "n_brain_voxels", "n_k2_valid", "auc_mean",
    "k2_signed", "k2_reference_available",
    "k2_reference_total_voxels", "k2_reference_selected_voxels",
    "k2_reference_selected_fraction", "k2_reference_baseline_frames",
    "k2_reference_tail_frames", "k2_reference_selection",
    "k2_reference_selection_domain", "k2_reference_peak",
    "k2_reference_failure_reason", "leakage_correction_disabled",
}


def _scatter_dce_uncertainties(
    maps: "_DCEParameterMaps",
    coords: tuple[int, int, int],
    model_name: str,
    uncertainties: dict[str, float],
) -> None:
    """Scatter model-applicable standard errors/SDs with AIF inflation."""
    x, y, z = coords
    if (
        model_name != "2cxm"
        and "Ktrans" in uncertainties
        and np.isfinite(uncertainties["Ktrans"])
    ):
        maps.ktrans_std[x, y, z] = uncertainties["Ktrans"] * (
            maps.sigma_factor.get("Ktrans", 1.0)
        )
    if (
        model_name in ("tofts", "extended_tofts", "2cxm")
        and "ve" in uncertainties
        and np.isfinite(uncertainties["ve"])
    ):
        maps.ve_std[x, y, z] = uncertainties["ve"] * (
            maps.sigma_factor.get("ve", 1.0)
        )
    if (
        model_name in ("extended_tofts", "patlak", "2cxm")
        and "vp" in uncertainties
        and np.isfinite(uncertainties["vp"])
    ):
        maps.vp_std[x, y, z] = uncertainties["vp"] * (
            maps.sigma_factor.get("vp", 1.0)
        )
    if model_name == "2cxm":
        if "Fp" in uncertainties and np.isfinite(uncertainties["Fp"]):
            maps.fp_std[x, y, z] = uncertainties["Fp"] * (
                maps.sigma_factor.get("Fp", 1.0)
            )
        if "PS" in uncertainties and np.isfinite(uncertainties["PS"]):
            maps.ps_std[x, y, z] = uncertainties["PS"] * (
                maps.sigma_factor.get("PS", 1.0)
            )


def _scatter_dce_fit_artifacts(
    maps: "_DCEParameterMaps",
    coords: tuple[int, int, int],
    result: dict[str, Any],
) -> None:
    """Scatter fitted-curve artifacts."""
    x, y, z = coords
    fitted_curve = result.get("fitted_curve")
    if maps.fitted_curves is not None and fitted_curve is not None:
        maps.fitted_curves[x, y, z, :] = fitted_curve.astype(np.float32)


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
    r_squared: float,
) -> None:
    """Scatter the model-applicable sentinels for a rejected fit."""
    x, y, z = coords
    fill_value = np.nan
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
        int(constrained),
        int(unconstrained_fallback),
        float(result["r_squared"]) if converged else None,
    )


def _scatter_dce_fit_results(
    voxel_data: list["VoxelInput"],
    results: list[dict[str, Any]],
    maps: "_DCEParameterMaps",
    model_name: str,
    is_auto: bool,
    inference: str,
    constraint_mode: Any,
    constraint_vp_uncertainty: float,
) -> "_DCEFitSummary":
    """Scatter fitted voxels and accumulate their QC diagnostics."""
    n_constrained = 0
    n_optimizer_converged = n_retained = 0
    n_prefit_rejected = n_execution_exceptions = n_fit_unsuccessful = 0
    n_unconstrained_fallback = n_rejected_postfit = 0
    r2_values: list[float] = []
    failure_reasons: dict[str, int] = {}
    vp_prior_departures: list[float] = []
    n_inference_fallback = 0
    for voxel, result in zip(voxel_data, results):
        if result["failed"]:
            failure_stage = result.get("failure_stage", "fit")
            if failure_stage == "prefit":
                n_prefit_rejected += 1
            elif failure_stage == "execution":
                n_execution_exceptions += 1
            else:
                n_fit_unsuccessful += 1
            reason = result.get("failure_reason") or "unknown"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            continue
        n_optimizer_converged += int(bool(result.get("converged", False)))
        coords = tuple(result["coords"])
        maps.optimizer_converged[coords] = bool(result.get("converged", False))
        params = result["params"]
        selected_model = result.get("selected_model_name", model_name)
        maps.model_selection[coords] = maps.model_selection_codes.get(selected_model, 0)
        maps.delta_bic[coords] = float(result.get("delta_bic", np.nan))
        if _is_degenerate_dce_fit(selected_model, params):
            _scatter_degenerate_dce_fit(
                maps, coords, selected_model, result["r_squared"]
            )
            n_rejected_postfit += 1
            continue
        n_retained += 1
        maps.retained_nondegenerate[coords] = True
        n_inference_fallback += int(bool(result.get("inference_fallback", False)))
        _scatter_dce_parameters(
            maps, coords, selected_model, params, result["r_squared"],
        )
        if result["uncertainties"]:
            _scatter_dce_uncertainties(
                maps, coords, selected_model, result["uncertainties"],
            )
        _scatter_dce_fit_artifacts(maps, coords, result)
        departure = _compute_vp_prior_departure(
            constraint_mode,
            inference,
            selected_model,
            voxel.voxel_vp,
            params.get("vp", 0.0),
            constraint_vp_uncertainty,
        )
        if departure is not None:
            vp_prior_departures.append(departure)
        accounting = _account_retained_dce_fit(voxel, result, constraint_mode)
        n_constrained += accounting.n_constrained
        n_unconstrained_fallback += accounting.n_unconstrained_fallback
        if accounting.r_squared is not None:
            r2_values.append(accounting.r_squared)
    return _DCEFitSummary(
        n_optimizer_converged,
        n_prefit_rejected,
        n_execution_exceptions,
        n_fit_unsuccessful,
        n_retained,
        n_constrained,
        n_unconstrained_fallback,
        n_rejected_postfit,
        r2_values,
        failure_reasons,
        vp_prior_departures,
        n_inference_fallback,
    )


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


def _summarize_inference_fallback(
    inference: str,
    n_retained: int,
    n_nlls_fallback: int,
) -> Optional[dict[str, str | int | float]]:
    """Summarize retained Bayesian requests supplied by NLLS fallback fits."""
    if inference == "nlls":
        return None
    return {
        "requested_inference": inference,
        "fallback_inference": "nlls",
        "n_retained_voxels": n_retained,
        "n_nlls_fallback": n_nlls_fallback,
        "fallback_fraction_of_retained": n_nlls_fallback / max(1, n_retained),
    }


def _summarize_dce_model_selection(
    is_auto: bool,
    candidate_names: list[str],
    model_selection: npt.NDArray[Any],
    model_selection_codes: dict[str, int],
) -> dict[str, Any]:
    """Summarize the model-selection categorical map."""
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
    n_voxels: int,
    n_brain_voxels: int,
    n_optimizer_converged: int,
    n_prefit_rejected: int,
    n_execution_exceptions: int,
    n_fit_unsuccessful: int,
    n_retained: int,
    n_constrained: int,
    n_unconstrained_fallback: int,
    n_rejected_postfit: int,
    failure_reasons: dict[str, int],
) -> dict[str, Any]:
    """Assemble aggregate DCE fit outcome counters."""
    accounting = {
        "denominators": {
            "brain": int(n_brain_voxels),
            "attempted": int(n_voxels),
            "retained": int(n_retained),
        },
        "brain_mask": {"count": int(n_brain_voxels)},
        "fitting_eligible": {
            "count": int(n_voxels),
            "fraction_of_brain": n_voxels / max(1, n_brain_voxels),
        },
        "attempted": {
            "count": int(n_voxels),
            "fraction_of_fitting_eligible": 1.0 if n_voxels else 0.0,
        },
        "optimizer_converged": {
            "count": int(n_optimizer_converged),
            "fraction_of_attempted": n_optimizer_converged / max(1, n_voxels),
        },
        "prefit_rejected": {
            "count": int(n_prefit_rejected),
            "fraction_of_attempted": n_prefit_rejected / max(1, n_voxels),
        },
        "execution_exception": {
            "count": int(n_execution_exceptions),
            "fraction_of_attempted": n_execution_exceptions / max(1, n_voxels),
        },
        "fit_unsuccessful": {
            "count": int(n_fit_unsuccessful),
            "fraction_of_attempted": n_fit_unsuccessful / max(1, n_voxels),
        },
        "postfit_degenerate": {
            "count": int(n_rejected_postfit),
            "fraction_of_attempted": n_rejected_postfit / max(1, n_voxels),
        },
        "retained_nondegenerate": {
            "count": int(n_retained),
            "fraction_of_attempted": n_retained / max(1, n_voxels),
        },
    }
    return {
        "n_constrained": n_constrained,
        "n_unconstrained_fallback": n_unconstrained_fallback,
        "failure_reasons": failure_reasons,
        "voxel_accounting": accounting,
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
    optimizer_converged: npt.NDArray[np.bool_],
    retained_nondegenerate: npt.NDArray[np.bool_],
    tsnr: npt.NDArray[Any],
    signal_qc_available: bool,
    ktrans: npt.NDArray[Any],
    r_squared: npt.NDArray[Any],
    ve: npt.NDArray[Any],
    vp: npt.NDArray[Any],
    kep: npt.NDArray[Any],
    model_selection: npt.NDArray[Any],
    model_selection_codes: dict[str, int],
    is_auto: bool,
    model_name: str,
    tsnr_threshold: float = 0.0,
    *, identification_pass: npt.NDArray[np.bool_],
) -> npt.NDArray[np.uint8]:
    """Combine reported-fit quality with selected-NLLS identification."""
    return _evaluate_dce_qc_mask(
        fitting_mask,
        optimizer_converged,
        retained_nondegenerate,
        tsnr,
        signal_qc_available,
        ktrans,
        r_squared,
        ve,
        vp,
        kep,
        model_selection,
        model_selection_codes,
        is_auto,
        model_name,
        tsnr_threshold,
        identification_pass=identification_pass,
    ).mask


class _DCEQCMaskEvaluation(NamedTuple):
    """DCE quantitative-mask result and overlapping component failures."""

    mask: npt.NDArray[np.uint8]
    component_failures: dict[str, int]
    acceptance_policy: dict[str, Any]


def _evaluate_dce_qc_mask(
    fitting_mask: npt.NDArray[np.bool_],
    optimizer_converged: npt.NDArray[np.bool_],
    retained_nondegenerate: npt.NDArray[np.bool_],
    tsnr: npt.NDArray[Any],
    signal_qc_available: bool,
    ktrans: npt.NDArray[Any],
    r_squared: npt.NDArray[Any],
    ve: npt.NDArray[Any],
    vp: npt.NDArray[Any],
    kep: npt.NDArray[Any],
    model_selection: npt.NDArray[Any],
    model_selection_codes: dict[str, int],
    is_auto: bool,
    model_name: str,
    tsnr_threshold: float = 0.0,
    *, identification_pass: npt.NDArray[np.bool_],
) -> _DCEQCMaskEvaluation:
    """Conjoin reported-fit quality and the corresponding NLLS identification."""
    from pmriprep.models.dce.identification import identification_policy

    if identification_pass.shape != fitting_mask.shape or identification_pass.dtype != np.bool_:
        raise ValueError("Identification must be a boolean volume matching the fitting mask.")
    tsnr_ok = (
        tsnr >= tsnr_threshold
        if signal_qc_available and tsnr_threshold > 0
        else np.ones(fitting_mask.shape, dtype=bool)
    )
    ktrans_ok = np.isfinite(ktrans) & (ktrans > 0) & (ktrans < KTRANS_BOUNDS.max)

    # R² > 0 requires the fitted curve to outperform its temporal mean.
    r_squared_ok = np.isfinite(r_squared) & (r_squared > 0.0)
    patlak_volume_ok = (
        np.isfinite(vp)
        & (vp > _VOLUME_QC_ENDPOINT_ATOL)
        & (vp < VP_BOUNDS.max - _VOLUME_QC_ENDPOINT_ATOL)
    )
    tofts_volume_ok = (
        np.isfinite(ve)
        & (ve > _VOLUME_QC_ENDPOINT_ATOL)
        & (ve < VE_BOUNDS.max - _VOLUME_QC_ENDPOINT_ATOL)
    )
    extended_volume_ok = (
        np.isfinite(ve)
        & np.isfinite(vp)
        & (ve > _VOLUME_QC_ENDPOINT_ATOL)
        & (vp > _VOLUME_QC_ENDPOINT_ATOL)
        & ((ve + vp) < VP_VE_TOTAL_MAX - _VOLUME_QC_ENDPOINT_ATOL)
    )
    reversible_kep_ok = (
        np.isfinite(kep)
        & (kep > KEP_BOUNDS.min)
        & (kep <= KEP_BOUNDS.max)
    )

    if is_auto:
        is_patlak = model_selection == model_selection_codes.get("patlak", -1)
        is_tofts = model_selection == model_selection_codes.get("tofts", -1)
        is_extended = model_selection == model_selection_codes.get(
            "extended_tofts", -1
        )
        is_2cxm = model_selection == model_selection_codes.get("2cxm", -1)
        volume_ok = (
            (is_patlak & patlak_volume_ok)
            | (is_tofts & tofts_volume_ok)
            | ((is_extended | is_2cxm) & extended_volume_ok)
        )
        kep_ok = is_patlak | ((is_tofts | is_extended | is_2cxm) & reversible_kep_ok)
    elif model_name == "patlak":
        volume_ok = patlak_volume_ok
        kep_ok = np.ones(fitting_mask.shape, dtype=bool)
    elif model_name == "tofts":
        volume_ok = tofts_volume_ok
        kep_ok = reversible_kep_ok
    elif model_name in ("extended_tofts", "2cxm"):
        volume_ok = extended_volume_ok
        kep_ok = reversible_kep_ok
    else:
        volume_ok = np.zeros(fitting_mask.shape, dtype=bool)
        kep_ok = np.zeros(fitting_mask.shape, dtype=bool)

    components = {
        "optimizer_converged": optimizer_converged,
        "retained_nondegenerate": retained_nondegenerate,
        "tsnr": tsnr_ok,
        "ktrans_range": ktrans_ok,
        "r_squared": r_squared_ok,
        "volume_interior": volume_ok,
        "kep_range": kep_ok,
        "primary_identification": identification_pass,
    }
    qc_mask = fitting_mask.copy()
    for component in components.values():
        qc_mask &= component
    failures = {
        name: int(np.count_nonzero(fitting_mask & ~component))
        for name, component in components.items()
    }
    policy = {
        "acquisition": {
            "minimum_tsnr": (
                tsnr_threshold
                if signal_qc_available and tsnr_threshold > 0
                else None
            ),
        },
        "fit": {
            "minimum_r_squared_exclusive": 0.0,
            "requires_optimizer_convergence": True,
            "requires_retained_nondegenerate_fit": True,
            "ktrans_minimum_per_min_exclusive": 0.0,
            "ktrans_maximum_per_min_exclusive": KTRANS_BOUNDS.max,
            "uncertainty_used_for_acceptance": True,
            "volume_endpoint_absolute_tolerance": _VOLUME_QC_ENDPOINT_ATOL,
            "model_specific_volume_checks": {
                "patlak": {
                    "vp_minimum_exclusive": _VOLUME_QC_ENDPOINT_ATOL,
                    "vp_maximum_exclusive": (
                        VP_BOUNDS.max - _VOLUME_QC_ENDPOINT_ATOL
                    ),
                },
                "tofts": {
                    "ve_minimum_exclusive": _VOLUME_QC_ENDPOINT_ATOL,
                    "ve_maximum_exclusive": (
                        VE_BOUNDS.max - _VOLUME_QC_ENDPOINT_ATOL
                    ),
                },
                "extended_tofts": {
                    "ve_minimum_exclusive": _VOLUME_QC_ENDPOINT_ATOL,
                    "vp_minimum_exclusive": _VOLUME_QC_ENDPOINT_ATOL,
                    "ve_plus_vp_maximum_exclusive": (
                        VP_VE_TOTAL_MAX - _VOLUME_QC_ENDPOINT_ATOL
                    ),
                },
                "2cxm": {
                    "ve_minimum_exclusive": _VOLUME_QC_ENDPOINT_ATOL,
                    "vp_minimum_exclusive": _VOLUME_QC_ENDPOINT_ATOL,
                    "ve_plus_vp_maximum_exclusive": (
                        VP_VE_TOTAL_MAX - _VOLUME_QC_ENDPOINT_ATOL
                    ),
                },
            },
            "reversible_model_kep_minimum_per_min_exclusive": KEP_BOUNDS.min,
            "reversible_model_kep_maximum_per_min_inclusive": KEP_BOUNDS.max,
        },
    }
    policy["identification"] = identification_policy()
    result: npt.NDArray[np.uint8] = qc_mask.astype(np.uint8)
    return _DCEQCMaskEvaluation(result, failures, policy)


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
    }
    write_map(maps.iauc60, outputs["out_iauc60"])
    write_map(maps.iauc90, outputs["out_iauc90"])
    write_map(maps.iauc120, outputs["out_iauc120"])
    if maps.peak_enhancement is not None:
        write_map(maps.peak_enhancement, outputs["out_peak_enhancement"])
    write_map(maps.time_to_peak, outputs["out_time_to_peak"])
    write_map(maps.signal_enhancement_ratio, outputs["out_ser"])
    return outputs


def _write_dce_parameter_maps(
    maps: "_DCEParameterMaps",
    save_ve: bool,
    save_vp: bool,
    generate_path: Callable[[str], str],
    write_map: Callable[..., None],
) -> dict[str, str]:
    """Write fitted DCE parameter maps and return applicable result paths."""
    outputs = {
        "out_ktrans": generate_path("_Ktrans"),
        "out_r2": generate_path("_R2"),
    }
    write_map(maps.ktrans, outputs["out_ktrans"])
    write_map(maps.r_squared, outputs["out_r2"])
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


def _write_dce_uncertainty_maps(
    maps: "_DCEParameterMaps",
    is_auto: bool,
    save_ve: bool,
    save_vp: bool,
    generate_path: Callable[[str], str],
    write_map: Callable[..., None],
) -> dict[str, str]:
    """Write applicable DCE parameter-uncertainty derivatives."""
    outputs: dict[str, str] = {}

    def write(name: str, suffix: str, values: npt.NDArray[Any]) -> None:
        outputs[name] = generate_path(suffix)
        write_map(values, outputs[name])

    def write_parameter_uncertainty(
        stem: str,
        standard_deviation: npt.NDArray[Any],
    ) -> None:
        key = stem.lower() if stem != "Fp" else "fp"
        write(f"out_{key}_std", f"_{stem}_std", standard_deviation)

    if is_auto or not maps.save_2cxm:
        write_parameter_uncertainty("Ktrans", maps.ktrans_std)
    if save_ve:
        write_parameter_uncertainty("ve", maps.ve_std)
    if save_vp:
        write_parameter_uncertainty("vp", maps.vp_std)
    if maps.save_2cxm:
        write_parameter_uncertainty("Fp", maps.fp_std)
        write_parameter_uncertainty("ps", maps.ps_std)

    return outputs


def _write_dce_terminal_outputs(
    maps: "_DCEParameterMaps",
    fitting_mask: npt.NDArray[np.bool_],
    tsnr: npt.NDArray[Any],
    peak_enhancement: Optional[npt.NDArray[Any]],
    is_auto: bool,
    model_name: str,
    save_fitted_curves: bool,
    affine: npt.NDArray[Any],
    generate_path: Callable[[str], str],
    write_map: Callable[..., None],
    qc_metrics: Optional[dict[str, Any]] = None,
    tsnr_threshold: float = 0.0,
) -> dict[str, str]:
    """Write terminal QC, model-selection, and fitted-curve outputs."""
    outputs = {
        "out_tsnr": generate_path("_tSNR"),
        "out_fitting_mask": generate_path("_fitting_mask"),
        "out_optimizer_converged_mask": generate_path(
            "_optimizer_converged_mask"
        ),
        "out_retained_nondegenerate_mask": generate_path(
            "_retained_nondegenerate_mask"
        ),
    }
    write_map(tsnr, outputs["out_tsnr"])
    write_map(
        fitting_mask,
        outputs["out_fitting_mask"],
        dtype=np.uint8,
    )
    write_map(
        maps.optimizer_converged,
        outputs["out_optimizer_converged_mask"],
        dtype=np.uint8,
    )
    write_map(
        maps.retained_nondegenerate,
        outputs["out_retained_nondegenerate_mask"],
        dtype=np.uint8,
    )
    qc_evaluation = _evaluate_dce_qc_mask(
        fitting_mask,
        maps.optimizer_converged,
        maps.retained_nondegenerate,
        tsnr,
        peak_enhancement is not None,
        maps.ktrans,
        maps.r_squared,
        maps.ve,
        maps.vp,
        maps.kep,
        maps.model_selection,
        maps.model_selection_codes,
        is_auto,
        model_name,
        tsnr_threshold,
        identification_pass=maps.primary_identification,
    )
    qc_mask = qc_evaluation.mask
    if qc_metrics is not None:
        n_accepted = int(np.count_nonzero(qc_mask))
        accounting = qc_metrics.get("voxel_accounting")
        if isinstance(accounting, dict):
            denominators = accounting.get("denominators", {})
            attempted = int(denominators.get("attempted", 0))
            retained = int(denominators.get("retained", 0))
            brain = int(denominators.get("brain", 0))
            accounting["quantitative_qc_accepted"] = {
                "count": n_accepted,
                "fraction_of_retained": n_accepted / max(1, retained),
                "fraction_of_attempted": n_accepted / max(1, attempted),
                "fraction_of_brain": n_accepted / max(1, brain),
            }
        qc_metrics["qc_component_failures"] = {
            "scope": "fitting_eligible_voxels",
            "counts_are_overlapping": True,
            **qc_evaluation.component_failures,
        }
        qc_metrics["qc_acceptance_policy"] = qc_evaluation.acceptance_policy
    outputs["out_qc_mask"] = generate_path("_qc_mask")
    write_map(qc_mask, outputs["out_qc_mask"], dtype=np.uint8)
    if is_auto:
        outputs["out_model_selection"] = generate_path("_modelSelection")
        outputs["out_delta_bic"] = generate_path("_deltaBIC")
        write_map(
            maps.model_selection,
            outputs["out_model_selection"],
            dtype=np.int8,
        )
        write_map(maps.delta_bic, outputs["out_delta_bic"])
    if save_fitted_curves and maps.fitted_curves is not None:
        outputs["out_fitted_curves"] = generate_path("_fitted_curves")
        _write_dce_fitted_curves(
            maps.fitted_curves,
            outputs["out_fitted_curves"],
            affine,
        )
    return outputs


def _initialize_dce_qc_metrics(
    prepared: "_DCEFitPreparation",
    model_name: str,
    constraint_qc: dict[str, Any],
    delay_correction: dict[str, Any],
    two_cxm_max_tr_seconds: float,
    tr_comparison_tolerance_seconds: float,
) -> dict[str, Any]:
    """Assemble DCE QC state before fitting and result reduction."""
    return {
        "model": "auto" if prepared.is_auto else model_name,
        "inference": prepared.inference,
        "tsnr_threshold": prepared.tsnr_threshold,
        "baseline_frames": prepared.baseline_frames,
        "selection_tr_seconds": prepared.selection_tr_seconds,
        "acquisition_duration_seconds": (
            (prepared.n_timepoints - 1) * prepared.selection_tr_seconds
        ),
        "two_cxm_auto_sampling_eligible": (
            prepared.selection_tr_seconds
            <= two_cxm_max_tr_seconds + tr_comparison_tolerance_seconds
        ),
        "dsc_cbv_constraint": constraint_qc,
        "delay_correction": delay_correction,
    }


def _validate_dce_time_axis(
    time_points: npt.ArrayLike,
    n_dce_frames: int,
    n_aif_samples: int,
    volume_tr: Optional[float] = None,
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
    volume_tr : float, optional
        Volume interval in seconds. When supplied, must be finite, positive
        and consistent with the time-axis interval.

    Returns
    -------
    numpy.ndarray
        Validated one-dimensional time axis as ``float64``.

    Raises
    ------
    ValueError
        If the time axis is not one-dimensional, finite, length-matched,
        strictly increasing, or uniformly spaced, or if volume_tr is invalid
        or disagrees with its interval.

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
    if volume_tr is not None:
        if not np.isfinite(volume_tr) or volume_tr <= 0.0:
            raise ValueError("DCE volume_tr must be finite and positive (seconds).")
        if not np.isclose(volume_tr / 60.0, intervals[0], rtol=1e-5, atol=1e-8):
            raise ValueError(
                f"DCE volume_tr ({volume_tr:.9g} seconds) does not match the "
                f"time-axis interval ({intervals[0] * 60.0:.9g} seconds). "
                "time_points_file must contain minutes; volume_tr uses seconds."
            )
    return axis


class VoxelInput(NamedTuple):
    """Typed container for per-voxel fitting input data."""

    x: int
    y: int
    z: int
    tissue_curve: npt.NDArray[Any]
    voxel_vp: Optional[float]


class _DCEFitPreparation(NamedTuple):
    """Validated DCE inputs and pre-fit fitting-domain state."""

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
    save_fitted_curves: bool


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
    iauc_qc: dict[str, Any]
    ser_qc: dict[str, Any]


class _DCEParameterMaps(NamedTuple):
    """Allocated mutable PK, uncertainty, status, and fit-curve maps."""

    ktrans: npt.NDArray[Any]
    ve: npt.NDArray[Any]
    vp: npt.NDArray[Any]
    kep: npt.NDArray[Any]
    r_squared: npt.NDArray[Any]
    fp: npt.NDArray[Any]
    ps: npt.NDArray[Any]
    extraction_fraction: npt.NDArray[Any]
    optimizer_converged: npt.NDArray[Any]
    retained_nondegenerate: npt.NDArray[Any]
    primary_identification: npt.NDArray[np.bool_]
    delay: npt.NDArray[Any]
    model_selection: npt.NDArray[Any]
    model_selection_codes: dict[str, int]
    delta_bic: npt.NDArray[Any]
    ktrans_std: npt.NDArray[Any]
    ve_std: npt.NDArray[Any]
    vp_std: npt.NDArray[Any]
    fp_std: npt.NDArray[Any]
    ps_std: npt.NDArray[Any]
    sigma_factor: dict[str, float]
    fitted_curves: Optional[npt.NDArray[Any]]
    save_2cxm: bool


class _DCEFitAccountingUpdate(NamedTuple):
    """Constraint diagnostics contributed by one retained DCE fit result."""

    n_constrained: int
    n_unconstrained_fallback: int
    r_squared: Optional[float]


class _DCEFitSummary(NamedTuple):
    """Aggregate counters and diagnostics produced while scattering fits."""

    n_optimizer_converged: int
    n_prefit_rejected: int
    n_execution_exceptions: int
    n_fit_unsuccessful: int
    n_retained: int
    n_constrained: int
    n_unconstrained_fallback: int
    n_rejected_postfit: int
    r2_values: list[float]
    failure_reasons: dict[str, int]
    vp_prior_departures: list[float]
    n_inference_fallback: int


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
        desc="Volume interval in seconds; must agree with time_points_file when both are supplied",
    )
    time_points_file = File(
        exists=True,
        desc=(
            "Optional 1D text file with uniformly spaced time points (minutes); "
            "must agree with volume_tr when both are supplied"
        ),
    )

    # Inference engine options
    inference = traits.Enum(
        "nlls",
        "vi",
        usedefault=True,
        desc="Inference engine: nlls (bounded least squares) or vi (variational Bayes).",
    )
    # tSNR filtering
    tsnr_threshold = traits.Float(
        DEFAULT_TSNR_THRESHOLD,
        usedefault=True,
        desc="tSNR threshold for voxel inclusion (0 = no threshold)",
    )
    ser_early_time_s = traits.Float(60.0, usedefault=True)
    ser_late_time_s = traits.Float(300.0, usedefault=True)
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
    nawm_exclusion_reason = traits.Str(
        desc="Reason an automatically generated NAWM mask was excluded from "
             "quantitative use; reported when the DSC-informed vp prior skips.",
    )
    dsc_aif_qc = traits.Dict(
        mandatory=False,
        desc="DSC detection QC containing whole-brain tissue CNR (optional)",
    )
    dsc_preload_qc = traits.Dict(
        mandatory=False,
        desc="DSC leakage QC and preload provenance from leakage_correct node",
    )
    dsc_preprocessing_stats = traits.Dict(
        mandatory=False,
        desc="DSC preprocessing QC stats (fraction_cnr_gt_4, mean_cnr)",
    )
    dsc_registration_qc = traits.Dict(
        mandatory=False,
        desc="DSC-to-T1w registration QC for spatial vp-prior qualification",
    )
    dce_registration_qc = traits.Dict(
        mandatory=False,
        desc="T1w-to-DCE registration QC for spatial vp-prior qualification",
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

    bolus_arrival_frame = traits.Int(
        desc="Bolus arrival frame index for model-free enhancement metrics",
    )

    aif_detection_method = traits.Str(
        "population", usedefault=True,
        desc="AIF source for provenance; all sources use the configured delay policy.",
    )
    delay_correction = traits.Bool(
        True, usedefault=True,
        desc="Estimate local effective AIF delay; false fixes delay to zero.",
    )

    aif_uncertainty_inflation_factor = traits.Float(
        1.0,
        usedefault=True,
        desc="Heuristic AIF-confidence input to parameter uncertainty scaling. "
             "Values above 1 increase the configured parameter-specific "
             "standard-error or posterior-SD factors; the policy leaves ve "
             "unchanged.",
    )

    # Test-mode settings
    sloppy = traits.Bool(
        False,
        usedefault=True,
        desc="Use fast, low-quality settings for testing (NOT for production)",
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

    # Parameter uncertainty maps (standard error for NLLS; posterior SD otherwise)
    out_ktrans_std = File(desc="Ktrans parameter uncertainty map")
    out_ve_std = File(desc="ve parameter uncertainty map")
    out_vp_std = File(desc="vp parameter uncertainty map")
    out_fp_std = File(desc="Fp parameter uncertainty map (2CXM only)")
    out_ps_std = File(desc="PS parameter uncertainty map (2CXM only)")

    # Model-free maps
    out_iauc60 = File(desc="IAUC at 60s (mM·min)")
    out_iauc90 = File(desc="IAUC at 90s (mM·min)")
    out_iauc120 = File(desc="IAUC at 120s (mM·min)")
    out_peak_enhancement = File(desc="Peak enhancement percentage")
    out_time_to_peak = File(desc="Time to peak (seconds)")
    out_ser = File(desc="Signal Enhancement Ratio (washout characterization)")

    # Model selection map (auto mode)
    out_model_selection = File(desc="Model selection map (int8: 0=unfitted, 1=patlak, 2=tofts, 3=extended_tofts, 4=2cxm)")

    # Delta-BIC confidence map (auto mode)
    out_delta_bic = File(desc="Delta-BIC confidence map (runner-up BIC minus winner BIC; >2 = positive evidence)")

    # QC outputs
    out_r2 = File(desc="R² (goodness of fit) map")
    out_tsnr = File(desc="tSNR map (always computed)")
    out_qc_mask = File(desc="Binary QC: reported-fit quality AND selected-NLLS primary identification")
    out_delay = File(exists=True, desc="Native selected AIF-relative model delay in seconds")
    out_fitting_mask = File(desc="Internal fitting-eligibility mask")
    out_optimizer_converged_mask = File(
        desc="Internal optimizer-convergence mask"
    )
    out_retained_nondegenerate_mask = File(
        desc="Internal mask of fits retained in primary parameter maps"
    )
    out_fitted_curves = File(desc="Fitted concentration curves (if save_fitted_curves=True)")
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
    - patlak: Patlak model fitted in the concentration domain (Ktrans, vp)
    - 2cxm: Two-Compartment Exchange Model (Fp, PS, ve, vp)

    Auto-selection logic (when model='auto'):

    - Patlak is always included
    - Extended Tofts is always included
    - 2CXM considered if the volume interval is <= 2 s, with numerical tolerance,
      and admitted only when the conditional ET fit passes the detection and
      residual-improvement requirements
    - Converged candidates compared by BIC; a more complex winner must improve
      BIC by at least 6 over a converged Patlak fit
    - Standard Tofts is available through explicit model selection

    Output maps:

    - Ktrans: Volume transfer constant (min⁻¹)
    - ve: Extravascular extracellular volume fraction (unavailable for Patlak)
    - vp: Plasma volume fraction (extended_tofts, patlak, 2cxm)
    - kep: Efflux rate constant (Ktrans/ve, min⁻¹; unavailable for Patlak)

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
        Temporal spacing between DCE volumes in seconds. Must match the
        time-file interval when both timing inputs are supplied.
    time_points_file : str, optional
        Text file with uniformly spaced time points in minutes. If omitted,
        the time axis is generated from volume_tr. The validated axis controls
        fitting and model eligibility.
    inference : str
        Public inference engines: nlls (default, bounded least squares) or
        vi (variational Bayes).
    delay_correction : bool
        Estimate local effective input delay, or fix delay to zero when false.
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
        Ktrans standard error for NLLS or posterior standard deviation for
        Bayesian fitting, where available. Not propagated for derived 2CXM
        Ktrans.
    out_ve_std : str, optional
        ve standard error or posterior standard deviation, where applicable.
    out_vp_std : str, optional
        vp standard error or posterior standard deviation, where applicable.
    out_fp_std : str, optional
        Fp standard error or posterior standard deviation (2CXM only).
    out_ps_std : str, optional
        PS standard error or posterior standard deviation (2CXM only).
    out_r2 : str
        R² goodness-of-fit map.
    out_tsnr : str
        Baseline signal tSNR map; unavailable in-brain values are NaN when
        raw DCE signal is not supplied.
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
    >>> compute.inputs.inference = 'vi'
    >>> result = compute.run()  # doctest: +SKIP

    Notes
    -----
    Time units:

    - Time points are in minutes for model fitting
    - Ktrans and kep are in min⁻¹
    - Concentration is in mM (millimolar)

    Uncertainty maps contain NLLS standard errors or Bayesian posterior
    standard deviations, conditional on the selected family and delay, with
    parameter-specific AIF-confidence inflation.
    Unavailable estimates remain NaN within the brain mask.

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
        mask_bool = mask > 0
        baseline_end = min(baseline_frames, data.shape[-1])

        if baseline_end < 2:
            tsnr[mask_bool] = np.nan
            return tsnr

        baseline_data = data[..., :baseline_end]

        # Compute mean and std along time axis
        with np.errstate(divide="ignore", invalid="ignore"):
            baseline_mean = np.mean(baseline_data, axis=3)
            baseline_std = np.std(baseline_data, axis=3, ddof=1)
            valid = (
                mask_bool
                & np.isfinite(baseline_mean)
                & np.isfinite(baseline_std)
                & (baseline_mean > EPSILON_VE)
                & (baseline_std > EPSILON_VE)
            )
            tsnr[mask_bool] = np.nan
            np.divide(baseline_mean, baseline_std, out=tsnr, where=valid)

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

    # 2CXM needs high temporal resolution to separate its fast components.
    _MAX_VOLUME_TR_2CXM = 2.0
    # Absorb only floating-point representation error at the sampling boundary.
    _VOLUME_TR_COMPARISON_TOLERANCE = 1e-6

    def _build_candidate_set(self, volume_tr: float) -> list[str]:
        """Build BIC candidate model set gated by temporal resolution.

        Auto selection includes Patlak and Extended Tofts, both with an
        explicit plasma-volume term. It also includes 2CXM when the volume
        interval is at most 2 s, allowing numerical tolerance at the boundary.
        Standard Tofts remains available via explicit ``model=tofts`` selection.

        Parameters
        ----------
        volume_tr : float
            Volume TR in seconds.

        Returns
        -------
        list
            List of candidate model names, ordered by complexity.
        """
        candidates = ["patlak", "extended_tofts"]
        if volume_tr <= (
            self._MAX_VOLUME_TR_2CXM + self._VOLUME_TR_COMPARISON_TOLERANCE
        ):
            candidates.append("2cxm")
        return candidates

    def _prepare_fit_inputs(self) -> _DCEFitPreparation:
        """Load and validate DCE inputs and construct the pre-fit fit domain.

        Returns
        -------
        _DCEFitPreparation
            Image geometry, validated time axis, model-selection state, signal
            QC maps, and the voxel mask admitted to pharmacokinetic fitting.

        Raises
        ------
        ValueError
            If scan timing is missing or fails the shared time-axis validation,
            raw-signal baseline validation fails, or a positive tSNR threshold is
            requested but tSNR is unavailable in every brain voxel.
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

        volume_tr = (
            float(self.inputs.volume_tr)
            if isdefined(self.inputs.volume_tr) and self.inputs.volume_tr is not None
            else None
        )
        if self.inputs.time_points_file:
            time_points = np.loadtxt(self.inputs.time_points_file)
        elif volume_tr is not None:
            vol_tr_minutes = volume_tr / 60.0
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
            volume_tr=volume_tr,
        )
        # Fitting and model eligibility share the validated axis.
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
        finite_signal_mask: Optional[npt.NDArray[np.bool_]] = None
        peak_enhancement: Optional[npt.NDArray[Any]]
        if isdefined(self.inputs.dce_signal_file):
            signal_img = nib.load(self.inputs.dce_signal_file)
            signal_data = signal_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]
            tsnr_map = self._compute_tsnr(signal_data, mask, baseline_frames)
            finite_signal_mask = np.all(np.isfinite(signal_data), axis=-1)
            from pmriprep.utils.dce_model_free import compute_peak_enhancement

            peak_enhancement = compute_peak_enhancement(signal_data, baseline_frames)
            del signal_data
        else:
            logger.warning(
                "dce_signal_file not provided; baseline tSNR is unavailable"
            )
            tsnr_map = np.where(mask > 0, np.nan, 0.0)
            peak_enhancement = None

        # Admit only brain voxels with finite inputs, configured signal quality,
        # and a positive concentration enhancement.
        # Signal quality uses baseline tSNR, a dimensionless signal/noise ratio.
        mask_bool = mask > 0
        if tsnr_threshold > 0:
            if not np.any(mask_bool & np.isfinite(tsnr_map)):
                raise ValueError(
                    "A positive tSNR threshold was requested, but baseline tSNR "
                    "is unavailable in every brain voxel"
                )
            fitting_mask = mask_bool & (tsnr_map >= tsnr_threshold)
        else:
            fitting_mask = mask_bool
        finite_concentration = np.all(np.isfinite(conc_data), axis=-1)
        enhancement_mask = np.max(conc_data, axis=-1) > MIN_ENHANCEMENT_MM
        fitting_mask &= (
            finite_concentration
            & enhancement_mask
        )
        if finite_signal_mask is not None:
            fitting_mask &= finite_signal_mask
        if not np.any(fitting_mask):
            logger.error(
                "DCE fitting mask is empty after signal-quality and enhancement gates"
            )

        inference = str(self.inputs.inference)
        save_fitted_curves = bool(self.inputs.save_fitted_curves)
        # Preserve the identifiability warning for every 2CXM fitting method.
        if model_name == "2cxm":
            logger.warning(
                "2CXM individual parameter estimates (especially PS) may be "
                "unreliable due to PS/ve coupling degeneracy. Consider "
                "Extended Tofts if Ktrans is "
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
            save_fitted_curves,
        )

    def _prepare_vp_constraint(
        self,
        prepared: _DCEFitPreparation,
    ) -> _DCEVPConstraint:
        """Prepare the DSC-derived vp prior and its QC state.

        Parameters
        ----------
        prepared : _DCEFitPreparation
            Validated DCE inputs, fitting domain, and initial model selection.

        Returns
        -------
        _DCEVPConstraint
            Prior map, constraint configuration, QC summary, and any
            constraint-driven adjustment to automatic model selection.

        Raises
        ------
        ValueError
            If the DSC-informed vp prior has a different spatial shape from
            the DCE fitting mask.

        Notes
        -----
        The prior remains NAWM-anchored as ``vp = NAWM_VP_REFERENCE * nCBV``.
        Models with vp share the same qualification rules whether selected
        explicitly or automatically. Models without vp and explicit prior
        disabling return before loading DSC data.
        """
        from pmriprep.utils.dsc_cbv_constraint import (
            NO_DSC_RUN_REASON_STRING,
            SKIP_REASON_NO_DSC_RUN,
            SKIP_REASON_NLLS_UNSUPPORTED,
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
        needs_vp_constraint = self._has_vp(model_name) or prepared.is_auto
        dsc_inputs_present = (
            (isdefined(self.inputs.dsc_ncbv_file) and bool(self.inputs.dsc_ncbv_file))
            or (isdefined(self.inputs.dsc_aif_qc) and bool(self.inputs.dsc_aif_qc))
            or (isdefined(self.inputs.dsc_preload_qc) and bool(self.inputs.dsc_preload_qc))
        )

        if not needs_vp_constraint or self.inputs.constraint_mode == "skip":
            if not needs_vp_constraint:
                reason = "The selected DCE model has no vp parameter."
                skip_reason = "model_without_vp"
            else:
                reason = "The DSC-informed vp prior was disabled by configuration."
                skip_reason = "disabled_by_configuration"
            return _DCEVPConstraint(
                None,
                ConstraintMode.SKIP,
                0.0,
                {"mode": "skip", "reason": reason, "skip_reason": skip_reason},
                model_name,
                candidate_names,
            )

        if (
            prepared.inference == "nlls"
            and needs_vp_constraint
            and dsc_inputs_present
            and ncbv_available
        ):
            return _DCEVPConstraint(
                None,
                ConstraintMode.SKIP,
                0.0,
                {
                    "mode": "skip",
                    "reason": (
                        "DSC-informed vp priors are supported only for Bayesian "
                        "fitting; NLLS used the model's ordinary bounds."
                    ),
                    "skip_reason": SKIP_REASON_NLLS_UNSUPPORTED,
                },
                model_name,
                candidate_names,
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
                    dsc_qc_data=aif_qc_data,
                    brain_mask_file=self.inputs.brain_mask_file,
                    mode_override=str(self.inputs.constraint_mode),
                    k2_metrics=k2_metrics,
                    dose=float(self.inputs.dsc_dose),
                    agent_standard_dose=float(self.inputs.dsc_agent_standard_dose),
                    dsc_registration_qc=(
                        dict(self.inputs.dsc_registration_qc)
                        if isdefined(self.inputs.dsc_registration_qc)
                        else None
                    ),
                    dce_registration_qc=(
                        dict(self.inputs.dce_registration_qc)
                        if isdefined(self.inputs.dce_registration_qc)
                        else None
                    ),
                    nawm_exclusion_reason=(
                        str(self.inputs.nawm_exclusion_reason)
                        if isdefined(self.inputs.nawm_exclusion_reason)
                        and bool(self.inputs.nawm_exclusion_reason)
                        else None
                    ),
                )
                vp_constraint_map = vp_result.vp_map
                constraint_mode = vp_result.config.mode
                constraint_vp_uncertainty = vp_result.config.vp_uncertainty
                constraint_qc = vp_result.qc_report
                if (
                    vp_constraint_map is not None
                    and vp_constraint_map.shape != prepared.mask.shape
                ):
                    raise ValueError(
                        "DSC-informed vp prior shape does not match the DCE domain: "
                        f"{vp_constraint_map.shape} != {prepared.mask.shape}"
                    )

            except Exception as exc:
                logger.warning(
                    "DSC-informed vp prior preparation failed; continuing without "
                    "the prior.",
                    exc_info=exc,
                )
                vp_constraint_map = None
                constraint_mode = ConstraintMode.SKIP
                constraint_vp_uncertainty = 0.0
                constraint_qc = {
                    "mode": "skip",
                    "reason": (
                        "The DSC-informed vp prior could not be prepared; review the "
                        "processing log."
                    ),
                    "skip_reason": "constraint_preparation_failed",
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
            compute_iauc,
            compute_signal_enhancement_ratio,
            compute_time_to_peak,
        )

        concentration = prepared.concentration
        time_points = prepared.time_points
        detected_bolus_frame: Optional[int] = None
        if isdefined(self.inputs.bolus_arrival_frame):
            detected_bolus_frame = int(self.inputs.bolus_arrival_frame)
        iauc_reference_frame = prepared.baseline_frames
        if detected_bolus_frame is not None:
            iauc_reference_frame = min(iauc_reference_frame, detected_bolus_frame)
        reference_time_s = float(time_points[iauc_reference_frame] * 60.0)
        available_duration_s = float(
            (time_points[-1] - time_points[iauc_reference_frame]) * 60.0
        )
        iauc_windows: dict[str, Any] = {}
        for cutoff_s in (60.0, 90.0, 120.0):
            complete = available_duration_s + 6e-8 >= cutoff_s
            endpoint_min = time_points[iauc_reference_frame] + cutoff_s / 60.0
            interpolated = complete and not np.any(
                np.isclose(time_points, endpoint_min, rtol=0.0, atol=1e-9)
            )
            iauc_windows[str(int(cutoff_s))] = {
                "requested_duration_s": cutoff_s,
                "complete": bool(complete),
                "cutoff_interpolated": bool(interpolated),
            }
            if not complete:
                logger.warning(
                    "IAUC%d unavailable: %.3f s acquired after reference frame %d",
                    int(cutoff_s), available_duration_s, iauc_reference_frame,
                )
        iauc_qc = {
            "reference_frame": iauc_reference_frame,
            "reference_time_s": reference_time_s,
            "configured_baseline_frames": int(self.inputs.baseline_frames),
            "effective_baseline_frames": prepared.baseline_frames,
            "detected_bolus_arrival_frame": detected_bolus_frame,
            "reference_selection": "earlier_of_baseline_count_and_detected_arrival",
            "available_post_reference_duration_s": available_duration_s,
            "windows": iauc_windows,
        }
        ser_early_time_s = float(self.inputs.ser_early_time_s)
        ser_late_time_s = float(self.inputs.ser_late_time_s)
        ser_complete = available_duration_s + 6e-8 >= ser_late_time_s
        ser_qc = {
            "reference_frame": iauc_reference_frame,
            "reference_time_s": reference_time_s,
            "reference_selection": "earlier_of_baseline_count_and_detected_arrival",
            "early_sample_time_s": ser_early_time_s,
            "late_sample_time_s": ser_late_time_s,
            "available_post_reference_duration_s": available_duration_s,
            "complete": bool(ser_complete),
        }
        if not ser_complete:
            logger.warning(
                "SER unavailable: %.3f s acquired after reference frame %d; "
                "late sample requires %.3f s",
                available_duration_s, iauc_reference_frame, ser_late_time_s,
            )
        ser: npt.NDArray[Any]
        if not isdefined(self.inputs.dce_signal_file):
            ser = np.full(concentration.shape[:3], np.nan, dtype=np.float64)
        else:
            import nibabel as nib  # type: ignore[import-untyped]
            signal_img = nib.load(self.inputs.dce_signal_file)
            signal_data: npt.NDArray[Any] = signal_img.get_fdata().astype(  # type: ignore[attr-defined]
                np.float64
            )
            ser = compute_signal_enhancement_ratio(
                signal_data,
                time_points,
                bolus_arrival_frame=iauc_reference_frame,
                baseline_frames=prepared.baseline_frames,
                early_time_s=ser_early_time_s,
                late_time_s=ser_late_time_s,
            )
        return _DCEModelFreeMaps(
            compute_iauc(
                concentration, time_points, cutoff_s=60.0,
                bolus_arrival_frame=iauc_reference_frame,
            ),
            compute_iauc(
                concentration, time_points, cutoff_s=90.0,
                bolus_arrival_frame=iauc_reference_frame,
            ),
            compute_iauc(
                concentration, time_points, cutoff_s=120.0,
                bolus_arrival_frame=iauc_reference_frame,
            ),
            prepared.peak_enhancement,
            compute_time_to_peak(
                concentration,
                time_points,
                reference_frame=iauc_reference_frame,
                baseline_frames=prepared.baseline_frames,
                brain_mask=prepared.mask,
            ),
            ser,
            iauc_qc,
            ser_qc,
        )

    def _initialize_parameter_maps(
        self,
        prepared: _DCEFitPreparation,
        model_name: str,
        candidate_names: list[str],
    ) -> _DCEParameterMaps:
        """Allocate mutable output state before pharmacokinetic fitting."""
        from pmriprep.utils.dce_constants import DCE_AIF_G6_SIGMA_INFLATION

        spatial_shape = prepared.concentration.shape[:3]
        # Unsupported and rejected voxels are missing estimates in every
        # model-selection mode. A numerical zero is reserved for a retained
        # fit whose parameter estimate is genuinely zero.
        fill_value = np.nan
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
            np.full(prepared.concentration.shape, np.nan, dtype=np.float32)
            if prepared.save_fitted_curves
            else None
        )
        return _DCEParameterMaps(
            ktrans=np.full(spatial_shape, fill_value, dtype=np.float64),
            ve=np.full(spatial_shape, fill_value, dtype=np.float64),
            vp=np.full(spatial_shape, fill_value, dtype=np.float64),
            kep=np.full(spatial_shape, fill_value, dtype=np.float64),
            r_squared=np.full(spatial_shape, np.nan, dtype=np.float64),
            fp=np.full(spatial_shape, fill_value, dtype=np.float64),
            ps=np.full(spatial_shape, fill_value, dtype=np.float64),
            extraction_fraction=np.full(
                spatial_shape, fill_value, dtype=np.float64
            ),
            optimizer_converged=np.zeros(spatial_shape, dtype=bool),
            retained_nondegenerate=np.zeros(spatial_shape, dtype=bool),
            primary_identification=np.zeros(spatial_shape, dtype=bool),
            delay=np.full(spatial_shape, np.nan, dtype=np.float32),
            model_selection=np.zeros(spatial_shape, dtype=np.int8),
            model_selection_codes=model_selection_codes,
            delta_bic=np.full(spatial_shape, np.nan, dtype=np.float32),
            ktrans_std=np.full(spatial_shape, np.nan, dtype=np.float64),
            ve_std=np.full(spatial_shape, np.nan, dtype=np.float64),
            vp_std=np.full(spatial_shape, np.nan, dtype=np.float64),
            fp_std=np.full(spatial_shape, np.nan, dtype=np.float64),
            ps_std=np.full(spatial_shape, np.nan, dtype=np.float64),
            sigma_factor=sigma_factor,
            fitted_curves=fitted_curves,
            save_2cxm=(
                (prepared.is_auto and "2cxm" in candidate_names)
                or model_name == "2cxm"
            ),
        )

    def _fit_local_slices(
        self, prepared: _DCEFitPreparation, constraint: _DCEVPConstraint,
        maps: _DCEParameterMaps, qc: dict[str, Any],
    ) -> _DCEFitSummary:
        """Fit, refine and scatter each complete slice before releasing profiles.

        Parameters
        ----------
        prepared, constraint : named tuples
            Validated fitting domain, original input/time axis and qualified prior.
        maps : _DCEParameterMaps
            Mutable volume outputs. Only these and small summaries span slices.
        qc : dict
            Run diagnostics updated with actual timing, inference and memory use.

        Returns
        -------
        _DCEFitSummary
            Aggregate fit accounting across all attempted slices.
        """
        from collections import Counter
        from pmriprep.models.dce.delay_profiles import MAX_LOCAL_DELAY_SECONDS
        from pmriprep.models.dce.fixed_inference import selected_fit_result
        from pmriprep.models.dce.identification import profile_excursion
        from pmriprep.models.dce.local_refinement import ConditionalInput
        from pmriprep.models.dce.local_runner import LocalDelayRunner
        from pmriprep.models.dce.local_selection import SelectionMode
        from pmriprep.models.dce.retention import rate_ceiling_hits
        from pmriprep.utils.dsc_cbv_constraint import ConstraintMode

        summaries: list[_DCEFitSummary] = []
        fallbacks: Counter[str] = Counter()
        timing_counts: Counter[str] = Counter({
            "selected_at_lower_bound": 0, "selected_at_upper_bound": 0,
            "reference_at_lower_bound": 0, "reference_at_upper_bound": 0,
        })
        failure_counts: Counter[str] = Counter()
        identification_counts: dict[str, dict[str, Any]] = {}
        identification_worker_seconds = refinement_worker_seconds = 0.0
        retention_counts: Counter[str] = Counter({
            "selected_2cxm": 0, "central_2cxm_ceiling_rejections": 0,
        })
        support_counts: Counter[int] = Counter()
        radii: set[float] = set()
        max_profile_bytes = 0
        start = time.monotonic()
        n_procs = max(1, int(self.inputs.n_procs))
        mode = cast(SelectionMode, "auto" if prepared.is_auto else prepared.model_name)
        engine = cast(Literal["nlls", "vi"], prepared.inference)
        delay_enabled, sloppy = bool(self.inputs.delay_correction), bool(self.inputs.sloppy)
        fit_times = prepared.time_points
        interval = float(fit_times[1] - fit_times[0])
        if not np.allclose(np.diff(fit_times), interval, rtol=1e-12, atol=1e-14):
            # The input validator has already accepted uniformity at text-file
            # precision. Resolve serialization jitter for the exact operator;
            # original metric times and the validated frame interval stay intact.
            fit_times = fit_times[0] + np.arange(len(fit_times)) * interval
        qc["delay_correction"]["time_axis_rounding_max_seconds"] = float(
            np.max(np.abs(fit_times - prepared.time_points)) * 60
        )
        with LocalDelayRunner(n_procs) as runner:
            for z in range(prepared.concentration.shape[2]):
                xy = np.argwhere(prepared.fitting_mask[:, :, z])
                if not len(xy):
                    continue
                xyz = np.column_stack([xy, np.full(len(xy), z, dtype=int)])
                curves = prepared.concentration[tuple(xyz.T)]
                fits = runner.fit_slice(
                    prepared.aif, fit_times, curves, xyz, prepared.affine,
                    mode=mode, delay_correction=delay_enabled, sloppy=sloppy,
                )
                max_profile_bytes = max(max_profile_bytes, sum(
                    getattr(p, key).nbytes for p in fits.profiles.values()
                    for key in ("params", "sse", "status", "evaluations", "optimality", "seed_index")
                ))
                for profile in fits.profiles.values():
                    timing_counts["retried_profile_points"] += len(profile.retries)
                    for failure in profile.failures.values():
                        failure_counts[failure.reason] += 1
                inputs: list[ConditionalInput] = []
                refined_members: list[int] = []
                voxels: list[VoxelInput] = []
                results: list[dict[str, Any]] = []
                for member, coordinates in enumerate(xyz):
                    coords = (int(coordinates[0]), int(coordinates[1]), int(coordinates[2]))
                    prior = None
                    if (engine == "vi" and constraint.mode == ConstraintMode.SOFT
                            and constraint.vp_map is not None):
                        value = float(constraint.vp_map[coords])
                        if np.isfinite(value) and value > 0:
                            prior = value
                    voxels.append(VoxelInput(*coords, curves[member], prior))
                    choice = fits.selections[member]
                    family = choice.family
                    timing = choice.timing.get(family) if family is not None else None
                    if family is None or timing is None or timing.reason is not None:
                        reason = timing.reason if timing is not None else "no_eligible_model"
                        results.append(dict(coords=coords, failed=True, failure_stage="fit", failure_reason=reason))
                        continue
                    timing_counts["selected"] += 1
                    maps.delay[coords] = fits.profiles[family].delays_seconds[timing.index]
                    if delay_enabled:
                        grid = fits.profiles[family].delays_seconds
                        for name, index in (("selected", timing.index), ("reference", timing.reference)):
                            timing_counts[name + "_at_lower_bound"] += int(grid[index] == 0.0)
                            timing_counts[name + "_at_upper_bound"] += int(grid[index] == MAX_LOCAL_DELAY_SECONDS)
                    support_counts[timing.support] += 1
                    if fits.radius_mm is not None:
                        radii.add(float(fits.radius_mm))
                    if timing.boundary_unavailable_reason is not None:
                        timing_counts["boundary_unavailable_" + timing.boundary_unavailable_reason] += 1
                    timing_counts["free_delay"] += int(timing.free)
                    if timing.fallback is not None:
                        timing_counts[timing.fallback] += 1
                    for flag in ("free_optimum_near_lower", "free_optimum_near_upper", "reference_at_bound"):
                        timing_counts[flag] += int(getattr(timing, flag) is True)
                    if timing.upper_edge is not None:
                        timing_counts[timing.upper_edge] += 1
                    profile = fits.profiles[family]
                    seed = selected_fit_result(
                        profile, timing, member, prepared.aif, fit_times, curves[member],
                    )
                    if family == "2cxm":
                        retention_counts["selected_2cxm"] += 1
                        hits = rate_ceiling_hits(family, seed.parameters)
                        retention_counts["central_2cxm_ceiling_rejections"] += bool(hits)
                        for name in hits:
                            retention_counts["central_2cxm_" + name + "_ceiling"] += 1
                    results.append(dict(
                        coords=coords, failed=False, converged=True, params=seed.parameters,
                        selected_model_name=family, delta_bic=choice.delta_bic,
                        r_squared=seed.r_squared, uncertainties={},
                    ))
                    if _is_degenerate_dce_fit(family, seed.parameters):
                        retention_counts["central_degenerate"] += 1
                        retention_counts["central_degenerate_2cxm"] += int(family == "2cxm")
                        continue
                    inputs.append(ConditionalInput(
                        family, prepared.aif, fit_times, profile.observed,
                        curves[member], float(profile.delays_seconds[timing.index]), seed,
                        engine, profile_excursion(profile, member, timing.index),
                        float(profile.sse[timing.index, member]),
                        prior, constraint.vp_uncertainty, sloppy,
                    ))
                    refined_members.append(member)
                refinements = runner.refine_selected(inputs)
                for member, conditional in zip(refined_members, refinements, strict=True):
                    refined, identification = conditional.refinement, conditional.identification
                    identification_worker_seconds += conditional.identification_seconds
                    refinement_worker_seconds += conditional.refinement_seconds
                    result = results[member]
                    maps.primary_identification[tuple(result["coords"])] = identification.primary_pass
                    counts = identification_counts.setdefault(refined.family, {
                        "evaluated": 0, "primary_pass": 0, "parameters": {},
                    })
                    counts["evaluated"] += 1
                    counts["primary_pass"] += int(identification.primary_pass)
                    for parameter, reasons in zip(identification.names, identification.reasons, strict=True):
                        record = counts["parameters"].setdefault(parameter, {"passed": 0, "failures": {}})
                        record["passed"] += int(not reasons)
                        for reason in reasons:
                            record["failures"][reason] = record["failures"].get(reason, 0) + 1
                    fit = refined.fit
                    result.update(
                        params=fit.parameters, r_squared=fit.r_squared,
                        uncertainties=fit.uncertainties, converged=fit.converged,
                        constrained=(refined.actual_engine == "vi" and voxels[member].voxel_vp is not None
                                     and self._has_vp(refined.family)),
                        inference_fallback=refined.fallback,
                        fitted_curve=fit.fitted_curve if prepared.save_fitted_curves else None,
                    )
                    if refined.fallback:
                        fallbacks[refined.reason or "unknown"] += 1
                summaries.append(_scatter_dce_fit_results(
                    voxels, results, maps, prepared.model_name, prepared.is_auto,
                    engine, constraint.mode, constraint.vp_uncertainty,
                ))
                logger.info("DCE slice %d/%d: %d attempted voxels", z + 1, prepared.concentration.shape[2], len(xy))
                # Consume profile-dependent diagnostics here, before the next slice.
                del fits, profile, curves, inputs, refinements, results, voxels
        qc["delay_correction"].update(
            grid_step_seconds=(1.0 if sloppy else 0.1) if delay_enabled else None,
            maximum_seconds=MAX_LOCAL_DELAY_SECONDS if delay_enabled else 0.0,
            approximate_test_mode=sloppy, counts=dict(timing_counts),
            family_numerical_failures=dict(failure_counts),
            support_histogram={str(k): v for k, v in sorted(support_counts.items())},
            effective_radii_mm=sorted(radii),
            selected_summary_denominator="selected count before numerical retention and identification QC; applies to support and selected-family boundary/fallback counts",
            boundary_count_definitions={
                "selected_at_lower_bound": "Selected delay equals zero with delay correction enabled",
                "selected_at_upper_bound": "Selected delay equals maximum_seconds with delay correction enabled",
                "reference_at_lower_bound": "Selected family's reference equals zero with delay correction enabled, including zero-reference fallbacks",
                "reference_at_upper_bound": "Selected family's reference equals maximum_seconds with delay correction enabled",
                "free_optimum_near_lower_or_upper": "Centre voxel's own selected-family profile optimum, irrespective of whether its free delay was selected",
                "flat_edge_or_truncated": "Centre voxel's own free-optimum upper-edge classification; does not classify the pooled reference or count selected upper-bound delays",
            },
            retention_counts=dict(retention_counts),
            retention_count_denominator="2CXM ceiling counts use selected_2cxm; Fp/PS contacts may overlap",
        )
        qc["inference_fallback_reasons"] = dict(fallbacks)
        qc["identification_qc"] = {
            "basis": "selected_nlls",
            "scope": "selected_nondegenerate_nlls_fits_before_vi",
            "denominator": "evaluated count within each model; parameter failure counts overlap",
            "unevaluated_primary_pass": False,
            "by_model": identification_counts,
        }
        qc["timing_diagnostics"] = {
            "basis": "selected_nlls", "report_only": True,
            "frame_interval_seconds": interval * 60,
            "acquisition_frame_interval_seconds": prepared.selection_tr_seconds,
            "observation_indices": list(range(prepared.n_timepoints)),
            "observation_times_seconds": (fit_times * 60).tolist(),
            "original_observation_times_seconds": (prepared.time_points * 60).tolist(),
            "n_observations": prepared.n_timepoints,
            "scope": "selected_nlls_fits_before_numerical_retention_and_vi",
            "sensitivity_refits_performed": False,
            "endpoint_convention": "PCHIP with first/last AIF values held outside acquired support",
            "approximate_test_mode": sloppy,
        }
        qc["local_fitting_execution"] = dict(
            n_procs=n_procs, chunk_size=128, peak_slice_profile_bytes=max_profile_bytes,
            elapsed_seconds=time.monotonic() - start,
            includes="profiles, selection, NLLS identification, conditional uncertainty, VI and scattering",
            identification_worker_seconds=identification_worker_seconds,
            refinement_worker_seconds=refinement_worker_seconds,
            worker_timing_basis="summed worker elapsed seconds, not parallel wall time",
        )
        combined: dict[str, Any] = {}
        for field in _DCEFitSummary._fields:
            values = [getattr(s, field) for s in summaries]
            if field == "failure_reasons":
                combined_counts: Counter[str] = Counter()
                for value in values:
                    combined_counts.update(value)
                combined[field] = dict(combined_counts)
            elif field in ("r2_values", "vp_prior_departures"):
                combined[field] = [item for value in values for item in value]
            else:
                combined[field] = sum(values)
        return _DCEFitSummary(**combined)

    def _run_interface(self, runtime: Any) -> Any:
        """Execute DCE map computation."""
        prepared = self._prepare_fit_inputs()
        conc_data = prepared.concentration
        affine = prepared.affine
        header = prepared.header
        mask = prepared.mask
        model_name = prepared.model_name
        is_auto = prepared.is_auto
        candidate_names = prepared.candidate_names
        tsnr_threshold = prepared.tsnr_threshold
        tsnr_map = prepared.tsnr_map
        peak_enh = prepared.peak_enhancement
        fitting_mask = prepared.fitting_mask
        mask_bool = mask > 0
        inference = prepared.inference
        save_fitted_curves = prepared.save_fitted_curves

        vp_constraint = self._prepare_vp_constraint(prepared)
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
        retained_nondegenerate_map = parameter_maps.retained_nondegenerate
        model_selection_map = parameter_maps.model_selection
        _MODEL_SELECTION_CODES = parameter_maps.model_selection_codes
        delta_bic_map = parameter_maps.delta_bic
        ktrans_std = parameter_maps.ktrans_std
        ve_std = parameter_maps.ve_std
        vp_std = parameter_maps.vp_std
        fp_std = parameter_maps.fp_std
        ps_std = parameter_maps.ps_std
        fitted_curves = parameter_maps.fitted_curves
        save_2cxm = parameter_maps.save_2cxm

        # Get voxel coordinates to fit
        mask_coords = np.where(fitting_mask)
        n_voxels = len(mask_coords[0])

        qc_metrics = _initialize_dce_qc_metrics(
            prepared,
            model_name,
            constraint_qc,
            {"enabled": bool(self.inputs.delay_correction), "method": "local_profile"},
            self._MAX_VOLUME_TR_2CXM,
            self._VOLUME_TR_COMPARISON_TOLERANCE,
        )
        qc_metrics["iauc"] = model_free.iauc_qc
        qc_metrics["ser"] = model_free.ser_qc

        qc_metrics["temporal_weighting"] = "unit"
        save_ve = is_auto or self._has_ve(model_name)
        save_vp = is_auto or self._has_vp(model_name)
        fit_summary = self._fit_local_slices(prepared, vp_constraint, parameter_maps, qc_metrics)

        # Compile QC metrics
        qc_metrics.update(
            _summarize_dce_fit_outcomes(
                n_voxels,
                int(np.count_nonzero(mask_bool)),
                fit_summary.n_optimizer_converged,
                fit_summary.n_prefit_rejected,
                fit_summary.n_execution_exceptions,
                fit_summary.n_fit_unsuccessful,
                fit_summary.n_retained,
                fit_summary.n_constrained,
                fit_summary.n_unconstrained_fallback,
                fit_summary.n_rejected_postfit,
                fit_summary.failure_reasons,
            )
        )

        # Report the fraction of finite tissue peaks above the 0.8 mM FXL-risk
        # cutoff within the fitting mask. _generate_dce_pk_fit_qc also reports
        # the fraction stratified by the tumor mask.
        qc_metrics["frac_fxl_risk_whole_brain"] = _fraction_fxl_risk(
            conc_data,
            fitting_mask,
        )

        qc_metrics.update(
            _summarize_dce_model_selection(
                is_auto,
                candidate_names,
                model_selection_map,
                _MODEL_SELECTION_CODES,
            )
        )

        qc_metrics.update(_summarize_dce_r2(fit_summary.r2_values))

        qc_metrics["parameter_uncertainty"] = {
            "aif_inflation_input_factor": float(
                self.inputs.aif_uncertainty_inflation_factor
            ),
            "aif_parameter_scale_factors": dict(parameter_maps.sigma_factor),
        }

        inference_fallback = _summarize_inference_fallback(
            inference,
            fit_summary.n_retained,
            fit_summary.n_inference_fallback,
        )
        if inference_fallback is not None:
            qc_metrics["inference_fallback"] = inference_fallback

        qc_metrics.update(
            _summarize_vp_prior_departures(fit_summary.vp_prior_departures)
        )

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
        retained_nondegenerate_map = _zero_dce_background(
            retained_nondegenerate_map, mask_bool
        )
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
                retained_nondegenerate=retained_nondegenerate_map,
            ),
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
                model_free.iauc_qc,
                model_free.ser_qc,
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
        delay_path = self._gen_outfilename("_delay")
        _save_dce_map(parameter_maps.delay, delay_path)
        self._results["out_delay"] = delay_path

        # Summarize the reported retained reversible-model timescale without refitting.
        applicable = parameter_maps.retained_nondegenerate & np.isin(parameter_maps.model_selection, (2, 3))
        valid_tau = applicable & np.isfinite(parameter_maps.ve) & np.isfinite(parameter_maps.ktrans) & (parameter_maps.ktrans > 0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            tau = 60 * parameter_maps.ve[valid_tau] / parameter_maps.ktrans[valid_tau]
        finite_tau = np.isfinite(tau)
        flagged = finite_tau & (tau < prepared.selection_tr_seconds)
        flagged_et = flagged & (parameter_maps.model_selection[valid_tau] == 3)
        low_vp = flagged_et & (parameter_maps.vp[valid_tau] <= 1e-6)
        qc_metrics["timing_diagnostics"]["tau_e"] = {
            "basis": "reported_retained_et_or_tofts", "units": "s",
            "applicable_retained": int(applicable.sum()), "defined": int(finite_tau.sum()),
            "undefined": int(applicable.sum() - finite_tau.sum()),
            "below_frame": int(flagged.sum()), "flagged_et": int(flagged_et.sum()),
            "low_vp_threshold": 1e-6, "flagged_et_low_vp": int(low_vp.sum()),
            "low_vp_fraction_of_flagged_et": float(low_vp.sum() / flagged_et.sum()) if flagged_et.any() else None,
        }

        # Save method-specific parameter uncertainty maps.
        self._results.update(
            _write_dce_uncertainty_maps(
                parameter_maps._replace(
                    ktrans=ktrans,
                    ve=ve,
                    vp=vp,
                    fp=fp_map,
                    ps=ps_map,
                    ktrans_std=ktrans_std,
                    ve_std=ve_std,
                    vp_std=vp_std,
                    fp_std=fp_std,
                    ps_std=ps_std,
                ),
                is_auto,
                save_ve,
                save_vp,
                self._gen_outfilename,
                _save_dce_map,
            )
        )

        self._results.update(
            _write_dce_terminal_outputs(
                parameter_maps._replace(
                    ktrans=ktrans,
                    ve=ve,
                    vp=vp,
                    kep=kep,
                    r_squared=r2_map,
                    model_selection=model_selection_map,
                    delta_bic=delta_bic_map,
                    fitted_curves=fitted_curves,
                ),
                fitting_mask,
                tsnr_map,
                peak_enh,
                is_auto,
                model_name,
                save_fitted_curves,
                affine,
                self._gen_outfilename,
                _save_dce_map,
                qc_metrics,
                tsnr_threshold=tsnr_threshold,
            )
        )

        return runtime

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List interface outputs.

        Returns only explicitly set results from _run_interface, avoiding
        stale defaults from output_spec().get() (which returns '.' for
        optional File traits without exists=True).
        """
        if hasattr(self, "_results"):
            return dict(self._results)
        return self.output_spec().get()  # type: ignore[misc,no-any-return]
