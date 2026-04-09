"""DCE map computation interface for pharmacokinetic modeling.

This module implements the main Nipype interface for computing DCE
pharmacokinetic parameter maps (Ktrans, ve, vp, kep) from concentration
curves and AIF using various PK models.
"""

import logging
from pathlib import Path
from typing import Any, NamedTuple, Optional

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
    FP_BOUNDS,
    HEMATOCRIT_ARTERIAL,
    KTRANS_BOUNDS,
    MIN_ENHANCEMENT_MM,
    PS_BOUNDS,
    VE_BOUNDS,
    VP_BOUNDS,
)
from pmriprep.utils.dce_voxel_prep import prepare_voxel_for_fitting

logger = logging.getLogger(__name__)


class VoxelInput(NamedTuple):
    """Typed container for per-voxel fitting input data."""

    x: int
    y: int
    z: int
    tissue_curve: npt.NDArray[Any]
    voxel_vp: Optional[float]


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

    Args:
        fd_values: Framewise displacement per volume (mm).
        bolus_frame: Bolus arrival frame index.
        n_frames: Number of time frames.
        min_voxel_dim: Minimum voxel dimension (mm) for threshold scaling.

    Returns:
        Per-frame weight array (1.0 = full weight, <1.0 = downweighted).

    References:
        Melbourne A, et al. MRM 2007.
        Buckley DL. MRM 2002.
        Ortiz-Ramon R, et al. Eur Radiol 2017.
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
    bayesian: str,
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
    if bayesian != "none":
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
    _worker_ctx["bayesian"] = bayesian
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

    Args:
        voxel_data: VoxelInput with (x, y, z, tissue_curve, voxel_vp).

    Returns:
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
    bayesian = _worker_ctx["bayesian"]
    mcmc_samples = _worker_ctx["mcmc_samples"]
    baseline_frames = _worker_ctx["baseline_frames"]
    constraint_mode = ConstraintMode(_worker_ctx["constraint_mode_str"])
    constraint_vp_uncertainty = _worker_ctx["constraint_vp_uncertainty"]
    model_name = _worker_ctx["model_name"]
    motion_weights = _worker_ctx["motion_weights"]
    sloppy = _worker_ctx.get("sloppy", False)
    is_auto = _worker_ctx.get("is_auto", False)

    # Validate voxel (skip concentration-based variation check — raw signal
    # variation is filtered upstream in ComputeDCEMaps, matching perf-plus)
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

        # HARD constraint: use fit_constrained with fixed vp
        if (
            fixed_vp is not None
            and constraint_mode == ConstraintMode.HARD
            and hasattr(model, "fit_constrained")
        ):
            result = model.fit_constrained(
                prepared_curve, aif, time_points, fixed_vp=fixed_vp,
                weights=motion_weights,
            )
            constrained = True

        # SOFT constraint: Bayesian with informative vp prior
        elif (
            fixed_vp is not None
            and constraint_mode == ConstraintMode.SOFT
            and bayesian != "none"
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
            elif bayesian == "vi":
                soft_fitter = BayesianPKFitter(
                    model, vp_prior_override=vp_prior
                )
                result = soft_fitter.fit_vi(
                    prepared_curve, aif, time_points,
                    weights=motion_weights,
                    sloppy=sloppy,
                )
                constrained = True
            elif bayesian == "map":
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
            elif bayesian == "mcmc":
                soft_fitter = BayesianPKFitter(
                    model, vp_prior_override=vp_prior
                )
                result = soft_fitter.fit_mcmc(
                    prepared_curve, aif, time_points, n_samples=mcmc_samples,
                    sloppy=sloppy, weights=motion_weights,
                )
                constrained = True

        # SOFT constraint with NLLS: tighten vp bounds around DSC value
        elif (
            fixed_vp is not None
            and constraint_mode == ConstraintMode.SOFT
            and bayesian == "none"
        ):
            sigma = constraint_vp_uncertainty if constraint_vp_uncertainty > 0 else 0.2
            vp_lo = max(fixed_vp * (1.0 - sigma), VP_BOUNDS.min)
            vp_hi = min(fixed_vp * (1.0 + sigma), VP_BOUNDS.max)
            if vp_lo >= vp_hi:
                vp_lo, vp_hi = VP_BOUNDS.min, VP_BOUNDS.max
            result = model.fit(
                prepared_curve, aif, time_points,
                weights=motion_weights,
                vp_bounds=(vp_lo, vp_hi),
            )
            constrained = True

        # Unconstrained fallback
        if result is None:
            if is_auto or bayesian == "none":
                # is_auto: NLLS for BIC comparison; MAP applied to winner later
                result = model.fit(
                    prepared_curve, aif, time_points,
                    weights=motion_weights,
                )
            elif bayesian == "vi" and fitter is not None:
                result = fitter.fit_vi(
                    prepared_curve, aif, time_points,
                    weights=motion_weights,
                    sloppy=sloppy,
                )
            elif bayesian == "map" and fitter is not None:
                result = fitter.fit_map(
                    prepared_curve, aif, time_points,
                    compute_uncertainties=not sloppy,
                    weights=motion_weights,
                    sloppy=sloppy,
                )
            elif bayesian == "mcmc" and fitter is not None:
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
            # Kass & Raftery (1995): ΔBIC ≥ 6 = "strong evidence".
            # Prevents noise-driven model selection in low-enhancement
            # voxels where complex models have unidentifiable parameters.
            # Refs: Bagher-Ebadian MRM 2012, Sourbron & Buckley MRM 2013.
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
            if bayesian == "vi":
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

            elif bayesian == "map":
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

            elif bayesian == "mcmc":
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

        return {
            "coords": (x, y, z),
            "params": dict(result.parameters),
            "r_squared": result.r_squared,
            "converged": result.converged,
            "uncertainties": uncertainties,
            "fitted_curve": fitted_curve,
            "quality_flag": getattr(result, "quality_flag", "good"),
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

#: K2 leakage correction metric keys to extract from preload QC dict.
_K2_QC_KEYS = {
    "k2_mean", "k2_std", "k2_max", "k2_median",
    "percent_leaky_voxels", "percent_severe_leakage",
    "percent_above_clinical_max",
    "n_brain_voxels", "auc_mean",
}


def _normalize_param_name(pname: str) -> str:
    """Normalize batched result param name to output format.

    Args:
        pname: Raw parameter name from batched result (e.g. 'ktrans', 'fp', 'ps').

    Returns:
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
) -> None:
    """Compute derived parameters (kep, E, Ktrans) in place.

    Args:
        params_out: Parameter dict to augment with derived values.
        model_name: Model name for determining which derivations to apply.
    """
    if model_name in ("tofts", "extended_tofts"):
        ve_val = params_out.get("ve", EPSILON_VE)
        params_out["kep"] = params_out.get("Ktrans", 0.0) / max(ve_val, EPSILON_VE)
    elif model_name == "2cxm":
        fp_val = params_out.get("Fp", 0.0)
        ps_val = params_out.get("PS", 0.0)
        E = ps_val / (fp_val + ps_val + 1e-10)
        params_out["E"] = E
        params_out["Ktrans"] = E * fp_val / 100.0
        ve_val = params_out.get("ve", EPSILON_VE)
        params_out["kep"] = params_out["Ktrans"] / max(ve_val, EPSILON_VE)


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

    Args:
        r_const: FitResult for constrained voxels.
        const_idx: Global indices of constrained voxels.
        r_free: FitResult for free (unconstrained) voxels.
        free_idx: Global indices of free voxels.
        n_voxels: Total number of voxels.
        model_name: Model name for the merged result.

    Returns:
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

    return FitResult(
        params=params,
        r_squared=r_squared,
        converged=converged,
        fitted_curves=fitted_curves,
        n_iterations=n_iterations,
        model_name=model_name,
        uncertainties=uncertainties,
    )


def _fit_with_split_constraint(
    fit_fn: Any,
    tissue_curves: npt.NDArray[np.floating],
    fixed_vp: Optional[npt.NDArray[np.floating]],
    model_name: str,
    n_voxels: int,
    **fit_kwargs: Any,
) -> Any:
    """Handle mixed constrained/unconstrained voxels by splitting batches.

    When fixed_vp contains a mix of zeros (unconstrained) and positive values
    (constrained), splits into two batches, fits separately, and merges.

    Args:
        fit_fn: Fitting function (batched_lm_fit or batched_patlak_lstsq).
        tissue_curves: Observed tissue curves, shape (N, T).
        fixed_vp: Fixed vp values or None. Zeros = unconstrained.
        model_name: Model name.
        n_voxels: Total number of voxels.
        **fit_kwargs: Additional kwargs passed to fit_fn.

    Returns:
        FitResult with all voxels correctly handled.
    """
    if fixed_vp is None or model_name not in _MODEL_HAS_VP:
        return fit_fn(tissue_curves, fixed_vp=None, **fit_kwargs)

    has_constraint = fixed_vp > 0
    if np.all(has_constraint):
        return fit_fn(tissue_curves, fixed_vp=fixed_vp, **fit_kwargs)
    if not np.any(has_constraint):
        return fit_fn(tissue_curves, fixed_vp=None, **fit_kwargs)

    # Mixed: split, fit separately, merge
    const_idx = np.where(has_constraint)[0]
    free_idx = np.where(~has_constraint)[0]
    r_const = fit_fn(tissue_curves[const_idx], fixed_vp=fixed_vp[const_idx], **fit_kwargs)
    r_free = fit_fn(tissue_curves[free_idx], fixed_vp=None, **fit_kwargs)
    return _merge_fit_results(
        r_const, const_idx, r_free, free_idx, n_voxels, model_name,
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
    tr = traits.Float(
        desc="Pulse TR in seconds for time vector (RepetitionTimeExcitation)",
    )
    volume_tr = traits.Float(
        desc="Volume TR in seconds for model auto-selection (RepetitionTime, temporal resolution)",
    )
    time_points_file = File(
        exists=True,
        desc="Optional 1D text file with time points (minutes); if not provided, uses TR",
    )

    # Bayesian options
    bayesian = traits.Enum(
        "none",
        "vi",
        "map",
        usedefault=True,
        desc="Bayesian inference method: none (NLLS), vi, or map",
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
    # Use traits.Str instead of File: when DSC is unavailable, the upstream
    # resample node returns "" which nipype File trait cannot validate.
    dsc_cbv_file = traits.Str(
        desc="DSC-derived CBV map for vp constraint (optional, empty when unavailable)",
    )
    dsc_aif_qc = traits.Dict(
        mandatory=False,
        desc="DSC AIF QC metrics dict for CBV quality assessment (optional)",
    )
    dsc_preload_qc = traits.Dict(
        mandatory=False,
        desc="DSC preload + K2 QC metrics dict from preload_correct node",
    )
    constraint_mode = traits.Enum(
        "auto",
        "hard",
        "soft",
        "skip",
        usedefault=True,
        desc="DSC CBV constraint mode: auto, hard, soft, or skip",
    )
    hematocrit = traits.Float(
        HEMATOCRIT_ARTERIAL,
        usedefault=True,
        desc="Arterial hematocrit. Capillary Hct derived via Fahraeus ratio.",
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

    # AIF detection method (for BAT correction logic)
    aif_detection_method = traits.Str(
        "population",
        usedefault=True,
        desc="AIF detection method: 'population', 'automatic', 'atlas'. "
             "BAT correction is applied only for population AIFs.",
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
    out_quality_flag = File(desc="Quality flag map (0=unfitted, 1=good, 2=poor_convergence, 3=high_uncertainty)")
    out_r2 = File(desc="R² (goodness of fit) map")
    out_tsnr = File(desc="tSNR map (always computed)")
    out_qc_mask = File(desc="QC mask (1=good fit, 0=unfitted/poor)")
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
    - patlak: Patlak model with graphical analysis (Ktrans, vp)
    - 2cxm: Two-Compartment Exchange Model (Fp, PS, ve, vp)

    Auto-selection logic (when model='auto'):
    - If TR < 5s: Extended Tofts (better temporal resolution)
    - If TR >= 5s: Standard Tofts (reduced parameters for stability)

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
    tr : float, optional
        Repetition time in seconds (required for model='auto').
    time_points_file : str, optional
        Text file with time points in minutes; if not provided, uses TR.
    bayesian : str
        Bayesian inference: none (NLLS), map, or mcmc. Default: 'none'.
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
    >>> compute.inputs.bayesian = 'map'
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

    References
    ----------
    Tofts PS, Kermode AG. "Measurement of the blood-brain barrier
    permeability and leakage space using dynamic MR imaging."
    Magn Reson Med. 1991;17(2):357-367.

    Tofts PS, et al. "Estimating kinetic parameters from dynamic
    contrast-enhanced T1-weighted MRI of a diffusable tracer."
    J Magn Reson Imaging. 1999;10(3):223-232.

    Sourbron SP, Buckley DL. "On the scope and interpretation of
    the Tofts models for DCE-MRI." Magn Reson Med. 2011;66(3):735-745.
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

    def _select_model(self, tr_seconds: float) -> str:
        """Auto-select model based on TR.

        Args:
            tr_seconds: Repetition time in seconds.

        Returns:
            Model name: 'extended_tofts' if TR < 5s, else 'tofts'.
        """
        if tr_seconds < 5.0:
            return "extended_tofts"
        else:
            return "tofts"

    def _compute_tsnr(
        self,
        data: npt.NDArray[Any],
        mask: npt.NDArray[Any],
        baseline_frames: int,
    ) -> npt.NDArray[Any]:
        """Compute temporal SNR from baseline frames.

        tSNR = mean(baseline) / std(baseline)

        Args:
            data: 4D concentration data (x, y, z, t).
            mask: 3D brain mask.
            baseline_frames: Number of baseline frames.

        Returns:
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

        Args:
            data: 4D concentration data.
            mask: 3D brain mask.

        Returns:
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

        Args:
            model_name: Model name.

        Returns:
            True if model outputs vp.
        """
        return model_name in ("extended_tofts", "patlak", "2cxm")

    def _has_ve(self, model_name: str) -> bool:
        """Check if model has ve parameter.

        ve (and derived kep = Ktrans/ve) is estimated by tofts,
        extended_tofts, and 2cxm. Patlak does not estimate ve.

        Args:
            model_name: Model name.

        Returns:
            True if model outputs ve.
        """
        return model_name in ("tofts", "extended_tofts", "2cxm")

    # Temporal resolution gates for BIC candidate set
    _MAX_VOLUME_TR_2CXM = 3.0       # Nadav et al. JMRI 2017
    _MAX_VOLUME_TR_ETOFTS = 10.0    # Conservative clinical limit

    def _build_candidate_set(self, volume_tr: float) -> list[str]:
        """Build BIC candidate model set gated by temporal resolution.

        Standard Tofts is excluded from auto selection because its vp=0
        assumption is never valid in brain tissue. The intravascular
        contribution vp×Cp(t) is absorbed into Ktrans, causing systematic
        overestimation (Sourbron & Buckley MRM 2011). Patlak (2 params,
        has vp) and Extended Tofts (3 params, has vp) form a complete
        nested hierarchy for brain DCE. Standard Tofts remains available
        via explicit ``model=tofts`` selection.

        Args:
            volume_tr: Volume TR in seconds.

        Returns:
            List of candidate model names, ordered by complexity.
        """
        candidates = ["patlak"]
        if volume_tr <= self._MAX_VOLUME_TR_ETOFTS:
            candidates.append("extended_tofts")
        if volume_tr <= self._MAX_VOLUME_TR_2CXM:
            candidates.append("2cxm")
        return candidates

    def _run_interface(self, runtime: Any) -> Any:
        """Execute DCE map computation."""
        import nibabel as nib  # type: ignore[import-untyped]

        from pmriprep.utils.dsc_cbv_constraint import (
            ConstraintMode,
            prepare_vp_constraint_from_data,
        )

        # Load data
        conc_img = nib.load(self.inputs.concentration_file)
        conc_data: npt.NDArray[Any] = conc_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]
        affine = conc_img.affine  # type: ignore[attr-defined]
        header = conc_img.header  # type: ignore[attr-defined]

        mask_img = nib.load(self.inputs.brain_mask_file)
        mask: npt.NDArray[Any] = mask_img.get_fdata()  # type: ignore[attr-defined]

        # Load AIF
        aif = np.loadtxt(self.inputs.aif_file)

        # Determine time points
        n_timepoints = conc_data.shape[3]

        if self.inputs.time_points_file:
            time_points = np.loadtxt(self.inputs.time_points_file)
        elif isdefined(self.inputs.volume_tr) and self.inputs.volume_tr is not None:
            # Use volume TR (temporal resolution between 3D acquisitions) for
            # the time vector — this is the actual time between DCE frames.
            # Note: self.inputs.tr is the pulse TR (RF excitation interval),
            # which is ~1000x shorter for 3D sequences like VIBE.
            vol_tr_minutes = float(self.inputs.volume_tr) / 60.0
            time_points = np.arange(n_timepoints, dtype=np.float64) * vol_tr_minutes
        elif self.inputs.tr:
            # Fallback to pulse TR (correct for 2D sequences where
            # pulse TR == volume TR)
            tr_minutes = float(self.inputs.tr) / 60.0
            time_points = np.arange(n_timepoints, dtype=np.float64) * tr_minutes
        else:
            raise ValueError("Either tr or time_points_file must be provided")

        # Get TR for auto-model selection
        # Use volume_tr (temporal resolution) if available, else fall back to tr
        if isdefined(self.inputs.volume_tr) and self.inputs.volume_tr is not None:
            selection_tr_seconds = float(self.inputs.volume_tr)
        elif isdefined(self.inputs.tr) and self.inputs.tr is not None:
            selection_tr_seconds = float(self.inputs.tr)
        else:
            # Estimate TR from time points (minutes -> seconds)
            selection_tr_seconds = float(time_points[1] - time_points[0]) * 60.0

        # Select model
        model_name = str(self.inputs.model)
        is_auto = model_name == "auto"
        candidate_names: list[str] = []
        if is_auto:
            candidate_names = self._build_candidate_set(selection_tr_seconds)
            model_name = candidate_names[-1]  # ceiling = most complex eligible

        # Baseline frames for tSNR
        baseline_frames = int(self.inputs.baseline_frames)
        if baseline_frames == 0:
            baseline_frames = self._detect_baseline_frames(conc_data, mask)

        # Always compute tSNR for QC (even if threshold is 0)
        # Use raw DCE signal for tSNR when available (matches perf-plus)
        tsnr_threshold = float(self.inputs.tsnr_threshold)
        signal_std_map: Optional[npt.NDArray[Any]] = None
        if isdefined(self.inputs.dce_signal_file):
            signal_img = nib.load(self.inputs.dce_signal_file)
            signal_data = signal_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]
            tsnr_map = self._compute_tsnr(signal_data, mask, baseline_frames)
            # Raw signal variation map (matches perf-plus dce_pk_modeling.py:1576)
            signal_std_map = np.std(signal_data, axis=-1)
            # Peak enhancement must use raw signal (concentration baseline ~0
            # by definition → division by near-zero → overflow)
            from pmriprep.utils.dce_model_free import compute_peak_enhancement
            peak_enh = compute_peak_enhancement(signal_data, baseline_frames)
            del signal_data
        else:
            logger.warning(
                "dce_signal_file not provided; computing tSNR from concentration "
                "data (values will be low — consider wiring dce_preprocess.out_file)"
            )
            tsnr_map = self._compute_tsnr(conc_data, mask, baseline_frames)
            peak_enh = None  # Cannot compute without raw signal

        # Create fitting mask (brain mask + tSNR threshold if specified)
        mask_bool = mask > 0
        if tsnr_threshold > 0:
            fitting_mask = mask_bool & (tsnr_map >= tsnr_threshold)
        else:
            fitting_mask = mask_bool

        # Raw signal variation filter (perf-plus: std(S_voxel) >= 1.0)
        if signal_std_map is not None:
            fitting_mask = fitting_mask & (signal_std_map >= 1.0)

        # Pre-fit: exclude only voxels with no measurable enhancement (CSF, skull, air).
        # Concentration-domain CNR is degraded by Ernst equation noise amplification
        # (~1/(E1*TR)), making CNR > 3 reject 50-66% of brain tissue with adequate
        # signal-domain tSNR. Quality gating deferred to post-fit QC mask (R² >= 0.5,
        # tSNR >= 10, physiological bounds). Matches perf-plus approach.
        # Ref: Buckley MRM 2002 CNR guideline is informational, not a hard cutoff.
        enhancement_mask = np.max(conc_data, axis=-1) > MIN_ENHANCEMENT_MM
        fitting_mask = fitting_mask & enhancement_mask

        # Parameters
        bayesian = str(self.inputs.bayesian)
        mcmc_samples = int(self.inputs.mcmc_samples)
        save_fitted_curves = bool(self.inputs.save_fitted_curves)

        # 2CXM parameter identifiability warning (all fitting methods)
        if model_name == "2cxm":
            logger.warning(
                "2CXM individual parameter estimates (especially PS) may be "
                "unreliable due to PS/ve coupling degeneracy (Sourbron & "
                "Buckley, MRM 2011). Consider Extended Tofts if Ktrans is "
                "the primary parameter of interest."
            )

        # =====================================================================
        # BAT Correction
        # =====================================================================
        # Population AIF is pre-aligned to baseline_frames during generation
        # (see _generate_population_aif in dce_aif.py). The Parker AIF has
        # ~7s of intrinsic rise time before reaching significant levels,
        # which naturally aligns with the first post-baseline tissue
        # enhancement frame. The Tofts/2CXM convolution handles residual
        # sub-TR timing offsets. BAT correction was evaluated but found to
        # worsen test-retest reproducibility: at coarse TR (~5s), tissue BAT
        # detection has ~1-frame quantization uncertainty that introduces
        # per-visit noise, breaking the consistency of identical AIF placement
        # across visits.
        # Auto-detected AIFs share the same temporal frame as tissue data
        # and do not require BAT correction.
        aif_detection_method = str(self.inputs.aif_detection_method)
        bat_correction: dict[str, Any] = {
            "applied": False,
            "reason": (
                "population_aif_pre_aligned"
                if aif_detection_method == "population"
                else "not_population_aif"
            ),
        }

        spatial_shape = conc_data.shape[:3]

        # =====================================================================
        # DSC CBV vp Constraint
        # =====================================================================
        vp_constraint_map: Optional[npt.NDArray[Any]] = None
        constraint_mode = ConstraintMode.SKIP
        constraint_vp_uncertainty: float = 0.2
        constraint_qc: dict[str, Any] = {"mode": "skip", "reason": "No DSC CBV provided"}

        has_dsc_cbv = (
            isdefined(self.inputs.dsc_cbv_file)
            and self.inputs.dsc_cbv_file  # non-empty string
            and (model_name in ("extended_tofts", "2cxm") or is_auto)
        )

        if has_dsc_cbv:
            try:
                # Get AIF QC dict if available
                aif_qc_data = (
                    self.inputs.dsc_aif_qc
                    if isdefined(self.inputs.dsc_aif_qc)
                    else {}
                )
                preload_qc = (
                    dict(self.inputs.dsc_preload_qc)
                    if isdefined(self.inputs.dsc_preload_qc) else {}
                )
                preload_class = preload_qc.get("preload_classification")
                k2_metrics = (
                    {k: v for k, v in preload_qc.items() if k in _K2_QC_KEYS}
                    or None
                )

                vp_result = prepare_vp_constraint_from_data(
                    cbv_file=self.inputs.dsc_cbv_file,
                    aif_qc_data=aif_qc_data,
                    brain_mask_file=self.inputs.brain_mask_file,
                    mode_override=str(self.inputs.constraint_mode),
                    hematocrit=float(self.inputs.hematocrit),
                    preload_classification=preload_class,
                    k2_metrics=k2_metrics,
                )
                vp_constraint_map = vp_result.vp_map
                constraint_mode = vp_result.config.mode
                constraint_vp_uncertainty = vp_result.config.vp_uncertainty
                constraint_qc = vp_result.qc_report

                # 2CXM: vp enters nonlinearly (eigenvalues, coefficients).
                # Hard-fixing vp propagates DSC CBV error into Fp and PS
                # via k_out=Fp/vp and k_pe=PS/vp. Default to SOFT.
                if (
                    model_name == "2cxm"
                    and constraint_mode == ConstraintMode.HARD
                    and str(self.inputs.constraint_mode) == "auto"
                ):
                    constraint_mode = ConstraintMode.SOFT
                    constraint_vp_uncertainty = 0.2
                    constraint_qc["downgrade_from"] = "hard"
                    constraint_qc["mode"] = "soft"
                    constraint_qc["reason"] = (
                        "2CXM model: hard->soft (vp enters nonlinearly)"
                    )

                # Bayesian (MAP/MCMC): hard constraint bypasses the
                # Bayesian fitter and uses L-BFGS-B, which converges
                # non-enhancing voxels to exact bounds (Ktrans=0.0)
                # triggering mass post-fit rejection. SOFT uses the
                # Bayesian fitter with an informative DSC prior on vp.
                # Must run BEFORE scale mismatch so SOFT mode can
                # widen uncertainty instead of downgrading to SKIP.
                if (
                    str(self.inputs.bayesian) != "none"
                    and constraint_mode == ConstraintMode.HARD
                    and str(self.inputs.constraint_mode) == "auto"
                ):
                    constraint_mode = ConstraintMode.SOFT
                    constraint_vp_uncertainty = 0.2
                    constraint_qc["downgrade_from"] = constraint_qc.get(
                        "downgrade_from", "hard"
                    )
                    constraint_qc["mode"] = "soft"
                    constraint_qc["reason"] = (
                        "Bayesian fitting: hard->soft "
                        "(informative prior on vp)"
                    )

                # Scale mismatch check: if vp*Cp overshoots tissue
                # concentration in many voxels, the constraint is
                # incompatible (e.g., DSC/DCE AIF scale differences,
                # partial volume, half-dose acquisitions).
                # For HARD (NLLS): downgrade to SKIP (cannot adapt).
                # For SOFT (Bayesian): widen uncertainty instead.
                if (
                    constraint_mode
                    in (ConstraintMode.HARD, ConstraintMode.SOFT)
                    and str(self.inputs.constraint_mode) == "auto"
                    and vp_constraint_map is not None
                ):
                    aif_peak = float(np.max(aif))
                    if aif_peak > 0:
                        valid_constrained = (
                            fitting_mask
                            & np.isfinite(vp_constraint_map)
                            & (vp_constraint_map > 0)
                        )
                        n_valid = int(np.sum(valid_constrained))
                        if n_valid > 100:
                            vp_cp_peak = (
                                vp_constraint_map[valid_constrained] * aif_peak
                            )
                            ct_peak = np.max(
                                conc_data[valid_constrained], axis=-1
                            )
                            overshoot_frac = float(
                                np.mean(vp_cp_peak > ct_peak)
                            )
                            if overshoot_frac > 0.3:
                                median_vp = float(
                                    np.nanmedian(
                                        vp_constraint_map[valid_constrained]
                                    )
                                )
                                if constraint_mode == ConstraintMode.HARD:
                                    # NLLS: cannot recover from scale mismatch
                                    logger.warning(
                                        "DSC CBV hard constraint incompatible: "
                                        "vp*Cp_peak exceeds tissue peak in "
                                        "%.0f%% of voxels (AIF peak=%.3f mM, "
                                        "median vp=%.4f). Downgrading to "
                                        "unconstrained fitting.",
                                        overshoot_frac * 100,
                                        aif_peak,
                                        median_vp,
                                    )
                                    constraint_qc["downgrade_from"] = (
                                        constraint_qc.get(
                                            "downgrade_from",
                                            constraint_qc.get("mode", "hard"),
                                        )
                                    )
                                    constraint_mode = ConstraintMode.SKIP
                                    constraint_qc["mode"] = "skip"
                                    constraint_qc["reason"] = (
                                        f"Scale mismatch: vp*Cp overshoots "
                                        f"tissue in {overshoot_frac*100:.0f}% "
                                        f"of voxels -> skip"
                                    )
                                    constraint_qc["scale_mismatch"] = {
                                        "overshoot_fraction": overshoot_frac,
                                        "aif_peak_mM": aif_peak,
                                        "median_vp": median_vp,
                                        "action": "downgrade_to_skip",
                                    }
                                else:
                                    # Bayesian SOFT: widen prior uncertainty
                                    # instead of killing constraint
                                    logger.info(
                                        "DSC CBV scale mismatch detected "
                                        "(vp*Cp_peak exceeds tissue peak in "
                                        "%.0f%% of voxels, AIF peak=%.3f mM, "
                                        "median vp=%.4f). Widening SOFT prior "
                                        "uncertainty to 0.5.",
                                        overshoot_frac * 100,
                                        aif_peak,
                                        median_vp,
                                    )
                                    constraint_vp_uncertainty = 0.5
                                    constraint_qc["reason"] = (
                                        f"Scale mismatch: widened soft prior "
                                        f"uncertainty to 0.5 "
                                        f"({overshoot_frac*100:.0f}% overshoot)"
                                    )
                                    constraint_qc["scale_mismatch"] = {
                                        "overshoot_fraction": overshoot_frac,
                                        "aif_peak_mM": aif_peak,
                                        "median_vp": median_vp,
                                        "action": "widen_soft_uncertainty",
                                    }
            except Exception as exc:
                constraint_qc = {
                    "mode": "skip",
                    "reason": f"Failed to load DSC CBV constraint: {exc}",
                }

            # Auto mode: when DSC-CBV passes quality, ensure extended_tofts is
            # in the candidate set and set ceiling to most complex eligible
            if is_auto and constraint_mode != ConstraintMode.SKIP:
                if "extended_tofts" not in candidate_names:
                    candidate_names.append("extended_tofts")
                model_name = candidate_names[-1]

        # =====================================================================
        # Compute model-free maps (vectorized, fast, before PK fitting loop)
        # =====================================================================
        from pmriprep.utils.dce_model_free import (
            compute_bolus_arrival_time,
            compute_curve_type,
            compute_iauc,
            compute_signal_enhancement_ratio,
            compute_time_to_peak,
            compute_wash_in_slope,
            compute_wash_out_slope,
        )

        iauc60 = compute_iauc(conc_data, time_points, cutoff_s=60.0)
        iauc90 = compute_iauc(conc_data, time_points, cutoff_s=90.0)
        iauc120 = compute_iauc(conc_data, time_points, cutoff_s=120.0)
        # peak_enh already computed from raw signal above (not concentration)
        ttp = compute_time_to_peak(conc_data, time_points)
        ser = compute_signal_enhancement_ratio(
            conc_data, time_points, bolus_arrival_frame=baseline_frames,
        )

        # New semi-quantitative maps: wash-in slope, wash-out slope, BAT, curve type
        semiquant_vol_tr_s: Optional[float] = (
            float(self.inputs.volume_tr)
            if isdefined(self.inputs.volume_tr) and self.inputs.volume_tr is not None
            else None
        )
        wash_in_slope = compute_wash_in_slope(
            conc_data, time_points, bolus_arrival_frame=baseline_frames,
            volume_tr_s=semiquant_vol_tr_s,
        )
        wash_out_slope = compute_wash_out_slope(
            conc_data, time_points, volume_tr_s=semiquant_vol_tr_s,
        )
        bat = compute_bolus_arrival_time(
            conc_data, time_points, baseline_frames=baseline_frames,
            brain_mask=mask,
        )
        curve_type = compute_curve_type(ser)

        # =====================================================================
        # Initialize output arrays
        # =====================================================================
        fill_val = np.nan if is_auto else 0.0
        ktrans = np.full(spatial_shape, fill_val, dtype=np.float64)
        ve = np.full(spatial_shape, fill_val, dtype=np.float64)
        vp = np.full(spatial_shape, fill_val, dtype=np.float64)
        kep = np.full(spatial_shape, fill_val, dtype=np.float64)
        r2_map = np.zeros(spatial_shape, dtype=np.float64)

        # 2CXM-specific (only if 2CXM is in candidates or explicitly selected)
        save_2cxm = (is_auto and "2cxm" in candidate_names) or model_name == "2cxm"
        fp_map = np.zeros(spatial_shape, dtype=np.float64)
        ps_map = np.zeros(spatial_shape, dtype=np.float64)
        e_map = np.zeros(spatial_shape, dtype=np.float64)

        # Quality flag map: 0=unfitted, 1=good, 2=poor_convergence, 3=high_uncertainty
        quality_flag_map = np.zeros(spatial_shape, dtype=np.int8)
        _QUALITY_FLAG_CODES = {"good": 1, "poor_convergence": 2, "high_uncertainty": 3}

        # Model selection map: complexity-ordered encoding
        _MODEL_SELECTION_CODES: dict[str, int] = {
            "patlak": 1,          # 2 params
            "tofts": 2,           # 2 params
            "extended_tofts": 3,  # 3 params
            "2cxm": 4,            # 4 params
        }
        model_selection_map = np.zeros(spatial_shape, dtype=np.int8)

        # Delta-BIC confidence map (auto mode only)
        delta_bic_map = np.full(spatial_shape, np.nan, dtype=np.float32)

        # Uncertainty maps
        ktrans_std = np.zeros(spatial_shape, dtype=np.float64)
        ve_std = np.zeros(spatial_shape, dtype=np.float64)
        vp_std = np.zeros(spatial_shape, dtype=np.float64)
        fp_std = np.zeros(spatial_shape, dtype=np.float64)
        ps_std = np.zeros(spatial_shape, dtype=np.float64)

        # Fitted curves
        fitted_curves: Optional[npt.NDArray[Any]] = None
        if save_fitted_curves:
            fitted_curves = np.zeros(conc_data.shape, dtype=np.float32)

        # Get voxel coordinates to fit
        mask_coords = np.where(fitting_mask)
        n_voxels = len(mask_coords[0])

        # QC tracking
        n_brain = int(np.sum(mask_bool))
        qc_metrics: dict[str, Any] = {
            "model": "auto" if is_auto else model_name,
            "bayesian": bayesian,
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

        # Compute motion weights from framewise displacement (if available)
        motion_weights: Optional[npt.NDArray[np.floating]] = None
        if (
            isdefined(self.inputs.fd_values)
            and self.inputs.fd_values
            and isdefined(self.inputs.bolus_arrival_frame)
        ):
            voxel_dims = header.get_zooms()[:3]  # type: ignore[attr-defined]
            min_voxel_dim = float(min(voxel_dims))
            motion_weights = _compute_motion_weights(
                fd_values=self.inputs.fd_values,
                bolus_frame=int(self.inputs.bolus_arrival_frame),
                n_frames=n_timepoints,
                min_voxel_dim=min_voxel_dim,
            )
            n_downweighted = int(np.sum(motion_weights < 0.95))
            bolus_frame = int(self.inputs.bolus_arrival_frame)
            washin_end = min(bolus_frame + 4, n_timepoints - 1)
            qc_metrics["motion_weighting"] = {
                "applied": True,
                "n_downweighted_frames": n_downweighted,
                "min_weight": float(np.min(motion_weights)),
                "washin_weights": motion_weights[bolus_frame:washin_end + 1].tolist(),
            }
            logger.info(
                "Motion weighting: %d/%d frames downweighted (min weight=%.3f)",
                n_downweighted, n_timepoints, float(np.min(motion_weights)),
            )
        else:
            qc_metrics["motion_weighting"] = {"applied": False}

        # Build voxel data list for parallel/serial processing
        voxel_data_list: list[VoxelInput] = []
        for i in range(n_voxels):
            x, y, z = (
                int(mask_coords[0][i]),
                int(mask_coords[1][i]),
                int(mask_coords[2][i]),
            )
            tissue_curve = conc_data[x, y, z, :]
            voxel_vp: Optional[float] = None
            if vp_constraint_map is not None and constraint_mode != ConstraintMode.SKIP:
                vp_val = float(vp_constraint_map[x, y, z])
                if np.isfinite(vp_val) and vp_val > 0:
                    voxel_vp = vp_val
            voxel_data_list.append(VoxelInput(x, y, z, tissue_curve, voxel_vp))

        # Worker initializer args (shared across all paths)
        sloppy = bool(self.inputs.sloppy)
        random_seed = (
            int(self.inputs.random_seed) if isdefined(self.inputs.random_seed) else None
        )
        init_args = (
            aif, time_points, model_name, bayesian, mcmc_samples,
            baseline_frames, constraint_mode.value, constraint_vp_uncertainty,
            save_fitted_curves, motion_weights, sloppy, random_seed,
            candidate_names if is_auto else None,
        )

        n_procs = max(1, int(self.inputs.n_procs))

        # Always use batched fitting (vectorized operations)
        results = self._fit_batched(
            conc_data=conc_data,
            mask_coords=mask_coords,
            aif=aif,
            time_points=time_points,
            model_name=model_name,
            bayesian=bayesian,
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
            ktrans_val = params.get("Ktrans", 0.0)
            ve_val = params.get("ve", 0.0)
            vp_val = params.get("vp", 0.0)
            is_constrained = res.get("constrained", False)
            # For constrained fits (vp fixed from DSC-CBV), Ktrans=0 is
            # the correct physical answer for non-enhancing tissue —
            # only reject strictly negative (numerical error). vp checks
            # are skipped because vp was fixed and guaranteed positive.
            ktrans_lo = ktrans_val < 0 if is_constrained else ktrans_val <= 0
            if effective_model_name == "2cxm":
                fp_val = params.get("Fp", 0.0)
                ps_val = params.get("PS", 0.0)
                degenerate = (
                    fp_val <= 0 or fp_val >= FP_BOUNDS.max
                    or ps_val < 0 or ps_val >= PS_BOUNDS.max
                    or ve_val <= 0 or ve_val >= VE_BOUNDS.max
                    or vp_val <= 0 or vp_val >= VP_BOUNDS.max
                )
            elif effective_model_name == "patlak":
                # Patlak: Ktrans=0 is valid (non-enhancing tissue), only
                # reject strictly negative. ve is not a Patlak parameter.
                degenerate = (
                    (ktrans_val < 0) or ktrans_val >= KTRANS_BOUNDS.max
                    or (not is_constrained and (vp_val <= 0 or vp_val >= VP_BOUNDS.max))
                )
            elif effective_model_name == "extended_tofts":
                degenerate = (
                    ktrans_lo or ktrans_val >= KTRANS_BOUNDS.max
                    or ve_val <= 0 or ve_val >= VE_BOUNDS.max
                    or (not is_constrained and (vp_val <= 0 or vp_val >= VP_BOUNDS.max))
                )
            else:  # tofts
                degenerate = (
                    ktrans_val <= 0 or ktrans_val >= KTRANS_BOUNDS.max
                    or ve_val <= 0 or ve_val >= VE_BOUNDS.max
                )
            if degenerate:
                degen_fill = np.nan if is_auto else 0.0
                ktrans[x, y, z] = degen_fill
                if self._has_ve(effective_model_name):
                    ve[x, y, z] = degen_fill
                    kep[x, y, z] = degen_fill
                if self._has_vp(effective_model_name):
                    vp[x, y, z] = degen_fill
                r2_map[x, y, z] = res["r_squared"]
                n_rejected_postfit += 1
                continue

            ktrans[x, y, z] = ktrans_val
            if self._has_ve(effective_model_name):
                ve[x, y, z] = ve_val
                kep[x, y, z] = params.get("kep", 0.0)
            r2_map[x, y, z] = res["r_squared"]

            if self._has_vp(effective_model_name):
                vp[x, y, z] = params.get("vp", 0.0)

            # 2CXM-specific parameters
            if effective_model_name == "2cxm":
                fp_map[x, y, z] = params.get("Fp", 0.0)
                ps_map[x, y, z] = params.get("PS", 0.0)
                e_map[x, y, z] = params.get("E", 0.0)

            # Uncertainty (Bayesian only)
            if bayesian != "none" and res["uncertainties"]:
                ktrans_std[x, y, z] = res["uncertainties"].get("Ktrans", 0.0)
                if self._has_ve(effective_model_name):
                    ve_std[x, y, z] = res["uncertainties"].get("ve", 0.0)
                if self._has_vp(effective_model_name):
                    vp_std[x, y, z] = res["uncertainties"].get("vp", 0.0)
                # 2CXM-specific: Fp and PS uncertainties
                if effective_model_name == "2cxm":
                    fp_std[x, y, z] = res["uncertainties"].get("Fp", 0.0)
                    ps_std[x, y, z] = res["uncertainties"].get("PS", 0.0)

            # Fitted curve
            if save_fitted_curves and fitted_curves is not None:
                if res["fitted_curve"] is not None:
                    fitted_curves[x, y, z, :] = res["fitted_curve"].astype(np.float32)

            # Set quality flag
            qf = res.get("quality_flag", "good")
            quality_flag_map[x, y, z] = _QUALITY_FLAG_CODES.get(qf, 1)

            # Track convergence and constraints
            if res.get("constrained", False):
                n_constrained += 1
            elif vd.voxel_vp is not None and constraint_mode != ConstraintMode.SKIP:
                # Had vp constraint but fell back to unconstrained
                n_unconstrained_fallback += 1

            # z_departure: how much DCE data pulled vp away from DSC prior
            # Only meaningful for SOFT + Bayesian (where a soft prior was used)
            if (
                constraint_mode == ConstraintMode.SOFT
                and bayesian != "none"
                and self._has_vp(effective_model_name)
                and vd.voxel_vp is not None
                and vd.voxel_vp > 0
            ):
                vp_fitted = params.get("vp", 0.0)
                vp_prior = vd.voxel_vp
                prior_sigma = max(
                    vp_prior * constraint_vp_uncertainty, 0.001
                )
                z = abs(vp_fitted - vp_prior) / prior_sigma
                z_departure_list.append(z)

            if res["converged"]:
                n_converged += 1
                r2_list.append(res["r_squared"])
            else:
                n_failed += 1

            # Accumulate MCMC diagnostics (rhat/ess are empty for non-MCMC)
            voxel_rhat = res.get("rhat", {})
            voxel_ess = res.get("ess", {})
            if voxel_rhat:
                n_mcmc_voxels += 1
                for param, val in voxel_rhat.items():
                    mcmc_rhat_lists.setdefault(param, []).append(val)
                for param, val in voxel_ess.items():
                    mcmc_ess_lists.setdefault(param, []).append(val)

        # Compile QC metrics
        qc_metrics["n_converged"] = n_converged
        qc_metrics["n_failed"] = n_failed
        qc_metrics["convergence_rate"] = n_converged / max(1, n_voxels)
        qc_metrics["n_constrained"] = n_constrained
        qc_metrics["n_unconstrained_fallback"] = n_unconstrained_fallback
        qc_metrics["n_rejected_postfit"] = n_rejected_postfit
        qc_metrics["failure_reasons"] = failure_reason_counts

        qc_metrics["model_selection"] = "auto" if is_auto else "fixed"

        # Per-model selection stats (auto mode)
        if is_auto:
            qc_metrics["candidate_models"] = candidate_names
            n_total = int(np.sum(model_selection_map > 0))
            for code_name in candidate_names:
                count = int(np.sum(
                    model_selection_map == _MODEL_SELECTION_CODES[code_name]
                ))
                qc_metrics[f"n_{code_name}_selected"] = count
                qc_metrics[f"{code_name}_fraction"] = (
                    count / max(1, n_total)
                )

        # Quality flag breakdown
        qf_vals = quality_flag_map[fitting_mask]
        qc_metrics["quality_n_good"] = int(np.sum(qf_vals == 1))
        qc_metrics["quality_n_poor_convergence"] = int(np.sum(qf_vals == 2))
        qc_metrics["quality_n_high_uncertainty"] = int(np.sum(qf_vals == 3))

        if r2_list:
            qc_metrics["r2_mean"] = float(np.mean(r2_list))
            qc_metrics["r2_std"] = float(np.std(r2_list))
            qc_metrics["r2_median"] = float(np.median(r2_list))

        # MCMC convergence diagnostics summary
        if n_mcmc_voxels > 0:
            from pmriprep.models.dce.bayesian import QC_RHAT_MAX, QC_ESS_MIN

            mcmc_diag: dict[str, Any] = {"n_mcmc_voxels": n_mcmc_voxels}
            for param, vals in mcmc_rhat_lists.items():
                arr = np.array(vals)
                mcmc_diag[f"rhat_median_{param}"] = float(np.median(arr))
                mcmc_diag[f"rhat_frac_poor_{param}"] = float(
                    np.mean(arr >= QC_RHAT_MAX)
                )
            for param, vals in mcmc_ess_lists.items():
                arr = np.array(vals)
                mcmc_diag[f"ess_median_{param}"] = float(np.median(arr))
                mcmc_diag[f"ess_frac_low_{param}"] = float(
                    np.mean(arr < QC_ESS_MIN)
                )
            qc_metrics["mcmc_diagnostics"] = mcmc_diag

        # z_departure stats: soft constraint prior departure
        if z_departure_list:
            z_arr = np.array(z_departure_list)
            qc_metrics["z_departure_median"] = float(np.median(z_arr))
            qc_metrics["z_departure_p90"] = float(np.percentile(z_arr, 90))
            qc_metrics["frac_prior_dominated"] = float(np.mean(z_arr < 0.5))
            qc_metrics["frac_strong_departure"] = float(np.mean(z_arr > 2.0))

        # Compute summary statistics on fitted voxels
        # For auto mode, use np.isfinite guard (NaN = non-applicable parameter)
        fitted_mask = fitting_mask & (np.isfinite(ktrans)) & (ktrans > 0)
        if np.any(fitted_mask):
            qc_metrics["ktrans_mean"] = float(np.nanmean(ktrans[fitted_mask]))
            qc_metrics["ktrans_median"] = float(np.nanmedian(ktrans[fitted_mask]))

            if save_ve:
                ve_vals = ve[fitted_mask]
                finite_ve = np.isfinite(ve_vals)
                if np.any(finite_ve):
                    qc_metrics["ve_mean"] = float(np.nanmean(ve_vals[finite_ve]))
                    qc_metrics["ve_median"] = float(np.nanmedian(ve_vals[finite_ve]))

            if save_vp:
                vp_vals = vp[fitted_mask]
                finite_vp = np.isfinite(vp_vals)
                if np.any(finite_vp):
                    qc_metrics["vp_mean"] = float(np.nanmean(vp_vals[finite_vp]))
                    qc_metrics["vp_median"] = float(np.nanmedian(vp_vals[finite_vp]))

            if save_2cxm:
                fp_vals = fp_map[fitted_mask]
                finite_fp = np.isfinite(fp_vals) & (fp_vals > 0)
                if np.any(finite_fp):
                    qc_metrics["fp_median"] = float(np.nanmedian(fp_vals[finite_fp]))
                ps_vals = ps_map[fitted_mask]
                finite_ps = np.isfinite(ps_vals) & (ps_vals > 0)
                if np.any(finite_ps):
                    qc_metrics["ps_median"] = float(np.nanmedian(ps_vals[finite_ps]))

            # Histogram statistics for fitted parameters
            ktrans_vals = ktrans[fitted_mask]
            hist_counts, hist_edges = np.histogram(ktrans_vals[np.isfinite(ktrans_vals)], bins=50)
            qc_metrics["ktrans_histogram"] = hist_counts.tolist()
            qc_metrics["ktrans_histogram_edges"] = hist_edges.tolist()

            if save_ve:
                ve_hist = ve[fitted_mask]
                ve_hist = ve_hist[np.isfinite(ve_hist)]
                if len(ve_hist) > 0:
                    hist_counts_ve, hist_edges_ve = np.histogram(ve_hist, bins=50)
                    qc_metrics["ve_histogram"] = hist_counts_ve.tolist()
                    qc_metrics["ve_histogram_edges"] = hist_edges_ve.tolist()

        # Generate output filenames
        out_ktrans = self._gen_outfilename("_Ktrans")
        out_r2 = self._gen_outfilename("_R2")

        # Save main outputs
        ktrans_img = nib.Nifti1Image(ktrans.astype(np.float32), affine, header)  # type: ignore[attr-defined]
        nib.save(ktrans_img, out_ktrans)  # type: ignore[attr-defined]

        # ve and kep are model-dependent: produced by Tofts, ExtTofts, 2CXM
        # but NOT by Patlak (which only estimates Ktrans and vp).
        # In auto mode, always save (NaN where not applicable).
        if save_ve:
            out_ve = self._gen_outfilename("_ve")
            ve_img = nib.Nifti1Image(ve.astype(np.float32), affine, header)  # type: ignore[attr-defined]
            nib.save(ve_img, out_ve)  # type: ignore[attr-defined]

            out_kep = self._gen_outfilename("_kep")
            kep_img = nib.Nifti1Image(kep.astype(np.float32), affine, header)  # type: ignore[attr-defined]
            nib.save(kep_img, out_kep)  # type: ignore[attr-defined]

        r2_img = nib.Nifti1Image(r2_map.astype(np.float32), affine, header)  # type: ignore[attr-defined]
        nib.save(r2_img, out_r2)  # type: ignore[attr-defined]

        # Save quality flag map
        out_quality_flag = self._gen_outfilename("_quality_flag")
        qf_img = nib.Nifti1Image(quality_flag_map, affine, header)  # type: ignore[attr-defined]
        nib.save(qf_img, out_quality_flag)  # type: ignore[attr-defined]

        # Save model-free maps
        out_iauc60_path = self._gen_outfilename("_IAUC60")
        out_iauc90_path = self._gen_outfilename("_IAUC90")
        out_iauc120_path = self._gen_outfilename("_IAUC120")
        out_peak_enh_path = self._gen_outfilename("_peak_enhancement")
        out_ttp_path = self._gen_outfilename("_time_to_peak")
        out_ser_path = self._gen_outfilename("_SER")

        out_wash_in_path = self._gen_outfilename("_wash_in_slope")
        out_wash_out_path = self._gen_outfilename("_wash_out_slope")
        out_bat_path = self._gen_outfilename("_BAT")
        out_curve_type_path = self._gen_outfilename("_curve_type")

        save_list: list[tuple[Any, str]] = [
            (iauc60, out_iauc60_path),
            (iauc90, out_iauc90_path),
            (iauc120, out_iauc120_path),
            (ttp, out_ttp_path),
            (ser, out_ser_path),
            (wash_in_slope, out_wash_in_path),
            (wash_out_slope, out_wash_out_path),
            (bat, out_bat_path),
        ]
        if peak_enh is not None:
            peak_enh[~mask_bool] = 0.0
            save_list.insert(3, (peak_enh, out_peak_enh_path))
        for arr, path in save_list:
            img = nib.Nifti1Image(arr.astype(np.float32), affine, header)  # type: ignore[attr-defined]
            nib.save(img, path)  # type: ignore[attr-defined]

        # Save curve type as int8 (categorical)
        ct_img = nib.Nifti1Image(curve_type, affine, header)  # type: ignore[attr-defined]
        nib.save(ct_img, out_curve_type_path)  # type: ignore[attr-defined]

        # Initialize results
        self._results: dict[str, Any] = {
            "out_ktrans": out_ktrans,
            "out_r2": out_r2,
            "out_quality_flag": out_quality_flag,
            "out_iauc60": out_iauc60_path,
            "out_iauc90": out_iauc90_path,
            "out_iauc120": out_iauc120_path,
            "out_peak_enhancement": out_peak_enh_path,
            "out_time_to_peak": out_ttp_path,
            "out_ser": out_ser_path,
            "out_wash_in_slope": out_wash_in_path,
            "out_wash_out_slope": out_wash_out_path,
            "out_bat": out_bat_path,
            "out_curve_type": out_curve_type_path,
            "qc_metrics": qc_metrics,
            "constraint_qc": constraint_qc,
        }

        if save_ve:
            self._results["out_ve"] = out_ve
            self._results["out_kep"] = out_kep

        # Save vp if applicable
        if save_vp:
            out_vp = self._gen_outfilename("_vp")
            vp_img = nib.Nifti1Image(vp.astype(np.float32), affine, header)  # type: ignore[attr-defined]
            nib.save(vp_img, out_vp)  # type: ignore[attr-defined]
            self._results["out_vp"] = out_vp

        # Save 2CXM-specific outputs
        if save_2cxm:
            out_fp = self._gen_outfilename("_Fp")
            out_ps = self._gen_outfilename("_PS")
            out_e = self._gen_outfilename("_E")

            fp_img = nib.Nifti1Image(fp_map.astype(np.float32), affine, header)  # type: ignore[attr-defined]
            nib.save(fp_img, out_fp)  # type: ignore[attr-defined]

            ps_img = nib.Nifti1Image(ps_map.astype(np.float32), affine, header)  # type: ignore[attr-defined]
            nib.save(ps_img, out_ps)  # type: ignore[attr-defined]

            e_img = nib.Nifti1Image(e_map.astype(np.float32), affine, header)  # type: ignore[attr-defined]
            nib.save(e_img, out_e)  # type: ignore[attr-defined]

            self._results["out_fp"] = out_fp
            self._results["out_ps"] = out_ps
            self._results["out_e"] = out_e

        # Save uncertainty maps
        if bayesian != "none":
            out_ktrans_std = self._gen_outfilename("_Ktrans_std")

            ktrans_std_img = nib.Nifti1Image(ktrans_std.astype(np.float32), affine, header)  # type: ignore[attr-defined]
            nib.save(ktrans_std_img, out_ktrans_std)  # type: ignore[attr-defined]

            self._results["out_ktrans_std"] = out_ktrans_std

            if save_ve:
                out_ve_std = self._gen_outfilename("_ve_std")
                ve_std_img = nib.Nifti1Image(ve_std.astype(np.float32), affine, header)  # type: ignore[attr-defined]
                nib.save(ve_std_img, out_ve_std)  # type: ignore[attr-defined]
                self._results["out_ve_std"] = out_ve_std

            if save_vp:
                out_vp_std = self._gen_outfilename("_vp_std")
                vp_std_img = nib.Nifti1Image(vp_std.astype(np.float32), affine, header)  # type: ignore[attr-defined]
                nib.save(vp_std_img, out_vp_std)  # type: ignore[attr-defined]
                self._results["out_vp_std"] = out_vp_std

            # 2CXM-specific: save Fp and PS uncertainty maps
            if save_2cxm:
                out_fp_std_file = self._gen_outfilename("_Fp_std")
                fp_std_img = nib.Nifti1Image(fp_std.astype(np.float32), affine, header)  # type: ignore[attr-defined]
                nib.save(fp_std_img, out_fp_std_file)  # type: ignore[attr-defined]
                self._results["out_fp_std"] = out_fp_std_file

                out_ps_std_file = self._gen_outfilename("_PS_std")
                ps_std_img = nib.Nifti1Image(ps_std.astype(np.float32), affine, header)  # type: ignore[attr-defined]
                nib.save(ps_std_img, out_ps_std_file)  # type: ignore[attr-defined]
                self._results["out_ps_std"] = out_ps_std_file

            # Compute and save Coefficient of Variation (CV = std / mean)
            # Use safe division to avoid divide-by-zero
            cv_fill = np.nan if is_auto else 0.0
            with np.errstate(divide="ignore", invalid="ignore"):
                ktrans_cv = np.where(
                    ktrans > EPSILON_VE, ktrans_std / ktrans, cv_fill
                ).astype(np.float32)

            out_ktrans_cv = self._gen_outfilename("_Ktrans_cv")
            nib.save(nib.Nifti1Image(ktrans_cv, affine, header), out_ktrans_cv)  # type: ignore[attr-defined]
            self._results["out_ktrans_cv"] = out_ktrans_cv

            if save_ve:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ve_cv = np.where(
                        ve > EPSILON_VE, ve_std / ve, cv_fill
                    ).astype(np.float32)
                out_ve_cv = self._gen_outfilename("_ve_cv")
                nib.save(nib.Nifti1Image(ve_cv, affine, header), out_ve_cv)  # type: ignore[attr-defined]
                self._results["out_ve_cv"] = out_ve_cv

            if save_vp:
                with np.errstate(divide="ignore", invalid="ignore"):
                    vp_cv = np.where(
                        vp > EPSILON_VE, vp_std / vp, cv_fill
                    ).astype(np.float32)
                out_vp_cv = self._gen_outfilename("_vp_cv")
                nib.save(nib.Nifti1Image(vp_cv, affine, header), out_vp_cv)  # type: ignore[attr-defined]
                self._results["out_vp_cv"] = out_vp_cv

            # Compute and save 95% Credible Intervals (mean ± 1.96 * std)
            # Assuming approximately normal posterior (valid for MAP/MLE)
            ktrans_ci95_lower = np.maximum(ktrans - 1.96 * ktrans_std, 0.0).astype(np.float32)
            ktrans_ci95_upper = (ktrans + 1.96 * ktrans_std).astype(np.float32)

            out_ktrans_ci95_lower = self._gen_outfilename("_Ktrans_ci95lower")
            out_ktrans_ci95_upper = self._gen_outfilename("_Ktrans_ci95upper")

            nib.save(nib.Nifti1Image(ktrans_ci95_lower, affine, header), out_ktrans_ci95_lower)  # type: ignore[attr-defined]
            nib.save(nib.Nifti1Image(ktrans_ci95_upper, affine, header), out_ktrans_ci95_upper)  # type: ignore[attr-defined]

            self._results["out_ktrans_ci95_lower"] = out_ktrans_ci95_lower
            self._results["out_ktrans_ci95_upper"] = out_ktrans_ci95_upper

            if save_ve:
                ve_ci95_lower = np.clip(ve - 1.96 * ve_std, 0.0, 1.0).astype(np.float32)
                ve_ci95_upper = np.clip(ve + 1.96 * ve_std, 0.0, 1.0).astype(np.float32)

                out_ve_ci95_lower = self._gen_outfilename("_ve_ci95lower")
                out_ve_ci95_upper = self._gen_outfilename("_ve_ci95upper")

                nib.save(nib.Nifti1Image(ve_ci95_lower, affine, header), out_ve_ci95_lower)  # type: ignore[attr-defined]
                nib.save(nib.Nifti1Image(ve_ci95_upper, affine, header), out_ve_ci95_upper)  # type: ignore[attr-defined]

                self._results["out_ve_ci95_lower"] = out_ve_ci95_lower
                self._results["out_ve_ci95_upper"] = out_ve_ci95_upper

            if save_vp:
                vp_ci95_lower = np.clip(vp - 1.96 * vp_std, 0.0, 1.0).astype(np.float32)
                vp_ci95_upper = np.clip(vp + 1.96 * vp_std, 0.0, 1.0).astype(np.float32)

                out_vp_ci95_lower = self._gen_outfilename("_vp_ci95lower")
                out_vp_ci95_upper = self._gen_outfilename("_vp_ci95upper")

                nib.save(nib.Nifti1Image(vp_ci95_lower, affine, header), out_vp_ci95_lower)  # type: ignore[attr-defined]
                nib.save(nib.Nifti1Image(vp_ci95_upper, affine, header), out_vp_ci95_upper)  # type: ignore[attr-defined]

                self._results["out_vp_ci95_lower"] = out_vp_ci95_lower
                self._results["out_vp_ci95_upper"] = out_vp_ci95_upper

        # Save tSNR map (always computed)
        out_tsnr = self._gen_outfilename("_tSNR")
        tsnr_img = nib.Nifti1Image(tsnr_map.astype(np.float32), affine, header)  # type: ignore[attr-defined]
        nib.save(tsnr_img, out_tsnr)  # type: ignore[attr-defined]
        self._results["out_tsnr"] = out_tsnr

        # Create QC mask using physiological bounds (matches perf-plus)
        # Include voxels with: good R², physiological parameter values, adequate SNR
        qc_mask = np.ones(spatial_shape, dtype=bool)
        qc_mask &= fitting_mask  # Within brain mask
        qc_mask &= (tsnr_map >= 10)  # Adequate tSNR (perf-plus threshold)
        if peak_enh is not None:
            qc_mask &= (peak_enh >= 10)  # Visible enhancement
        qc_mask &= (ktrans > 0) & (ktrans < 5.0)  # Physiological Ktrans
        qc_mask &= (r2_map >= 0.5)  # Reasonable model fit

        # Model-specific constraints (per-voxel model-aware for BIC selection)
        is_patlak_voxel = model_selection_map == _MODEL_SELECTION_CODES.get("patlak", -1)
        if is_auto or model_name in ["extended_tofts", "2cxm"]:
            # Volume fraction constraint: ve+vp < 0.95
            # NaN-safe: treat NaN as 0 so Patlak voxels pass
            ve_safe = np.where(np.isfinite(ve), ve, 0.0)
            vp_safe = np.where(np.isfinite(vp), vp, 0.0)
            etofts_volume_ok = (ve_safe + vp_safe) < 0.95
            qc_mask &= (etofts_volume_ok | is_patlak_voxel)
        if is_auto or model_name != "patlak":
            # kep check: NaN kep is fine for Patlak voxels (kep undefined)
            kep_ok = np.where(np.isfinite(kep), kep > 0.001, True)
            qc_mask &= kep_ok

        qc_mask = qc_mask.astype(np.uint8)
        out_qc_mask = self._gen_outfilename("_qc_mask")
        qc_mask_img = nib.Nifti1Image(qc_mask, affine, header)  # type: ignore[attr-defined]
        nib.save(qc_mask_img, out_qc_mask)  # type: ignore[attr-defined]
        self._results["out_qc_mask"] = out_qc_mask

        # Save model selection map (auto/BIC mode)
        if is_auto:
            out_model_sel = self._gen_outfilename("_modelSelection")
            model_sel_img = nib.Nifti1Image(model_selection_map, affine, header)  # type: ignore[attr-defined]
            nib.save(model_sel_img, out_model_sel)  # type: ignore[attr-defined]
            self._results["out_model_selection"] = out_model_sel

            # Save delta-BIC confidence map
            out_delta_bic = self._gen_outfilename("_deltaBIC")
            delta_bic_img = nib.Nifti1Image(delta_bic_map.astype(np.float32), affine, header)  # type: ignore[attr-defined]
            nib.save(delta_bic_img, out_delta_bic)  # type: ignore[attr-defined]
            self._results["out_delta_bic"] = out_delta_bic

        # Save fitted curves
        if save_fitted_curves and fitted_curves is not None:
            out_fitted = self._gen_outfilename("_fitted_curves")
            fitted_img = nib.Nifti1Image(fitted_curves, affine)  # type: ignore[attr-defined]
            nib.save(fitted_img, out_fitted)  # type: ignore[attr-defined]
            self._results["out_fitted_curves"] = out_fitted

        return runtime

    def _fit_batched(
        self,
        conc_data: npt.NDArray[Any],
        mask_coords: tuple[npt.NDArray[Any], ...],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        model_name: str,
        bayesian: str,
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

        Args:
            conc_data: Concentration data, shape (X, Y, Z, T).
            mask_coords: Tuple of coordinate arrays from np.where(fitting_mask).
            aif: AIF, shape (T,).
            time_points: Time points, shape (T,).
            model_name: Primary model name.
            bayesian: Bayesian mode ('none', 'map', 'vi', 'mcmc').
            is_auto: Whether auto-mode BIC selection is active.
            candidate_names: List of candidate model names for auto mode.
            vp_constraint_map: vp constraint map or None.
            constraint_mode: Constraint mode (SKIP, HARD, SOFT).
            constraint_vp_uncertainty: Uncertainty for soft constraint.
            motion_weights: Per-frame motion weights or None.
            save_fitted_curves: Whether to save fitted curves.
            sloppy: Whether to use sloppy mode.
            mcmc_samples: Number of MCMC samples.
            baseline_frames: Number of baseline frames.
            n_procs: Number of CPU processes for MAP/MCMC refinement.
            random_seed: Random seed or None.
            init_args: Worker init args for CPU fallback.
            voxel_data_list: List of VoxelInput for CPU fallback paths.

        Returns:
            List of result dicts matching _fit_single_voxel output format.
        """
        from pmriprep.models.dce.batched_fitting import (
            FitResult,
            batched_lm_fit,
            batched_patlak_lstsq,
        )
        from pmriprep.utils.dsc_cbv_constraint import ConstraintMode

        n_voxels = len(mask_coords[0])
        n_timepoints = len(aif)

        # VI/MCMC: fall through to CPU multiprocessing (unchanged)
        if bayesian in ("vi", "mcmc"):
            logger.info("Batched mode with %s: falling through to CPU multiprocessing", bayesian)
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

        results: list[dict[str, Any]] = []

        if not is_auto:
            # Single-model batched fit with split constraint handling
            if model_name == "patlak":
                def _patlak_fit(tc: npt.NDArray[np.floating], **kw: Any) -> FitResult:
                    return batched_patlak_lstsq(
                        tc, aif, time_points, **kw,
                    )
                fit_result = _fit_with_split_constraint(
                    _patlak_fit, tissue_curves, fixed_vp, model_name, n_voxels,
                    weights=motion_weights,
                    save_fitted_curves=save_fitted_curves,
                    compute_uncertainties=bayesian == "none",
                )
            else:
                def _lm_fit(tc: npt.NDArray[np.floating], **kw: Any) -> FitResult:
                    return batched_lm_fit(
                        model_name, tc, aif, time_points, **kw,
                    )
                fit_result = _fit_with_split_constraint(
                    _lm_fit, tissue_curves, fixed_vp, model_name, n_voxels,
                    weights=motion_weights,
                    save_fitted_curves=save_fitted_curves or bayesian == "map",
                    compute_uncertainties=bayesian == "none",
                )

            results = self._fit_result_to_dicts(
                fit_result, coords, model_name,
                fixed_vp=fixed_vp,
            )

            # MAP refinement: use batched fit as warm-start for CPU MAP
            if bayesian == "map":
                results = self._refine_map_from_batched(
                    results, voxel_data_list, init_args, n_procs,
                )

            return results

        # Auto mode: fit ALL candidate models using batched fitting, select via BIC
        from pmriprep.models.dce.base import EPSILON_RSS

        fit_results: dict[str, FitResult] = {}
        for cname in candidate_names:
            try:
                if cname == "patlak":
                    def _patlak_auto(tc: npt.NDArray[np.floating], **kw: Any) -> FitResult:
                        return batched_patlak_lstsq(tc, aif, time_points, **kw)
                    fit_results[cname] = _fit_with_split_constraint(
                        _patlak_auto, tissue_curves, fixed_vp, cname, n_voxels,
                        weights=motion_weights,
                        save_fitted_curves=True,
                    )
                else:
                    def _lm_auto(tc: npt.NDArray[np.floating], *, _model: str = cname, **kw: Any) -> FitResult:
                        return batched_lm_fit(_model, tc, aif, time_points, **kw)
                    fit_results[cname] = _fit_with_split_constraint(
                        _lm_auto, tissue_curves, fixed_vp, cname, n_voxels,
                        weights=motion_weights,
                        save_fitted_curves=True,
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
            # Per-voxel effective param count: subtract 1 only where vp is fixed
            k_base = _N_PARAMS[cname]
            if cname in _MODEL_HAS_VP and fixed_vp is not None:
                k_arr = np.where(
                    fixed_vp > 0, k_base - 1, k_base,
                ).astype(float)
            else:
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
        # Refs: Kass & Raftery (1995), Bagher-Ebadian MRM 2012.
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

            _compute_derived_params(params_out, winner_name)

            # Uncertainties
            uncertainties: dict[str, float] = {}
            if winner_result.uncertainties:
                for pname, uarr in winner_result.uncertainties.items():
                    uncertainties[_normalize_param_name(pname)] = float(uarr[i])

            fitted_curve = None
            if save_fitted_curves and winner_result.fitted_curves is not None:
                fitted_curve = winner_result.fitted_curves[i]

            results.append({
                "coords": (x, y, z),
                "params": params_out,
                "r_squared": float(winner_result.r_squared[i]),
                "converged": bool(winner_result.converged[i]),
                "uncertainties": uncertainties,
                "fitted_curve": fitted_curve,
                "quality_flag": "good" if winner_result.converged[i] else "poor_convergence",
                "rhat": {},
                "ess": {},
                "failed": not bool(winner_result.converged[i]) and float(winner_result.r_squared[i]) < -0.5,
                "failure_reason": None,
                "constrained": is_constrained,
                "selected_model_name": winner_name,
                "delta_bic": float(delta_bic[i]),
            })

        # MAP refinement of BIC winners
        if bayesian == "map":
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

        Args:
            fit_result: FitResult from batched fitting.
            coords: Voxel coordinates, shape (N, 3).
            model_name: Model name.
            fixed_vp: Fixed vp values or None.

        Returns:
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

            _compute_derived_params(params_out, model_name)

            # Uncertainties
            uncertainties: dict[str, float] = {}
            if fit_result.uncertainties:
                for pname, uarr in fit_result.uncertainties.items():
                    uncertainties[_normalize_param_name(pname)] = float(uarr[i])

            fitted_curve = None
            if fit_result.fitted_curves is not None:
                fitted_curve = fit_result.fitted_curves[i]

            results.append({
                "coords": (x, y, z),
                "params": params_out,
                "r_squared": float(fit_result.r_squared[i]),
                "converged": bool(fit_result.converged[i]),
                "uncertainties": uncertainties,
                "fitted_curve": fitted_curve,
                "quality_flag": "good" if fit_result.converged[i] else "poor_convergence",
                "rhat": {},
                "ess": {},
                "failed": not bool(fit_result.converged[i]) and float(fit_result.r_squared[i]) < -0.5,
                "failure_reason": None,
                "constrained": is_constrained,
                "selected_model_name": model_name,
                "delta_bic": np.nan,
            })

        return results

    def _refine_map_from_batched(
        self,
        batched_results: list[dict[str, Any]],
        voxel_data_list: list[VoxelInput],
        init_args: tuple[Any, ...],
        n_procs: int,
    ) -> list[dict[str, Any]]:
        """Runs CPU MAP fitting on all voxels.

        Uses CPU MAP where it converges, falling back to batched NLLS otherwise.

        Args:
            batched_results: Per-voxel result dicts from batched fitting.
            voxel_data_list: VoxelInput list for CPU fitting.
            init_args: Worker initialization args.
            n_procs: Number of CPU processes.

        Returns:
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
