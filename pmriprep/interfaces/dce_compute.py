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
    KTRANS_BOUNDS,
    VE_BOUNDS,
)
from pmriprep.utils.dce_voxel_prep import prepare_voxel_for_fitting

logger = logging.getLogger("nipype.interface")


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
) -> None:
    """Initialize per-worker model and fitter objects (called once per process)."""
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

        # HARD constraint: use fit_constrained with fixed vp
        if (
            voxel_vp is not None
            and constraint_mode == ConstraintMode.HARD
            and hasattr(model, "fit_constrained")
        ):
            result = model.fit_constrained(
                prepared_curve, aif, time_points, fixed_vp=voxel_vp
            )
            constrained = True

        # SOFT constraint: Bayesian with informative vp prior
        elif (
            voxel_vp is not None
            and constraint_mode == ConstraintMode.SOFT
            and bayesian != "none"
        ):
            vp_prior = create_vp_prior_from_dsc(
                voxel_vp, relative_uncertainty=constraint_vp_uncertainty
            )
            soft_fitter = BayesianPKFitter(
                model, vp_prior_override=vp_prior
            )
            if bayesian == "map":
                result = soft_fitter.fit_map(
                    prepared_curve, aif, time_points,
                    compute_uncertainties=False,
                )
            else:
                result = soft_fitter.fit_mcmc(
                    prepared_curve, aif, time_points, n_samples=mcmc_samples
                )
            constrained = True

        # SOFT constraint with NLLS: fall back to hard constraint
        elif (
            voxel_vp is not None
            and constraint_mode == ConstraintMode.SOFT
            and bayesian == "none"
            and hasattr(model, "fit_constrained")
        ):
            result = model.fit_constrained(
                prepared_curve, aif, time_points, fixed_vp=voxel_vp
            )
            constrained = True

        # Unconstrained fallback
        if result is None:
            if bayesian == "none":
                result = model.fit(prepared_curve, aif, time_points)
            elif bayesian == "map" and fitter is not None:
                result = fitter.fit_map(
                    prepared_curve, aif, time_points,
                    compute_uncertainties=False,
                )
            elif bayesian == "mcmc" and fitter is not None:
                result = fitter.fit_mcmc(
                    prepared_curve, aif, time_points,
                    n_samples=mcmc_samples,
                )
            else:
                result = model.fit(prepared_curve, aif, time_points)

        # Build output dict
        has_vp = model_name in ("extended_tofts", "patlak", "2cxm")
        uncertainties = {}
        if result.has_uncertainty():
            uncertainties["Ktrans"] = result.get_uncertainty("Ktrans", 0.0)
            uncertainties["ve"] = result.get_uncertainty("ve", 0.0)
            if has_vp:
                uncertainties["vp"] = result.get_uncertainty("vp", 0.0)

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
            "failed": False,
            "failure_reason": None,
            "constrained": constrained,
        }

    except Exception:
        return {
            "coords": (x, y, z),
            "failed": True,
            "failure_reason": "exception",
        }


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
    t1_map_file = File(
        exists=True,
        desc="Optional T1 map in milliseconds (for model initialization)",
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
        "map",
        "mcmc",
        usedefault=True,
        desc="Bayesian inference method: none (NLLS), map, or mcmc",
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
    dsc_cbv_file = File(
        desc="DSC-derived CBV map for vp constraint (optional)",
    )
    dsc_aif_qc = traits.Dict(
        desc="DSC AIF QC metrics dict for CBV quality assessment (optional)",
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
        0.40,
        usedefault=True,
        desc="Blood hematocrit for CBV-to-vp conversion (default 0.40)",
    )

    # Processing options
    n_procs = traits.Int(
        1,
        usedefault=True,
        desc="Number of parallel processes for voxel fitting (1 = serial)",
    )
    use_gpu = traits.Bool(
        False,
        usedefault=True,
        desc="Use GPU acceleration when available",
    )
    save_uncertainty = traits.Bool(
        False,
        usedefault=True,
        desc="Save uncertainty maps from Bayesian inference",
    )
    save_fitted_curves = traits.Bool(
        False,
        usedefault=True,
        desc="Save fitted concentration curves (large output)",
    )

    # AIF detection method (for BAT correction logic)
    aif_detection_method = traits.Str(
        "population",
        usedefault=True,
        desc="AIF detection method: 'population', 'automatic', 'atlas'. "
             "BAT correction is applied only for population AIFs.",
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
    out_ve = File(exists=True, desc="ve map (extravascular extracellular volume fraction)")
    out_vp = File(desc="vp map (plasma volume fraction, if applicable)")
    out_kep = File(exists=True, desc="kep map (efflux rate constant, min^-1)")

    # 2CXM-specific outputs
    out_fp = File(desc="Fp map (plasma flow rate, mL/min/100mL, 2cxm only)")
    out_ps = File(desc="PS map (permeability-surface area product, mL/min/100mL, 2cxm only)")
    out_e = File(desc="E map (extraction fraction, 2cxm only)")

    # Uncertainty maps (Bayesian only)
    out_ktrans_std = File(desc="Ktrans standard deviation map (Bayesian)")
    out_ve_std = File(desc="ve standard deviation map (Bayesian)")
    out_vp_std = File(desc="vp standard deviation map (Bayesian)")

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

    # QC outputs
    out_quality_flag = File(desc="Quality flag map (0=unfitted, 1=good, 2=poor_convergence, 3=high_uncertainty)")
    out_r2 = File(desc="R² (goodness of fit) map")
    out_tsnr = File(desc="tSNR map (always computed)")
    out_qc_mask = File(desc="QC mask (1=good fit, 0=unfitted/poor)")
    out_fitted_curves = File(desc="Fitted concentration curves (if save_fitted_curves=True)")
    qc_metrics = traits.Dict(desc="QC metrics dictionary")
    constraint_qc = traits.Dict(desc="DSC CBV constraint QC report")

    # PK QC plots
    pk_modeling_qc_plot = File(desc="PK modeling QC plot PNG path")
    pk_param_maps_qc_plot = File(desc="PK parameter maps QC plot PNG path")


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
    t1_map_file : str, optional
        T1 map for model initialization (not currently used).
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
    use_gpu : bool
        Use GPU acceleration. Default: False.
    save_uncertainty : bool
        Save uncertainty maps (Bayesian only). Default: False.
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

    For Bayesian methods (map, mcmc), uncertainty maps are also generated
    when save_uncertainty=True.

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

        Args:
            model_name: Model name.

        Returns:
            True if model outputs vp.
        """
        return model_name in ("extended_tofts", "patlak", "2cxm")

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
        if model_name == "auto":
            model_name = self._select_model(selection_tr_seconds)

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
            del signal_data
        else:
            logger.warning(
                "dce_signal_file not provided; computing tSNR from concentration "
                "data (values will be low — consider wiring dce_preprocess.out_file)"
            )
            tsnr_map = self._compute_tsnr(conc_data, mask, baseline_frames)

        # Create fitting mask (brain mask + tSNR threshold if specified)
        mask_bool = mask > 0
        if tsnr_threshold > 0:
            fitting_mask = mask_bool & (tsnr_map >= tsnr_threshold)
        else:
            fitting_mask = mask_bool

        # Raw signal variation filter (perf-plus: std(S_voxel) >= 1.0)
        if signal_std_map is not None:
            fitting_mask = fitting_mask & (signal_std_map >= 1.0)

        # Pre-fit enhancement screen: skip voxels with no meaningful
        # contrast enhancement (these produce R2 < 0 fits on noise)
        peak_enhancement = np.max(conc_data, axis=-1)
        baseline_std = np.std(conc_data[..., :baseline_frames], axis=-1)
        enhancement_mask = peak_enhancement > 3.0 * np.maximum(baseline_std, 1e-10)
        fitting_mask = fitting_mask & enhancement_mask

        # Parameters
        bayesian = str(self.inputs.bayesian)
        mcmc_samples = int(self.inputs.mcmc_samples)
        use_gpu = bool(self.inputs.use_gpu)
        save_uncertainty = bool(self.inputs.save_uncertainty)
        save_fitted_curves = bool(self.inputs.save_fitted_curves)

        # =====================================================================
        # BAT Correction for Population AIFs
        # =====================================================================
        aif_detection_method = str(self.inputs.aif_detection_method)
        bat_correction: dict[str, Any] = {
            "applied": False,
            "reason": "not_population_aif",
        }

        if aif_detection_method == "population":
            from pmriprep.utils.dce_bat_correction import (
                apply_aif_time_shift,
                detect_aif_bat,
                detect_tissue_bat,
            )

            # Use raw DCE signal for tissue BAT detection (matches perf-plus)
            if isdefined(self.inputs.dce_signal_file):
                bat_signal_img = nib.load(self.inputs.dce_signal_file)
                bat_signal_data = bat_signal_img.get_fdata().astype(np.float64)  # type: ignore[attr-defined]
            else:
                # Fallback to concentration data
                bat_signal_data = conc_data

            # Time vector in seconds for BAT detection
            time_vector_s = time_points * 60.0  # minutes -> seconds

            tissue_bat = detect_tissue_bat(
                bat_signal_data, mask, time_vector_s, baseline_frames
            )
            aif_bat = detect_aif_bat(aif, time_vector_s)

            time_shift_s = tissue_bat["tissue_bat_s"] - aif_bat["aif_bat_s"]

            logger.info(
                "BAT correction: tissue BAT=%.1fs (frame %d), AIF BAT=%.1fs "
                "(frame %d), shift=%.1fs",
                tissue_bat["tissue_bat_s"],
                tissue_bat["tissue_bat_frame"],
                aif_bat["aif_bat_s"],
                aif_bat["aif_bat_frame"],
                time_shift_s,
            )

            if abs(time_shift_s) > 2.0:
                aif = apply_aif_time_shift(aif, time_vector_s, time_shift_s)
                new_peak_frame = int(np.argmax(aif))
                new_peak_s = float(time_vector_s[new_peak_frame])
                logger.info(
                    "BAT correction applied: shifted AIF by +%.1fs "
                    "(new peak at %.1fs, was %.1fs)",
                    time_shift_s,
                    new_peak_s,
                    aif_bat["aif_peak_s"],
                )
                if abs(time_shift_s) > 60:
                    logger.warning(
                        "Large BAT shift (%.0fs > 60s) — check baseline "
                        "frames or injection timing",
                        time_shift_s,
                    )
                bat_correction = {
                    "applied": True,
                    "time_shift_s": time_shift_s,
                    "tissue_bat_s": tissue_bat["tissue_bat_s"],
                    "tissue_bat_frame": tissue_bat["tissue_bat_frame"],
                    "aif_bat_s": aif_bat["aif_bat_s"],
                    "aif_peak_s": aif_bat["aif_peak_s"],
                    "corrected_aif_peak_s": new_peak_s,
                    "detection_confidence": tissue_bat["confidence"],
                    "detection_method": tissue_bat["method"],
                }
            else:
                logger.info("BAT shift < 2s, no correction needed")
                bat_correction = {
                    "applied": False,
                    "time_shift_s": time_shift_s,
                    "reason": "shift_too_small",
                }

            # Clean up temporary signal data
            if isdefined(self.inputs.dce_signal_file):
                del bat_signal_data

        spatial_shape = conc_data.shape[:3]

        # =====================================================================
        # DSC CBV vp Constraint
        # =====================================================================
        vp_constraint_map: Optional[npt.NDArray[Any]] = None
        constraint_mode = ConstraintMode.SKIP
        constraint_vp_uncertainty: float = 0.2
        constraint_qc: dict[str, Any] = {"constraint_mode": "skip", "reason": "No DSC CBV provided"}

        has_dsc_cbv = (
            isdefined(self.inputs.dsc_cbv_file)
            and model_name == "extended_tofts"
        )

        if has_dsc_cbv:
            try:
                # Get AIF QC dict if available
                aif_qc_data = (
                    self.inputs.dsc_aif_qc
                    if isdefined(self.inputs.dsc_aif_qc)
                    else {}
                )
                vp_result = prepare_vp_constraint_from_data(
                    cbv_file=self.inputs.dsc_cbv_file,
                    aif_qc_data=aif_qc_data,
                    brain_mask_file=self.inputs.brain_mask_file,
                    mode_override=str(self.inputs.constraint_mode),
                    hematocrit=float(self.inputs.hematocrit),
                )
                vp_constraint_map = vp_result.vp_map
                constraint_mode = vp_result.config.mode
                constraint_vp_uncertainty = vp_result.config.vp_uncertainty
                constraint_qc = vp_result.qc_report
            except Exception as exc:
                constraint_qc = {
                    "constraint_mode": "skip",
                    "reason": f"Failed to load DSC CBV constraint: {exc}",
                }

        # =====================================================================
        # Compute model-free maps (vectorized, fast, before PK fitting loop)
        # =====================================================================
        from pmriprep.utils.dce_model_free import (
            compute_iauc,
            compute_peak_enhancement,
            compute_signal_enhancement_ratio,
            compute_time_to_peak,
        )

        iauc60 = compute_iauc(conc_data, time_points, cutoff_s=60.0)
        iauc90 = compute_iauc(conc_data, time_points, cutoff_s=90.0)
        iauc120 = compute_iauc(conc_data, time_points, cutoff_s=120.0)
        peak_enh = compute_peak_enhancement(conc_data, baseline_frames)
        ttp = compute_time_to_peak(conc_data, time_points)
        ser = compute_signal_enhancement_ratio(conc_data, time_points)

        # =====================================================================
        # Initialize output arrays
        # =====================================================================
        ktrans = np.zeros(spatial_shape, dtype=np.float64)
        ve = np.zeros(spatial_shape, dtype=np.float64)
        vp = np.zeros(spatial_shape, dtype=np.float64)
        kep = np.zeros(spatial_shape, dtype=np.float64)
        r2_map = np.zeros(spatial_shape, dtype=np.float64)

        # 2CXM-specific
        fp_map = np.zeros(spatial_shape, dtype=np.float64)
        ps_map = np.zeros(spatial_shape, dtype=np.float64)
        e_map = np.zeros(spatial_shape, dtype=np.float64)

        # Quality flag map: 0=unfitted, 1=good, 2=poor_convergence, 3=high_uncertainty
        quality_flag_map = np.zeros(spatial_shape, dtype=np.int8)
        _QUALITY_FLAG_CODES = {"good": 1, "poor_convergence": 2, "high_uncertainty": 3}

        # Uncertainty maps
        ktrans_std = np.zeros(spatial_shape, dtype=np.float64)
        ve_std = np.zeros(spatial_shape, dtype=np.float64)
        vp_std = np.zeros(spatial_shape, dtype=np.float64)

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
            "model": model_name,
            "bayesian": bayesian,
            "n_voxels_brain": n_brain,
            "n_voxels_fitted": n_voxels,
            "n_voxels_signal_filtered": n_brain - n_voxels,
            "tsnr_threshold": tsnr_threshold,
            "baseline_frames": baseline_frames,
            "selection_tr_seconds": selection_tr_seconds,  # TR used for model selection
            "constraint_mode": constraint_mode.value,
            "bat_correction": bat_correction,
        }

        n_converged = 0
        n_failed = 0
        n_constrained = 0
        n_unconstrained_fallback = 0
        n_rejected_postfit = 0
        r2_list: list[float] = []
        failure_reason_counts: dict[str, int] = {}

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
        init_args = (
            aif, time_points, model_name, bayesian, mcmc_samples,
            baseline_frames, constraint_mode.value, constraint_vp_uncertainty,
            save_fitted_curves,
        )

        n_procs = max(1, int(self.inputs.n_procs))

        if use_gpu:
            # GPU path: future batched_tofts_model_gpu integration (unchanged)
            _init_worker(*init_args)
            results = [_fit_single_voxel(vd) for vd in voxel_data_list]
        elif n_procs > 1:
            from concurrent.futures import ProcessPoolExecutor

            chunk_size = max(1, n_voxels // (n_procs * 10))
            with ProcessPoolExecutor(
                max_workers=n_procs,
                initializer=_init_worker,
                initargs=init_args,
            ) as executor:
                results = list(executor.map(
                    _fit_single_voxel, voxel_data_list, chunksize=chunk_size,
                ))
        else:
            # Serial path (n_procs=1): avoid multiprocessing overhead
            _init_worker(*init_args)
            results = [_fit_single_voxel(vd) for vd in voxel_data_list]

        # Clear worker state to avoid stale references across runs
        _worker_ctx.clear()

        # Scatter results into output arrays
        has_vp = self._has_vp(model_name)
        for vd, res in zip(voxel_data_list, results):
            if res["failed"]:
                n_failed += 1
                reason = res.get("failure_reason", "unknown")
                failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1
                continue

            x, y, z = res["coords"]
            params = res["params"]

            # Post-fit validation: reject degenerate fits (matching perf-plus)
            # Voxels where parameters hit exact optimizer bounds or R² < 0
            # are unreliable and should be zeroed.
            ktrans_val = params.get("Ktrans", 0.0)
            ve_val = params.get("ve", 0.0)
            r2_val = res["r_squared"]
            degenerate = (
                ktrans_val <= 0 or ktrans_val >= KTRANS_BOUNDS.max
                or ve_val <= 0 or ve_val >= VE_BOUNDS.max
                or r2_val < 0
            )
            if degenerate:
                ktrans[x, y, z] = 0.0
                ve[x, y, z] = 0.0
                kep[x, y, z] = 0.0
                r2_map[x, y, z] = 0.0
                if has_vp:
                    vp[x, y, z] = 0.0
                n_rejected_postfit += 1
                continue

            ktrans[x, y, z] = ktrans_val
            ve[x, y, z] = ve_val
            kep[x, y, z] = params.get("kep", 0.0)
            r2_map[x, y, z] = r2_val

            if has_vp:
                vp[x, y, z] = params.get("vp", 0.0)

            # 2CXM-specific parameters
            if model_name == "2cxm":
                fp_map[x, y, z] = params.get("Fp", 0.0)
                ps_map[x, y, z] = params.get("PS", 0.0)
                e_map[x, y, z] = params.get("E", 0.0)

            # Uncertainty (Bayesian only)
            if save_uncertainty and res["uncertainties"]:
                ktrans_std[x, y, z] = res["uncertainties"].get("Ktrans", 0.0)
                ve_std[x, y, z] = res["uncertainties"].get("ve", 0.0)
                if has_vp:
                    vp_std[x, y, z] = res["uncertainties"].get("vp", 0.0)

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

            if res["converged"]:
                n_converged += 1
                r2_list.append(res["r_squared"])
            else:
                n_failed += 1

        # Compile QC metrics
        qc_metrics["n_converged"] = n_converged
        qc_metrics["n_failed"] = n_failed
        qc_metrics["convergence_rate"] = n_converged / max(1, n_voxels)
        qc_metrics["n_constrained"] = n_constrained
        qc_metrics["n_unconstrained_fallback"] = n_unconstrained_fallback
        qc_metrics["n_rejected_postfit"] = n_rejected_postfit
        qc_metrics["failure_reasons"] = failure_reason_counts

        # Quality flag breakdown
        qf_vals = quality_flag_map[fitting_mask]
        qc_metrics["quality_n_good"] = int(np.sum(qf_vals == 1))
        qc_metrics["quality_n_poor_convergence"] = int(np.sum(qf_vals == 2))
        qc_metrics["quality_n_high_uncertainty"] = int(np.sum(qf_vals == 3))

        if r2_list:
            qc_metrics["r2_mean"] = float(np.mean(r2_list))
            qc_metrics["r2_std"] = float(np.std(r2_list))
            qc_metrics["r2_median"] = float(np.median(r2_list))

        # Compute summary statistics on fitted voxels
        fitted_mask = fitting_mask & (ktrans > 0)
        if np.any(fitted_mask):
            qc_metrics["ktrans_mean"] = float(np.mean(ktrans[fitted_mask]))
            qc_metrics["ktrans_median"] = float(np.median(ktrans[fitted_mask]))
            qc_metrics["ve_mean"] = float(np.mean(ve[fitted_mask]))
            qc_metrics["ve_median"] = float(np.median(ve[fitted_mask]))

            if self._has_vp(model_name):
                qc_metrics["vp_mean"] = float(np.mean(vp[fitted_mask]))
                qc_metrics["vp_median"] = float(np.median(vp[fitted_mask]))

            # Histogram statistics for fitted parameters
            ktrans_vals = ktrans[fitted_mask]
            hist_counts, hist_edges = np.histogram(ktrans_vals, bins=50)
            qc_metrics["ktrans_histogram"] = hist_counts.tolist()
            qc_metrics["ktrans_histogram_edges"] = hist_edges.tolist()

            ve_vals = ve[fitted_mask]
            hist_counts_ve, hist_edges_ve = np.histogram(ve_vals, bins=50)
            qc_metrics["ve_histogram"] = hist_counts_ve.tolist()
            qc_metrics["ve_histogram_edges"] = hist_edges_ve.tolist()

        # Generate output filenames
        out_ktrans = self._gen_outfilename("_Ktrans")
        out_ve = self._gen_outfilename("_ve")
        out_kep = self._gen_outfilename("_kep")
        out_r2 = self._gen_outfilename("_R2")

        # Save main outputs
        ktrans_img = nib.Nifti1Image(ktrans.astype(np.float32), affine, header)  # type: ignore[attr-defined]
        nib.save(ktrans_img, out_ktrans)  # type: ignore[attr-defined]

        ve_img = nib.Nifti1Image(ve.astype(np.float32), affine, header)  # type: ignore[attr-defined]
        nib.save(ve_img, out_ve)  # type: ignore[attr-defined]

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

        for arr, path in [
            (iauc60, out_iauc60_path),
            (iauc90, out_iauc90_path),
            (iauc120, out_iauc120_path),
            (peak_enh, out_peak_enh_path),
            (ttp, out_ttp_path),
            (ser, out_ser_path),
        ]:
            img = nib.Nifti1Image(arr.astype(np.float32), affine, header)  # type: ignore[attr-defined]
            nib.save(img, path)  # type: ignore[attr-defined]

        # Initialize results
        self._results: dict[str, Any] = {
            "out_ktrans": out_ktrans,
            "out_ve": out_ve,
            "out_kep": out_kep,
            "out_r2": out_r2,
            "out_quality_flag": out_quality_flag,
            "out_iauc60": out_iauc60_path,
            "out_iauc90": out_iauc90_path,
            "out_iauc120": out_iauc120_path,
            "out_peak_enhancement": out_peak_enh_path,
            "out_time_to_peak": out_ttp_path,
            "out_ser": out_ser_path,
            "qc_metrics": qc_metrics,
            "constraint_qc": constraint_qc,
        }

        # Save vp if applicable
        if self._has_vp(model_name):
            out_vp = self._gen_outfilename("_vp")
            vp_img = nib.Nifti1Image(vp.astype(np.float32), affine, header)  # type: ignore[attr-defined]
            nib.save(vp_img, out_vp)  # type: ignore[attr-defined]
            self._results["out_vp"] = out_vp

        # Save 2CXM-specific outputs
        if model_name == "2cxm":
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
        if save_uncertainty and bayesian != "none":
            out_ktrans_std = self._gen_outfilename("_Ktrans_std")
            out_ve_std = self._gen_outfilename("_ve_std")

            ktrans_std_img = nib.Nifti1Image(ktrans_std.astype(np.float32), affine, header)  # type: ignore[attr-defined]
            nib.save(ktrans_std_img, out_ktrans_std)  # type: ignore[attr-defined]

            ve_std_img = nib.Nifti1Image(ve_std.astype(np.float32), affine, header)  # type: ignore[attr-defined]
            nib.save(ve_std_img, out_ve_std)  # type: ignore[attr-defined]

            self._results["out_ktrans_std"] = out_ktrans_std
            self._results["out_ve_std"] = out_ve_std

            if self._has_vp(model_name):
                out_vp_std = self._gen_outfilename("_vp_std")
                vp_std_img = nib.Nifti1Image(vp_std.astype(np.float32), affine, header)  # type: ignore[attr-defined]
                nib.save(vp_std_img, out_vp_std)  # type: ignore[attr-defined]
                self._results["out_vp_std"] = out_vp_std

            # Compute and save Coefficient of Variation (CV = std / mean)
            # Use safe division to avoid divide-by-zero
            with np.errstate(divide="ignore", invalid="ignore"):
                ktrans_cv = np.where(
                    ktrans > EPSILON_VE, ktrans_std / ktrans, 0.0
                ).astype(np.float32)
                ve_cv = np.where(
                    ve > EPSILON_VE, ve_std / ve, 0.0
                ).astype(np.float32)

            out_ktrans_cv = self._gen_outfilename("_Ktrans_cv")
            out_ve_cv = self._gen_outfilename("_ve_cv")

            nib.save(nib.Nifti1Image(ktrans_cv, affine, header), out_ktrans_cv)  # type: ignore[attr-defined]
            nib.save(nib.Nifti1Image(ve_cv, affine, header), out_ve_cv)  # type: ignore[attr-defined]

            self._results["out_ktrans_cv"] = out_ktrans_cv
            self._results["out_ve_cv"] = out_ve_cv

            if self._has_vp(model_name):
                with np.errstate(divide="ignore", invalid="ignore"):
                    vp_cv = np.where(
                        vp > EPSILON_VE, vp_std / vp, 0.0
                    ).astype(np.float32)
                out_vp_cv = self._gen_outfilename("_vp_cv")
                nib.save(nib.Nifti1Image(vp_cv, affine, header), out_vp_cv)  # type: ignore[attr-defined]
                self._results["out_vp_cv"] = out_vp_cv

            # Compute and save 95% Credible Intervals (mean ± 1.96 * std)
            # Assuming approximately normal posterior (valid for MAP/MLE)
            ktrans_ci95_lower = np.maximum(ktrans - 1.96 * ktrans_std, 0.0).astype(np.float32)
            ktrans_ci95_upper = (ktrans + 1.96 * ktrans_std).astype(np.float32)
            ve_ci95_lower = np.clip(ve - 1.96 * ve_std, 0.0, 1.0).astype(np.float32)
            ve_ci95_upper = np.clip(ve + 1.96 * ve_std, 0.0, 1.0).astype(np.float32)

            out_ktrans_ci95_lower = self._gen_outfilename("_Ktrans_ci95lower")
            out_ktrans_ci95_upper = self._gen_outfilename("_Ktrans_ci95upper")
            out_ve_ci95_lower = self._gen_outfilename("_ve_ci95lower")
            out_ve_ci95_upper = self._gen_outfilename("_ve_ci95upper")

            nib.save(nib.Nifti1Image(ktrans_ci95_lower, affine, header), out_ktrans_ci95_lower)  # type: ignore[attr-defined]
            nib.save(nib.Nifti1Image(ktrans_ci95_upper, affine, header), out_ktrans_ci95_upper)  # type: ignore[attr-defined]
            nib.save(nib.Nifti1Image(ve_ci95_lower, affine, header), out_ve_ci95_lower)  # type: ignore[attr-defined]
            nib.save(nib.Nifti1Image(ve_ci95_upper, affine, header), out_ve_ci95_upper)  # type: ignore[attr-defined]

            self._results["out_ktrans_ci95_lower"] = out_ktrans_ci95_lower
            self._results["out_ktrans_ci95_upper"] = out_ktrans_ci95_upper
            self._results["out_ve_ci95_lower"] = out_ve_ci95_lower
            self._results["out_ve_ci95_upper"] = out_ve_ci95_upper

            if self._has_vp(model_name):
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
        qc_mask &= (peak_enh >= 10)  # Visible enhancement
        qc_mask &= (ktrans > 0) & (ktrans < 5.0)  # Physiological Ktrans
        qc_mask &= (r2_map >= 0.5)  # Reasonable model fit

        # Model-specific constraints
        if model_name in ["extended_tofts", "2cxm"]:
            qc_mask &= ((ve + vp) < 0.95)  # Volume fraction constraint
        qc_mask &= (kep > 0.001)  # Non-zero efflux rate

        qc_mask = qc_mask.astype(np.uint8)
        out_qc_mask = self._gen_outfilename("_qc_mask")
        qc_mask_img = nib.Nifti1Image(qc_mask, affine, header)  # type: ignore[attr-defined]
        nib.save(qc_mask_img, out_qc_mask)  # type: ignore[attr-defined]
        self._results["out_qc_mask"] = out_qc_mask

        # Save fitted curves
        if save_fitted_curves and fitted_curves is not None:
            out_fitted = self._gen_outfilename("_fitted_curves")
            fitted_img = nib.Nifti1Image(fitted_curves, affine)  # type: ignore[attr-defined]
            nib.save(fitted_img, out_fitted)  # type: ignore[attr-defined]
            self._results["out_fitted_curves"] = out_fitted

        # Generate PK QC plots
        try:
            from pmriprep.utils.dce_pk_qc_plots import (
                generate_pk_modeling_qc_plot,
                generate_pk_parameter_maps_qc_plot,
            )

            # Reload signal data for QC voxel (if dce_signal_file provided)
            signal_for_qc: Optional[npt.NDArray[Any]] = None
            if isdefined(self.inputs.dce_signal_file):
                signal_for_qc = nib.load(self.inputs.dce_signal_file).get_fdata()  # type: ignore[attr-defined]

            pk_modeling_qc_path = generate_pk_modeling_qc_plot(
                conc_data=conc_data,
                signal_data=signal_for_qc,
                aif=aif,
                time_points_s=time_points * 60.0,  # min -> sec
                ktrans=ktrans,
                ve=ve,
                vp=vp if self._has_vp(model_name) else None,
                r2_map=r2_map,
                brain_mask=mask_bool.astype(np.float64),
                fitted_curves=fitted_curves,
                model_name=model_name,
                baseline_frames=baseline_frames,
                qc_mask=qc_mask,
            )
            if pk_modeling_qc_path:
                self._results["pk_modeling_qc_plot"] = pk_modeling_qc_path

            param_maps_dict: dict[str, npt.NDArray[Any]] = {
                "Ktrans": ktrans,
                "ve": ve,
            }
            if self._has_vp(model_name):
                param_maps_dict["vp"] = vp

            pk_param_maps_qc_path = generate_pk_parameter_maps_qc_plot(
                param_maps=param_maps_dict,
                brain_mask=mask_bool.astype(np.float64),
            )
            if pk_param_maps_qc_path:
                self._results["pk_param_maps_qc_plot"] = pk_param_maps_qc_path

        except Exception as e:
            logger.warning("Failed to generate PK QC plots: %s", e, exc_info=True)

        return runtime

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List interface outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]

        if hasattr(self, "_results"):
            outputs.update(self._results)

        return outputs
