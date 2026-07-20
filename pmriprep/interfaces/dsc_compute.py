"""DSC map computation interface for perfusion quantification.

This module implements the main Nipype interface for computing DSC
perfusion parameter maps (CBF, CBV, MTT, Tmax, PSR, PH) from
concentration curves and AIF using various deconvolution methods.
"""

import json
import logging
from pathlib import Path
from typing import Any, NamedTuple, Optional, cast

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

from pmriprep.utils.dsc_constants import (
    EPSILON_CONCENTRATION,
    EPSILON_SHAPE,
    EPSILON_SIGNAL,
    DEFAULT_SVD_THRESHOLD,
    DEFAULT_OI_THRESHOLD,
)
from pmriprep.utils.dsc_signal import compute_psr_legacy, compute_peak_height_legacy, compute_ttp_legacy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level worker for parallel parametric deconvolution
# ---------------------------------------------------------------------------
class _DSCVoxelInput(NamedTuple):
    """Lightweight struct for a single voxel's data."""

    x: int
    y: int
    z: int
    tissue_curve: Any  # npt.NDArray


_dsc_worker_ctx: dict[str, Any] = {}


def _compute_dsc_motion_weights(
    fd_values: list[float],
    bolus_frame: int,
    n_frames: int,
    min_voxel_dim: float,
    tr: float,
) -> npt.NDArray[np.floating]:
    """Compute per-frame motion weights for DSC parametric fitting.

    Same weight function as DCE: w_i = (threshold / FD_i)^2 for FD > threshold.
    DSC-specific: critical window = entire first-pass (~15s), not just bolus ± 4.

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
    tr : float
        Repetition time (seconds) for first-pass duration calculation.

    Returns
    -------
    npt.NDArray[np.floating]
        Per-frame weight array (1.0 = full weight, <1.0 = downweighted).

    """
    weights = np.ones(n_frames, dtype=np.float64)
    fd_arr = np.array(fd_values[:n_frames], dtype=np.float64)
    base_threshold = max(min_voxel_dim * 0.25, 0.1)  # floor at 0.1mm

    # DSC: entire first-pass is critical (typically 5-10 frames at 1.5-2s TR)
    first_pass_frames = int(np.ceil(15.0 / max(tr, 0.5)))
    first_pass_end = min(bolus_frame + first_pass_frames, n_frames - 1)

    for i in range(n_frames):
        if bolus_frame <= i <= first_pass_end:
            eff_thresh = base_threshold * 0.5  # 2x stricter during first-pass
        else:
            eff_thresh = base_threshold

        if fd_arr[i] > eff_thresh:
            weights[i] = (eff_thresh / fd_arr[i]) ** 2

    return weights


def _warn_invalid_cpi_combo(
    method: str, inference: str, cpi_infer_cpt: bool,
) -> None:
    """Emit log warnings for known-bad CPI inference + cpi_infer_cpt combos.

    DRO matrix data
    (``tests/benchmarks/run_cpi_infer_cpt_map_vs_vb.py``) shows:

    - inference='nlls' + cpi_infer_cpt=False produces 2-3x larger CBF bias
      than VI/MAP because no prior shrinkage compensates for the rigid CP
      layout.
    - inference='map' + cpi_infer_cpt=True is unstable: MAP point estimate
      fails to find the multimodal posterior basin and CBF variance is
      ~7x higher than VI at 5% noise.

    Parameters
    ----------
    method : str
        DSC deconvolution method name.
    inference : str
        Parametric inference engine ('nlls', 'map', 'vi', 'svb').
    cpi_infer_cpt : bool
        Whether CPI control point times are inferred.
    """
    if method != "cpi":
        return
    if inference == "nlls" and not cpi_infer_cpt:
        logger.warning(
            "DSC CPI: inference='nlls' with cpi_infer_cpt=False produces "
            "2-3x larger CBF bias than VI/MAP on the gamma-variate DRO "
            "(no prior shrinkage compensates for the rigid CP layout). "
            "Recommend cpi_infer_cpt=True or inference='vi'."
        )
    if inference == "map" and cpi_infer_cpt:
        logger.warning(
            "DSC CPI: inference='map' with cpi_infer_cpt=True is unstable "
            "(MAP point estimate fails to find the multimodal posterior "
            "basin; CBF variance ~7x higher than VI at 5% noise). "
            "Recommend inference='vi' for cpi_infer_cpt=True cases."
        )


def _dsc_init_worker(
    aif: Any,
    time_points: Any,
    method: str,
    inference: str,
    mcmc_samples: int,
    cpi_control_points: int,
    bolus_duration_frames: int,
    dt: float,
    motion_weights: Any = None,
    sloppy: bool = False,
    random_seed: Optional[int] = None,
    cpi_infer_cpt: bool = True,
    cpi_infer_delay: bool = False,
    mean_tissue: Any = None,
) -> None:
    """Initialise shared state once per worker process."""
    from pmriprep.utils.parallel import limit_worker_threads
    limit_worker_threads()

    from pmriprep.models.dsc import (
        VascularModel,
        CPIModel,
        select_n_control_points,
    )

    # Seed RNG in worker process for reproducibility — offset by PID
    # so parallel workers explore different random sequences
    if random_seed is not None:
        import os
        np.random.seed(random_seed + os.getpid() % (2**31))

    _dsc_worker_ctx["aif"] = aif
    _dsc_worker_ctx["time_points"] = time_points
    _dsc_worker_ctx["method"] = method
    _dsc_worker_ctx["inference"] = inference
    _dsc_worker_ctx["mcmc_samples"] = mcmc_samples
    _dsc_worker_ctx["dt"] = dt
    _dsc_worker_ctx["motion_weights"] = motion_weights
    _dsc_worker_ctx["sloppy"] = sloppy

    # Pre-build model template so each voxel can cheaply instantiate
    if method == "vascular":
        _dsc_worker_ctx["model_factory"] = lambda: VascularModel()
    else:  # cpi
        n_cp = cpi_control_points
        if n_cp == 0:
            n_cp = select_n_control_points(
                bolus_duration_frames,
                infer_cpt=bool(cpi_infer_cpt),
                mean_tissue=mean_tissue,
                dt=float(dt),
            )
        n_frames = int(len(time_points))
        _dsc_worker_ctx["model_factory"] = (
            lambda _ncp=n_cp, _nf=n_frames, _dt=dt, _ic=bool(cpi_infer_cpt), _id=bool(cpi_infer_delay): CPIModel(
                n_control_points=_ncp,
                n_acquisition_frames=_nf,
                dt=_dt,
                infer_cpt=_ic,
                infer_delay=_id,
            )
        )


def _dsc_fit_single_voxel(voxel_data: _DSCVoxelInput) -> dict[str, Any]:
    """Fit a single voxel — called from pool workers."""
    from pmriprep.models.dsc import fit_dsc_model

    x, y, z, tissue_curve = voxel_data

    # Skip near-zero curves
    if np.max(np.abs(tissue_curve)) < EPSILON_CONCENTRATION:
        return {"coords": (x, y, z), "failed": True}

    method = _dsc_worker_ctx["method"]
    inference = _dsc_worker_ctx["inference"]
    aif = _dsc_worker_ctx["aif"]
    time_points = _dsc_worker_ctx["time_points"]
    dt = _dsc_worker_ctx["dt"]
    motion_weights = _dsc_worker_ctx.get("motion_weights")
    sloppy = _dsc_worker_ctx.get("sloppy", False)

    try:
        model_instance = _dsc_worker_ctx["model_factory"]()

        result = fit_dsc_model(
            model=model_instance,
            concentration=tissue_curve,
            aif=aif,
            time_points=time_points,
            method=inference,
            mcmc_samples=_dsc_worker_ctx["mcmc_samples"],
            compute_uncertainties=not sloppy,
            weights=motion_weights,
            sloppy=sloppy,
        )

        out: dict[str, Any] = {"coords": (x, y, z), "failed": False}
        cbf_val = result.parameters.get("cbf", result.parameters.get("CBF", 0.0))
        out["cbf"] = cbf_val

        if method == "vascular":
            alpha = result.parameters.get("alpha", 1.0)
            beta = result.parameters.get("beta", 1.0)
            delay = result.parameters.get("delay", 0.0)

            # Compute residue for derived parameters:
            # CBV = integral of residue, MTT = CBV / CBF (central volume theorem)
            model = _dsc_worker_ctx["model_factory"]()
            residue = model.residue_function(
                [cbf_val, alpha, beta, 0.0], time_points  # delay=0 for clean residue
            )
            out["cbv"] = float(np.trapezoid(residue, dx=dt))
            out["mtt"] = out["cbv"] / cbf_val if cbf_val > EPSILON_CONCENTRATION else 0.0
            out["tmax"] = delay
            out["alpha"] = alpha
            out["beta"] = beta
            out["cth"] = beta * np.sqrt(alpha)
        else:  # cpi
            if result.fitted_residue is not None:
                res = result.fitted_residue
                cbv_val = float(np.trapezoid(res, dx=dt))
                out["cbv"] = cbv_val
                out["mtt"] = cbv_val / cbf_val if cbf_val > EPSILON_CONCENTRATION else 0.0
                # CPI reports the fitted bolus delay (FABBER inferdelay) when
                # enabled; the residue (CBV/MTT) is delay-invariant. Without
                # infer_delay the parameter is absent, so tmax is 0.
                out["tmax"] = float(result.parameters.get("delay", 0.0))
            else:
                out["cbv"] = 0.0
                out["mtt"] = 0.0
                out["tmax"] = 0.0

        # Extract uncertainty only for directly fitted CBF. Derived CBV depends
        # on correlated fitted parameters, but DSCFitResult does not retain
        # their joint covariance or derived posterior samples.
        if result.uncertainties:
            out["cbf_std"] = result.uncertainties.get("cbf", 0.0)

        # Extract credible intervals only for directly fitted CBF.
        if result.credible_intervals:
            cbf_ci = result.credible_intervals.get("cbf", (0.0, 0.0))
            out["cbf_ci95_lower"] = cbf_ci[0]
            out["cbf_ci95_upper"] = cbf_ci[1]

        out["r_squared"] = result.r_squared
        out["converged"] = result.passes_convergence()

        # Flag poor fits for oSVD fallback. R² gate applies to every
        # inference method; the `converged` flag only gates NLLS/MAP
        # because for VI/SVB the FABBER 10-iter schedule rarely satisfies
        # the strict FE-plateau convergence test even on usable fits.
        bad_r2 = result.r_squared < 0.5
        bad_solver_conv = (
            inference in ("nlls", "map") and not result.converged
        )
        if bad_r2 or bad_solver_conv:
            out["needs_fallback"] = True

        return out

    except Exception:
        return {"coords": (x, y, z), "failed": True, "needs_fallback": True}


#: Minimum reference voxels for the first-pass-CBV WM median to be trusted;
#: below this the map has no valid anchor and is not produced.
_FIRSTPASS_MIN_REFERENCE_VOXELS: int = 50
#: Fixed FWHM (mm) of the light spatial pre-fit smoothing applied before the
#: per-voxel gamma fit, to raise its effective CNR (standard DSC preprocessing).
_FIRSTPASS_PREFIT_FWHM_MM: float = 5.0


class _DSCComputeInputs(NamedTuple):
    """Validated image arrays and acquisition settings for DSC fitting."""

    concentration: npt.NDArray[Any]
    affine: npt.NDArray[Any]
    header: Any
    mask: npt.NDArray[Any]
    mask_bool: npt.NDArray[np.bool_]
    aif: npt.NDArray[Any]
    signal: Optional[npt.NDArray[Any]]
    method: str
    dt: float
    svd_threshold: float
    oi_threshold: float
    cpi_control_points: int
    cpi_infer_cpt: bool
    cpi_infer_delay: bool
    inference: str
    mcmc_samples: int
    baseline_frames: int
    tail_frames: int
    te: float


class _DSCMotionWeighting(NamedTuple):
    """Frame weights and matching QC state for parametric DSC fitting."""

    weights: Optional[npt.NDArray[np.floating]]
    qc: dict[str, Any]


class _DSCBaselinePreparation(NamedTuple):
    """Baseline-adjusted DSC state and leakage-correction provenance."""

    concentration: npt.NDArray[Any]
    aif: npt.NDArray[Any]
    recomputed_raw_concentration: Optional[npt.NDArray[Any]]
    detected_frames: int
    n_baseline: int
    detection_info: dict[str, Any]
    k2_map_path: Optional[str]


class _DSCSignalMetrics(NamedTuple):
    """Signal-derived DSC maps and their QC provenance."""

    psr: npt.NDArray[Any]
    peak_height: npt.NDArray[Any]
    qc: dict[str, Any]


class _DSCTimingMaps(NamedTuple):
    """Raw, AIF-corrected, and optional NAWM-relative DSC timing maps."""

    ttp: npt.NDArray[Any]
    ttp_corrected: npt.NDArray[Any]
    relative_ttp: Optional[npt.NDArray[Any]]
    aif_bat_seconds: float


class _DSCAUCMaps(NamedTuple):
    """DSC AUC decomposition and measured-AIF CBV state."""

    raw: npt.NDArray[Any]
    corrected: npt.NDArray[Any]
    difference: npt.NDArray[Any]
    cbv: npt.NDArray[Any]
    qc: dict[str, Any]


class _DSCQualityMaps(NamedTuple):
    """Concentration CNR and raw-signal SNR/tSNR maps."""

    cnr: npt.NDArray[Any]
    snr: npt.NDArray[Any]
    tsnr: npt.NDArray[Any]


class _DSCFinalMaps(NamedTuple):
    """Clipped, unit-scaled, and brain-masked DSC output maps."""

    rel_cbf: npt.NDArray[Any]
    rel_cbv: npt.NDArray[Any]
    rel_cbv_auc: npt.NDArray[Any]
    mtt: npt.NDArray[Any]
    tmax: npt.NDArray[Any]
    ttp: npt.NDArray[Any]
    ttp_corrected: npt.NDArray[Any]
    relative_ttp: Optional[npt.NDArray[Any]]
    auc_raw: npt.NDArray[Any]
    auc_corrected: npt.NDArray[Any]
    auc_difference: npt.NDArray[Any]
    rel_auc_raw: npt.NDArray[Any]
    rel_auc_corrected: npt.NDArray[Any]
    cnr: npt.NDArray[Any]
    snr: npt.NDArray[Any]
    tsnr: npt.NDArray[Any]
    psr: npt.NDArray[Any]
    peak_height: npt.NDArray[Any]
    delay: Optional[npt.NDArray[Any]]
    residues: Optional[npt.NDArray[Any]]
    cbf_std: Optional[npt.NDArray[Any]]
    cbf_ci_lower: Optional[npt.NDArray[Any]]
    cbf_ci_upper: Optional[npt.NDArray[Any]]


class _DSCDeconvolutionMaps(NamedTuple):
    """Common map state returned by either DSC deconvolution family."""

    cbf: npt.NDArray[Any]
    cbv: npt.NDArray[Any]
    mtt: npt.NDArray[Any]
    tmax: npt.NDArray[Any]
    qc: dict[str, Any]
    delay: Optional[npt.NDArray[Any]] = None
    residues: Optional[npt.NDArray[Any]] = None
    alpha: Optional[npt.NDArray[Any]] = None
    beta: Optional[npt.NDArray[Any]] = None
    cth: Optional[npt.NDArray[Any]] = None
    cbf_std: Optional[npt.NDArray[Any]] = None
    cbv_std: Optional[npt.NDArray[Any]] = None
    cbf_ci_lower: Optional[npt.NDArray[Any]] = None
    cbf_ci_upper: Optional[npt.NDArray[Any]] = None
    cbv_ci_lower: Optional[npt.NDArray[Any]] = None
    cbv_ci_upper: Optional[npt.NDArray[Any]] = None
    parametric_r2: Optional[npt.NDArray[Any]] = None
    fallback_mask: Optional[npt.NDArray[Any]] = None


class _DSCPostprocessing(NamedTuple):
    """Final DSC maps, source concentration, and accumulated QC state."""

    raw_concentration: npt.NDArray[Any]
    maps: _DSCFinalMaps
    qc: dict[str, Any]


class ComputeDSCMapsInputSpec(BaseInterfaceInputSpec):
    """Input specification for ComputeDSCMaps interface."""

    concentration_file = File(
        exists=True,
        mandatory=True,
        desc="4D NIfTI concentration curves (output from signal_to_concentration)",
    )
    aif_file = File(
        exists=True,
        mandatory=True,
        desc="1D text file with AIF concentration curve",
    )
    measured_aif_file = File(
        exists=True,
        mandatory=True,
        desc="Measured AIF curve (pre-gamma) for full-bolus AUC CBV denominator. "
             "Required — wiring must connect AIFDetection.out_measured_aif_curve.",
    )
    brain_mask_file = File(
        exists=True,
        mandatory=True,
        desc="3D brain mask NIfTI",
    )
    signal_file = File(
        exists=True,
        mandatory=False,
        desc="Optional 4D raw signal for PSR/PH calculation (if not using concentration)",
    )
    raw_concentration_file = File(
        exists=True,
        mandatory=False,
        desc="Pre-K2 concentration (for auc_raw when K2 correction was applied)",
    )
    te = traits.Float(
        0.035,
        usedefault=True,
        desc="Echo time in seconds (for S0 re-computation)",
    )
    aif_mask_file = File(
        exists=True,
        mandatory=False,
        desc="3D AIF mask NIfTI (for re-extracting AIF after S0 update)",
    )
    aif_detection_level = traits.Str(
        "atlas",
        usedefault=True,
        desc="AIF detection level: 'atlas_strict', 'atlas_relaxed', 'brain_*', or 'synthetic'",
    )
    preload_classification = traits.Str(
        "none",
        usedefault=True,
        desc="Effective classification: K2 adequacy ('adequate', 'partial', "
        "'inadequate', 'unknown') on the auto path, or preload-dose class "
        "('none', 'minimal', 'half_dose', 'significant', 'severe') on the user path",
    )

    # Deconvolution method
    method = traits.Enum(
        "sSVD",
        "oSVD",
        "tikhonov",
        "vascular",
        "cpi",
        usedefault=True,
        desc="Deconvolution method: sSVD, oSVD, tikhonov, vascular, or cpi",
    )

    # Method-specific parameters
    svd_threshold = traits.Float(
        DEFAULT_SVD_THRESHOLD,
        usedefault=True,
        desc="SVD truncation threshold (fraction of max singular value)",
    )
    oi_threshold = traits.Float(
        DEFAULT_OI_THRESHOLD,
        usedefault=True,
        desc="Oscillation index threshold for oSVD method",
    )
    cpi_control_points = traits.Int(
        0,
        usedefault=True,
        desc="Number of CPI control points (0 = auto-select)",
    )
    cpi_infer_cpt = traits.Bool(
        True,
        usedefault=True,
        desc="Infer control point times via log-spacing parameters "
             "(default True). Lets CPs migrate to optimal locations; cuts "
             "bias on long-MTT/low-CBF tissues by ~50%. Costs ~3x per-voxel "
             "compute. Set False to match FABBER --infer-cpt false.",
    )
    cpi_infer_delay = traits.Bool(
        False,
        usedefault=True,
        desc="Infer a per-voxel bolus arrival delay for CPI (FABBER "
             "inferdelay), off by default. Applied as a post-hoc shift of the "
             "convolved curve; the residue (CBV/MTT) is unaffected.",
    )

    # Time parameters
    tr = traits.Float(
        1.5,
        usedefault=True,
        desc="Repetition time in seconds",
    )
    baseline_frames = traits.Int(
        5,
        usedefault=True,
        desc="Pre-bolus baseline frames (wired from AIF detection)",
    )
    fallback_baseline_frames = traits.Int(
        5,
        usedefault=True,
        desc="Fallback baseline when AIF-detected value is unavailable",
    )
    tail_frames = traits.Int(
        5,
        usedefault=True,
        desc="Number of tail frames for PSR calculation",
    )

    # Inference options for parametric methods (cpi, vascular)
    inference = traits.Enum(
        "vi",
        "svb",
        "map",
        "nlls",
        usedefault=True,
        desc="Inference engine for parametric deconvolution (cpi, vascular): "
             "vi (FABBER VB, default), svb (spatial VB), map, or nlls "
             "(Levenberg-Marquardt). SVD methods ignore this option.",
    )
    mcmc_samples = traits.Int(
        500,
        usedefault=True,
        desc="Number of MCMC samples for Bayesian inference",
    )

    # Processing options
    n_procs = traits.Int(
        1,
        usedefault=True,
        desc="Number of parallel processes for voxel fitting (1 = serial)",
    )
    save_residue = traits.Bool(
        False,
        usedefault=True,
        desc="Save residue functions (large output)",
    )
    # Motion weighting inputs (from motion_correct)
    fd_values = traits.List(
        traits.Float,
        desc="Framewise displacement per volume (mm) from motion correction",
    )

    k2_map_input_file = File(
        mandatory=False,
        desc="K2 map from preload_correct (passed through when baseline not recomputed)",
    )
    enhancing_mask_file = traits.Either(
        File(exists=True),
        traits.Enum(""),
        mandatory=False,
        desc="Enhancing tumor mask in DSC space for QC voxel selection",
    )
    nawm_mask_file = traits.Str(
        desc="NAWM mask in DSC space for rTTP normalization",
    )
    wm_mask_dsc = traits.Str(
        desc="Eroded whole-WM mask in DSC space (FAST-derived) used as the "
        "first-pass-CBV normalization reference. Empty string when "
        "unavailable, in which case the first-pass map is not produced.",
    )

    # First-pass CBV — opt-in additive stroke/non-tumor output.
    first_pass_cbv = traits.Bool(
        False,
        usedefault=True,
        desc="Compute the additive gamma-variate first-pass relative CBV "
        "map. Inert: nothing internal consumes it.",
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

    # Output files — quantification-independent
    out_mtt = File(genfile=True, desc="Output MTT map NIfTI")
    out_tmax = File(genfile=True, desc="Output Tmax map NIfTI")
    out_psr = File(genfile=True, desc="Output PSR map NIfTI")
    out_ph = File(genfile=True, desc="Output peak height map NIfTI")
    out_ttp = File(genfile=True, desc="Output TTP map NIfTI")
    out_cnr = File(genfile=True, desc="Output CNR map NIfTI")

    # Output files — AUC decomposition
    out_auc_raw = File(genfile=True, desc="Output AUC raw (pre-K2) map NIfTI")
    out_auc_corr = File(genfile=True, desc="Output AUC corrected (post-K2) map NIfTI")
    out_auc_diff = File(genfile=True, desc="Output AUC diff (leakage) map NIfTI")

    # Output files — relative maps (always produced)
    out_rel_cbf = File(genfile=True, desc="Output relative CBF map NIfTI")
    out_rel_cbv = File(genfile=True, desc="Output relative CBV map NIfTI (deconvolution)")
    out_rel_cbv_auc = File(genfile=True, desc="Output AUC-based relative CBV map NIfTI")
    out_rel_cbv_firstpass = File(
        genfile=True,
        desc="Output gamma-variate first-pass relative CBV map NIfTI "
        "(optional; only when first_pass_cbv is set)",
    )
    out_rel_auc_raw = File(genfile=True, desc="Output relative AUC raw map NIfTI")
    out_rel_auc_corr = File(genfile=True, desc="Output relative AUC corrected map NIfTI")

    # Output files — quality metrics
    out_snr = File(genfile=True, desc="Output SNR (baseline) map NIfTI")
    out_tsnr = File(genfile=True, desc="Output tSNR (temporal) map NIfTI")


class ComputeDSCMapsOutputSpec(TraitedSpec):
    """Output specification for ComputeDSCMaps interface."""

    # Quantification-independent (always produced)
    out_mtt = File(exists=True, desc="MTT map (mean transit time)")
    out_tmax = File(exists=True, desc="Tmax map (time to maximum of residue)")
    out_psr = File(exists=True, desc="PSR map (percent signal recovery)")
    out_ph = File(exists=True, desc="PH map (peak height)")
    out_ttp = File(exists=True, desc="TTP map (time to peak concentration)")
    out_ttp_corrected = File(desc="AIF-corrected TTP map (BAT subtracted)")
    out_rttp = File(desc="NAWM-relative TTP map")
    out_cnr = File(exists=True, desc="CNR map (contrast-to-noise ratio)")

    # AUC decomposition (always produced)
    out_auc_raw = File(exists=True, desc="AUC raw (pre-K2) map")
    out_auc_corr = File(exists=True, desc="AUC corrected (post-K2) map")
    out_auc_diff = File(exists=True, desc="AUC diff (leakage contribution) map")

    # Relative maps (always produced)
    out_rel_cbf = File(exists=True, desc="Relative CBF map")
    out_rel_cbv = File(exists=True, desc="Relative CBV map (deconvolution)")
    out_rel_cbv_auc = File(exists=True, desc="AUC-based relative CBV map")
    out_rel_auc_raw = File(exists=True, desc="Relative AUC raw map")
    out_rel_auc_corr = File(exists=True, desc="Relative AUC corrected map")

    # Quality metrics
    out_snr = File(exists=True, desc="SNR map (baseline signal)")
    out_tsnr = File(exists=True, desc="tSNR map (temporal signal)")

    # K2 leakage map (recomputed or pass-through)
    out_k2 = File(desc="K2 leakage correction map (from recomputed or pass-through)")

    # QC fit data (all deconvolution methods)
    out_qc_fit_data = File(desc="JSON with QC voxel fit data for fit QC plot")

    # Optional extras
    out_rel_cbv_firstpass = File(
        desc="Gamma-variate first-pass relative CBV map (stroke/non-tumor "
        "convention). Empty string when first_pass_cbv is off."
    )
    out_delay = File(desc="Delay map (oSVD method)")
    out_residues = File(desc="Residue functions (if save_residue=True)")
    out_alpha = File(desc="Alpha map (vascular method only)")
    out_beta = File(desc="Beta map (vascular method only)")
    out_cth = File(desc="CTH map — capillary transit time heterogeneity (vascular only)")
    out_cbf_std = File(desc="CBF uncertainty (std) map (Bayesian only)")
    out_cbv_std = File(desc="CBV uncertainty (std) map (Bayesian only)")
    out_cbf_cv = File(desc="CBF coefficient of variation (Bayesian only)")
    out_cbv_cv = File(desc="CBV coefficient of variation (Bayesian only)")
    out_cbf_ci95_lower = File(desc="CBF 95% CI lower bound (VI/MCMC only)")
    out_cbf_ci95_upper = File(desc="CBF 95% CI upper bound (VI/MCMC only)")
    out_cbv_ci95_lower = File(desc="CBV 95% CI lower bound (VI/MCMC only)")
    out_cbv_ci95_upper = File(desc="CBV 95% CI upper bound (VI/MCMC only)")
    out_parametric_r2 = File(
        desc="Per-voxel R² of parametric fit (CPI / Vascular only). "
        "Raw value preserved on fallback voxels; see out_fallback_mask."
    )
    out_fallback_mask = File(
        desc="Binary mask: 1 where parametric fit fell back to oSVD "
        "(CPI / Vascular only)"
    )
    qc_metrics = traits.Dict(desc="QC metrics dictionary")


class ComputeDSCMaps(BaseInterface):
    """Compute DSC perfusion parameter maps.

    This interface computes cerebral perfusion parameter maps from DSC-MRI
    concentration curves and an arterial input function (AIF) using various
    deconvolution methods.

    Supported deconvolution methods:
    - sSVD: Standard SVD (Østergaard method)
    - oSVD: Oscillation-index limited SVD (adaptive regularization, delay-insensitive)
    - tikhonov: L-curve-selected first-order Tikhonov regularization
    - vascular: Gamma-variate vascular model
    - cpi: Control Point Interpolation model

    Output maps:
    - CBF: Cerebral blood flow (max of residue function)
    - CBV: Cerebral blood volume (integral of residue function)
    - MTT: Mean transit time (CBV/CBF, central volume theorem)
    - Tmax: Time to maximum of residue function
    - PSR: Percent signal recovery (BBB integrity indicator)
    - PH: Peak height (contrast effect)

    Parameters
    ----------
    concentration_file : str
        4D NIfTI with concentration time courses.
    aif_file : str
        1D text file with AIF concentration values.
    brain_mask_file : str
        3D brain mask NIfTI.
    signal_file : str, optional
        4D raw signal for PSR/PH (if not using concentration).
    method : str
        Deconvolution method. Default: 'sSVD'.
    svd_threshold : float
        SVD truncation threshold. Default: 0.2.
    oi_threshold : float
        Oscillation index target for oSVD. Default :data:`DEFAULT_OI_THRESHOLD`.
    cpi_control_points : int
        Number of CPI control points (0=auto). Default: 0.
    tr : float
        Repetition time in seconds. Default: 1.5.
    baseline_frames : int
        Pre-bolus baseline frames. Default: 5.
    tail_frames : int
        Tail frames for PSR. Default: 5.
    inference : str
        Inference engine for parametric methods (cpi/vascular):
        nlls, map, vi, svb. Default: 'vi'.
    mcmc_samples : int
        MCMC samples for Bayesian. Default: 500.
    save_residue : bool
        Save residue functions. Default: False.

    Returns
    -------
    out_rel_cbf, out_rel_cbv, out_rel_cbv_auc, out_rel_auc_raw, out_rel_auc_corr : str
        Paths to relative perfusion maps (always produced).
    out_mtt, out_tmax, out_psr, out_ph, out_ttp, out_cnr : str
        Paths to quantification-independent maps.
    out_auc_raw, out_auc_corr, out_auc_diff : str
        Paths to AUC decomposition maps.
    out_snr, out_tsnr : str
        Paths to SNR and tSNR quality metric maps.
    out_delay : str, optional
        Path to delay map (oSVD).
    out_residues : str, optional
        Path to residue functions (if save_residue=True).
    qc_metrics : dict
        QC metrics including deconvolution statistics.

    Examples
    --------
    >>> compute = ComputeDSCMaps()
    >>> compute.inputs.concentration_file = 'concentration_4d.nii.gz'
    >>> compute.inputs.aif_file = 'aif_curve.txt'
    >>> compute.inputs.brain_mask_file = 'brain_mask.nii.gz'
    >>> compute.inputs.method = 'oSVD'
    >>> result = compute.run()  # doctest: +SKIP

    Notes
    -----
    CBF/CBV/MTT are in relative units unless the input concentration was
    computed with absolute r2* values.

    For Bayesian methods (map, vi, mcmc), uncertainty maps are also
    generated when method is 'vascular' or 'cpi'.

    The oSVD method additionally outputs a delay map showing the
    estimated tracer arrival delay relative to the AIF.

    """

    input_spec = ComputeDSCMapsInputSpec
    output_spec = ComputeDSCMapsOutputSpec

    def _load_compute_inputs(self) -> _DSCComputeInputs:
        """Load and validate the DSC arrays and acquisition settings.

        Returns
        -------
        _DSCComputeInputs
            Immutable input state consumed by baseline preparation and fitting.

        Raises
        ------
        RuntimeError
            If the brain mask is not on the native DSC spatial grid.

        """
        import nibabel as nib  # type: ignore[import-untyped]

        conc_img = nib.load(self.inputs.concentration_file)
        concentration: npt.NDArray[Any] = conc_img.get_fdata()  # type: ignore[attr-defined]
        affine = conc_img.affine  # type: ignore[attr-defined]
        header = conc_img.header  # type: ignore[attr-defined]

        mask_img = nib.load(self.inputs.brain_mask_file)
        mask: npt.NDArray[Any] = mask_img.get_fdata()  # type: ignore[attr-defined]
        if mask.shape[:3] != concentration.shape[:3]:
            raise RuntimeError(
                f"Brain mask shape {mask.shape[:3]} does not match "
                f"concentration data shape {concentration.shape[:3]}. "
                "The brain mask must be in native DSC space. "
                "Ensure transform_brain_mask node is connected upstream."
            )

        signal: Optional[npt.NDArray[Any]] = None
        if self.inputs.signal_file:
            signal_img = nib.load(self.inputs.signal_file)
            signal = signal_img.get_fdata()  # type: ignore[attr-defined]

        return _DSCComputeInputs(
            concentration=concentration,
            affine=affine,
            header=header,
            mask=mask,
            mask_bool=mask > 0,
            aif=np.loadtxt(self.inputs.aif_file),
            signal=signal,
            method=str(self.inputs.method),
            dt=float(self.inputs.tr),
            svd_threshold=float(self.inputs.svd_threshold),
            oi_threshold=float(self.inputs.oi_threshold),
            cpi_control_points=int(self.inputs.cpi_control_points),
            cpi_infer_cpt=bool(self.inputs.cpi_infer_cpt),
            cpi_infer_delay=bool(self.inputs.cpi_infer_delay),
            inference=str(self.inputs.inference),
            mcmc_samples=int(self.inputs.mcmc_samples),
            baseline_frames=int(self.inputs.baseline_frames),
            tail_frames=int(self.inputs.tail_frames),
            te=float(self.inputs.te),
        )

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

    def _prepare_baseline_and_leakage(
        self, inputs: _DSCComputeInputs
    ) -> _DSCBaselinePreparation:
        """Detect the baseline and reconcile S0, AIF, concentration, and K2.

        Parameters
        ----------
        inputs : _DSCComputeInputs
            Validated DSC image arrays and acquisition settings.

        Returns
        -------
        _DSCBaselinePreparation
            Baseline-subtracted fitting arrays, any recomputed raw
            concentration, detection diagnostics, and K2 output provenance.

        """
        import nibabel as nib  # type: ignore[import-untyped]

        concentration = inputs.concentration
        aif = inputs.aif
        detected_frames, detection_info = self._detect_baseline_frames(
            aif,
            inputs.baseline_frames,
            concentration_data=concentration,
            mask=inputs.mask,
        )

        recomputed_raw: Optional[npt.NDArray[Any]] = None
        k2_result: Any = None
        if detected_frames != inputs.baseline_frames and inputs.signal is not None:
            new_s0 = np.mean(
                inputs.signal[..., :detected_frames], axis=-1, keepdims=True
            )
            new_s0 = np.maximum(new_s0, EPSILON_SIGNAL)
            signal_ratio = np.maximum(inputs.signal / new_s0, EPSILON_SIGNAL)
            recomputed_raw = -np.log(signal_ratio) / inputs.te

            if isdefined(self.inputs.aif_mask_file) and self.inputs.aif_mask_file:
                aif_mask_img = nib.load(self.inputs.aif_mask_file)
                aif_mask_data = np.asarray(aif_mask_img.dataobj) > 0.5  # type: ignore[attr-defined]
                if aif_mask_data.shape[:3] == recomputed_raw.shape[:3]:
                    roi_voxels = recomputed_raw[aif_mask_data]
                    if roi_voxels.ndim == 2 and roi_voxels.shape[0] > 0:
                        aif = np.mean(roi_voxels, axis=0)
                        if len(aif) >= 5:
                            from scipy.signal import savgol_filter  # type: ignore[import-untyped]

                            from pmriprep.utils.dsc_aif import get_aif_smoothing_window

                            window_length = get_aif_smoothing_window(len(aif), inputs.dt)
                            aif = savgol_filter(
                                aif, window_length=window_length, polyorder=2
                            )

            from pmriprep.utils.dsc_preload import apply_k2_correction

            k2_result = apply_k2_correction(
                concentration=recomputed_raw,
                brain_mask=inputs.mask.astype(bool),
                tr=inputs.dt,
                baseline_frames=detected_frames,
            )
            concentration = k2_result.concentration_corrected
            detection_info["s0_recomputed"] = True
            detection_info["s0_recomputed_frames"] = detected_frames
            detection_info["k2_recomputed"] = True

        k2_map_path: Optional[str] = None
        if detected_frames < inputs.baseline_frames and recomputed_raw is not None:
            k2_map_path = self._gen_outfilename("_k2")
            k2_img = nib.Nifti1Image(  # type: ignore[attr-defined]
                k2_result.k2_map.astype(np.float32), inputs.affine, inputs.header
            )
            nib.save(k2_img, k2_map_path)  # type: ignore[attr-defined]
        elif (
            isdefined(self.inputs.k2_map_input_file)
            and self.inputs.k2_map_input_file
            and Path(str(self.inputs.k2_map_input_file)).is_file()
        ):
            k2_map_path = str(self.inputs.k2_map_input_file)

        n_baseline = max(detected_frames, 4)
        aif_baseline_offset = float(np.mean(aif[:n_baseline]))
        aif = aif - aif_baseline_offset
        concentration_offset = np.mean(
            concentration[..., :n_baseline], axis=-1, keepdims=True
        )
        concentration = concentration - concentration_offset
        detection_info["aif_baseline_offset"] = round(aif_baseline_offset, 4)
        detection_info["conc_median_baseline_offset"] = round(
            float(np.median(concentration_offset[inputs.mask_bool])), 4
        )

        return _DSCBaselinePreparation(
            concentration=concentration,
            aif=aif,
            recomputed_raw_concentration=recomputed_raw,
            detected_frames=detected_frames,
            n_baseline=n_baseline,
            detection_info=detection_info,
            k2_map_path=k2_map_path,
        )

    def _prepare_motion_weighting(
        self,
        method: str,
        header: Any,
        n_frames: int,
        dt: float,
    ) -> _DSCMotionWeighting:
        """Prepare phase-aware motion weights for parametric DSC fitting.

        Parameters
        ----------
        method : str
            Selected DSC deconvolution method.
        header : object
            NIfTI header providing native spatial voxel dimensions.
        n_frames : int
            Number of acquired DSC frames.
        dt : float
            Frame interval in seconds.

        Returns
        -------
        _DSCMotionWeighting
            Optional fitting weights and QC values describing that exact array.

        """
        if (
            method not in ("vascular", "cpi")
            or not isdefined(self.inputs.fd_values)
            or not self.inputs.fd_values
        ):
            return _DSCMotionWeighting(None, {"applied": False})

        voxel_dims = header.get_zooms()[:3]
        min_voxel_dim = float(min(voxel_dims))
        bolus_frame = int(self.inputs.baseline_frames)
        weights = _compute_dsc_motion_weights(
            fd_values=self.inputs.fd_values,
            bolus_frame=bolus_frame,
            n_frames=n_frames,
            min_voxel_dim=min_voxel_dim,
            tr=dt,
        )
        first_pass_frames = int(np.ceil(15.0 / max(dt, 0.5)))
        first_pass_end = min(bolus_frame + first_pass_frames, n_frames - 1)
        n_downweighted = int(np.sum(weights < 0.95))
        qc = {
            "applied": True,
            "n_downweighted_frames": n_downweighted,
            "min_weight": float(np.min(weights)),
            "first_pass_weights": weights[bolus_frame:first_pass_end + 1].tolist(),
        }
        logger.info(
            "DSC motion weighting: %d/%d frames downweighted (min weight=%.3f)",
            n_downweighted,
            n_frames,
            float(np.min(weights)),
        )
        return _DSCMotionWeighting(weights, qc)

    def _compute_signal_metrics(
        self,
        signal: Optional[npt.NDArray[Any]],
        spatial_shape: tuple[int, ...],
        baseline_frames: int,
        tail_frames: int,
    ) -> _DSCSignalMetrics:
        """Compute PSR and peak height from raw DSC signal when available.

        Parameters
        ----------
        signal : npt.NDArray, optional
            Raw DSC signal array. Concentration is intentionally not used as a
            substitute because its PSR/peak-height interpretation differs.
        spatial_shape : tuple of int
            Output map shape.
        baseline_frames : int
            Detected pre-bolus frame count.
        tail_frames : int
            Number of recovery frames used for PSR.

        Returns
        -------
        _DSCSignalMetrics
            PSR and peak-height arrays plus unavailable-signal QC state.

        """
        if signal is None:
            zeros = np.zeros(spatial_shape, dtype=np.float64)
            return _DSCSignalMetrics(
                psr=zeros,
                peak_height=zeros.copy(),
                qc={"psr_ph_from_signal": False},
            )

        psr_result = compute_psr_legacy(signal, baseline_frames, tail_frames)
        peak_height_result = compute_peak_height_legacy(signal, baseline_frames)
        psr = (
            np.full(spatial_shape, psr_result, dtype=np.float64)
            if isinstance(psr_result, float)
            else psr_result
        )
        peak_height = (
            np.full(spatial_shape, peak_height_result, dtype=np.float64)
            if isinstance(peak_height_result, float)
            else peak_height_result
        )
        return _DSCSignalMetrics(psr=psr, peak_height=peak_height, qc={})

    def _select_raw_concentration(
        self,
        corrected: npt.NDArray[Any],
        recomputed: Optional[npt.NDArray[Any]],
    ) -> npt.NDArray[Any]:
        """Select pre-K2 concentration for model-free DSC measurements.

        Parameters
        ----------
        corrected : npt.NDArray
            Baseline-subtracted, potentially K2-corrected concentration.
        recomputed : npt.NDArray, optional
            Raw concentration recomputed after baseline refinement.

        Returns
        -------
        npt.NDArray
            Recomputed raw concentration, configured raw input, or corrected
            concentration, in that precedence order.

        """
        if recomputed is not None:
            return recomputed
        if self.inputs.raw_concentration_file:
            import nibabel as nib  # type: ignore[import-untyped]

            raw_img = nib.load(self.inputs.raw_concentration_file)
            return raw_img.get_fdata()  # type: ignore[attr-defined,no-any-return]
        return corrected

    def _compute_timing_maps(
        self,
        raw_concentration: npt.NDArray[Any],
        mask_bool: npt.NDArray[np.bool_],
        detected_baseline: int,
        detection_info: dict[str, Any],
        dt: float,
    ) -> _DSCTimingMaps:
        """Compute DSC timing maps while preserving unclipped derivations.

        Parameters
        ----------
        raw_concentration : npt.NDArray
            Pre-K2 concentration used for bolus timing.
        mask_bool : npt.NDArray
            Native-space brain mask.
        detected_baseline : int
            Detected pre-bolus frame count.
        detection_info : dict
            Baseline diagnostics containing the AIF arrival index when known.
        dt : float
            Frame interval in seconds.

        Returns
        -------
        _DSCTimingMaps
            Clipped raw TTP for output, AIF-corrected TTP, optional NAWM-relative
            TTP, and the AIF arrival time in seconds.

        """
        import nibabel as nib  # type: ignore[import-untyped]

        ttp_raw = compute_ttp_legacy(raw_concentration, detected_baseline, dt)
        aif_bat_frames = detection_info.get("arrival_index", detected_baseline)
        aif_bat_seconds = float(aif_bat_frames) * dt
        ttp_corrected = ttp_raw - aif_bat_seconds

        n_negative = int(np.sum(mask_bool & (ttp_corrected < 0)))
        n_mask = int(np.sum(mask_bool))
        if n_mask > 0 and n_negative / n_mask > 0.05:
            logger.warning(
                "%.1f%% of brain voxels have negative TTPc (ttp < aif_bat=%.1fs). "
                "Check AIF QC.",
                100.0 * n_negative / n_mask,
                aif_bat_seconds,
            )

        ttp_corrected = np.clip(ttp_corrected, 0.0, 30.0)
        ttp_corrected = np.where(np.isnan(ttp_raw), np.nan, ttp_corrected)

        relative_ttp: Optional[npt.NDArray[Any]] = None
        if (
            isdefined(self.inputs.nawm_mask_file)
            and str(self.inputs.nawm_mask_file) not in ("", "None")
            and Path(self.inputs.nawm_mask_file).exists()
        ):
            nawm_img = nib.load(self.inputs.nawm_mask_file)
            nawm_mask = nawm_img.get_fdata() > 0  # type: ignore[attr-defined]
            nawm_ttp = ttp_raw[nawm_mask]
            nawm_ttp = nawm_ttp[np.isfinite(nawm_ttp)]
            if len(nawm_ttp) > 100:
                relative_ttp = ttp_raw - np.median(nawm_ttp)
                relative_ttp = np.where(
                    np.isnan(ttp_raw), np.nan, relative_ttp
                )

        return _DSCTimingMaps(
            ttp=np.clip(ttp_raw, 0.0, 60.0),
            ttp_corrected=ttp_corrected,
            relative_ttp=relative_ttp,
            aif_bat_seconds=aif_bat_seconds,
        )

    def _compute_auc_maps(
        self,
        raw_concentration: npt.NDArray[Any],
        corrected_concentration: npt.NDArray[Any],
        n_baseline: int,
        detected_baseline: int,
        detection_info: dict[str, Any],
        dt: float,
    ) -> _DSCAUCMaps:
        """Compute AUC decomposition and measured-AIF full-bolus CBV.

        Parameters
        ----------
        raw_concentration : npt.NDArray
            Pre-K2 concentration, potentially carrying a residual baseline.
        corrected_concentration : npt.NDArray
            Baseline-subtracted, K2-corrected tissue concentration.
        n_baseline : int
            Baseline window used for raw tissue and measured-AIF subtraction.
        detected_baseline : int
            Detected baseline used when no explicit arrival index is available.
        detection_info : dict
            Baseline diagnostics containing the AIF arrival index when known.
        dt : float
            Frame interval in seconds.

        Returns
        -------
        _DSCAUCMaps
            Raw/corrected/leakage AUC, full-bolus CBV, and matching QC state.

        """
        raw_for_auc = raw_concentration - np.mean(
            raw_concentration[..., :n_baseline], axis=-1, keepdims=True
        )
        auc_corrected = np.minimum(
            np.trapezoid(corrected_concentration, dx=dt, axis=-1), 10000.0
        )
        auc_raw = np.minimum(np.trapezoid(raw_for_auc, dx=dt, axis=-1), 10000.0)
        auc_difference = auc_raw - auc_corrected

        spatial_shape = corrected_concentration.shape[:3]
        n_timepoints = corrected_concentration.shape[-1]
        bat = int(detection_info.get("arrival_index", detected_baseline))
        bolus_slice = slice(bat, n_timepoints)
        qc: dict[str, Any] = {}

        if (
            isdefined(self.inputs.measured_aif_file)
            and self.inputs.measured_aif_file
            and Path(str(self.inputs.measured_aif_file)).is_file()
        ):
            measured_aif = np.loadtxt(self.inputs.measured_aif_file)
            aif_for_auc = measured_aif - float(np.mean(measured_aif[:n_baseline]))
            aif_auc = float(np.trapezoid(aif_for_auc[bolus_slice], dx=dt))
            tissue_auc_bolus = np.trapezoid(
                corrected_concentration[..., bolus_slice], dx=dt, axis=-1
            )
            if aif_auc < EPSILON_SIGNAL:
                logger.warning(
                    "AIF AUC near-zero (%.6f); AUC CBV unavailable", aif_auc
                )
                cbv = np.full(spatial_shape, np.nan, dtype=np.float64)
            else:
                cbv = np.minimum(tissue_auc_bolus / aif_auc, 10.0)
            qc.update(
                cbv_auc_bat_frame=bat,
                cbv_auc_end_frame=n_timepoints,
                cbv_auc_aif_integral=round(aif_auc, 6),
            )
        else:
            logger.warning("Measured AIF not available; AUC CBV set to NaN")
            cbv = np.full(spatial_shape, np.nan, dtype=np.float64)
            qc["cbv_auc_measured_aif_missing"] = True

        _PRELOAD_INDICATORS = {"half_dose", "significant", "severe"}
        preload_class = str(self.inputs.preload_classification)
        if preload_class in _PRELOAD_INDICATORS:
            qc["cbv_auc_preload_warning"] = (
                f"Preload classification '{preload_class}': residual "
                "leakage may bias full-bolus AUC CBV low (T1-dip, BSW "
                "under-recovers) or high (positive T2* tail, BSW-uncorrectable). "
                "Interpret with caution."
            )

        return _DSCAUCMaps(
            raw=auc_raw,
            corrected=auc_corrected,
            difference=auc_difference,
            cbv=cbv,
            qc=qc,
        )

    def _compute_quality_maps(
        self,
        raw_concentration: npt.NDArray[Any],
        signal: Optional[npt.NDArray[Any]],
        detected_baseline: int,
    ) -> _DSCQualityMaps:
        """Compute concentration CNR and raw-signal SNR/tSNR.

        Parameters
        ----------
        raw_concentration : npt.NDArray
            Pre-K2 concentration used for baseline noise and peak contrast.
        signal : npt.NDArray, optional
            Raw DSC signal used for baseline SNR and whole-series tSNR.
        detected_baseline : int
            Detected pre-bolus frame count.

        Returns
        -------
        _DSCQualityMaps
            Unclipped CNR, SNR, and tSNR maps.

        """
        baseline_std = np.std(
            raw_concentration[..., :detected_baseline], axis=-1
        )
        peak_concentration = np.max(raw_concentration, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cnr = np.where(
                baseline_std > EPSILON_CONCENTRATION,
                peak_concentration / baseline_std,
                0.0,
            )

        spatial_shape = raw_concentration.shape[:3]
        if signal is None:
            zeros = np.zeros(spatial_shape, dtype=np.float64)
            return _DSCQualityMaps(cnr=cnr, snr=zeros, tsnr=zeros.copy())

        snr_baseline_frames = min(
            detected_baseline, raw_concentration.shape[-1] - 1
        )
        baseline_signal = signal[..., :snr_baseline_frames]
        baseline_mean = np.mean(baseline_signal, axis=-1)
        baseline_signal_std = np.std(baseline_signal, axis=-1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(
                baseline_signal_std > EPSILON_SIGNAL,
                baseline_mean / baseline_signal_std,
                0.0,
            )
            all_mean = np.mean(signal, axis=-1)
            all_std = np.std(signal, axis=-1, ddof=1)
            tsnr = np.where(all_std > EPSILON_SIGNAL, all_mean / all_std, 0.0)
        return _DSCQualityMaps(cnr=cnr, snr=snr, tsnr=tsnr)

    @staticmethod
    def _finalize_mtt(
        method: str,
        fitted_mtt: npt.NDArray[Any],
        cbf: npt.NDArray[Any],
        auc_cbv: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        """Apply measured-AIF central-volume MTT to SVD-family methods.

        Parameters
        ----------
        method : str
            Selected DSC deconvolution method.
        fitted_mtt : npt.NDArray
            MTT returned by the deconvolution implementation.
        cbf : npt.NDArray
            Unscaled CBF in inverse seconds.
        auc_cbv : npt.NDArray
            Measured-AIF full-bolus CBV.

        Returns
        -------
        npt.NDArray
            AUC-CBV/CBF MTT for SVD methods, otherwise the fitted map unchanged.

        """
        if method not in ("sSVD", "oSVD", "tikhonov"):
            return fitted_mtt
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                (cbf > EPSILON_CONCENTRATION) & np.isfinite(auc_cbv),
                auc_cbv / cbf,
                0.0,
            )

    @staticmethod
    def _summarize_dsc_maps(
        rel_cbf: npt.NDArray[Any],
        rel_cbv: npt.NDArray[Any],
        rel_cbv_auc: npt.NDArray[Any],
        mtt: npt.NDArray[Any],
        tmax: npt.NDArray[Any],
        mask_bool: npt.NDArray[np.bool_],
        aif_level: str,
        preload_class: str,
        baseline_info: dict[str, Any],
        aif_bat_seconds: float,
    ) -> dict[str, Any]:
        """Reduce finalized DSC maps and acquisition provenance to QC values."""
        has_voxels = bool(np.any(mask_bool))
        qc = {
            "cbf_mean": float(np.mean(rel_cbf[mask_bool])) if has_voxels else 0.0,
            "cbf_std": float(np.std(rel_cbf[mask_bool])) if has_voxels else 0.0,
            "cbv_mean": float(np.mean(rel_cbv[mask_bool])) if has_voxels else 0.0,
            "cbv_auc_mean": (
                float(np.nanmean(rel_cbv_auc[mask_bool])) if has_voxels else 0.0
            ),
            "mtt_mean": float(np.mean(mtt[mask_bool])) if has_voxels else 0.0,
            "tmax_mean": float(np.mean(tmax[mask_bool])) if has_voxels else 0.0,
            "cbf_median": (
                float(np.median(rel_cbf[mask_bool])) if has_voxels else 0.0
            ),
            "cbv_median": (
                float(np.median(rel_cbv[mask_bool])) if has_voxels else 0.0
            ),
            "cbv_auc_median": (
                float(np.nanmedian(rel_cbv_auc[mask_bool])) if has_voxels else 0.0
            ),
            "mtt_median": float(np.median(mtt[mask_bool])) if has_voxels else 0.0,
            "tmax_median": (
                float(np.median(tmax[mask_bool])) if has_voxels else 0.0
            ),
            "n_voxels": int(np.sum(mask_bool)),
            "aif_detection_level": aif_level,
            "preload_classification": preload_class,
        }
        qc.update(baseline_info)
        qc["aif_bat_seconds"] = aif_bat_seconds
        qc["cbv_literature_range_applicable"] = aif_level != "synthetic"
        return qc

    @staticmethod
    def _scale_cbf_uncertainty(
        standard_deviation: Optional[npt.NDArray[Any]],
        ci_lower: Optional[npt.NDArray[Any]],
        ci_upper: Optional[npt.NDArray[Any]],
    ) -> tuple[
        Optional[npt.NDArray[Any]],
        Optional[npt.NDArray[Any]],
        Optional[npt.NDArray[Any]],
    ]:
        """Convert optional CBF uncertainty maps from inverse seconds to minutes."""
        return tuple(
            None if value is None else value * 60.0
            for value in (standard_deviation, ci_lower, ci_upper)
        )  # type: ignore[return-value]

    @classmethod
    def _finalize_dsc_maps(
        cls,
        *,
        cbf: npt.NDArray[Any],
        cbv: npt.NDArray[Any],
        cbv_auc: npt.NDArray[Any],
        mtt: npt.NDArray[Any],
        tmax: npt.NDArray[Any],
        ttp: npt.NDArray[Any],
        ttp_corrected: npt.NDArray[Any],
        relative_ttp: Optional[npt.NDArray[Any]],
        auc_raw: npt.NDArray[Any],
        auc_corrected: npt.NDArray[Any],
        auc_difference: npt.NDArray[Any],
        cnr: npt.NDArray[Any],
        snr: npt.NDArray[Any],
        tsnr: npt.NDArray[Any],
        psr: npt.NDArray[Any],
        peak_height: npt.NDArray[Any],
        delay: Optional[npt.NDArray[Any]],
        residues: Optional[npt.NDArray[Any]],
        cbf_std: Optional[npt.NDArray[Any]],
        cbf_ci_lower: Optional[npt.NDArray[Any]],
        cbf_ci_upper: Optional[npt.NDArray[Any]],
        mask_bool: npt.NDArray[Any],
    ) -> _DSCFinalMaps:
        """Apply established output ranges, CBF units, and brain masking."""
        psr = np.clip(psr, 0.0, 200.0)
        peak_height = np.clip(peak_height, 0.0, 100.0)
        tmax = np.clip(tmax, 0.0, 60.0)
        mtt = np.clip(mtt, 0.0, 30.0)
        cbf = np.clip(cbf, 0.0, 100.0)
        cbv = np.clip(cbv, 0.0, 10.0)
        cnr = np.clip(cnr, 0.0, 1000.0)

        rel_cbf = np.clip(cbf * 60.0, 0.0, 100.0)
        rel_cbv = np.clip(cbv, 0.0, 10.0)
        cbf_std, cbf_ci_lower, cbf_ci_upper = cls._scale_cbf_uncertainty(
            cbf_std, cbf_ci_lower, cbf_ci_upper
        )

        return _DSCFinalMaps(
            rel_cbf=np.where(mask_bool, rel_cbf, 0.0),
            rel_cbv=np.where(mask_bool, rel_cbv, 0.0),
            rel_cbv_auc=np.where(mask_bool, cbv_auc, 0.0),
            mtt=np.where(mask_bool, mtt, 0.0),
            tmax=np.where(mask_bool, tmax, 0.0),
            ttp=np.where(mask_bool, ttp, 0.0),
            ttp_corrected=np.where(
                mask_bool, np.where(np.isnan(ttp_corrected), 0.0, ttp_corrected), 0.0
            ),
            relative_ttp=(
                None
                if relative_ttp is None
                else np.where(
                    mask_bool,
                    np.where(np.isnan(relative_ttp), 0.0, relative_ttp),
                    0.0,
                )
            ),
            auc_raw=np.where(mask_bool, auc_raw, 0.0),
            auc_corrected=np.where(mask_bool, auc_corrected, 0.0),
            auc_difference=np.where(mask_bool, auc_difference, 0.0),
            rel_auc_raw=np.where(mask_bool, auc_raw, 0.0),
            rel_auc_corrected=np.where(mask_bool, auc_corrected, 0.0),
            cnr=np.where(mask_bool, cnr, 0.0),
            snr=np.where(mask_bool, snr, 0.0),
            tsnr=np.where(mask_bool, tsnr, 0.0),
            psr=np.where(mask_bool, psr, 0.0),
            peak_height=np.where(mask_bool, peak_height, 0.0),
            delay=None if delay is None else np.where(mask_bool, delay, 0.0),
            residues=(
                None
                if residues is None
                else np.where(mask_bool[..., np.newaxis], residues, 0.0)
            ),
            cbf_std=cbf_std,
            cbf_ci_lower=cbf_ci_lower,
            cbf_ci_upper=cbf_ci_upper,
        )

    def _save_map(
        self,
        data: npt.NDArray[Any],
        suffix: str,
        affine: npt.NDArray[Any],
        header: Any,
        dtype: Any = np.float32,
    ) -> str:
        """Save a DSC map with an explicit dtype and clean copied header."""
        import nibabel as nib  # type: ignore[import-untyped]

        path = self._gen_outfilename(suffix)
        cast_data = data.astype(dtype)
        local_header = cast("nib.Nifti1Header", header.copy())
        local_header.set_data_dtype(dtype)
        img = nib.Nifti1Image(cast_data, affine, local_header)  # type: ignore[attr-defined]
        nib.save(img, path)  # type: ignore[attr-defined]
        return path

    def _prepare_final_outputs(
        self,
        *,
        deconvolution: _DSCDeconvolutionMaps,
        concentration: npt.NDArray[Any],
        recomputed_raw_concentration: Optional[npt.NDArray[Any]],
        signal: Optional[npt.NDArray[Any]],
        spatial_shape: tuple[int, ...],
        mask_bool: npt.NDArray[Any],
        detected_baseline: int,
        n_baseline: int,
        baseline_info: dict[str, Any],
        dt: float,
        tail_frames: int,
        method: str,
        aif_level: str,
        preload_class: str,
    ) -> _DSCPostprocessing:
        """Compose model-free maps, final masking, and DSC QC reduction."""
        qc = deconvolution.qc
        signal_metrics = self._compute_signal_metrics(
            signal, spatial_shape, detected_baseline, tail_frames
        )
        qc.update(signal_metrics.qc)

        raw_concentration = self._select_raw_concentration(
            concentration, recomputed_raw_concentration
        )
        timing = self._compute_timing_maps(
            raw_concentration,
            mask_bool,
            detected_baseline,
            baseline_info,
            dt,
        )
        auc = self._compute_auc_maps(
            raw_concentration,
            concentration,
            n_baseline,
            detected_baseline,
            baseline_info,
            dt,
        )
        qc.update(auc.qc)
        quality = self._compute_quality_maps(
            raw_concentration, signal, detected_baseline
        )
        mtt = self._finalize_mtt(
            method,
            deconvolution.mtt,
            deconvolution.cbf,
            auc.cbv,
        )
        maps = self._finalize_dsc_maps(
            cbf=deconvolution.cbf,
            cbv=deconvolution.cbv,
            cbv_auc=auc.cbv,
            mtt=mtt,
            tmax=deconvolution.tmax,
            ttp=timing.ttp,
            ttp_corrected=timing.ttp_corrected,
            relative_ttp=timing.relative_ttp,
            auc_raw=auc.raw,
            auc_corrected=auc.corrected,
            auc_difference=auc.difference,
            cnr=quality.cnr,
            snr=quality.snr,
            tsnr=quality.tsnr,
            psr=signal_metrics.psr,
            peak_height=signal_metrics.peak_height,
            delay=deconvolution.delay,
            residues=deconvolution.residues,
            cbf_std=deconvolution.cbf_std,
            cbf_ci_lower=deconvolution.cbf_ci_lower,
            cbf_ci_upper=deconvolution.cbf_ci_upper,
            mask_bool=mask_bool,
        )
        qc.update(
            self._summarize_dsc_maps(
                maps.rel_cbf,
                maps.rel_cbv,
                maps.rel_cbv_auc,
                maps.mtt,
                maps.tmax,
                mask_bool,
                aif_level,
                preload_class,
                baseline_info,
                timing.aif_bat_seconds,
            )
        )
        return _DSCPostprocessing(raw_concentration, maps, qc)

    def _save_base_maps(
        self,
        maps: _DSCFinalMaps,
        affine: npt.NDArray[Any],
        header: Any,
        qc_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Serialize the unconditional DSC maps and assemble interface results."""
        outputs = {
            "out_mtt": (maps.mtt, "_mtt"),
            "out_tmax": (maps.tmax, "_tmax"),
            "out_psr": (maps.psr, "_psr"),
            "out_ph": (maps.peak_height, "_ph"),
            "out_ttp": (maps.ttp, "_ttp"),
            "out_ttp_corrected": (maps.ttp_corrected, "_ttp_corrected"),
            "out_cnr": (maps.cnr, "_cnr"),
            "out_auc_raw": (maps.auc_raw, "_auc_raw"),
            "out_auc_corr": (maps.auc_corrected, "_auc_corr"),
            "out_auc_diff": (maps.auc_difference, "_auc_diff"),
            "out_rel_cbf": (maps.rel_cbf, "_rel_cbf"),
            "out_rel_cbv": (maps.rel_cbv, "_rel_cbv"),
            "out_rel_cbv_auc": (maps.rel_cbv_auc, "_rel_cbv_auc"),
            "out_rel_auc_raw": (maps.rel_auc_raw, "_rel_auc_raw"),
            "out_rel_auc_corr": (maps.rel_auc_corrected, "_rel_auc_corr"),
            "out_snr": (maps.snr, "_snr"),
            "out_tsnr": (maps.tsnr, "_tsnr"),
        }
        results: dict[str, Any] = {
            name: self._save_map(data, suffix, affine, header)
            for name, (data, suffix) in outputs.items()
        }
        results["qc_metrics"] = qc_metrics
        return results

    def _save_parametric_maps(
        self,
        *,
        rel_cbf: npt.NDArray[Any],
        rel_cbv: npt.NDArray[Any],
        cbf_std: Optional[npt.NDArray[Any]],
        cbv_std: Optional[npt.NDArray[Any]],
        cbf_ci_lower: Optional[npt.NDArray[Any]],
        cbf_ci_upper: Optional[npt.NDArray[Any]],
        cbv_ci_lower: Optional[npt.NDArray[Any]],
        cbv_ci_upper: Optional[npt.NDArray[Any]],
        parametric_r2: Optional[npt.NDArray[Any]],
        fallback_mask: Optional[npt.NDArray[Any]],
        mask_bool: npt.NDArray[Any],
        inference: str,
        affine: npt.NDArray[Any],
        header: Any,
    ) -> dict[str, str]:
        """Mask and serialize optional Bayesian and fit-diagnostic maps."""
        outputs: dict[str, str] = {}

        for name, suffix, standard_deviation, mean in (
            ("cbf", "_cbf", cbf_std, rel_cbf),
            ("cbv", "_cbv", cbv_std, rel_cbv),
        ):
            if standard_deviation is None:
                continue
            masked_std = np.where(mask_bool, standard_deviation, 0.0)
            outputs[f"out_{name}_std"] = self._save_map(
                masked_std, f"{suffix}_std", affine, header
            )
            cv = np.zeros_like(masked_std)
            valid = np.abs(mean) > EPSILON_CONCENTRATION
            np.divide(masked_std, np.abs(mean), out=cv, where=valid)
            cv = np.where(mask_bool, cv, 0.0)
            outputs[f"out_{name}_cv"] = self._save_map(
                cv, f"{suffix}_cv", affine, header
            )

        if inference in ("vi", "svb", "mcmc"):
            for name, suffix, interval in (
                ("out_cbf_ci95_lower", "_cbf_ci95_lower", cbf_ci_lower),
                ("out_cbf_ci95_upper", "_cbf_ci95_upper", cbf_ci_upper),
                ("out_cbv_ci95_lower", "_cbv_ci95_lower", cbv_ci_lower),
                ("out_cbv_ci95_upper", "_cbv_ci95_upper", cbv_ci_upper),
            ):
                if interval is not None:
                    outputs[name] = self._save_map(
                        np.where(mask_bool, interval, 0.0), suffix, affine, header
                    )

        if parametric_r2 is not None:
            r2 = np.where(mask_bool, parametric_r2, np.nan).astype(np.float32)
            outputs["out_parametric_r2"] = self._save_map(
                r2, "_dscParametric_r2", affine, header
            )
        if fallback_mask is not None:
            fallback = np.where(mask_bool, fallback_mask, 0).astype(np.uint8)
            outputs["out_fallback_mask"] = self._save_map(
                fallback, "_dscFallback_mask", affine, header, dtype=np.uint8
            )

        return outputs

    def _add_fit_qc_metrics(
        self,
        rel_cbv: npt.NDArray[Any],
        qc_metrics: dict[str, Any],
    ) -> None:
        """Add the selected fit voxel and fit quality to DSC QC state."""
        if "out_qc_fit_data" not in self._results:
            return
        with open(self._results["out_qc_fit_data"]) as fit_file:
            fit = json.load(fit_file)
        vx, vy, vz = fit["voxel"]
        qc_metrics["qc_voxel"] = f"({vx}, {vy}, {vz})"
        qc_metrics["qc_voxel_cbv"] = round(float(rel_cbv[vx, vy, vz]), 4)
        qc_metrics["qc_voxel_rsquared"] = round(fit.get("r_squared", 0.0), 4)
        tier = getattr(self, "_qc_voxel_tier", None)
        if tier is not None:
            qc_metrics["qc_voxel_tier"] = tier

    def _run_deconvolution(
        self,
        *,
        concentration: npt.NDArray[Any],
        aif: npt.NDArray[Any],
        mask: npt.NDArray[Any],
        method: str,
        dt: float,
        threshold: float,
        oi_threshold: float,
        inference: str,
        mcmc_samples: int,
        cpi_control_points: int,
        cpi_infer_cpt: bool,
        cpi_infer_delay: bool,
        motion_weighting: _DSCMotionWeighting,
    ) -> _DSCDeconvolutionMaps:
        """Dispatch DSC deconvolution and normalize its typed return state."""
        if method in ("sSVD", "oSVD", "tikhonov"):
            cbf, cbv, mtt, tmax, delay, residues, qc = self._deconvolve_svd(
                concentration, aif, mask, method, dt, threshold, oi_threshold
            )
            return _DSCDeconvolutionMaps(
                cbf=cbf,
                cbv=cbv,
                mtt=mtt,
                tmax=tmax,
                qc=qc,
                delay=delay,
                residues=residues,
            )

        (
            cbf, cbv, mtt, tmax, qc, alpha, beta, cth, cbf_std, cbv_std,
            cbf_ci_lower, cbf_ci_upper, cbv_ci_lower, cbv_ci_upper,
            parametric_r2, fallback_mask,
        ) = self._deconvolve_parametric(
            concentration,
            aif,
            mask,
            method,
            dt,
            inference,
            mcmc_samples,
            cpi_control_points,
            cpi_infer_cpt,
            cpi_infer_delay,
            motion_weights=motion_weighting.weights,
        )
        qc["motion_weighting"] = motion_weighting.qc
        return _DSCDeconvolutionMaps(
            cbf=cbf,
            cbv=cbv,
            mtt=mtt,
            tmax=tmax,
            qc=qc,
            alpha=alpha,
            beta=beta,
            cth=cth,
            cbf_std=cbf_std,
            cbv_std=cbv_std,
            cbf_ci_lower=cbf_ci_lower,
            cbf_ci_upper=cbf_ci_upper,
            cbv_ci_lower=cbv_ci_lower,
            cbv_ci_upper=cbv_ci_upper,
            parametric_r2=parametric_r2,
            fallback_mask=fallback_mask,
        )

    def _save_fit_qc_data(
        self,
        *,
        rel_cbf: npt.NDArray[Any],
        concentration: npt.NDArray[Any],
        aif: npt.NDArray[Any],
        mask_bool: npt.NDArray[Any],
        dt: float,
        method: str,
        inference: str,
        mcmc_samples: int,
        cpi_control_points: int,
        cpi_infer_cpt: bool,
        cpi_infer_delay: bool,
        threshold: float,
        oi_threshold: float,
        signal: Optional[npt.NDArray[Any]],
    ) -> None:
        """Dispatch fit-QC serialization to the selected method family."""
        if method in ("vascular", "cpi"):
            self._save_qc_fit_data(
                rel_cbf,
                concentration,
                aif,
                mask_bool,
                dt,
                method,
                inference,
                mcmc_samples,
                cpi_control_points,
                cpi_infer_cpt,
                cpi_infer_delay,
                signal_data=signal,
            )
            return
        self._save_qc_fit_data_svd(
            rel_cbf,
            concentration,
            aif,
            mask_bool,
            dt,
            method,
            threshold,
            oi_threshold,
            signal_data=signal,
        )

    def _save_ancillary_maps(
        self,
        *,
        relative_ttp: Optional[npt.NDArray[Any]],
        k2_map_path: Optional[str],
        delay: Optional[npt.NDArray[Any]],
        alpha: Optional[npt.NDArray[Any]],
        beta: Optional[npt.NDArray[Any]],
        cth: Optional[npt.NDArray[Any]],
        residues: Optional[npt.NDArray[Any]],
        save_residue: bool,
        mask_bool: npt.NDArray[Any],
        spatial_shape: tuple[int, ...],
        affine: npt.NDArray[Any],
        header: Any,
    ) -> dict[str, str]:
        """Serialize optional timing/model maps and the required K2 output."""
        import nibabel as nib  # type: ignore[import-untyped]

        outputs: dict[str, str] = {}
        if relative_ttp is not None:
            outputs["out_rttp"] = self._save_map(
                relative_ttp, "_rttp", affine, header
            )

        if k2_map_path is not None:
            k2_data = np.asarray(
                nib.load(k2_map_path).get_fdata(),  # type: ignore[attr-defined]
                dtype=np.float32,
            )
            k2_data = np.where(mask_bool, k2_data, 0.0)
            outputs["out_k2"] = self._save_map(k2_data, "_k2", affine, header)
        else:
            k2_zeros = np.zeros(spatial_shape, dtype=np.float32)
            k2_fallback_path = self._gen_outfilename("_k2")
            k2_fallback_img = nib.Nifti1Image(k2_zeros, affine, header)  # type: ignore[attr-defined]
            nib.save(k2_fallback_img, k2_fallback_path)  # type: ignore[attr-defined]
            outputs["out_k2"] = k2_fallback_path

        if delay is not None:
            outputs["out_delay"] = self._save_map(delay, "_delay", affine, header)
        for name, suffix, data in (
            ("out_alpha", "_alpha", alpha),
            ("out_beta", "_beta", beta),
            ("out_cth", "_cth", cth),
        ):
            if data is not None:
                outputs[name] = self._save_map(
                    np.where(mask_bool, data, 0.0), suffix, affine, header
                )

        if residues is not None and save_residue:
            residue_path = self._gen_outfilename("_residues")
            residue_img = nib.Nifti1Image(residues.astype(np.float32), affine)  # type: ignore[attr-defined]
            nib.save(residue_img, residue_path)  # type: ignore[attr-defined]
            outputs["out_residues"] = residue_path
        return outputs

    def _assemble_outputs(
        self,
        *,
        inputs: _DSCComputeInputs,
        baseline: _DSCBaselinePreparation,
        deconvolution: _DSCDeconvolutionMaps,
        postprocessing: _DSCPostprocessing,
    ) -> dict[str, Any]:
        """Serialize all required and optional DSC interface outputs."""
        maps = postprocessing.maps
        qc = postprocessing.qc
        spatial_shape = baseline.concentration.shape[:3]
        results = self._save_base_maps(maps, inputs.affine, inputs.header, qc)
        self._results = results

        if self.inputs.first_pass_cbv:
            first_pass_path = self._compute_first_pass_cbv(
                raw_conc_data=postprocessing.raw_concentration,
                mask_bool=inputs.mask_bool,
                spatial_shape=spatial_shape,
                dt=inputs.dt,
                n_bl=baseline.n_baseline,
                affine=inputs.affine,
                header=inputs.header,
                qc_metrics=qc,
            )
            if first_pass_path:
                results["out_rel_cbv_firstpass"] = first_pass_path

        results.update(
            self._save_ancillary_maps(
                relative_ttp=maps.relative_ttp,
                k2_map_path=baseline.k2_map_path,
                delay=maps.delay,
                alpha=deconvolution.alpha,
                beta=deconvolution.beta,
                cth=deconvolution.cth,
                residues=maps.residues,
                save_residue=bool(self.inputs.save_residue),
                mask_bool=inputs.mask_bool,
                spatial_shape=spatial_shape,
                affine=inputs.affine,
                header=inputs.header,
            )
        )
        self._save_fit_qc_data(
            rel_cbf=maps.rel_cbf,
            concentration=baseline.concentration,
            aif=baseline.aif,
            mask_bool=inputs.mask_bool,
            dt=inputs.dt,
            method=inputs.method,
            inference=inputs.inference,
            mcmc_samples=inputs.mcmc_samples,
            cpi_control_points=inputs.cpi_control_points,
            cpi_infer_cpt=inputs.cpi_infer_cpt,
            cpi_infer_delay=inputs.cpi_infer_delay,
            threshold=inputs.svd_threshold,
            oi_threshold=inputs.oi_threshold,
            signal=inputs.signal,
        )
        self._add_fit_qc_metrics(maps.rel_cbv, qc)

        if inputs.method in ("cpi", "vascular"):
            results.update(
                self._save_parametric_maps(
                    rel_cbf=maps.rel_cbf,
                    rel_cbv=maps.rel_cbv,
                    cbf_std=maps.cbf_std,
                    cbv_std=deconvolution.cbv_std,
                    cbf_ci_lower=maps.cbf_ci_lower,
                    cbf_ci_upper=maps.cbf_ci_upper,
                    cbv_ci_lower=deconvolution.cbv_ci_lower,
                    cbv_ci_upper=deconvolution.cbv_ci_upper,
                    parametric_r2=deconvolution.parametric_r2,
                    fallback_mask=deconvolution.fallback_mask,
                    mask_bool=inputs.mask_bool,
                    inference=inputs.inference,
                    affine=inputs.affine,
                    header=inputs.header,
                )
            )
        return results

    def _first_pass_reference(
        self,
        fp_area: npt.NDArray[Any],
        mask_bool: npt.NDArray[Any],
    ) -> "tuple[float, int] | None":
        """Median first-pass area over the eroded whole-WM (FAST) reference.

        Returns ``(median, n_voxels)`` when a valid WM reference is available
        (a matching-grid mask with at least ``_FIRSTPASS_MIN_REFERENCE_VOXELS``
        finite voxels and a positive median), else ``None`` (the reference is
        rejected as too small, degenerate, or off-grid).

        There is no whole-brain fallback. A whole-brain percentile and a
        GMM-largest-component are both biased, cross-subject-unstable WM
        proxies on real data (the first-pass-AUC histogram has no clean GM/WM
        separation), so without a WM segmentation the map has no valid anchor
        and is not produced.

        The median is taken over all finite voxels (signed, not positive-
        filtered): dropping the negative-noise tail would bias the median
        upward and inflate the denominator, deflating the map, the same
        unbiased-negatives principle the primary AUC-CBV encodes.
        """
        import nibabel as nib  # type: ignore[import-untyped]

        wm = self.inputs.wm_mask_dsc
        if (
            not isdefined(wm)
            or str(wm) in ("", "None")
            or not Path(str(wm)).exists()
        ):
            return None
        wm_mask = nib.load(str(wm)).get_fdata() > 0  # type: ignore[attr-defined]
        # An off-grid WM mask (e.g. a partial-FOV reference) would raise on the
        # boolean AND; a mismatched reference is not usable, so reject it.
        if wm_mask.shape != mask_bool.shape:
            return None
        wm_vals = fp_area[wm_mask & mask_bool]
        wm_vals = wm_vals[np.isfinite(wm_vals)]
        if wm_vals.size < _FIRSTPASS_MIN_REFERENCE_VOXELS:
            return None
        med = float(np.median(wm_vals))
        if med <= EPSILON_CONCENTRATION:
            return None
        return med, int(wm_vals.size)

    def _compute_first_pass_cbv(
        self,
        *,
        raw_conc_data: npt.NDArray[Any],
        mask_bool: npt.NDArray[Any],
        spatial_shape: tuple[int, ...],
        dt: float,
        n_bl: int,
        affine: npt.NDArray[Any],
        header: Any,
        qc_metrics: dict[str, Any],
    ) -> str:
        """Compute + save the first-pass gamma-variate relative CBV map.

        Per-voxel gamma-variate fit to the first pass of the RAW concentration
        (recirculation removed by the gamma model), with the analytic gamma area
        as CBV, normalized tissue-to-tissue by the eroded whole-WM (FAST) median.
        Voxels whose fit is rejected are left NaN and counted in
        ``firstpass_unfittable_fraction``; there is no truncating fallback. A
        light fixed-FWHM spatial pre-smoothing raises the effective CNR of the
        per-voxel fit (standard DSC preprocessing), applied to a copy for this
        map only. Requires a WM segmentation: with no valid WM reference the map
        has no anchor and is not produced (returns ``""``). Emits
        ``firstpass_reference`` / ``firstpass_reference_n`` /
        ``firstpass_unfittable_fraction`` into ``qc_metrics``.
        """
        import nibabel as nib  # type: ignore[import-untyped]
        from scipy.ndimage import gaussian_filter

        from pmriprep.utils.dsc_aif import batched_gamma_firstpass_auc

        # No WM segmentation input at all: skip before the per-voxel fit, since
        # the map has no valid anchor without one.
        wm = self.inputs.wm_mask_dsc
        if (
            not isdefined(wm)
            or str(wm) in ("", "None")
            or not Path(str(wm)).exists()
        ):
            qc_metrics["firstpass_reference"] = "unavailable-no-wm-segmentation"
            logger.warning(
                "First-pass CBV requested but no WM segmentation is available; "
                "map not produced."
            )
            return ""

        # Light spatial pre-fit smoothing (fixed FWHM), applied per frame to a
        # copy so the primary maps are unaffected. Sigma is derived from the
        # voxel spacing so the FWHM is a fixed physical size across resolutions.
        zooms = [float(z) for z in header.get_zooms()[:3]]
        sigma_vox = [
            (_FIRSTPASS_PREFIT_FWHM_MM / 2.3548) / max(z, 1e-3) for z in zooms
        ]
        # Sanitize before smoothing: gaussian_filter would otherwise spread a
        # single non-finite voxel across a ~sigma neighbourhood (a hole in the
        # map). The fit's own nan_to_num runs too late to prevent that.
        smoothed = gaussian_filter(
            np.nan_to_num(raw_conc_data, nan=0.0, posinf=0.0, neginf=0.0),
            sigma=(sigma_vox[0], sigma_vox[1], sigma_vox[2], 0.0),
        )

        fp_res = batched_gamma_firstpass_auc(
            smoothed[mask_bool], tr=dt, baseline_frames=n_bl,
        )
        fp_area = np.full(spatial_shape, np.nan, dtype=np.float64)
        fp_area[mask_bool] = fp_res.auc  # NaN on unfittable voxels

        reference = self._first_pass_reference(fp_area, mask_bool)
        if reference is None:
            qc_metrics["firstpass_reference"] = "unavailable-wm-reference-rejected"
            logger.warning(
                "First-pass CBV: WM reference rejected (too few fitted voxels, "
                "degenerate median, or off-grid mask); map not produced."
            )
            return ""
        reference_value, reference_n = reference

        rel_cbv_fp = np.full(spatial_shape, np.nan, dtype=np.float64)
        np.divide(fp_area, reference_value, out=rel_cbv_fp, where=mask_bool)
        # Upper sentinel only (mirror the primary AUC-CBV clamp at 10). No lower
        # clip; half-rectifying biases low-CBV voxels and inflates the reference
        # median. Unfittable in-mask voxels stay NaN; the image background is 0.
        with np.errstate(invalid="ignore"):
            rel_cbv_fp = np.where(mask_bool, np.minimum(rel_cbv_fp, 10.0), 0.0)

        path = self._gen_outfilename("_rel_cbv_firstpass")
        cast_data = rel_cbv_fp.astype(np.float32)
        local_header = cast("nib.Nifti1Header", header.copy())
        local_header.set_data_dtype(np.float32)
        img = nib.Nifti1Image(cast_data, affine, local_header)  # type: ignore[attr-defined]
        nib.save(img, path)  # type: ignore[attr-defined]

        qc_metrics["firstpass_reference"] = "firstpass-WM-FAST"
        qc_metrics["firstpass_reference_n"] = int(reference_n)
        qc_metrics["firstpass_unfittable_fraction"] = round(
            float(fp_res.unfittable_fraction), 4
        )
        # Disclose the pre-fit smoothing so the effective-resolution change
        # lands in the map's own sidecar provenance, not only the docstring.
        qc_metrics["firstpass_prefit_fwhm_mm"] = float(_FIRSTPASS_PREFIT_FWHM_MM)
        return path

    def _detect_baseline_frames(
        self,
        aif: npt.NDArray[Any],
        configured_baseline: int,
        concentration_data: Optional[npt.NDArray[Any]] = None,
        mask: Optional[npt.NDArray[Any]] = None,
    ) -> tuple[int, dict[str, Any]]:
        """Validate and refine baseline using the AIF-detected value.

        The primary baseline comes from AIF detection (ensemble on p99
        brain concentration, wired via ``baseline_frames`` input).  This
        function cross-validates by running the ensemble on the AIF curve
        itself.  Falls back to the configured ``pre_bolus`` value when the
        primary is unavailable (e.g., synthetic AIF).

        Parameters
        ----------
        aif : npt.NDArray[Any]
            1D AIF concentration curve.
        configured_baseline : int
            AIF-detected baseline (or pre_bolus fallback).
        concentration_data : npt.NDArray[Any], optional
            Optional 4D concentration array for fallback.
        mask : npt.NDArray[Any], optional
            Optional 3D brain mask for fallback.

        Returns
        -------
        tuple of (int, dict)
            Tuple of (n_baseline, detection_info dict).
        """
        import logging as _logging

        from pmriprep.utils.baseline_detection import detect_baseline_ensemble

        logger = _logging.getLogger(__name__)
        fallback = int(self.inputs.fallback_baseline_frames)
        info: dict[str, Any] = {"baseline_frames_configured": configured_baseline}

        if len(aif) < 3:
            info.update(
                baseline_detection_method="fallback_short_aif",
                baseline_frames_detected=fallback,
            )
            return fallback, info

        # Primary: use AIF-detected value (from p99 ensemble upstream)
        primary = configured_baseline

        # Cross-validate: run ensemble on the AIF curve itself
        try:
            tr = float(self.inputs.tr) if hasattr(self.inputs, "tr") else 1.5
            validation, val_diag = detect_baseline_ensemble(
                np.maximum(aif, 0.0), tr=tr, min_baseline=2,
            )

            aif_confidence = val_diag.get("confidence", "low")
            info["validation_arrival"] = validation
            info["validation_confidence"] = aif_confidence
            info["validation_method"] = val_diag.get("method", "unknown")

            # Confidence-based override: AIF curve has cleaner bolus
            # signature than p99 brain concentration for preloaded data.
            if primary <= fallback and validation > fallback:
                # Primary was at default (no real detection) — use AIF
                primary = validation
                info["baseline_detection_method"] = "auto_aif_ensemble"
            elif (
                aif_confidence == "high"
                and abs(validation - primary) > 5
            ):
                # AIF strongly disagrees with high confidence — override.
                # The AIF curve is a clean arterial signal; p99 can be
                # fooled by preload washout fluctuations.
                logger.info(
                    "AIF ensemble overrides p99: AIF=%d (conf=%s), "
                    "p99=%d",
                    validation, aif_confidence, primary,
                )
                primary = validation
                info["baseline_detection_method"] = "auto_aif_override"
            elif validation > primary + 3:
                # Moderate disagreement — log but keep primary
                logger.warning(
                    "Baseline validation disagrees: AIF=%d, p99=%d "
                    "— keeping primary",
                    validation, primary,
                )
                info["baseline_detection_method"] = "primary_aif_detected"
            else:
                info["baseline_detection_method"] = "primary_aif_detected"

        except Exception:
            info["baseline_detection_method"] = "primary_no_validation"

        # Sanity: at least 2, at most half the acquisition
        if primary < 2 or primary > len(aif) // 2:
            info.update(
                baseline_detection_method="fallback_range",
                baseline_frames_detected=fallback,
                arrival_index=primary,
            )
            return fallback, info

        info["baseline_frames_detected"] = primary
        info["arrival_index"] = primary
        return primary, info

    def _deconvolve_svd(
        self,
        concentration_data: npt.NDArray[Any],
        aif: npt.NDArray[Any],
        mask: npt.NDArray[Any],
        method: str,
        dt: float,
        threshold: float,
        oi_threshold: float,
    ) -> tuple[
        npt.NDArray[Any],  # cbf
        npt.NDArray[Any],  # cbv
        npt.NDArray[Any],  # mtt
        npt.NDArray[Any],  # tmax
        Optional[npt.NDArray[Any]],  # delay (oSVD)
        Optional[npt.NDArray[Any]],  # residues
        dict[str, Any],  # qc_metrics
    ]:
        """Run SVD-based deconvolution methods."""
        from pmriprep.utils.dsc_deconvolution import (
            svd_deconvolve_standard_batch,
            svd_deconvolve_oscillation_batch,
            tikhonov_deconvolve_batch,
        )

        qc_metrics: dict[str, Any] = {"method": method}

        # Initialize output arrays
        delay: Optional[npt.NDArray[Any]] = None
        residues: Optional[npt.NDArray[Any]] = None

        mask_bool = mask.astype(bool)

        if method == "sSVD":
            res, cbf, cbv, mtt = svd_deconvolve_standard_batch(  # type: ignore[assignment]
                concentration_data,
                aif,
                dt=dt,
                threshold=threshold,
                mask=mask_bool,
            )
            # Extract tmax from residues
            tmax = np.argmax(res, axis=-1).astype(np.float64) * dt
            residues = res if self.inputs.save_residue else None

        elif method == "oSVD":
            res, cbf, cbv, mtt, oi_values, thresholds_used, delay = svd_deconvolve_oscillation_batch(  # type: ignore[assignment]
                concentration_data,
                aif,
                dt=dt,
                oi_target=oi_threshold,
                mask=mask_bool,
            )
            tmax = delay.copy()
            residues = res if self.inputs.save_residue else None
            qc_metrics["oi_mean"] = float(np.mean(oi_values[mask_bool]))
            qc_metrics["threshold_mean"] = float(np.mean(thresholds_used[mask_bool]))

        elif method == "tikhonov":
            res, cbf, cbv, mtt, lambda_opt, data_driven_lambda = (
                tikhonov_deconvolve_batch(  # type: ignore[assignment]
                    concentration_data,
                    aif,
                    dt=dt,
                    mask=mask_bool,
                )
            )
            tmax = np.argmax(res, axis=-1).astype(np.float64) * dt
            residues = res if self.inputs.save_residue else None
            qc_metrics["lambda_mean"] = float(np.mean(lambda_opt[mask_bool]))
            # Fraction of voxels where the Morozov discrepancy selector chose a
            # data-driven λ from the measured noise floor, vs the degenerate
            # near-noise-free fallback. A high fallback fraction on real data
            # would be unexpected (real acquisitions have measurable noise).
            n_mask = int(np.sum(mask_bool))
            if n_mask > 0:
                qc_metrics["data_driven_lambda_fraction"] = float(
                    np.sum(data_driven_lambda[mask_bool]) / n_mask
                )
                qc_metrics["degenerate_fallback_fraction"] = float(
                    1.0 - qc_metrics["data_driven_lambda_fraction"]
                )

        else:
            raise ValueError(f"Unknown SVD method: {method}")

        return cbf, cbv, mtt, tmax, delay, residues, qc_metrics

    def _deconvolve_parametric(
        self,
        concentration_data: npt.NDArray[Any],
        aif: npt.NDArray[Any],
        mask: npt.NDArray[Any],
        method: str,
        dt: float,
        inference: str,
        mcmc_samples: int,
        cpi_control_points: int,
        cpi_infer_cpt: bool,
        cpi_infer_delay: bool,
        motion_weights: Optional[npt.NDArray[np.floating]] = None,
    ) -> tuple[
        npt.NDArray[Any],  # cbf
        npt.NDArray[Any],  # cbv
        npt.NDArray[Any],  # mtt
        npt.NDArray[Any],  # tmax
        dict[str, Any],  # qc_metrics
        Optional[npt.NDArray[Any]],  # alpha_map (vascular only)
        Optional[npt.NDArray[Any]],  # beta_map (vascular only)
        Optional[npt.NDArray[Any]],  # cth_map (vascular only)
        Optional[npt.NDArray[Any]],  # cbf_std_map (parametric only)
        Optional[npt.NDArray[Any]],  # cbv_std_map (parametric only)
        Optional[npt.NDArray[Any]],  # cbf_ci_lower_map (parametric only)
        Optional[npt.NDArray[Any]],  # cbf_ci_upper_map (parametric only)
        Optional[npt.NDArray[Any]],  # cbv_ci_lower_map (parametric only)
        Optional[npt.NDArray[Any]],  # cbv_ci_upper_map (parametric only)
        Optional[npt.NDArray[Any]],  # parametric_r2_map (parametric only)
        Optional[npt.NDArray[Any]],  # fallback_mask (parametric only)
    ]:
        """Run parametric model deconvolution (vascular, CPI).

        Uses multiprocessing.Pool when n_procs > 1.
        """
        from pmriprep.models.dsc import estimate_bolus_duration

        spatial_shape = concentration_data.shape[:3]
        n_timepoints = concentration_data.shape[3]
        time_points = np.arange(n_timepoints, dtype=np.float64) * dt

        # Initialize output arrays
        cbf = np.zeros(spatial_shape, dtype=np.float64)
        cbv = np.zeros(spatial_shape, dtype=np.float64)
        mtt = np.zeros(spatial_shape, dtype=np.float64)
        tmax = np.zeros(spatial_shape, dtype=np.float64)
        alpha_map = np.zeros(spatial_shape, dtype=np.float64) if method == "vascular" else None
        beta_map = np.zeros(spatial_shape, dtype=np.float64) if method == "vascular" else None
        cth_map = np.zeros(spatial_shape, dtype=np.float64) if method == "vascular" else None
        cbf_std_map = np.zeros(spatial_shape, dtype=np.float64) if method in ("cpi", "vascular") else None
        # CBV is derived from correlated fitted parameters. Neither scalar nor
        # batched fit results retain the joint covariance needed to propagate
        # its uncertainty, so no CBV uncertainty map is emitted.
        cbv_std_map = None
        # Parametric-fit R² map and oSVD-fallback mask (parametric only).
        # R² inits to NaN so plotting tools render outside-brain voxels
        # transparent; raw parametric R² is preserved on fallback voxels
        # (the fallback mask separately flags which method produced
        # the saved CBF/CBV/MTT/Tmax in that voxel).
        parametric_r2_map = (
            np.full(spatial_shape, np.nan, dtype=np.float32)
            if method in ("cpi", "vascular") else None
        )
        fallback_mask = (
            np.zeros(spatial_shape, dtype=np.uint8)
            if method in ("cpi", "vascular") else None
        )
        has_ci = inference in ("vi", "svb", "mcmc", "map", "nlls")
        cbf_ci_lower_map = np.zeros(spatial_shape, dtype=np.float64) if has_ci else None
        cbf_ci_upper_map = np.zeros(spatial_shape, dtype=np.float64) if has_ci else None
        cbv_ci_lower_map = None
        cbv_ci_upper_map = None

        qc_metrics: dict[str, Any] = {
            "method": method,
            "inference": inference,
        }

        # Get mask indices
        mask_coords = np.where(mask > 0)
        n_voxels = len(mask_coords[0])

        # Estimate bolus duration once from AIF for CPI
        bolus_duration_frames = estimate_bolus_duration(aif)

        # Mean tissue across in-mask voxels for the MTT-adaptive auto-N
        # floor (used by select_n_control_points when cpi_infer_cpt=True).
        mean_tissue = np.mean(
            concentration_data[
                mask_coords[0], mask_coords[1], mask_coords[2], :
            ],
            axis=0,
        ).astype(np.float64)

        # Build voxel work items only for paths that consume them.
        # VI/SVB and MAP take concentration_data + mask_coords directly
        # via vectorized gather; pre-building N namedtuples + N×T copies
        # wastes ~40 MB at 50K voxels × 100 timepoints.
        voxel_data_list: list[_DSCVoxelInput] = []
        if inference == "nlls":
            for i in range(n_voxels):
                x, y, z = (
                    int(mask_coords[0][i]),
                    int(mask_coords[1][i]),
                    int(mask_coords[2][i]),
                )
                voxel_data_list.append(
                    _DSCVoxelInput(x, y, z, concentration_data[x, y, z, :])
                )

        # Worker initializer args
        sloppy = bool(self.inputs.sloppy)
        random_seed = (
            int(self.inputs.random_seed) if isdefined(self.inputs.random_seed) else None
        )
        init_args = (
            aif, time_points, method, inference, mcmc_samples,
            cpi_control_points, bolus_duration_frames, dt,
            motion_weights, sloppy, random_seed, cpi_infer_cpt,
            cpi_infer_delay, mean_tissue,
        )

        n_procs = max(1, int(self.inputs.n_procs))

        # Batched VI path: vectorized Variational Bayes in single process
        if inference in ("vi", "svb"):
            results = self._dsc_fit_batched_vi(
                concentration_data=concentration_data,
                mask_coords=mask_coords,
                aif=aif,
                time_points=time_points,
                method=method,
                dt=dt,
                n_voxels=n_voxels,
                cpi_control_points=cpi_control_points,
                cpi_infer_cpt=cpi_infer_cpt,
                cpi_infer_delay=cpi_infer_delay,
                bolus_duration_frames=bolus_duration_frames,
                mean_tissue=mean_tissue,
                motion_weights=motion_weights,
                sloppy=sloppy,
                spatial=(inference == "svb"),
            )
        elif inference == "map":
            results = self._dsc_fit_batched_map(
                concentration_data=concentration_data,
                mask_coords=mask_coords,
                aif=aif,
                time_points=time_points,
                method=method,
                dt=dt,
                n_voxels=n_voxels,
                cpi_control_points=cpi_control_points,
                cpi_infer_cpt=cpi_infer_cpt,
                cpi_infer_delay=cpi_infer_delay,
                bolus_duration_frames=bolus_duration_frames,
                mean_tissue=mean_tissue,
                motion_weights=motion_weights,
                sloppy=sloppy,
            )
        elif n_procs > 1:
            from concurrent.futures import ProcessPoolExecutor

            chunk_size = max(1, n_voxels // (n_procs * 10))
            executor = ProcessPoolExecutor(
                max_workers=n_procs,
                initializer=_dsc_init_worker,
                initargs=init_args,
            )
            try:
                results = list(executor.map(
                    _dsc_fit_single_voxel, voxel_data_list,
                    chunksize=chunk_size,
                ))
            except (KeyboardInterrupt, SystemExit):
                logger.warning("Interrupted during parallel DSC fitting, cancelling workers")
                raise
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        else:
            # Serial path: avoid multiprocessing overhead
            _dsc_init_worker(*init_args)
            results = [_dsc_fit_single_voxel(vd) for vd in voxel_data_list]
            _dsc_worker_ctx.clear()

        # Scatter results into output arrays
        n_converged = 0
        r_squared_list: list[float] = []

        for res in results:
            if res["failed"]:
                continue
            x, y, z = res["coords"]
            cbf[x, y, z] = res.get("cbf", 0.0)
            cbv[x, y, z] = res.get("cbv", 0.0)
            mtt[x, y, z] = res.get("mtt", 0.0)
            tmax[x, y, z] = res.get("tmax", 0.0)

            if alpha_map is not None and "alpha" in res:
                alpha_map[x, y, z] = res["alpha"]
            if beta_map is not None and "beta" in res:
                beta_map[x, y, z] = res["beta"]
            if cth_map is not None and "cth" in res:
                cth_map[x, y, z] = res["cth"]
            if cbf_std_map is not None and "cbf_std" in res:
                cbf_std_map[x, y, z] = res["cbf_std"]
            if cbv_std_map is not None and "cbv_std" in res:
                cbv_std_map[x, y, z] = res["cbv_std"]
            if cbf_ci_lower_map is not None and "cbf_ci95_lower" in res:
                cbf_ci_lower_map[x, y, z] = res["cbf_ci95_lower"]
            if cbf_ci_upper_map is not None and "cbf_ci95_upper" in res:
                cbf_ci_upper_map[x, y, z] = res["cbf_ci95_upper"]
            if cbv_ci_lower_map is not None and "cbv_ci95_lower" in res:
                cbv_ci_lower_map[x, y, z] = res["cbv_ci95_lower"]
            if cbv_ci_upper_map is not None and "cbv_ci95_upper" in res:
                cbv_ci_upper_map[x, y, z] = res["cbv_ci95_upper"]

            if res.get("r_squared") is not None:
                r_squared_list.append(res["r_squared"])
                # Scatter raw parametric R² into 3-D map even on
                # would-fallback voxels — keeps the close-call vs
                # catastrophic distinction visible in the saved map.
                if parametric_r2_map is not None:
                    parametric_r2_map[x, y, z] = float(res["r_squared"])
            if res.get("converged"):
                n_converged += 1

        # Fallback: run oSVD for voxels where parametric fit failed or R² < 0.5
        # Fall back to oSVD for poor vascular fits; ground truth maps
        # likely contain oSVD-derived values for many voxels.
        from pmriprep.utils.dsc_deconvolution import svd_deconvolve_oscillation

        fallback_coords = [
            res["coords"]
            for res in results
            if res.get("needs_fallback") or res.get("failed")
        ]
        n_fallback = 0
        for coords in fallback_coords:
            x, y, z = coords
            tissue = concentration_data[x, y, z, :]
            if np.max(np.abs(tissue)) < EPSILON_CONCENTRATION:
                continue
            try:
                svd_result = svd_deconvolve_oscillation(
                    tissue, aif, dt,
                    oi_target=float(self.inputs.oi_threshold),
                )
                cbf[x, y, z] = svd_result.cbf
                cbv[x, y, z] = svd_result.cbv
                mtt[x, y, z] = svd_result.mtt
                tmax[x, y, z] = svd_result.delay
                # Zero parametric/uncertainty maps for fallback voxels —
                # these are SVD-derived, not from parametric fit
                if alpha_map is not None:
                    alpha_map[x, y, z] = 0.0
                if beta_map is not None:
                    beta_map[x, y, z] = 0.0
                if cth_map is not None:
                    cth_map[x, y, z] = 0.0
                if cbf_std_map is not None:
                    cbf_std_map[x, y, z] = 0.0
                if cbv_std_map is not None:
                    cbv_std_map[x, y, z] = 0.0
                if cbf_ci_lower_map is not None:
                    cbf_ci_lower_map[x, y, z] = 0.0
                if cbf_ci_upper_map is not None:
                    cbf_ci_upper_map[x, y, z] = 0.0
                if cbv_ci_lower_map is not None:
                    cbv_ci_lower_map[x, y, z] = 0.0
                if cbv_ci_upper_map is not None:
                    cbv_ci_upper_map[x, y, z] = 0.0
                if fallback_mask is not None:
                    fallback_mask[x, y, z] = 1
                n_fallback += 1
            except Exception:
                pass  # leave as parametric result or zeros

        # Compile QC metrics
        if r_squared_list:
            qc_metrics["r_squared_mean"] = float(np.mean(r_squared_list))
            qc_metrics["r_squared_std"] = float(np.std(r_squared_list))

        if method in ("cpi", "vascular"):
            qc_metrics["n_converged"] = n_converged
            # FE/NLP plateau rate is informational only (FABBER 10-iter
            # spec rarely satisfies the strict plateau in 10 iters by
            # design); fit quality is gated by R² in the fallback path.
            qc_metrics["fe_plateau_rate"] = n_converged / max(1, n_voxels)

        qc_metrics["n_fallback_osvd"] = n_fallback
        # Fallback rate is the actual fit-health signal for parametric
        # methods: voxels rejected by the R² < 0.5 gate fall through to
        # oSVD. Always populated (zero for non-parametric methods).
        qc_metrics["fallback_rate"] = n_fallback / max(1, n_voxels)

        return (cbf, cbv, mtt, tmax, qc_metrics, alpha_map, beta_map, cth_map,
                cbf_std_map, cbv_std_map,
                cbf_ci_lower_map, cbf_ci_upper_map, cbv_ci_lower_map, cbv_ci_upper_map,
                parametric_r2_map, fallback_mask)

    @staticmethod
    def _build_dsc_parametric_setup(
        method: str,
        concentration_data: npt.NDArray[Any],
        mask_coords: tuple[npt.NDArray[Any], ...],
        aif: npt.NDArray[np.floating],
        dt: float,
        cpi_control_points: int,
        bolus_duration_frames: int,
        cpi_infer_cpt: bool = True,
        cpi_infer_delay: bool = False,
        mean_tissue: Optional[npt.NDArray[np.floating]] = None,
    ) -> dict[str, Any]:
        """Build model, priors, and fitting kwargs for batched DSC fitting.

        Shared by both batched VI and batched MAP paths.

        Returns
        -------
        dict
            Dict with keys: tissue_curves, fitting_names, fitting_bounds,
            mono_prec, mono_eta, cp_times, conv_matrix, means_nat, stds_nat,
            means_t, stds_t, transforms, ubs.
        """
        from pmriprep.models.bayesian_utils import build_priors_from_params
        from pmriprep.models.dsc import CPIModel, VascularModel, select_n_control_points
        from pmriprep.utils.dsc_deconvolution import build_volterra_matrix

        tissue_curves = concentration_data[
            mask_coords[0], mask_coords[1], mask_coords[2], :
        ]
        tissue_curves = np.nan_to_num(tissue_curves, nan=0.0, posinf=0.0, neginf=0.0)

        if method == "vascular":
            model = VascularModel()
            fitting_defs = model.get_fitting_parameter_definitions()
            priors = build_priors_from_params(fitting_defs)
            fitting_names = [p.name for p in fitting_defs]
            fitting_bounds = [(p.bounds[0], p.bounds[1]) for p in fitting_defs]
            mono_prec = None
            mono_eta = None
            cp_times = None
            conv_matrix = None
            cpi_t_terminal = None
        else:  # cpi
            n_cp = cpi_control_points
            if n_cp == 0:
                n_cp = select_n_control_points(
                    bolus_duration_frames,
                    infer_cpt=bool(cpi_infer_cpt),
                    mean_tissue=mean_tissue,
                    dt=float(dt),
                )
            n_timepoints = int(concentration_data.shape[3])
            cpi_model = CPIModel(
                n_control_points=n_cp,
                n_acquisition_frames=n_timepoints,
                dt=float(dt),
                infer_cpt=bool(cpi_infer_cpt),
                infer_delay=bool(cpi_infer_delay),
            )
            fitting_defs = list(cpi_model.get_fitting_parameter_definitions())
            priors = build_priors_from_params(fitting_defs)
            fitting_names = [p.name for p in fitting_defs]
            fitting_bounds = [(p.bounds[0], p.bounds[1]) for p in fitting_defs]
            mono_prec = None
            mono_eta = None
            cp_times = np.array(cpi_model._control_point_times, dtype=np.float64)
            conv_matrix = build_volterra_matrix(aif, dt)
            cpi_t_terminal = float(cpi_model.terminal_time)

        means_nat = np.array([p.mean for p in priors], dtype=np.float64)
        stds_nat = np.array([p.std for p in priors], dtype=np.float64)
        means_t = np.array([p.mean_transformed for p in priors], dtype=np.float64)
        stds_t = np.array([p.std_transformed for p in priors], dtype=np.float64)
        transforms = [p.transform for p in priors]
        ubs: list[float | None] = [p.upper_bound for p in priors]

        return dict(
            tissue_curves=tissue_curves,
            fitting_names=fitting_names,
            fitting_bounds=fitting_bounds,
            mono_prec=mono_prec,
            mono_eta=mono_eta,
            cp_times=cp_times,
            conv_matrix=conv_matrix,
            cpi_t_terminal=cpi_t_terminal,
            cpi_infer_delay=bool(cpi_infer_delay),
            means_nat=means_nat,
            stds_nat=stds_nat,
            means_t=means_t,
            stds_t=stds_t,
            transforms=transforms,
            ubs=ubs,
        )

    def _dsc_fit_batched_vi(
        self,
        concentration_data: npt.NDArray[Any],
        mask_coords: tuple[npt.NDArray[Any], ...],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        method: str,
        dt: float,
        n_voxels: int,
        cpi_control_points: int,
        cpi_infer_cpt: bool,
        cpi_infer_delay: bool,
        bolus_duration_frames: int,
        motion_weights: Optional[npt.NDArray[np.floating]],
        sloppy: bool,
        spatial: bool = False,
        mean_tissue: Optional[npt.NDArray[np.floating]] = None,
    ) -> list[dict[str, Any]]:
        """Batched VI for DSC parametric models.

        Extracts priors from model definitions and calls batched_dsc_vi_fit.
        Runs in a single process with vectorized numpy operations.

        Parameters
        ----------
        concentration_data : npt.NDArray[Any]
            Concentration data, shape (X, Y, Z, T).
        mask_coords : tuple of npt.NDArray[Any]
            Tuple of coordinate arrays.
        aif : npt.NDArray[np.floating]
            AIF, shape (T,).
        time_points : npt.NDArray[np.floating]
            Time points (seconds), shape (T,).
        method : str
            'cpi' or 'vascular'.
        dt : float
            Time step (seconds).
        n_voxels : int
            Number of voxels.
        cpi_control_points : int
            Number of CPI control points.
        bolus_duration_frames : int
            Bolus duration in frames.
        motion_weights : npt.NDArray[np.floating] or None
            Per-frame weights or None.
        sloppy : bool
            Whether to use sloppy mode.
        spatial : bool
            Use spatial VB wrapper (MRF neighbor coupling).

        Returns
        -------
        list of dict
            List of per-voxel result dicts.
        """
        from pmriprep.models.dsc.batched_fitting import batched_dsc_vi_fit

        s = self._build_dsc_parametric_setup(
            method, concentration_data, mask_coords, aif, dt,
            cpi_control_points, bolus_duration_frames,
            cpi_infer_cpt=cpi_infer_cpt,
            cpi_infer_delay=cpi_infer_delay,
            mean_tissue=mean_tissue,
        )

        vi_label = "SVB" if spatial else "VI"
        logger.info("DSC batched %s: fitting %s on %d voxels", vi_label, method, n_voxels)

        vi_kwargs: dict[str, Any] = dict(
            model_type=method,
            tissue_curves=s["tissue_curves"],
            aif=aif,
            time_points=time_points,
            prior_means_nat=s["means_nat"],
            prior_stds_nat=s["stds_nat"],
            transforms=s["transforms"],
            upper_bounds=s["ubs"],
            fitting_param_names=s["fitting_names"],
            fitting_bounds=s["fitting_bounds"],
            mono_prec=s["mono_prec"],
            mono_eta=s["mono_eta"],
            control_point_times=s["cp_times"],
            toeplitz_matrix=s["conv_matrix"],
            cpi_t_terminal=s["cpi_t_terminal"],
            cpi_infer_delay=s["cpi_infer_delay"],
            prior_means_transformed=s["means_t"],
            prior_stds_transformed=s["stds_t"],
            weights=motion_weights,
            sloppy=sloppy,
            save_fitted_curves=True,
        )

        if spatial:
            from pmriprep.models.dsc.spatial_vb import batched_dsc_spatial_vi_fit
            mc3: tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]] = (
                mask_coords[0], mask_coords[1], mask_coords[2],
            )
            batched_result = batched_dsc_spatial_vi_fit(
                **vi_kwargs,
                mask_coords=mc3,
                spatial_shape=concentration_data.shape[:3],
            )
        else:
            # Parallel batched VI when --nprocs > 1 (spatial path
            # remains single-process — voxels are coupled via the
            # M-prior, see batched_fitting.py guard).
            batched_result = batched_dsc_vi_fit(
                **vi_kwargs, nprocs=int(self.inputs.n_procs),
            )

        return self._batched_result_to_voxel_dicts(
            batched_result, mask_coords, method, n_voxels, dt,
            s["cp_times"], time_points, s["cpi_t_terminal"],
            s["cpi_infer_delay"],
        )

    def _batched_result_to_voxel_dicts(
        self,
        batched_result: Any,
        mask_coords: tuple[npt.NDArray[Any], ...],
        method: str,
        n_voxels: int,
        dt: float,
        cp_times: Optional[npt.NDArray[np.floating]],
        time_points: npt.NDArray[np.floating],
        cpi_t_terminal: Optional[float] = None,
        cpi_infer_delay: bool = False,
    ) -> list[dict[str, Any]]:
        """Convert DSCBatchedFitResult to per-voxel result dicts.

        Shared by both batched VI and batched MAP paths.

        Parameters
        ----------
        batched_result : Any
            DSCBatchedFitResult from VI or MAP.
        mask_coords : tuple of npt.NDArray[Any]
            Tuple of coordinate arrays.
        method : str
            'cpi' or 'vascular'.
        n_voxels : int
            Number of voxels.
        dt : float
            Time step (seconds).
        cp_times : npt.NDArray[np.floating] or None
            CPI control point times or None.
        time_points : npt.NDArray[np.floating]
            Time points (seconds), shape (T,).
        cpi_t_terminal : float or None, optional
            CPI terminal anchor time (seconds); must match the value the forward
            model used so the readout integrates the residue that was fitted.
        cpi_infer_delay : bool, optional
            Whether the CPI fit carried a per-voxel ``delay`` parameter; when
            True ``out["tmax"]`` reports it (CBV/MTT stay delay-invariant).

        Returns
        -------
        list of dict
            List of per-voxel result dicts.
        """
        from pmriprep.models.dsc.batched_fitting import batched_cpi_residue

        results: list[dict[str, Any]] = []

        # Precompute the CPI residue for every voxel via the SAME helper the
        # forward model uses, so the readout integrates the residue that was
        # fitted (parity with the NLLS fitted_residue integral). Reassemble the
        # (N, P) parameter matrix in canonical order [cbf, f1..fN (, s1..sN)].
        cpi_r_norm: Optional[npt.NDArray[np.floating]] = None
        if method == "cpi" and cp_times is not None:
            n_cp_local = len(cp_times)
            cols = [np.asarray(batched_result.params["cbf"], dtype=np.float64)]
            cols += [
                np.asarray(batched_result.params[f"f{k}"], dtype=np.float64)
                for k in range(1, n_cp_local + 1)
            ]
            if "s1" in batched_result.params:
                cols += [
                    np.asarray(batched_result.params[f"s{k}"], dtype=np.float64)
                    for k in range(1, n_cp_local + 1)
                ]
            if cpi_infer_delay and "delay" in batched_result.params:
                cols.append(
                    np.asarray(batched_result.params["delay"], dtype=np.float64)
                )
            params_matrix = np.column_stack(cols)
            cpi_r_norm = batched_cpi_residue(
                params_matrix, time_points, cp_times,
                t_terminal=cpi_t_terminal, infer_delay=cpi_infer_delay,
            )

        for i in range(n_voxels):
            x = int(mask_coords[0][i])
            y = int(mask_coords[1][i])
            z = int(mask_coords[2][i])

            out: dict[str, Any] = {
                "coords": (x, y, z),
                "failed": False,
            }

            if method == "vascular":
                cbf_val = float(batched_result.params["cbf"][i])
                mtt_val = float(batched_result.params["mtt"][i])
                alpha_val = float(batched_result.params["alpha"][i])
                delay_val = float(batched_result.params["delay"][i])
                beta_val = mtt_val / max(alpha_val, EPSILON_SHAPE)

                out["cbf"] = cbf_val
                out["cbv"] = cbf_val * mtt_val
                out["mtt"] = mtt_val
                out["tmax"] = delay_val
                out["alpha"] = alpha_val
                out["beta"] = beta_val
                out["cth"] = beta_val * np.sqrt(alpha_val)
            else:  # cpi
                cbf_val = float(batched_result.params["cbf"][i])
                out["cbf"] = cbf_val
                if cbf_val > EPSILON_CONCENTRATION and cpi_r_norm is not None:
                    cbv_val = cbf_val * float(np.trapezoid(cpi_r_norm[i], dx=dt))
                else:
                    cbv_val = 0.0
                out["cbv"] = cbv_val
                out["mtt"] = cbv_val / cbf_val if cbf_val > EPSILON_CONCENTRATION else 0.0
                if cpi_infer_delay and "delay" in batched_result.params:
                    out["tmax"] = float(batched_result.params["delay"][i])
                else:
                    out["tmax"] = 0.0

            cbf_std_val = 0.0
            if batched_result.uncertainties:
                cbf_std_val = float(batched_result.uncertainties.get(
                    "cbf", np.zeros(n_voxels))[i])
                out["cbf_std"] = cbf_std_val

            out["cbf_ci95_lower"] = max(cbf_val - 1.96 * cbf_std_val, 0.0)
            out["cbf_ci95_upper"] = cbf_val + 1.96 * cbf_std_val

            out["r_squared"] = float(batched_result.r_squared[i])
            out["converged"] = bool(batched_result.converged[i])
            # Fallback gate: fit quality only. The `converged` flag tracks
            # the strict FE-plateau test, but FABBER's protocol is a fixed
            # 10-iter schedule (see batched_fitting.py:1006-1013) — most
            # voxels never satisfy the 1e-4 relative FE plateau in that
            # window even when their parameter fits are perfectly usable.
            # Gating on `not converged` would silently fall back ~100% of
            # voxels to oSVD; gate on R² instead.
            out["needs_fallback"] = float(batched_result.r_squared[i]) < 0.5
            out["failed"] = float(batched_result.r_squared[i]) < -0.5

            results.append(out)

        return results

    def _dsc_fit_batched_map(
        self,
        concentration_data: npt.NDArray[Any],
        mask_coords: tuple[npt.NDArray[Any], ...],
        aif: npt.NDArray[np.floating],
        time_points: npt.NDArray[np.floating],
        method: str,
        dt: float,
        n_voxels: int,
        cpi_control_points: int,
        cpi_infer_cpt: bool,
        cpi_infer_delay: bool,
        bolus_duration_frames: int,
        motion_weights: Optional[npt.NDArray[np.floating]],
        sloppy: bool,
        mean_tissue: Optional[npt.NDArray[np.floating]] = None,
    ) -> list[dict[str, Any]]:
        """Batched MAP for DSC parametric models.

        Extracts priors from model definitions and calls batched_dsc_map_fit.
        Runs in a single process with vectorized numpy operations.

        Parameters
        ----------
        concentration_data : npt.NDArray[Any]
            Concentration data, shape (X, Y, Z, T).
        mask_coords : tuple of npt.NDArray[Any]
            Tuple of coordinate arrays.
        aif : npt.NDArray[np.floating]
            AIF, shape (T,).
        time_points : npt.NDArray[np.floating]
            Time points (seconds), shape (T,).
        method : str
            'cpi' or 'vascular'.
        dt : float
            Time step (seconds).
        n_voxels : int
            Number of voxels.
        cpi_control_points : int
            Number of CPI control points.
        bolus_duration_frames : int
            Bolus duration in frames.
        motion_weights : npt.NDArray[np.floating] or None
            Per-frame weights or None.
        sloppy : bool
            Whether to use sloppy mode.

        Returns
        -------
        list of dict
            List of per-voxel result dicts.
        """
        from pmriprep.models.dsc.batched_fitting import batched_dsc_map_fit

        s = self._build_dsc_parametric_setup(
            method, concentration_data, mask_coords, aif, dt,
            cpi_control_points, bolus_duration_frames,
            cpi_infer_cpt=cpi_infer_cpt,
            cpi_infer_delay=cpi_infer_delay,
            mean_tissue=mean_tissue,
        )

        batched_result = batched_dsc_map_fit(
            model_type=method,
            tissue_curves=s["tissue_curves"],
            aif=aif,
            time_points=time_points,
            prior_means_nat=s["means_nat"],
            prior_stds_nat=s["stds_nat"],
            transforms=s["transforms"],
            upper_bounds=s["ubs"],
            fitting_param_names=s["fitting_names"],
            fitting_bounds=s["fitting_bounds"],
            mono_prec=s["mono_prec"],
            mono_eta=s["mono_eta"],
            control_point_times=s["cp_times"],
            toeplitz_matrix=s["conv_matrix"],
            cpi_t_terminal=s["cpi_t_terminal"],
            cpi_infer_delay=s["cpi_infer_delay"],
            prior_means_transformed=s["means_t"],
            prior_stds_transformed=s["stds_t"],
            weights=motion_weights,
            sloppy=sloppy,
            save_fitted_curves=True,
            nprocs=int(self.inputs.n_procs),
        )

        return self._batched_result_to_voxel_dicts(
            batched_result, mask_coords, method, n_voxels, dt,
            s["cp_times"], time_points, s["cpi_t_terminal"],
            s["cpi_infer_delay"],
        )

    def _save_qc_fit_data(
        self,
        cbf_map: npt.NDArray[Any],
        conc_data: npt.NDArray[Any],
        aif: npt.NDArray[Any],
        mask_bool: npt.NDArray[np.bool_],
        dt: float,
        method: str,
        inference: str,
        mcmc_samples: int,
        cpi_control_points: int,
        cpi_infer_cpt: bool,
        cpi_infer_delay: bool,
        signal_data: npt.NDArray[Any] | None = None,
    ) -> None:
        """Select a QC voxel and save its fit data as JSON.

        The QC voxel is chosen near the median CBF in the eroded brain mask.
        Its tissue curve is re-fit to produce the model prediction, which
        is saved alongside the measured concentration and AIF for the
        fit QC plot.

        Parameters
        ----------
        cbf_map : npt.NDArray[Any]
            3D relative CBF map (masked).
        conc_data : npt.NDArray[Any]
            4D concentration data.
        aif : npt.NDArray[Any]
            1D AIF concentration curve.
        mask_bool : npt.NDArray[np.bool_]
            3D boolean brain mask.
        dt : float
            Temporal resolution in seconds.
        method : str
            Deconvolution method ('vascular' or 'cpi').
        inference : str
            Inference engine ('nlls', 'map', 'vi', 'svb').
        mcmc_samples : int
            Number of MCMC samples.
        cpi_control_points : int
            Number of CPI control points (0=auto).
        signal_data : npt.NDArray[Any] or None, optional
            4D raw signal data (optional, for QC plot).
        """
        import nibabel as nib  # type: ignore[import-untyped]
        from scipy.ndimage import binary_erosion  # type: ignore[import-untyped]
        from pmriprep.utils.dsc_qc_plots import _select_dsc_qc_voxel
        from pmriprep.models.dsc import (
            VascularModel,
            CPIModel,
            fit_dsc_model,
            estimate_bolus_duration,
            select_n_control_points,
        )

        try:
            # Load and resample enhancing mask if available
            enhancing_mask_data: npt.NDArray[Any] | None = None
            if (
                isdefined(self.inputs.enhancing_mask_file)
                and self.inputs.enhancing_mask_file
            ):
                enh_img = nib.load(self.inputs.enhancing_mask_file)
                conc_img = nib.load(self.inputs.concentration_file)
                enh_data = enh_img.get_fdata()  # type: ignore[attr-defined]
                needs_resample = (
                    enh_data.shape[:3] != conc_data.shape[:3]
                    or not np.allclose(
                        enh_img.affine, conc_img.affine, atol=1e-3  # type: ignore[attr-defined]
                    )
                )
                if needs_resample:
                    from nilearn.image import resample_to_img  # type: ignore[import-untyped]
                    resampled = resample_to_img(
                        enh_img, conc_img, interpolation="nearest"
                    )
                    enhancing_mask_data = (
                        resampled.get_fdata() > 0.5  # type: ignore[attr-defined]
                    ).astype(np.float64)
                else:
                    enhancing_mask_data = enh_data

            qc_voxel, qc_voxel_tier = _select_dsc_qc_voxel(
                cbf_map, mask_bool, binary_erosion,
                enhancing_mask=enhancing_mask_data,
            )
            x, y, z = qc_voxel
            self._qc_voxel_tier = qc_voxel_tier
            tissue_curve = conc_data[x, y, z, :].astype(np.float64)

            if np.max(np.abs(tissue_curve)) < EPSILON_CONCENTRATION:
                return

            n_timepoints = conc_data.shape[3]
            time_points = np.arange(n_timepoints, dtype=np.float64) * dt

            # Build model instance
            model_instance: Any
            if method == "vascular":
                model_instance = VascularModel()
            else:
                bolus_duration_frames = estimate_bolus_duration(aif)
                n_cp = cpi_control_points
                if n_cp == 0:
                    # Mean tissue across in-mask voxels for the
                    # MTT-adaptive auto-N floor.
                    mt_mean = np.mean(
                        conc_data[mask_bool, :], axis=0,
                    ).astype(np.float64)
                    n_cp = select_n_control_points(
                        bolus_duration_frames,
                        infer_cpt=bool(cpi_infer_cpt),
                        mean_tissue=mt_mean,
                        dt=float(dt),
                    )
                model_instance = CPIModel(
                    n_control_points=n_cp,
                    n_acquisition_frames=n_timepoints,
                    dt=float(dt),
                    infer_cpt=bool(cpi_infer_cpt),
                    infer_delay=bool(cpi_infer_delay),
                )

            # Fit the QC voxel
            result = fit_dsc_model(
                model=model_instance,
                concentration=tissue_curve,
                aif=aif,
                time_points=time_points,
                method=inference,
                mcmc_samples=mcmc_samples,
                compute_uncertainties=False,
            )

            if result is None:
                return

            # Use the model's fitted tissue curve directly (matches the
            # model's forward operator).
            model_prediction: npt.NDArray[Any]
            if result.fitted_tissue_curve is not None:
                model_prediction = result.fitted_tissue_curve
            elif result.fitted_residue is not None:
                # Fallback: reconstruct from residue via the Volterra
                # product-integration operator. Only parametric models (CPI,
                # vascular) reach this branch, and they all use the Volterra
                # forward, so this matches their forward operator.
                from pmriprep.utils.dsc_deconvolution import build_volterra_matrix
                model_prediction = build_volterra_matrix(aif, dt) @ result.fitted_residue
            else:
                return

            # Extract raw signal at QC voxel (for signal intensity panel)
            raw_signal: list[float] | None = None
            if signal_data is not None and signal_data.ndim == 4:
                raw_signal = signal_data[x, y, z, :].tolist()

            # Save as JSON
            base_path = self._gen_outfilename("_qc_fit_data")
            # _gen_outfilename appends .nii.gz; replace with .json
            qc_fit_path = base_path.replace(".nii.gz", ".json")
            qc_fit_dict = {
                "voxel": [x, y, z],
                "voxel_cbf": float(cbf_map[x, y, z]),
                "r_squared": float(result.r_squared) if result.r_squared else 0.0,
                "method": method,
                "tissue_concentration": tissue_curve.tolist(),
                "model_prediction": model_prediction.tolist(),
                "aif": aif.tolist(),
                "time_points": time_points.tolist(),
                "raw_signal": raw_signal,
            }

            from pmriprep.utils.sidecar_builders import dump_sidecar_json
            with open(qc_fit_path, "w") as f:
                f.write(dump_sidecar_json(qc_fit_dict))

            self._results["out_qc_fit_data"] = qc_fit_path

        except Exception as e:
            logger.warning(f"Failed to generate QC fit data: {e}")

    def _save_qc_fit_data_svd(
        self,
        cbf_map: npt.NDArray[Any],
        conc_data: npt.NDArray[Any],
        aif: npt.NDArray[Any],
        mask_bool: npt.NDArray[np.bool_],
        dt: float,
        method: str,
        threshold: float,
        oi_threshold: float,
        signal_data: npt.NDArray[Any] | None = None,
    ) -> None:
        """Select a QC voxel and save SVD deconvolution fit data as JSON.

        For SVD methods, the "fit" is the reconstruction of concentration
        from the recovered residue function: C_recon = AIF * R(t) * dt.
        The R² between measured and reconstructed concentration serves as
        a QC metric for deconvolution quality.

        Parameters
        ----------
        cbf_map : npt.NDArray[Any]
            3D relative CBF map (masked).
        conc_data : npt.NDArray[Any]
            4D concentration data.
        aif : npt.NDArray[Any]
            1D AIF concentration curve.
        mask_bool : npt.NDArray[np.bool_]
            3D boolean brain mask.
        dt : float
            Temporal resolution in seconds.
        method : str
            Deconvolution method ('sSVD', 'oSVD', 'tikhonov').
        threshold : float
            SVD truncation threshold (for sSVD).
        oi_threshold : float
            Oscillation index target (for oSVD).
        signal_data : npt.NDArray[Any] or None, optional
            4D raw signal data (optional, for QC plot).
        """
        import nibabel as nib  # type: ignore[import-untyped]
        from scipy.ndimage import binary_erosion  # type: ignore[import-untyped]
        from pmriprep.utils.dsc_qc_plots import _select_dsc_qc_voxel
        from pmriprep.utils.dsc_deconvolution import (
            svd_deconvolve_standard,
            svd_deconvolve_oscillation,
            tikhonov_deconvolve,
            build_block_circulant_matrix,
            build_volterra_matrix,
        )

        try:
            # Load and resample enhancing mask if available
            enhancing_mask_data: npt.NDArray[Any] | None = None
            if (
                isdefined(self.inputs.enhancing_mask_file)
                and self.inputs.enhancing_mask_file
            ):
                enh_img = nib.load(self.inputs.enhancing_mask_file)
                conc_img = nib.load(self.inputs.concentration_file)
                enh_data = enh_img.get_fdata()  # type: ignore[attr-defined]
                needs_resample = (
                    enh_data.shape[:3] != conc_data.shape[:3]
                    or not np.allclose(
                        enh_img.affine, conc_img.affine, atol=1e-3  # type: ignore[attr-defined]
                    )
                )
                if needs_resample:
                    from nilearn.image import resample_to_img  # type: ignore[import-untyped]
                    resampled = resample_to_img(
                        enh_img, conc_img, interpolation="nearest"
                    )
                    enhancing_mask_data = (
                        resampled.get_fdata() > 0.5  # type: ignore[attr-defined]
                    ).astype(np.float64)
                else:
                    enhancing_mask_data = enh_data

            qc_voxel, qc_voxel_tier = _select_dsc_qc_voxel(
                cbf_map, mask_bool, binary_erosion,
                enhancing_mask=enhancing_mask_data,
            )
            x, y, z = qc_voxel
            self._qc_voxel_tier = qc_voxel_tier
            tissue_curve = conc_data[x, y, z, :].astype(np.float64)

            if np.max(np.abs(tissue_curve)) < EPSILON_CONCENTRATION:
                return

            n_timepoints = conc_data.shape[3]
            time_points = np.arange(n_timepoints, dtype=np.float64) * dt

            # Deconvolve the single QC voxel and reconstruct concentration
            c_recon: npt.NDArray[Any]
            if method == "sSVD":
                residue = svd_deconvolve_standard(
                    tissue_curve, aif, dt=dt, threshold=threshold,
                ).residue
                c_recon = build_volterra_matrix(aif, dt) @ residue
            elif method == "oSVD":
                full_residue = svd_deconvolve_oscillation(
                    tissue_curve, aif, dt=dt, oi_target=oi_threshold,
                ).residue
                matrix = build_block_circulant_matrix(aif, dt)
                c_recon = (matrix @ full_residue)[:n_timepoints]
            elif method == "tikhonov":
                residue = tikhonov_deconvolve(
                    tissue_curve, aif, dt=dt,
                ).residue
                c_recon = build_volterra_matrix(aif, dt) @ residue
            else:
                return

            # Compute R²
            ss_res = np.sum((tissue_curve - c_recon) ** 2)
            ss_tot = np.sum((tissue_curve - np.mean(tissue_curve)) ** 2)
            r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Extract raw signal at QC voxel (for signal intensity panel)
            raw_signal: list[float] | None = None
            if signal_data is not None and signal_data.ndim == 4:
                raw_signal = signal_data[x, y, z, :].tolist()

            # Save as JSON (same format as parametric)
            base_path = self._gen_outfilename("_qc_fit_data")
            qc_fit_path = base_path.replace(".nii.gz", ".json")
            qc_fit_dict = {
                "voxel": [x, y, z],
                "voxel_cbf": float(cbf_map[x, y, z]),
                "r_squared": r_squared,
                "method": method,
                "tissue_concentration": tissue_curve.tolist(),
                "model_prediction": c_recon.tolist(),
                "aif": aif.tolist(),
                "time_points": time_points.tolist(),
                "raw_signal": raw_signal,
            }

            from pmriprep.utils.sidecar_builders import dump_sidecar_json
            with open(qc_fit_path, "w") as f:
                f.write(dump_sidecar_json(qc_fit_dict))

            self._results["out_qc_fit_data"] = qc_fit_path

        except Exception as e:
            logger.warning(f"Failed to generate SVD QC fit data: {e}")

    def _run_interface(self, runtime: Any) -> Any:
        """Execute DSC map computation."""
        inputs = self._load_compute_inputs()
        conc_data = inputs.concentration
        header = inputs.header
        mask = inputs.mask
        mask_bool = inputs.mask_bool
        aif = inputs.aif
        signal_data = inputs.signal
        method = inputs.method
        dt = inputs.dt
        threshold = inputs.svd_threshold
        oi_threshold = inputs.oi_threshold
        cpi_control_points = inputs.cpi_control_points
        cpi_infer_cpt = inputs.cpi_infer_cpt
        cpi_infer_delay = inputs.cpi_infer_delay
        inference = inputs.inference
        mcmc_samples = inputs.mcmc_samples

        # SVD methods are direct matrix decompositions — Bayesian/NLLS
        # inference only applies to parametric fitting (cpi, vascular).
        if method in ("sSVD", "oSVD", "tikhonov") and inference != "vi":
            logger.debug(
                "inference='%s' ignored for SVD method '%s'.",
                inference, method,
            )

        # Warn on known-bad CPI inference + infer_cpt combinations
        # (DRO matrix data — see tests/benchmarks/run_cpi_infer_cpt_map_vs_vb.py)
        _warn_invalid_cpi_combo(method, inference, cpi_infer_cpt)

        tail_frames = inputs.tail_frames
        baseline = self._prepare_baseline_and_leakage(inputs)
        conc_data = baseline.concentration
        aif = baseline.aif
        recomputed_raw_conc = baseline.recomputed_raw_concentration
        detected_baseline = baseline.detected_frames
        n_bl = baseline.n_baseline
        baseline_detection_info = baseline.detection_info
        spatial_shape = conc_data.shape[:3]

        motion_weighting = self._prepare_motion_weighting(
            method, header, conc_data.shape[3], dt
        )
        deconvolution = self._run_deconvolution(
            concentration=conc_data,
            aif=aif,
            mask=mask,
            method=method,
            dt=dt,
            threshold=threshold,
            oi_threshold=oi_threshold,
            inference=inference,
            mcmc_samples=mcmc_samples,
            cpi_control_points=cpi_control_points,
            cpi_infer_cpt=cpi_infer_cpt,
            cpi_infer_delay=cpi_infer_delay,
            motion_weighting=motion_weighting,
        )
        aif_level = str(self.inputs.aif_detection_level)
        preload_class = str(self.inputs.preload_classification)
        postprocessing = self._prepare_final_outputs(
            deconvolution=deconvolution,
            concentration=conc_data,
            recomputed_raw_concentration=recomputed_raw_conc,
            signal=signal_data,
            spatial_shape=spatial_shape,
            mask_bool=mask_bool,
            detected_baseline=detected_baseline,
            n_baseline=n_bl,
            baseline_info=baseline_detection_info,
            dt=dt,
            tail_frames=tail_frames,
            method=method,
            aif_level=aif_level,
            preload_class=preload_class,
        )
        self._results = self._assemble_outputs(
            inputs=inputs,
            baseline=baseline,
            deconvolution=deconvolution,
            postprocessing=postprocessing,
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
