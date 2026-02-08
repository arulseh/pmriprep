"""DSC map computation interface for perfusion quantification.

This module implements the main Nipype interface for computing DSC
perfusion parameter maps (CBF, CBV, MTT, Tmax, PSR, PH) from
concentration curves and AIF using various deconvolution methods.
"""

from pathlib import Path
from typing import Any, Literal, Optional

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
    EPSILON_SIGNAL,
    DEFAULT_SVD_THRESHOLD,
    DEFAULT_OI_THRESHOLD,
    calculate_k_h,
    VESSEL_CALIBRATION_GRE_3T,
    get_r2_star,
)
from pmriprep.utils.dsc_signal import compute_psr_legacy, compute_peak_height_legacy, compute_ttp_legacy


# ---------------------------------------------------------------------------
# Module-level worker for parallel parametric deconvolution
# ---------------------------------------------------------------------------
from typing import NamedTuple


class _DSCVoxelInput(NamedTuple):
    """Lightweight struct for a single voxel's data."""
    x: int
    y: int
    z: int
    tissue_curve: Any  # npt.NDArray


_dsc_worker_ctx: dict[str, Any] = {}


def _dsc_init_worker(
    aif: Any,
    time_points: Any,
    method: str,
    bayesian: str,
    mcmc_samples: int,
    cpi_control_points: int,
    bolus_duration_frames: int,
    dt: float,
) -> None:
    """Initialise shared state once per worker process."""
    from pmriprep.models.dsc import (
        VascularModel,
        CPIModel,
        select_n_control_points,
    )

    _dsc_worker_ctx["aif"] = aif
    _dsc_worker_ctx["time_points"] = time_points
    _dsc_worker_ctx["method"] = method
    _dsc_worker_ctx["bayesian"] = bayesian
    _dsc_worker_ctx["mcmc_samples"] = mcmc_samples
    _dsc_worker_ctx["dt"] = dt

    # Pre-build model template so each voxel can cheaply instantiate
    if method == "vascular":
        _dsc_worker_ctx["model_factory"] = lambda: VascularModel()
    else:  # cpi
        n_cp = cpi_control_points
        if n_cp == 0:
            n_cp = select_n_control_points(bolus_duration_frames, dt)
        _dsc_worker_ctx["model_factory"] = lambda _ncp=n_cp: CPIModel(n_control_points=_ncp)


def _dsc_fit_single_voxel(voxel_data: _DSCVoxelInput) -> dict[str, Any]:
    """Fit a single voxel — called from pool workers."""
    from pmriprep.models.dsc import fit_dsc_model
    from pmriprep.models.dsc.cpi import _fit_cpi_simple

    x, y, z, tissue_curve = voxel_data

    # Skip near-zero curves
    if np.max(np.abs(tissue_curve)) < EPSILON_CONCENTRATION:
        return {"coords": (x, y, z), "failed": True}

    method = _dsc_worker_ctx["method"]
    bayesian = _dsc_worker_ctx["bayesian"]
    aif = _dsc_worker_ctx["aif"]
    time_points = _dsc_worker_ctx["time_points"]
    dt = _dsc_worker_ctx["dt"]

    try:
        model_instance = _dsc_worker_ctx["model_factory"]()

        if bayesian == "none" and method == "cpi":
            # Fast path: direct NLLS without Bayesian prior/noise overhead
            result = _fit_cpi_simple(model_instance, tissue_curve, aif, time_points)
            if result is None:
                return {"coords": (x, y, z), "failed": True}
        elif bayesian == "none":
            result = fit_dsc_model(
                model=model_instance,
                concentration=tissue_curve,
                aif=aif,
                time_points=time_points,
                method="map",
                compute_uncertainties=False,
            )
        else:
            result = fit_dsc_model(
                model=model_instance,
                concentration=tissue_curve,
                aif=aif,
                time_points=time_points,
                method=bayesian,
                mcmc_samples=_dsc_worker_ctx["mcmc_samples"],
                compute_uncertainties=False,
            )

        out: dict[str, Any] = {"coords": (x, y, z), "failed": False}
        cbf_val = result.parameters.get("cbf", result.parameters.get("CBF", 0.0))
        out["cbf"] = cbf_val

        if method == "vascular":
            alpha = result.parameters.get("alpha", 1.0)
            beta = result.parameters.get("beta", 1.0)
            mtt_val = alpha * beta
            out["mtt"] = mtt_val
            out["cbv"] = cbf_val * mtt_val / 60.0
            out["tmax"] = result.parameters.get("delay", 0.0)
        else:  # cpi
            if result.fitted_residue is not None:
                res = result.fitted_residue
                cbv_val = float(np.sum(res) * dt)
                out["cbv"] = cbv_val
                out["mtt"] = cbv_val / cbf_val if cbf_val > EPSILON_CONCENTRATION else 0.0
                out["tmax"] = float(np.argmax(res)) * dt
            else:
                out["cbv"] = 0.0
                out["mtt"] = 0.0
                out["tmax"] = 0.0

        out["r_squared"] = result.r_squared
        out["converged"] = (bayesian != "none" and result.passes_convergence())
        return out

    except Exception:
        return {"coords": (x, y, z), "failed": True}


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
    aif_best_coord = traits.List(
        traits.Int,
        mandatory=False,
        desc="Best AIF voxel coordinate [x, y, z] for re-extraction",
    )
    field_strength = traits.Float(
        3.0,
        usedefault=True,
        desc="Magnetic field strength in Tesla (for r2_star auto-lookup)",
    )
    aif_detection_level = traits.Str(
        "atlas",
        usedefault=True,
        desc="AIF detection level: 'atlas_strict', 'atlas_relaxed', 'brain_*', or 'synthetic'",
    )
    preload_classification = traits.Str(
        "none",
        usedefault=True,
        desc="Preload severity: 'none', 'minimal', 'half_dose', 'significant', 'severe'",
    )

    # Deconvolution method
    method = traits.Enum(
        "sSVD",
        "cSVD",
        "oSVD",
        "tikhonov",
        "vascular",
        "cpi",
        usedefault=True,
        desc="Deconvolution method: sSVD, cSVD, oSVD, tikhonov, vascular, or cpi",
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

    # Time parameters
    tr = traits.Float(
        1.5,
        usedefault=True,
        desc="Repetition time in seconds",
    )
    baseline_frames = traits.Int(
        5,
        usedefault=True,
        desc="Number of pre-bolus baseline frames",
    )
    tail_frames = traits.Int(
        5,
        usedefault=True,
        desc="Number of tail frames for PSR calculation",
    )

    # Bayesian options
    bayesian = traits.Enum(
        "none",
        "map",
        "vi",
        "mcmc",
        usedefault=True,
        desc="Bayesian inference method: none, map, vi, or mcmc",
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
    use_gpu = traits.Bool(
        False,
        usedefault=True,
        desc="Use GPU acceleration when available",
    )
    save_residue = traits.Bool(
        False,
        usedefault=True,
        desc="Save residue functions (large output)",
    )
    k2_method = traits.Enum(
        "exponential",
        "integral",
        usedefault=True,
        desc="K2 leakage correction method: exponential (legacy) or integral (BSW)",
    )
    k2_map_input_file = File(
        mandatory=False,
        desc="K2 map from preload_correct (passed through when baseline not recomputed)",
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
    out_rel_cbv = File(genfile=True, desc="Output relative CBV map NIfTI")
    out_rel_auc_raw = File(genfile=True, desc="Output relative AUC raw map NIfTI")
    out_rel_auc_corr = File(genfile=True, desc="Output relative AUC corrected map NIfTI")

    # Output files — absolute maps (conditional on gating)
    out_abs_cbf = File(genfile=True, desc="Output absolute CBF map NIfTI")
    out_abs_cbv = File(genfile=True, desc="Output absolute CBV map NIfTI")
    out_abs_auc_raw = File(genfile=True, desc="Output absolute AUC raw map NIfTI")
    out_abs_auc_corr = File(genfile=True, desc="Output absolute AUC corrected map NIfTI")

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
    out_cnr = File(exists=True, desc="CNR map (contrast-to-noise ratio)")

    # AUC decomposition (always produced)
    out_auc_raw = File(exists=True, desc="AUC raw (pre-K2) map")
    out_auc_corr = File(exists=True, desc="AUC corrected (post-K2) map")
    out_auc_diff = File(exists=True, desc="AUC diff (leakage contribution) map")

    # Relative maps (always produced)
    out_rel_cbf = File(exists=True, desc="Relative CBF map")
    out_rel_cbv = File(exists=True, desc="Relative CBV map")
    out_rel_auc_raw = File(exists=True, desc="Relative AUC raw map")
    out_rel_auc_corr = File(exists=True, desc="Relative AUC corrected map")

    # Absolute maps (conditional — only when gating passes)
    out_abs_cbf = File(desc="Absolute CBF map (mL/100g/min)")
    out_abs_cbv = File(desc="Absolute CBV map (mL/100g)")
    out_abs_auc_raw = File(desc="Absolute AUC raw map (mM·s)")
    out_abs_auc_corr = File(desc="Absolute AUC corrected map (mM·s)")

    # Quality metrics
    out_snr = File(exists=True, desc="SNR map (baseline signal)")
    out_tsnr = File(exists=True, desc="tSNR map (temporal signal)")

    # K2 leakage map (recomputed or pass-through)
    out_k2 = File(desc="K2 leakage correction map (from recomputed or pass-through)")

    # Gating flag
    absolute_permitted = traits.Bool(desc="Whether absolute quantification gating passed")

    # Optional extras
    out_delay = File(desc="Delay map (only for cSVD method)")
    out_residues = File(desc="Residue functions (if save_residue=True)")
    qc_metrics = traits.Dict(desc="QC metrics dictionary")


class ComputeDSCMaps(BaseInterface):
    """Compute DSC perfusion parameter maps.

    This interface computes cerebral perfusion parameter maps from DSC-MRI
    concentration curves and an arterial input function (AIF) using various
    deconvolution methods.

    Supported deconvolution methods:
    - sSVD: Standard SVD (Østergaard method)
    - cSVD: Circulant SVD (delay-insensitive)
    - oSVD: Oscillation-index limited SVD (adaptive regularization)
    - tikhonov: Tikhonov regularization with GCV
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
        Oscillation index target for oSVD. Default: 0.065.
    cpi_control_points : int
        Number of CPI control points (0=auto). Default: 0.
    tr : float
        Repetition time in seconds. Default: 1.5.
    baseline_frames : int
        Pre-bolus baseline frames. Default: 5.
    tail_frames : int
        Tail frames for PSR. Default: 5.
    bayesian : str
        Bayesian inference: none, map, vi, mcmc. Default: 'none'.
    mcmc_samples : int
        MCMC samples for Bayesian. Default: 500.
    use_gpu : bool
        Use GPU acceleration. Default: False.
    save_residue : bool
        Save residue functions. Default: False.

    Returns
    -------
    out_rel_cbf, out_rel_cbv, out_rel_auc_raw, out_rel_auc_corr : str
        Paths to relative perfusion maps (always produced).
    out_abs_cbf, out_abs_cbv, out_abs_auc_raw, out_abs_auc_corr : str, optional
        Paths to absolute perfusion maps (only when gating passes).
    out_mtt, out_tmax, out_psr, out_ph, out_ttp, out_cnr : str
        Paths to quantification-independent maps.
    out_auc_raw, out_auc_corr, out_auc_diff : str
        Paths to AUC decomposition maps.
    out_snr, out_tsnr : str
        Paths to SNR and tSNR quality metric maps.
    absolute_permitted : bool
        Whether absolute quantification gating passed.
    out_delay : str, optional
        Path to delay map (cSVD only).
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

    The cSVD method additionally outputs a delay map showing the
    estimated tracer arrival delay relative to the AIF.

    References
    ----------
    Østergaard et al. (1996). High resolution measurement of cerebral
    blood flow using intravascular tracer bolus passages. Part I.
    Magn Reson Med 36(5):715-725.

    Wu et al. (2003). Tracer arrival timing-insensitive technique for
    estimating flow in MR perfusion-weighted imaging. Magn Reson Med 50(1):164-174.
    """

    input_spec = ComputeDSCMapsInputSpec
    output_spec = ComputeDSCMapsOutputSpec

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

    @staticmethod
    def _detect_baseline_frames(
        aif: npt.NDArray[Any],
        configured_baseline: int,
        concentration_data: Optional[npt.NDArray[Any]] = None,
        mask: Optional[npt.NDArray[Any]] = None,
    ) -> tuple[int, dict[str, Any]]:
        """Detect pre-bolus baseline frames from the AIF or concentration data.

        For real AIFs, applies Savitzky-Golay smoothing then 3-sigma threshold
        detection, mirroring the perf-plus pipeline. For synthetic AIFs (detected
        by leading zeros), falls back to peak-fraction detection on the mean
        brain concentration curve.

        Args:
            aif: 1D AIF concentration curve.
            configured_baseline: Fallback baseline frame count.
            concentration_data: Optional 4D concentration array for fallback.
            mask: Optional 3D brain mask for fallback.

        Returns:
            Tuple of (n_baseline, detection_info dict).
        """
        from pmriprep.utils.dsc_signal import detect_bolus_arrival

        info: dict[str, Any] = {"baseline_frames_configured": configured_baseline}

        if len(aif) < 3:
            info.update(
                baseline_detection_method="fallback_short_aif",
                baseline_frames_detected=configured_baseline,
            )
            return configured_baseline, info

        try:
            from scipy.signal import savgol_filter  # type: ignore[import-untyped]

            # Check if AIF is synthetic (leading zeros then abrupt rise)
            n_leading_zeros = 0
            for val in aif:
                if val == 0.0:
                    n_leading_zeros += 1
                else:
                    break
            aif_is_synthetic = n_leading_zeros >= 2 and aif[n_leading_zeros] > 0.5

            if aif_is_synthetic and concentration_data is not None and mask is not None:
                # Synthetic AIF: detect bolus onset from mean brain
                # concentration gradient (5% of max gradient threshold)
                mask_bool = mask > 0
                mean_conc = np.mean(concentration_data[mask_bool], axis=0)
                if len(mean_conc) >= 7:
                    smooth_conc = savgol_filter(
                        mean_conc, window_length=7, polyorder=2,
                    )
                    smooth_conc = np.maximum(smooth_conc, 0.0)
                    grad = np.diff(smooth_conc)
                    max_grad: float = float(np.max(grad))
                    if max_grad > 0:
                        thresh = max_grad * 0.05
                        above = np.where(grad > thresh)[0]
                        if len(above) > 0:
                            n_baseline = int(above[0])
                            if 2 <= n_baseline <= len(aif) // 2:
                                info.update(
                                    baseline_detection_method="auto_conc_gradient",
                                    baseline_frames_detected=n_baseline,
                                    arrival_index=n_baseline,
                                    aif_synthetic=True,
                                )
                                return n_baseline, info

            # Real AIF: smooth then detect (matches perf-plus lines 1478-1483).
            # Perf-plus applies savgol_filter before detect_bolus_arrival even
            # though the AIF was already smoothed once during ROI averaging.
            # Without re-smoothing, small noise spikes in the first few frames
            # can produce a false-positive 3-sigma threshold crossing.
            aif_smooth = savgol_filter(
                aif,
                window_length=min(5, len(aif)),
                polyorder=min(2, len(aif) - 1),
            )
            aif_smooth[aif_smooth < 0] = 0
            arrival_idx, n_baseline = detect_bolus_arrival(aif_smooth, baseline_frames=3)

            # Sanity: at least 2, at most half the acquisition
            if n_baseline < 2 or n_baseline > len(aif) // 2:
                info.update(
                    baseline_detection_method="fallback_range",
                    baseline_frames_detected=configured_baseline,
                    arrival_index=int(arrival_idx),
                )
                return configured_baseline, info

            info.update(
                baseline_detection_method="auto_aif",
                baseline_frames_detected=int(n_baseline),
                arrival_index=int(arrival_idx),
            )
            return int(n_baseline), info

        except Exception:
            info.update(
                baseline_detection_method="fallback_error",
                baseline_frames_detected=configured_baseline,
            )
            return configured_baseline, info

    def _deconvolve_svd(
        self,
        concentration_data: npt.NDArray[Any],
        aif: npt.NDArray[Any],
        mask: npt.NDArray[Any],
        method: str,
        dt: float,
        threshold: float,
        oi_threshold: float,
        use_gpu: bool,
    ) -> tuple[
        npt.NDArray[Any],  # cbf
        npt.NDArray[Any],  # cbv
        npt.NDArray[Any],  # mtt
        npt.NDArray[Any],  # tmax
        Optional[npt.NDArray[Any]],  # delay (cSVD only)
        Optional[npt.NDArray[Any]],  # residues
        dict[str, Any],  # qc_metrics
    ]:
        """Run SVD-based deconvolution methods."""
        from pmriprep.utils.dsc_deconvolution import (
            svd_deconvolve_standard_batch,
            svd_deconvolve_circulant_batch,
            svd_deconvolve_oscillation_batch,
            tikhonov_deconvolve_batch,
        )

        spatial_shape = concentration_data.shape[:3]
        n_timepoints = concentration_data.shape[3]
        qc_metrics: dict[str, Any] = {"method": method}

        # Initialize output arrays
        cbf = np.zeros(spatial_shape, dtype=np.float64)
        cbv = np.zeros(spatial_shape, dtype=np.float64)
        mtt = np.zeros(spatial_shape, dtype=np.float64)
        tmax = np.zeros(spatial_shape, dtype=np.float64)
        delay: Optional[npt.NDArray[Any]] = None
        residues: Optional[npt.NDArray[Any]] = None

        # Check for GPU deconvolution
        if use_gpu and method in ("sSVD", "cSVD"):
            try:
                from pmriprep.models.dsc.gpu_deconvolution import (
                    batched_svd_deconvolve_standard,
                    batched_svd_deconvolve_circulant,
                )

                if method == "sSVD":
                    result = batched_svd_deconvolve_standard(
                        concentration_data,
                        aif,
                        dt=dt,
                        threshold=threshold,
                        mask=mask.astype(bool),
                        return_residues=False,
                    )
                    cbf = result.cbf
                    cbv = result.cbv
                    mtt = result.mtt
                    tmax = result.tmax
                    qc_metrics["used_gpu"] = True

                elif method == "cSVD":
                    result = batched_svd_deconvolve_circulant(
                        concentration_data,
                        aif,
                        dt=dt,
                        threshold=threshold,
                        mask=mask.astype(bool),
                        return_residues=False,
                    )
                    cbf = result.cbf
                    cbv = result.cbv
                    mtt = result.mtt
                    tmax = result.tmax
                    delay = result.delay
                    qc_metrics["used_gpu"] = True

                return cbf, cbv, mtt, tmax, delay, residues, qc_metrics

            except (ImportError, RuntimeError):
                # Fall back to CPU
                qc_metrics["used_gpu"] = False

        # CPU-based deconvolution
        qc_metrics["used_gpu"] = False
        mask_bool = mask.astype(bool)

        if method == "sSVD":
            res, cbf, cbv, mtt = svd_deconvolve_standard_batch(
                concentration_data,
                aif,
                dt=dt,
                threshold=threshold,
                mask=mask_bool,
            )
            # Extract tmax from residues
            tmax = np.argmax(res, axis=-1).astype(np.float64) * dt
            residues = res if self.inputs.save_residue else None

        elif method == "cSVD":
            res, cbf, cbv, mtt, delay = svd_deconvolve_circulant_batch(
                concentration_data,
                aif,
                dt=dt,
                threshold=threshold,
                mask=mask_bool,
            )
            # For cSVD, tmax includes delay
            tmax = np.argmax(res, axis=-1).astype(np.float64) * dt
            residues = res if self.inputs.save_residue else None

        elif method == "oSVD":
            res, cbf, cbv, mtt, oi_values, thresholds_used = (
                svd_deconvolve_oscillation_batch(
                    concentration_data,
                    aif,
                    dt=dt,
                    oi_target=oi_threshold,
                    mask=mask_bool,
                )
            )
            tmax = np.argmax(res, axis=-1).astype(np.float64) * dt
            residues = res if self.inputs.save_residue else None
            qc_metrics["oi_mean"] = float(np.mean(oi_values[mask_bool]))
            qc_metrics["threshold_mean"] = float(np.mean(thresholds_used[mask_bool]))

        elif method == "tikhonov":
            res, cbf, cbv, mtt, lambda_opt = tikhonov_deconvolve_batch(
                concentration_data,
                aif,
                dt=dt,
                mask=mask_bool,
            )
            tmax = np.argmax(res, axis=-1).astype(np.float64) * dt
            residues = res if self.inputs.save_residue else None
            qc_metrics["lambda_mean"] = float(np.mean(lambda_opt[mask_bool]))

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
        bayesian: str,
        mcmc_samples: int,
        cpi_control_points: int,
    ) -> tuple[
        npt.NDArray[Any],  # cbf
        npt.NDArray[Any],  # cbv
        npt.NDArray[Any],  # mtt
        npt.NDArray[Any],  # tmax
        dict[str, Any],  # qc_metrics
    ]:
        """Run parametric model deconvolution (vascular, CPI).

        Uses multiprocessing.Pool when n_procs > 1 (matching perf-plus).
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

        qc_metrics: dict[str, Any] = {
            "method": method,
            "bayesian": bayesian,
        }

        # Get mask indices
        mask_coords = np.where(mask > 0)
        n_voxels = len(mask_coords[0])

        # Estimate bolus duration once from AIF for CPI
        bolus_duration_frames = estimate_bolus_duration(aif)

        # Build voxel work items
        voxel_data_list: list[_DSCVoxelInput] = []
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
        init_args = (
            aif, time_points, method, bayesian, mcmc_samples,
            cpi_control_points, bolus_duration_frames, dt,
        )

        n_procs = max(1, int(self.inputs.n_procs))

        if n_procs > 1:
            from concurrent.futures import ProcessPoolExecutor

            chunk_size = max(1, n_voxels // (n_procs * 10))
            with ProcessPoolExecutor(
                max_workers=n_procs,
                initializer=_dsc_init_worker,
                initargs=init_args,
            ) as executor:
                results = list(executor.map(
                    _dsc_fit_single_voxel, voxel_data_list,
                    chunksize=chunk_size,
                ))
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

            if res.get("r_squared") is not None:
                r_squared_list.append(res["r_squared"])
            if res.get("converged"):
                n_converged += 1

        # Compile QC metrics
        if r_squared_list:
            qc_metrics["r_squared_mean"] = float(np.mean(r_squared_list))
            qc_metrics["r_squared_std"] = float(np.std(r_squared_list))

        if bayesian != "none":
            qc_metrics["n_converged"] = n_converged
            qc_metrics["convergence_rate"] = n_converged / max(1, n_voxels)

        return cbf, cbv, mtt, tmax, qc_metrics

    def _run_interface(self, runtime: Any) -> Any:
        """Execute DSC map computation."""
        import nibabel as nib  # type: ignore[import-untyped]

        # Load data
        conc_img = nib.load(self.inputs.concentration_file)
        conc_data: npt.NDArray[Any] = conc_img.get_fdata()  # type: ignore[attr-defined]
        affine = conc_img.affine  # type: ignore[attr-defined]
        header = conc_img.header  # type: ignore[attr-defined]

        mask_img = nib.load(self.inputs.brain_mask_file)
        mask: npt.NDArray[Any] = mask_img.get_fdata()  # type: ignore[attr-defined]

        # Verify brain mask is in DSC space (guaranteed by transform_brain_mask node)
        if mask.shape[:3] != conc_data.shape[:3]:
            raise RuntimeError(
                f"Brain mask shape {mask.shape[:3]} does not match "
                f"concentration data shape {conc_data.shape[:3]}. "
                "The brain mask must be in native DSC space. "
                "Ensure transform_brain_mask node is connected upstream."
            )

        # Load AIF
        aif = np.loadtxt(self.inputs.aif_file)

        # Load signal for PSR/PH if provided
        signal_data: Optional[npt.NDArray[Any]] = None
        if self.inputs.signal_file:
            signal_img = nib.load(self.inputs.signal_file)
            signal_data = signal_img.get_fdata()  # type: ignore[attr-defined]

        # Parameters
        method = str(self.inputs.method)
        dt = float(self.inputs.tr)
        threshold = float(self.inputs.svd_threshold)
        oi_threshold = float(self.inputs.oi_threshold)
        cpi_control_points = int(self.inputs.cpi_control_points)
        bayesian = str(self.inputs.bayesian)
        mcmc_samples = int(self.inputs.mcmc_samples)
        use_gpu = bool(self.inputs.use_gpu)
        baseline_frames = int(self.inputs.baseline_frames)
        tail_frames = int(self.inputs.tail_frames)

        # Auto-detect baseline frames from AIF (matches perf-plus)
        detected_baseline, baseline_detection_info = self._detect_baseline_frames(
            aif, baseline_frames, concentration_data=conc_data, mask=mask
        )

        te = float(self.inputs.te)

        # Match perf-plus: re-compute S0/concentration with detected baseline
        # when it differs from configured baseline (perf-plus lines 5177-5205)
        recomputed_raw_conc: Optional[npt.NDArray[Any]] = None
        if detected_baseline != baseline_frames and signal_data is not None:
            new_s0 = np.mean(
                signal_data[..., :detected_baseline], axis=-1, keepdims=True
            )
            new_s0 = np.maximum(new_s0, EPSILON_SIGNAL)
            signal_ratio = signal_data / new_s0
            signal_ratio = np.clip(signal_ratio, EPSILON_SIGNAL, 1.0)
            recomputed_raw_conc = -np.log(signal_ratio) / te

            # Re-extract AIF from updated concentration
            # Match perf-plus: use stored best candidate coordinate (cost-function
            # selected) rather than scanning mask for highest peak
            aif_reextracted = False
            if isdefined(self.inputs.aif_best_coord) and len(self.inputs.aif_best_coord) == 3:
                cx, cy, cz = self.inputs.aif_best_coord
                if (0 <= cx < recomputed_raw_conc.shape[0] and
                        0 <= cy < recomputed_raw_conc.shape[1] and
                        0 <= cz < recomputed_raw_conc.shape[2]):
                    aif = recomputed_raw_conc[cx, cy, cz, :]
                    aif_reextracted = True
            if not aif_reextracted and isdefined(self.inputs.aif_mask_file):
                aif_mask_img = nib.load(self.inputs.aif_mask_file)
                aif_mask_data = aif_mask_img.get_fdata() > 0.5  # type: ignore[attr-defined]
                if aif_mask_data.shape[:3] == recomputed_raw_conc.shape[:3]:
                    aif_voxels = recomputed_raw_conc[aif_mask_data]
                    if aif_voxels.ndim == 2 and aif_voxels.shape[0] > 0:
                        peak_idx = int(np.argmax(np.max(aif_voxels, axis=1)))
                        aif = aif_voxels[peak_idx, :]

            # Re-apply K2 correction to updated concentration
            from pmriprep.utils.dsc_preload import apply_k2_correction

            k2_result = apply_k2_correction(
                concentration=recomputed_raw_conc,
                aif=aif,
                brain_mask=mask.astype(bool),
                tr=dt,
                method=str(self.inputs.k2_method),
                baseline_frames=detected_baseline,
            )
            conc_data = k2_result.concentration_corrected

            baseline_detection_info["s0_recomputed"] = True
            baseline_detection_info["s0_recomputed_frames"] = detected_baseline
            baseline_detection_info["k2_recomputed"] = True

        spatial_shape = conc_data.shape[:3]

        # Save K2 map: use recomputed K2 when baseline changed, else pass through
        k2_map_out_path: Optional[str] = None
        if detected_baseline != baseline_frames and recomputed_raw_conc is not None:
            # K2 was recomputed with detected baseline — save it
            k2_map_data = k2_result.k2_map  # type: ignore[possibly-undefined]
            k2_map_out_path = self._gen_outfilename("_k2")
            k2_img = nib.Nifti1Image(  # type: ignore[attr-defined]
                k2_map_data.astype(np.float32), affine, header
            )
            nib.save(k2_img, k2_map_out_path)  # type: ignore[attr-defined]
        elif isdefined(self.inputs.k2_map_input_file) and self.inputs.k2_map_input_file:
            # No recomputation — pass through the preload_correct K2 map
            k2_map_out_path = str(self.inputs.k2_map_input_file)

        # Run deconvolution based on method
        delay: Optional[npt.NDArray[Any]] = None
        residues: Optional[npt.NDArray[Any]] = None
        qc_metrics: dict[str, Any] = {}

        if method in ("sSVD", "cSVD", "oSVD", "tikhonov"):
            cbf, cbv, mtt, tmax, delay, residues, qc_metrics = self._deconvolve_svd(
                conc_data, aif, mask, method, dt, threshold, oi_threshold, use_gpu
            )
        else:  # vascular or cpi
            cbf, cbv, mtt, tmax, qc_metrics = self._deconvolve_parametric(
                conc_data, aif, mask, method, dt, bayesian, mcmc_samples, cpi_control_points
            )

        # Compute PSR and PH
        psr: npt.NDArray[Any]
        ph: npt.NDArray[Any]
        if signal_data is not None:
            # Use raw signal for PSR/PH
            psr_result = compute_psr_legacy(signal_data, detected_baseline, tail_frames)
            ph_result = compute_peak_height_legacy(signal_data, detected_baseline)
            # Convert to ndarray if scalar (shouldn't happen with 4D input)
            if isinstance(psr_result, float):
                psr = np.full(spatial_shape, psr_result, dtype=np.float64)
            else:
                psr = psr_result
            if isinstance(ph_result, float):
                ph = np.full(spatial_shape, ph_result, dtype=np.float64)
            else:
                ph = ph_result
        else:
            # Use concentration (note: interpretation differs)
            # For concentration, PSR/PH are computed from inverted signal
            psr = np.zeros(spatial_shape, dtype=np.float64)
            ph = np.zeros(spatial_shape, dtype=np.float64)
            # Note: PSR/PH from concentration requires different formula
            # For now, output zeros - proper implementation needs raw signal
            qc_metrics["psr_ph_from_signal"] = False

        # -----------------------------------------------------------
        # Load raw (pre-K2) concentration if available
        # Used for AUC, TTP, and CNR (these reflect first-pass bolus
        # timing, not BBB permeability from K2 correction)
        # -----------------------------------------------------------
        raw_conc_data: npt.NDArray[Any]
        if recomputed_raw_conc is not None:
            # S0 was re-computed with detected baseline — use that
            raw_conc_data = recomputed_raw_conc
        elif self.inputs.raw_concentration_file:
            raw_conc_img = nib.load(self.inputs.raw_concentration_file)
            raw_conc_data = raw_conc_img.get_fdata()  # type: ignore[attr-defined]
        else:
            # No K2 correction was applied — raw == corrected
            raw_conc_data = conc_data

        # Compute TTP from raw concentration (post-bolus argmax)
        ttp = compute_ttp_legacy(raw_conc_data, detected_baseline, dt)

        # -----------------------------------------------------------
        # AUC decomposition: raw (pre-K2), corrected (post-K2), diff
        # -----------------------------------------------------------
        # Rectangular integration (np.sum * dt) matches perf-plus ground
        # truth and is algebraically consistent with the rectangular
        # Toeplitz matrix used for deconvolution CBV (Ostergaard 1996).
        # For future absolute quantification, trapezoidal (O(dt^2)) can
        # be used instead:
        #   auc_corr = np.trapezoid(conc_data, dx=dt, axis=-1)
        auc_corr = np.sum(conc_data, axis=-1) * dt
        auc_raw = np.sum(raw_conc_data, axis=-1) * dt
        auc_diff = auc_raw - auc_corr

        # CNR: peak concentration / baseline std per voxel (from raw conc)
        # perf-plus uses ddof=0 (default) and full-timeseries peak
        baseline_std = np.std(raw_conc_data[..., :detected_baseline], axis=-1)
        peak_conc = np.max(raw_conc_data, axis=-1)
        cnr = np.where(
            baseline_std > EPSILON_CONCENTRATION,
            peak_conc / baseline_std,
            0.0,
        )

        # -----------------------------------------------------------
        # SNR / tSNR (from raw signal, not concentration)
        # Use auto-detected baseline frames from AIF bolus arrival
        # -----------------------------------------------------------
        n_timepoints = conc_data.shape[-1]
        snr_baseline_frames = min(detected_baseline, n_timepoints - 1)
        if signal_data is not None:
            baseline_sig = signal_data[..., :snr_baseline_frames]
            baseline_mean = np.mean(baseline_sig, axis=-1)
            baseline_std_sig = np.std(baseline_sig, axis=-1, ddof=1)
            snr = np.where(baseline_std_sig > EPSILON_SIGNAL, baseline_mean / baseline_std_sig, 0.0)

            all_mean = np.mean(signal_data, axis=-1)
            all_std = np.std(signal_data, axis=-1, ddof=1)
            tsnr = np.where(all_std > EPSILON_SIGNAL, all_mean / all_std, 0.0)
        else:
            snr = np.zeros(spatial_shape, dtype=np.float64)
            tsnr = np.zeros(spatial_shape, dtype=np.float64)

        # -----------------------------------------------------------
        # Clip maps to physiological ranges (matches perf-plus)
        # -----------------------------------------------------------
        psr = np.clip(psr, 0.0, 200.0)
        ph = np.clip(ph, 0.0, 100.0)
        tmax = np.clip(tmax, 0.0, 60.0)
        mtt = np.clip(mtt, 0.0, 30.0)
        cbf = np.clip(cbf, 0.0, 100.0)
        cbv = np.clip(cbv, 0.0, 10.0)
        cnr = np.clip(cnr, 0.0, 1000.0)

        # -----------------------------------------------------------
        # Relative maps (always produced)
        # -----------------------------------------------------------
        rel_cbf = cbf
        rel_cbv = cbv
        rel_auc_raw = auc_raw
        rel_auc_corr = auc_corr

        # -----------------------------------------------------------
        # Auto-absolute gating
        # -----------------------------------------------------------
        field_strength = float(self.inputs.field_strength) if self.inputs.field_strength else 0.0
        aif_level = str(self.inputs.aif_detection_level)
        preload_class = str(self.inputs.preload_classification)

        absolute_permitted = (
            field_strength > 0.0
            and aif_level != "synthetic"
            and preload_class in ("none", "minimal")
        )

        abs_cbf: Optional[npt.NDArray[Any]] = None
        abs_cbv: Optional[npt.NDArray[Any]] = None
        abs_auc_raw: Optional[npt.NDArray[Any]] = None
        abs_auc_corr: Optional[npt.NDArray[Any]] = None

        if absolute_permitted:
            k_h = calculate_k_h()
            vessel_cal = VESSEL_CALIBRATION_GRE_3T
            r2_star = get_r2_star(field_strength)

            abs_cbf = rel_cbf * k_h * vessel_cal * 60.0   # mL/100g/min
            abs_cbv = rel_cbv * k_h * vessel_cal           # mL/100g
            abs_auc_raw = auc_raw / r2_star                # mM·s
            abs_auc_corr = auc_corr / r2_star              # mM·s

        # -----------------------------------------------------------
        # Mask all outputs
        # -----------------------------------------------------------
        mask_bool = mask > 0
        rel_cbf = np.where(mask_bool, rel_cbf, 0.0)
        rel_cbv = np.where(mask_bool, rel_cbv, 0.0)
        mtt = np.where(mask_bool, mtt, 0.0)
        tmax = np.where(mask_bool, tmax, 0.0)
        ttp = np.where(mask_bool, ttp, 0.0)
        auc_raw = np.where(mask_bool, auc_raw, 0.0)
        auc_corr = np.where(mask_bool, auc_corr, 0.0)
        auc_diff = np.where(mask_bool, auc_diff, 0.0)
        rel_auc_raw = np.where(mask_bool, rel_auc_raw, 0.0)
        rel_auc_corr = np.where(mask_bool, rel_auc_corr, 0.0)
        cnr = np.where(mask_bool, cnr, 0.0)
        snr = np.where(mask_bool, snr, 0.0)
        tsnr = np.where(mask_bool, tsnr, 0.0)
        psr = np.where(mask_bool, psr, 0.0)
        ph = np.where(mask_bool, ph, 0.0)

        if absolute_permitted:
            abs_cbf = np.where(mask_bool, abs_cbf, 0.0)  # type: ignore[arg-type]
            abs_cbv = np.where(mask_bool, abs_cbv, 0.0)  # type: ignore[arg-type]
            abs_auc_raw = np.where(mask_bool, abs_auc_raw, 0.0)  # type: ignore[arg-type]
            abs_auc_corr = np.where(mask_bool, abs_auc_corr, 0.0)  # type: ignore[arg-type]

        # Add general QC metrics
        qc_metrics.update({
            "cbf_mean": float(np.mean(rel_cbf[mask_bool])) if np.any(mask_bool) else 0.0,
            "cbf_std": float(np.std(rel_cbf[mask_bool])) if np.any(mask_bool) else 0.0,
            "cbv_mean": float(np.mean(rel_cbv[mask_bool])) if np.any(mask_bool) else 0.0,
            "mtt_mean": float(np.mean(mtt[mask_bool])) if np.any(mask_bool) else 0.0,
            "tmax_mean": float(np.mean(tmax[mask_bool])) if np.any(mask_bool) else 0.0,
            "n_voxels": int(np.sum(mask_bool)),
            "absolute_permitted": absolute_permitted,
            "aif_detection_level": aif_level,
            "preload_classification": preload_class,
        })
        qc_metrics.update(baseline_detection_info)

        # -----------------------------------------------------------
        # Helper to save a 3D array as NIfTI
        # -----------------------------------------------------------
        def _save_map(data: npt.NDArray[Any], suffix: str) -> str:
            path = self._gen_outfilename(suffix)
            img = nib.Nifti1Image(data.astype(np.float32), affine, header)  # type: ignore[attr-defined]
            nib.save(img, path)  # type: ignore[attr-defined]
            return path

        # -----------------------------------------------------------
        # Save all output maps
        # -----------------------------------------------------------
        # Quantification-independent
        out_mtt = _save_map(mtt, "_mtt")
        out_tmax = _save_map(tmax, "_tmax")
        out_psr = _save_map(psr, "_psr")
        out_ph = _save_map(ph, "_ph")
        out_ttp = _save_map(ttp, "_ttp")
        out_cnr = _save_map(cnr, "_cnr")

        # AUC decomposition
        out_auc_raw = _save_map(auc_raw, "_auc_raw")
        out_auc_corr = _save_map(auc_corr, "_auc_corr")
        out_auc_diff = _save_map(auc_diff, "_auc_diff")

        # Relative maps
        out_rel_cbf = _save_map(rel_cbf, "_rel_cbf")
        out_rel_cbv = _save_map(rel_cbv, "_rel_cbv")
        out_rel_auc_raw = _save_map(rel_auc_raw, "_rel_auc_raw")
        out_rel_auc_corr = _save_map(rel_auc_corr, "_rel_auc_corr")

        # Quality metrics
        out_snr = _save_map(snr, "_snr")
        out_tsnr = _save_map(tsnr, "_tsnr")

        # Store results
        self._results: dict[str, Any] = {
            # Quantification-independent
            "out_mtt": out_mtt,
            "out_tmax": out_tmax,
            "out_psr": out_psr,
            "out_ph": out_ph,
            "out_ttp": out_ttp,
            "out_cnr": out_cnr,
            # AUC decomposition
            "out_auc_raw": out_auc_raw,
            "out_auc_corr": out_auc_corr,
            "out_auc_diff": out_auc_diff,
            # Relative maps
            "out_rel_cbf": out_rel_cbf,
            "out_rel_cbv": out_rel_cbv,
            "out_rel_auc_raw": out_rel_auc_raw,
            "out_rel_auc_corr": out_rel_auc_corr,
            # Quality metrics
            "out_snr": out_snr,
            "out_tsnr": out_tsnr,
            # Gating
            "absolute_permitted": absolute_permitted,
            "qc_metrics": qc_metrics,
        }

        # K2 map (recomputed or pass-through)
        if k2_map_out_path is not None:
            self._results["out_k2"] = k2_map_out_path

        # Absolute maps (only if gating passes)
        if absolute_permitted:
            self._results["out_abs_cbf"] = _save_map(abs_cbf, "_abs_cbf")  # type: ignore[arg-type]
            self._results["out_abs_cbv"] = _save_map(abs_cbv, "_abs_cbv")  # type: ignore[arg-type]
            self._results["out_abs_auc_raw"] = _save_map(abs_auc_raw, "_abs_auc_raw")  # type: ignore[arg-type]
            self._results["out_abs_auc_corr"] = _save_map(abs_auc_corr, "_abs_auc_corr")  # type: ignore[arg-type]

        # Save delay map if cSVD
        if delay is not None:
            self._results["out_delay"] = _save_map(delay, "_delay")

        # Save residues if requested
        if residues is not None and self.inputs.save_residue:
            out_residues = self._gen_outfilename("_residues")
            residues_img = nib.Nifti1Image(residues.astype(np.float32), affine)  # type: ignore[attr-defined]
            nib.save(residues_img, out_residues)  # type: ignore[attr-defined]
            self._results["out_residues"] = out_residues

        return runtime

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List interface outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]

        if hasattr(self, "_results"):
            outputs.update(self._results)

        return outputs
