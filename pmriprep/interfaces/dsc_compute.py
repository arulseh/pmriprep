"""DSC map computation interface for perfusion quantification.

This module implements the main Nipype interface for computing DSC
perfusion parameter maps (CBF, CBV, MTT, Tmax, PSR, PH). CPI fits original
signal; other deconvolution methods use concentration curves and an AIF.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, cast

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
    DEFAULT_OI_THRESHOLD,
    DEFAULT_SVD_THRESHOLD,
    EPSILON_CONCENTRATION,
    EPSILON_SIGNAL,
)
from pmriprep.utils.dsc_signal import compute_psr_legacy, compute_ttp_legacy

if TYPE_CHECKING:
    from pmriprep.models.dsc.signal_cpi.maps import CPIMaps

logger = logging.getLogger("nipype.interface")


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
    cpi_infer_delay: bool
    inference: str
    baseline_frames: int
    tail_frames: int
    leakage_method: str


class _DSCMotionWeighting(NamedTuple):
    """Frame weights and matching QC state for parametric DSC fitting."""

    weights: Optional[npt.NDArray[np.floating]]
    qc: dict[str, Any]


class _DSCBaselinePreparation(NamedTuple):
    """Finalized baseline-adjusted DSC state."""

    concentration: npt.NDArray[Any]
    aif: npt.NDArray[Any]
    detected_frames: int
    n_baseline: int
    detection_info: dict[str, Any]
    k2_map_path: Optional[str]


class _DSCSignalMetrics(NamedTuple):
    """Signal-derived DSC maps and their QC provenance."""

    psr: npt.NDArray[Any]
    qc: dict[str, Any]


class _DSCTimingMaps(NamedTuple):
    """AIF-arrival-referenced DSC tissue time-to-peak map."""

    ttp: npt.NDArray[Any]
    aif_bat_seconds: float


class _DSCAUCMaps(NamedTuple):
    """DSC AUC decomposition and measured-AIF CBV state."""

    raw: npt.NDArray[Any]
    corrected: npt.NDArray[Any]
    difference: npt.NDArray[Any]
    cbv: npt.NDArray[Any]
    qc: dict[str, Any]


class _DSCQualityMaps(NamedTuple):
    """Raw-signal CNR and baseline-tSNR maps."""

    cnr: npt.NDArray[Any]
    tsnr: npt.NDArray[Any]


class _DSCFinalMaps(NamedTuple):
    """Clipped, unit-scaled, and brain-masked DSC output maps."""

    rel_cbf: npt.NDArray[Any]
    rel_cbv: npt.NDArray[Any]
    rel_cbv_auc: npt.NDArray[Any]
    mtt: npt.NDArray[Any]
    tmax: npt.NDArray[Any]
    ttp: npt.NDArray[Any]
    auc_raw: npt.NDArray[Any]
    auc_corrected: npt.NDArray[Any]
    auc_difference: npt.NDArray[Any]
    cnr: npt.NDArray[Any]
    tsnr: npt.NDArray[Any]
    psr: npt.NDArray[Any]
    cbf_std: Optional[npt.NDArray[Any]]


class _DSCDeconvolutionMaps(NamedTuple):
    """Common map state returned by either DSC deconvolution family."""

    cbf: npt.NDArray[Any]
    cbv: npt.NDArray[Any]
    mtt: npt.NDArray[Any]
    tmax: npt.NDArray[Any]
    qc: dict[str, Any]
    cbf_std: Optional[npt.NDArray[Any]] = None
    cbv_std: Optional[npt.NDArray[Any]] = None
    parametric_r2: Optional[npt.NDArray[Any]] = None
    fallback_mask: Optional[npt.NDArray[Any]] = None
    cpi_fit: Optional["CPIMaps"] = None


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
        desc="Original 4D signal; required for CPI and used for signal-quality maps",
    )
    raw_concentration_file = File(
        exists=True,
        mandatory=False,
        desc="Pre-K2 concentration (for auc_raw when K2 correction was applied)",
    )
    aif_detection_level = traits.Str(
        "atlas",
        usedefault=True,
        desc="AIF detection level: 'atlas_strict', 'atlas_relaxed', 'brain_*', or 'synthetic'",
    )
    # Deconvolution method
    method = traits.Enum(
        "sSVD",
        "oSVD",
        "tikhonov",
        "cpi",
        usedefault=True,
        desc="Deconvolution method: sSVD, oSVD, tikhonov, or cpi",
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
    cpi_infer_delay = traits.Bool(
        True,
        usedefault=True,
        desc="Infer CPI arrival delay; false conditions it at zero and reports timing missing",
    )
    cpi_dispersion = traits.Enum("none", "auto", "gamma", usedefault=True,
                                 desc="CPI arterial dispersion selection")
    cpi_fallback = traits.Enum("osvd", "none", usedefault=True,
                               desc="Replace rejected CPI fits with oSVD or leave missing")
    te = traits.Float(desc="Echo time in seconds, required by signal-domain CPI")
    bsw_guidance_file = File(desc="Exact upstream BSW guidance archive for CPI")

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
    tail_frames = traits.Int(
        10,
        usedefault=True,
        desc="Number of tail frames for PSR calculation",
    )

    # Inference options for parametric methods (cpi)
    inference = traits.Enum(
        "vi",
        "svb",
        usedefault=True,
        desc="Inference engine for parametric deconvolution (cpi): "
             "vi (linearized VI, default) or svb (spatial VB). "
             "SVD methods ignore this option.",
    )

    # Processing options
    n_procs = traits.Int(
        1,
        usedefault=True,
        desc="Number of parallel processes for voxel fitting (1 = serial)",
    )
    fitting_memory_gb = traits.Float(
        desc="Scheduler memory allocation for CPI, used to limit fitting workers",
    )
    # Motion weighting inputs (from motion_correct)
    fd_values = traits.List(
        traits.Float,
        desc="Framewise displacement per volume (mm) from motion correction",
    )

    k2_map_input_file = File(
        mandatory=False,
        desc="K2 map from leakage_correct (passed through when baseline not recomputed)",
    )
    leakage_method = traits.Enum(
        "bsw",
        "none",
        usedefault=True,
        desc="Leakage-correction policy; only 'none' permits an absent K2 map",
    )
    enhancing_mask_file = traits.Either(
        File(exists=True),
        traits.Enum(""),
        mandatory=False,
        desc="Enhancing tumor mask in DSC space for QC voxel selection",
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
    out_tmax = File(genfile=True, desc="Output method-specific timing map NIfTI")
    out_psr = File(genfile=True, desc="Output PSR map NIfTI")
    out_cnr = File(genfile=True, desc="Output CNR map NIfTI")

    # Output files — AUC decomposition
    out_auc_raw = File(genfile=True, desc="Output AUC raw (pre-K2) map NIfTI")
    out_auc_corr = File(genfile=True, desc="Output AUC corrected (post-K2) map NIfTI")
    out_auc_diff = File(genfile=True, desc="Output AUC diff (leakage) map NIfTI")

    # Output files — relative maps (always produced)
    out_rel_cbf = File(genfile=True, desc="Output relative CBF map NIfTI")
    out_rel_cbv = File(genfile=True, desc="Output relative CBV map NIfTI (deconvolution)")
    out_rel_cbv_auc = File(genfile=True, desc="Output AUC-based relative CBV map NIfTI")

    # Output files — quality metrics
    out_tsnr = File(genfile=True, desc="Output baseline tSNR map NIfTI")


class ComputeDSCMapsOutputSpec(TraitedSpec):
    """Output specification for ComputeDSCMaps interface."""

    # Quantification-independent (always produced)
    out_mtt = File(exists=True, desc="MTT map (mean transit time)")
    out_tmax = File(exists=True, desc="Method-specific deconvolution timing map")
    out_psr = File(exists=True, desc="PSR map (percent signal recovery)")
    out_ttp = File(
        exists=True,
        desc="Tissue time to peak relative to AIF bolus arrival",
    )
    out_cnr = File(exists=True, desc="CNR map (contrast-to-noise ratio)")

    # AUC decomposition (always produced)
    out_auc_raw = File(exists=True, desc="AUC raw (pre-K2) map")
    out_auc_corr = File(exists=True, desc="AUC corrected (post-K2) map")
    out_auc_diff = File(exists=True, desc="AUC diff (leakage contribution) map")

    # Relative maps (always produced)
    out_rel_cbf = File(exists=True, desc="Relative CBF map")
    out_rel_cbv = File(exists=True, desc="Relative CBV map (deconvolution)")
    out_rel_cbv_auc = File(exists=True, desc="AUC-based relative CBV map")

    # Quality metrics
    out_tsnr = File(exists=True, desc="Baseline tSNR map")

    # K2 leakage map (recomputed or pass-through)
    out_k2 = File(desc="K2 leakage correction map (from recomputed or pass-through)")

    # QC fit data (all deconvolution methods)
    out_qc_fit_data = File(desc="JSON with QC voxel fit data for fit QC plot")

    # Optional extras
    out_cbf_std = File(desc="CBF parameter standard-error or posterior-SD map")
    out_parametric_r2 = File(
        desc="Per-voxel R² of parametric fit (CPI only). "
        "Raw value preserved on fallback voxels; see out_fallback_mask."
    )
    out_fallback_mask = File(
        desc="Binary mask: 1 where parametric fit fell back to oSVD "
        "(CPI only)"
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
    - tikhonov: Morozov-selected first-order Tikhonov regularization
    - cpi: Control Point Interpolation model

    Output maps:
    - CBF: Cerebral blood flow (max of residue function)
    - CBV: Cerebral blood volume (integral of residue function)
    - MTT: Mean transit time (CBV/CBF, central volume theorem)
    - Tmax: Time to maximum of residue function
    - PSR: Percent signal recovery (BBB integrity indicator)

    Parameters
    ----------
    concentration_file : str
        4D NIfTI with concentration time courses.
    aif_file : str
        1D text file with AIF concentration values.
    brain_mask_file : str
        3D brain mask NIfTI.
    signal_file : str, optional
        4D raw signal for PSR.
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
        Inference engine for parametric methods (cpi):
        vi or svb. Default: 'vi'.

    Returns
    -------
    out_rel_cbf, out_rel_cbv, out_rel_cbv_auc : str
        Paths to relative perfusion maps (always produced).
    out_mtt, out_tmax, out_psr, out_ttp, out_cnr : str
        Paths to quantification-independent maps.
    out_auc_raw, out_auc_corr, out_auc_diff : str
        Paths to AUC decomposition maps.
    out_tsnr : str
        Path to the baseline temporal-SNR quality map.
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

    For CPI with VI or spatial VB, uncertainty maps are also
    generated when method is 'cpi'.

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
            cpi_infer_delay=bool(self.inputs.cpi_infer_delay),
            inference=str(self.inputs.inference),
            baseline_frames=int(self.inputs.baseline_frames),
            tail_frames=int(self.inputs.tail_frames),
            leakage_method=str(self.inputs.leakage_method),
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
        """Center arrays using the baseline finalized by AIF detection.

        Parameters
        ----------
        inputs : _DSCComputeInputs
            Validated DSC image arrays and acquisition settings.

        Returns
        -------
        _DSCBaselinePreparation
            Baseline-subtracted fitting arrays and upstream K2 provenance.

        """
        concentration = inputs.concentration
        aif = inputs.aif
        detected_frames = inputs.baseline_frames
        detection_info: dict[str, Any] = {
            "baseline_detection_method": "aif_detection_finalized",
            "baseline_frames_detected": detected_frames,
            "arrival_index": detected_frames,
        }
        k2_map_path: Optional[str] = None
        if (
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
            float(np.median(concentration_offset[inputs.mask_bool]))
            if np.any(inputs.mask_bool) else float("nan"), 4
        )

        return _DSCBaselinePreparation(
            concentration=concentration,
            aif=aif,
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
        bolus_frame: int,
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
        bolus_frame : int
            Final detected pre-bolus frame count used by DSC fitting.

        Returns
        -------
        _DSCMotionWeighting
            Optional fitting weights and QC values describing that exact array.

        """
        if (
            method != "cpi"
            or not isdefined(self.inputs.fd_values)
            or not self.inputs.fd_values
        ):
            return _DSCMotionWeighting(None, {"applied": False})

        voxel_dims = header.get_zooms()[:3]
        min_voxel_dim = float(min(voxel_dims))
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
        brain_mask: npt.NDArray[np.bool_],
    ) -> _DSCSignalMetrics:
        """Compute PSR from raw, non-leakage-corrected DSC signal.

        Parameters
        ----------
        signal : npt.NDArray, optional
            Raw DSC signal array. Concentration is intentionally not used as a
            substitute because PSR is defined on the signal curve.
        spatial_shape : tuple of int
            Output map shape.
        baseline_frames : int
            Detected pre-bolus frame count.
        tail_frames : int
            Number of recovery frames used for PSR.

        Returns
        -------
        _DSCSignalMetrics
            PSR array plus availability and resolved-window QC state.

        """
        if signal is None:
            return _DSCSignalMetrics(
                psr=np.full(spatial_shape, np.nan, dtype=np.float64),
                qc={"psr_from_signal": False, "psr_valid_fraction": 0.0},
            )

        psr_result = compute_psr_legacy(
            signal,
            baseline_frames,
            tail_frames,
            brain_mask=brain_mask,
            min_drop_noise_sd=5.0,
        )
        psr = (
            np.full(spatial_shape, psr_result, dtype=np.float64)
            if isinstance(psr_result, float)
            else psr_result
        )
        n_brain = int(np.sum(brain_mask))
        n_valid = int(np.sum(brain_mask & np.isfinite(psr)))
        return _DSCSignalMetrics(
            psr=psr,
            qc={
                "psr_from_signal": True,
                "psr_baseline_frames": baseline_frames,
                "psr_recovery_tail_frames": tail_frames,
                "psr_min_drop_noise_sd": 5.0,
                "psr_valid_fraction": n_valid / max(n_brain, 1),
            },
        )

    def _select_raw_concentration(
        self,
        corrected: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        """Select pre-K2 concentration for model-free DSC measurements.

        Parameters
        ----------
        corrected : npt.NDArray
            Baseline-subtracted, potentially K2-corrected concentration.

        Returns
        -------
        npt.NDArray
            Configured raw input or corrected concentration.

        """
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
            AIF-arrival-referenced tissue TTP and the AIF arrival time in seconds.

        """
        ttp_raw = compute_ttp_legacy(raw_concentration, detected_baseline, dt)
        aif_bat_frames = detection_info.get("arrival_index", detected_baseline)
        aif_bat_seconds = float(aif_bat_frames) * dt
        ttp = ttp_raw - aif_bat_seconds

        n_negative = int(np.sum(mask_bool & (ttp < 0)))
        n_mask = int(np.sum(mask_bool))
        if n_mask > 0 and n_negative / n_mask > 0.05:
            logger.warning(
                "%.1f%% of brain voxels have negative TTP (ttp < aif_bat=%.1fs). "
                "Check AIF QC.",
                100.0 * n_negative / n_mask,
                aif_bat_seconds,
            )

        ttp = np.where(
            np.isfinite(ttp_raw) & (ttp >= 0.0),
            ttp,
            np.nan,
        )

        return _DSCTimingMaps(
            ttp=ttp,
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
        auc_corrected = np.trapezoid(
            corrected_concentration, dx=dt, axis=-1
        )
        auc_raw = np.trapezoid(raw_for_auc, dx=dt, axis=-1)
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
                cbv = tissue_auc_bolus / aif_auc
            qc.update(
                cbv_auc_bat_frame=bat,
                cbv_auc_end_frame=n_timepoints,
                cbv_auc_aif_integral=round(aif_auc, 6),
            )
        else:
            logger.warning("Measured AIF not available; AUC CBV set to NaN")
            cbv = np.full(spatial_shape, np.nan, dtype=np.float64)
            qc["cbv_auc_measured_aif_missing"] = True

        return _DSCAUCMaps(
            raw=auc_raw,
            corrected=auc_corrected,
            difference=auc_difference,
            cbv=cbv,
            qc=qc,
        )

    def _compute_quality_maps(
        self,
        signal: Optional[npt.NDArray[Any]],
        detected_baseline: int,
        spatial_shape: tuple[int, ...],
        brain_mask: npt.NDArray[np.bool_],
    ) -> _DSCQualityMaps:
        """Compute raw-signal CNR and baseline temporal SNR.

        Parameters
        ----------
        signal : npt.NDArray, optional
            Raw DSC signal used for CNR and baseline tSNR.
        detected_baseline : int
            Detected pre-bolus frame count.
        spatial_shape : tuple of int
            Output map shape when raw signal is unavailable.
        brain_mask : np.ndarray
            Brain domain used to pool the CNR noise floor.

        Returns
        -------
        _DSCQualityMaps
            Unclipped CNR and baseline-tSNR maps.

        """
        if signal is None:
            undefined = np.full(spatial_shape, np.nan, dtype=np.float64)
            return _DSCQualityMaps(cnr=undefined, tsnr=undefined.copy())

        baseline_frames = min(detected_baseline, signal.shape[-1] - 1)
        if baseline_frames < 2:
            undefined = np.full(spatial_shape, np.nan, dtype=np.float64)
            return _DSCQualityMaps(cnr=undefined, tsnr=undefined.copy())

        baseline_signal = signal[..., :baseline_frames]
        baseline_mean = np.mean(baseline_signal, axis=-1)
        baseline_noise = np.std(baseline_signal, axis=-1, ddof=1)
        post_baseline_minimum = np.min(signal[..., baseline_frames:], axis=-1)
        signal_drop = baseline_mean - post_baseline_minimum
        pool_values = baseline_noise[
            brain_mask & np.isfinite(baseline_noise) & (baseline_noise > EPSILON_SIGNAL)
        ]
        pooled_noise = float(np.median(pool_values)) if pool_values.size else np.nan
        effective_noise = np.maximum(baseline_noise, pooled_noise)
        valid_cnr = (
            np.isfinite(signal_drop)
            & np.isfinite(effective_noise)
            & (signal_drop > 0.0)
            & (effective_noise > EPSILON_SIGNAL)
        )
        valid_tsnr = (
            np.isfinite(baseline_mean)
            & np.isfinite(baseline_noise)
            & (baseline_mean > EPSILON_SIGNAL)
            & (baseline_noise > EPSILON_SIGNAL)
        )
        cnr = np.full(spatial_shape, np.nan, dtype=np.float64)
        tsnr = np.full(spatial_shape, np.nan, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(signal_drop, effective_noise, out=cnr, where=valid_cnr)
            np.divide(baseline_mean, baseline_noise, out=tsnr, where=valid_tsnr)
        return _DSCQualityMaps(cnr=cnr, tsnr=tsnr)

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
                np.nan,
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
        baseline_info: dict[str, Any],
        aif_bat_seconds: float,
        cbv_auc_qc_source: Optional[npt.NDArray[Any]] = None,
        method: str = "oSVD",
    ) -> dict[str, Any]:
        """Reduce finalized DSC maps and acquisition provenance to QC values."""
        def _finite_stat(
            data: npt.NDArray[Any],
            reducer: Any,
        ) -> float:
            values = data[mask_bool]
            values = values[np.isfinite(values)]
            return float(reducer(values)) if values.size else np.nan

        cbv_auc_for_qc = (
            rel_cbv_auc if cbv_auc_qc_source is None else cbv_auc_qc_source
        )
        finite_cbv_auc = mask_bool & np.isfinite(cbv_auc_for_qc)
        n_finite_cbv_auc = int(np.sum(finite_cbv_auc))
        n_negative_cbv_auc = int(
            np.sum(finite_cbv_auc & (cbv_auc_for_qc < 0.0))
        )
        timing_name = (
            "arrival_delay" if method == "cpi" else "tmax"
        )
        n_voxels = int(np.sum(mask_bool))
        timing_values = tmax[mask_bool]
        timing_values = timing_values[np.isfinite(timing_values)]
        n_finite_timing = int(timing_values.size)
        qc = {
            "cbf_mean": _finite_stat(rel_cbf, np.mean),
            "cbf_std": _finite_stat(rel_cbf, np.std),
            "cbv_mean": _finite_stat(rel_cbv, np.mean),
            "cbv_auc_mean": _finite_stat(rel_cbv_auc, np.mean),
            "mtt_mean": _finite_stat(mtt, np.mean),
            "cbf_median": _finite_stat(rel_cbf, np.median),
            "cbv_median": _finite_stat(rel_cbv, np.median),
            "cbv_auc_median": _finite_stat(rel_cbv_auc, np.median),
            "cbv_auc_finite_voxels": n_finite_cbv_auc,
            "cbv_auc_negative_voxels": n_negative_cbv_auc,
            "cbv_auc_negative_fraction": (
                n_negative_cbv_auc / n_finite_cbv_auc
                if n_finite_cbv_auc > 0
                else np.nan
            ),
            "mtt_median": _finite_stat(mtt, np.median),
            "n_voxels": n_voxels,
            "aif_detection_level": aif_level,
        }
        if timing_name == "tmax" or n_finite_timing > 0:
            qc[f"{timing_name}_mean"] = _finite_stat(tmax, np.mean)
            qc[f"{timing_name}_median"] = _finite_stat(tmax, np.median)
        if timing_name == "arrival_delay":
            qc["arrival_delay_finite_voxels"] = n_finite_timing
            qc["arrival_delay_finite_fraction"] = (
                n_finite_timing / n_voxels if n_voxels > 0 else np.nan
            )
        qc.update(baseline_info)
        qc["aif_bat_seconds"] = aif_bat_seconds
        qc["cbv_literature_range_applicable"] = aif_level != "synthetic"
        return qc

    @staticmethod
    def _scale_cbf_uncertainty(
        standard_deviation: Optional[npt.NDArray[Any]],
    ) -> Optional[npt.NDArray[Any]]:
        """Convert an optional CBF uncertainty map from inverse seconds to minutes."""
        return None if standard_deviation is None else standard_deviation * 60.0

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
        auc_raw: npt.NDArray[Any],
        auc_corrected: npt.NDArray[Any],
        auc_difference: npt.NDArray[Any],
        cnr: npt.NDArray[Any],
        tsnr: npt.NDArray[Any],
        psr: npt.NDArray[Any],
        cbf_std: Optional[npt.NDArray[Any]],
        mask_bool: npt.NDArray[Any],
    ) -> _DSCFinalMaps:
        """Apply CBF units and brain masking without post-fit clipping."""
        rel_cbf = np.where(np.isfinite(cbf) & (cbf >= 0.0), cbf * 60.0, np.nan)
        rel_cbv = np.where(np.isfinite(cbv) & (cbv >= 0.0), cbv, np.nan)
        rel_cbv_auc = np.where(
            np.isfinite(cbv_auc) & (cbv_auc >= 0.0), cbv_auc, np.nan
        )
        mtt = np.where(np.isfinite(mtt) & (mtt >= 0.0), mtt, np.nan)
        cbf_std = cls._scale_cbf_uncertainty(cbf_std)

        return _DSCFinalMaps(
            rel_cbf=np.where(mask_bool, rel_cbf, 0.0),
            rel_cbv=np.where(mask_bool, rel_cbv, 0.0),
            rel_cbv_auc=np.where(mask_bool, rel_cbv_auc, 0.0),
            mtt=np.where(mask_bool, mtt, 0.0),
            tmax=np.where(mask_bool, tmax, 0.0),
            ttp=np.where(mask_bool, ttp, 0.0),
            auc_raw=np.where(mask_bool, auc_raw, 0.0),
            auc_corrected=np.where(mask_bool, auc_corrected, 0.0),
            auc_difference=np.where(mask_bool, auc_difference, 0.0),
            cnr=np.where(mask_bool, cnr, 0.0),
            tsnr=np.where(mask_bool, tsnr, 0.0),
            psr=np.where(mask_bool, psr, 0.0),
            cbf_std=cbf_std,
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
    ) -> _DSCPostprocessing:
        """Compose model-free maps, final masking, and DSC QC reduction."""
        qc = deconvolution.qc
        signal_metrics = self._compute_signal_metrics(
            signal, spatial_shape, detected_baseline, tail_frames, mask_bool
        )
        qc.update(signal_metrics.qc)

        raw_concentration = self._select_raw_concentration(concentration)
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
            signal, detected_baseline, spatial_shape, mask_bool
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
            auc_raw=auc.raw,
            auc_corrected=auc.corrected,
            auc_difference=auc.difference,
            cnr=quality.cnr,
            tsnr=quality.tsnr,
            psr=signal_metrics.psr,
            cbf_std=deconvolution.cbf_std,
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
                baseline_info,
                timing.aif_bat_seconds,
                auc.cbv,
                method,
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
            "out_ttp": (maps.ttp, "_ttp"),
            "out_cnr": (maps.cnr, "_cnr"),
            "out_auc_raw": (maps.auc_raw, "_auc_raw"),
            "out_auc_corr": (maps.auc_corrected, "_auc_corr"),
            "out_auc_diff": (maps.auc_difference, "_auc_diff"),
            "out_rel_cbf": (maps.rel_cbf, "_rel_cbf"),
            "out_rel_cbv": (maps.rel_cbv, "_rel_cbv"),
            "out_rel_cbv_auc": (maps.rel_cbv_auc, "_rel_cbv_auc"),
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
        cbf_std: Optional[npt.NDArray[Any]],
        cbv_std: Optional[npt.NDArray[Any]],
        parametric_r2: Optional[npt.NDArray[Any]],
        fallback_mask: Optional[npt.NDArray[Any]],
        mask_bool: npt.NDArray[Any],
        affine: npt.NDArray[Any],
        header: Any,
    ) -> dict[str, str]:
        """Mask and serialize optional parameter-uncertainty and diagnostic maps."""
        outputs: dict[str, str] = {}
        valid_fit = mask_bool.copy()
        if fallback_mask is not None:
            valid_fit &= fallback_mask == 0

        for name, suffix, standard_deviation in (
            ("cbf", "_cbf", cbf_std),
            ("cbv", "_cbv", cbv_std),
        ):
            if standard_deviation is None:
                continue
            masked_std = np.where(
                valid_fit,
                standard_deviation,
                np.where(mask_bool, np.nan, 0.0),
            )
            outputs[f"out_{name}_std"] = self._save_map(
                masked_std, f"{suffix}_std", affine, header
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
    ) -> _DSCDeconvolutionMaps:
        """Dispatch DSC deconvolution and normalize its typed return state."""
        if method in ("sSVD", "oSVD", "tikhonov"):
            cbf, cbv, mtt, tmax, qc = self._deconvolve_svd(
                concentration, aif, mask, method, dt, threshold, oi_threshold
            )
            return _DSCDeconvolutionMaps(
                cbf=cbf,
                cbv=cbv,
                mtt=mtt,
                tmax=tmax,
                qc=qc,
            )

        raise ValueError(f"Unknown concentration deconvolution method: {method}")

    def _deconvolve_signal_cpi(
        self, inputs: _DSCComputeInputs, baseline: _DSCBaselinePreparation,
        motion: _DSCMotionWeighting,
    ) -> _DSCDeconvolutionMaps:
        """Fit original signal with exact upstream guidance and optional replacement."""
        import nibabel as nib
        from pmriprep.models.dsc.signal_cpi.acquisition import prepare_cpi_acquisition
        from pmriprep.models.dsc.signal_cpi.maps import fit_cpi_maps
        from pmriprep.models.dsc.signal_cpi.settings import CPIOptions
        from pmriprep.utils.bsw_guidance_io import load_bsw_guidance

        if inputs.signal is None or not isdefined(self.inputs.te) or self.inputs.te <= 0:
            raise ValueError("Signal-domain CPI requires original signal and positive echo time in seconds")
        if inputs.inference not in ("vi", "svb"):
            raise ValueError("CPI supports vi or svb inference")
        for path in (self.inputs.signal_file, self.inputs.brain_mask_file, self.inputs.raw_concentration_file):
            if isdefined(path) and path:
                image: Any = nib.load(path)
                if (image.shape[:3] != inputs.concentration.shape[:3]
                        or not np.allclose(image.affine, inputs.affine, rtol=0, atol=1e-5)):
                    raise ValueError("CPI signal, concentration and mask must share the native spatial grid")
        if inputs.signal.shape != inputs.concentration.shape:
            raise ValueError("CPI signal and concentration must have identical spatial/frame dimensions")
        guidance = None
        if inputs.leakage_method == "bsw":
            if not isdefined(self.inputs.bsw_guidance_file) or not self.inputs.bsw_guidance_file:
                raise ValueError("BSW-guided CPI requires the exact upstream guidance archive")
            guidance = load_bsw_guidance(self.inputs.bsw_guidance_file, affine=inputs.affine,
                                         shape=inputs.signal.shape, tr=inputs.dt)
        options = CPIOptions(inputs.cpi_control_points, inputs.cpi_infer_delay,
                              self.inputs.cpi_dispersion, self.inputs.cpi_fallback)
        acquisition = prepare_cpi_acquisition(inputs.signal, baseline.aif,
            np.arange(inputs.signal.shape[-1], dtype=float)*inputs.dt, float(self.inputs.te), guidance,
            options=options, inference_method=inputs.inference, temporal_weights=motion.weights,
            leakage_method=self.inputs.leakage_method, mask=inputs.mask_bool,
            baseline_frames=baseline.n_baseline, sloppy=bool(self.inputs.sloppy))
        raw = self._select_raw_concentration(baseline.concentration)
        if raw.shape != inputs.signal.shape:
            raise ValueError("CPI fallback concentration must match the signal acquisition")
        raw = raw - np.mean(raw[..., :baseline.n_baseline], axis=-1, keepdims=True)
        result = fit_cpi_maps(acquisition, raw, baseline.concentration, oi_target=inputs.oi_threshold,
            prediction_file=Path.cwd() / "cpi_signal_predictions.npy",
            n_procs=int(self.inputs.n_procs),
            memory_gb=(float(self.inputs.fitting_memory_gb)
                       if isdefined(self.inputs.fitting_memory_gb) else None))
        result.qc["motion_weighting"] = motion.qc
        return _DSCDeconvolutionMaps(result.cbf, result.cbv, result.mtt, result.delay, result.qc,
            cbf_std=result.cbf_std, parametric_r2=result.r_squared, fallback_mask=result.fallback_mask,
            cpi_fit=result)

    def _save_signal_cpi_qc(
        self, cbf: npt.NDArray[Any], concentration: npt.NDArray[Any], aif: npt.NDArray[Any],
        mask: npt.NDArray[Any], dt: float, signal: npt.NDArray[Any] | None,
        result: "CPIMaps",
    ) -> None:
        """Serialize an actual selected signal fit without rerunning inference."""
        from scipy.ndimage import binary_erosion
        from pmriprep.utils.dsc_qc_plots import _select_dsc_qc_voxel
        from pmriprep.utils.sidecar_builders import dump_sidecar_json

        eligible = mask & np.isfinite(result.dispersion) & np.all(np.isfinite(result.predictions), axis=-1)
        if signal is None or not np.any(eligible):
            return
        voxel, tier = _select_dsc_qc_voxel(
            cbf, eligible, binary_erosion,
            enhancing_mask=self._load_qc_enhancing_mask(concentration.shape[:3]),
        )
        index = tuple(voxel)
        path = self._gen_outfilename("_qc_fit_data").replace(".nii.gz", ".json")
        record = dict(voxel=list(voxel), voxel_cbf=float(cbf[index]), r_squared=float(result.r_squared[index]),
            method="cpi", fit_domain="signal", inference=str(self.inputs.inference),
            dispersion="gamma" if result.dispersion[index] else "none",
            converged=bool(result.converged[index]), fitted_k2=float(result.fitted_k2[index]),
            tissue_concentration=concentration[index].tolist(), model_prediction=result.predictions[index].tolist(),
            aif=aif.tolist(), time_points=(np.arange(aif.size)*dt).tolist(), raw_signal=signal[index].tolist(),
            prediction_source="selected_voxel_posterior")
        Path(path).write_text(dump_sidecar_json(record))
        self._results["out_qc_fit_data"] = path
        self._qc_voxel_tier = tier

    def _load_qc_enhancing_mask(
        self, spatial_shape: tuple[int, ...],
    ) -> npt.NDArray[Any] | None:
        """Load optional enhancing support on the native concentration grid.

        Parameters
        ----------
        spatial_shape : tuple of int
            Shape of the native concentration image used for QC selection.

        Returns
        -------
        ndarray or None
            Native-grid mask, nearest-neighbour resampled when necessary.
        """
        import nibabel as nib

        if not isdefined(self.inputs.enhancing_mask_file) or not self.inputs.enhancing_mask_file:
            return None
        enhancing: Any = nib.load(self.inputs.enhancing_mask_file)
        concentration: Any = nib.load(self.inputs.concentration_file)
        data = enhancing.get_fdata()
        if data.shape[:3] != spatial_shape or not np.allclose(
            enhancing.affine, concentration.affine, atol=1e-3
        ):
            from nilearn.image import resample_to_img

            data = (resample_to_img(enhancing, concentration, interpolation="nearest").get_fdata() > 0.5).astype(np.float64)
        return np.asarray(data)

    def _save_fit_qc_data(
        self,
        *,
        rel_cbf: npt.NDArray[Any],
        concentration: npt.NDArray[Any],
        aif: npt.NDArray[Any],
        mask_bool: npt.NDArray[Any],
        dt: float,
        method: str,
        threshold: float,
        oi_threshold: float,
        signal: Optional[npt.NDArray[Any]],
        cpi_fit: Optional["CPIMaps"] = None,
    ) -> None:
        """Dispatch fit-QC serialization to the selected method family."""
        if method == "cpi":
            if cpi_fit is None:
                raise ValueError("CPI QC requires the selected signal-fit record")
            self._save_signal_cpi_qc(rel_cbf, concentration, aif, mask_bool, dt, signal, cpi_fit)
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
        k2_map_path: Optional[str],
        leakage_method: str,
        mask_bool: npt.NDArray[Any],
        affine: npt.NDArray[Any],
        header: Any,
    ) -> dict[str, str]:
        """Serialize optional timing/model maps and the required K2 output."""
        import nibabel as nib  # type: ignore[import-untyped]

        outputs: dict[str, str] = {}
        if k2_map_path is not None:
            k2_data = np.asarray(
                nib.load(k2_map_path).get_fdata(),  # type: ignore[attr-defined]
                dtype=np.float32,
            )
            k2_data = np.where(mask_bool, k2_data, 0.0)
            outputs["out_k2"] = self._save_map(k2_data, "_k2", affine, header)
        elif leakage_method != "none":
            raise RuntimeError(
                "K2 correction produced no coefficient map; refusing to "
                "substitute an all-zero map"
            )

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
        results = self._save_base_maps(maps, inputs.affine, inputs.header, qc)
        self._results = results

        results.update(
            self._save_ancillary_maps(
                k2_map_path=baseline.k2_map_path,
                leakage_method=inputs.leakage_method,
                mask_bool=inputs.mask_bool,
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
            threshold=inputs.svd_threshold,
            oi_threshold=inputs.oi_threshold,
            signal=inputs.signal,
            cpi_fit=deconvolution.cpi_fit,
        )
        self._add_fit_qc_metrics(maps.rel_cbv, qc)

        if inputs.method == "cpi":
            results.update(
                self._save_parametric_maps(
                    cbf_std=maps.cbf_std,
                    cbv_std=deconvolution.cbv_std,
                    parametric_r2=deconvolution.parametric_r2,
                    fallback_mask=deconvolution.fallback_mask,
                    mask_bool=inputs.mask_bool,
                    affine=inputs.affine,
                    header=inputs.header,
                )
            )
        return results

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

        elif method == "oSVD":
            res, cbf, cbv, mtt, oi_values, thresholds_used, delay = svd_deconvolve_oscillation_batch(  # type: ignore[assignment]
                concentration_data,
                aif,
                dt=dt,
                oi_target=oi_threshold,
                mask=mask_bool,
            )
            tmax = delay.copy()
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

        return cbf, cbv, mtt, tmax, qc_metrics


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
        inference = inputs.inference

        # SVD methods are direct matrix decompositions — Bayesian
        # inference only applies to parametric fitting (cpi).
        if method in ("sSVD", "oSVD", "tikhonov") and inference != "vi":
            logger.debug(
                "inference='%s' ignored for SVD method '%s'.",
                inference, method,
            )

        tail_frames = inputs.tail_frames
        baseline = self._prepare_baseline_and_leakage(inputs)
        conc_data = baseline.concentration
        aif = baseline.aif
        detected_baseline = baseline.detected_frames
        n_bl = baseline.n_baseline
        baseline_detection_info = baseline.detection_info
        spatial_shape = conc_data.shape[:3]

        motion_weighting = self._prepare_motion_weighting(
            method, header, conc_data.shape[3], dt, detected_baseline
        )
        if method == "cpi":
            deconvolution = self._deconvolve_signal_cpi(inputs, baseline, motion_weighting)
        else:
            deconvolution = self._run_deconvolution(
                concentration=conc_data,
                aif=aif,
                mask=mask,
                method=method,
                dt=dt,
                threshold=threshold,
                oi_threshold=oi_threshold,
            )
        aif_level = str(self.inputs.aif_detection_level)
        postprocessing = self._prepare_final_outputs(
            deconvolution=deconvolution,
            concentration=conc_data,
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
