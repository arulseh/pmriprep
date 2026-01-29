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
from pmriprep.utils.dsc_signal import compute_psr, compute_peak_height


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
        10,
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
        Tail frames for PSR. Default: 10.
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
        """Run parametric model deconvolution (vascular, CPI)."""
        from scipy.integrate import trapezoid

        from pmriprep.models.dsc import (
            VascularModel,
            CPIModel,
            fit_dsc_model,
            select_n_control_points,
            estimate_bolus_duration,
        )

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

        # Track convergence for Bayesian
        n_converged = 0
        r_squared_list: list[float] = []

        # Estimate bolus duration once from AIF for CPI
        bolus_duration_frames = estimate_bolus_duration(aif)

        # Process voxels
        for i in range(n_voxels):
            x, y, z = (
                int(mask_coords[0][i]),
                int(mask_coords[1][i]),
                int(mask_coords[2][i]),
            )
            tissue_curve = concentration_data[x, y, z, :]

            # Skip if curve is mostly zeros
            if np.max(np.abs(tissue_curve)) < EPSILON_CONCENTRATION:
                continue

            try:
                # Select model
                model_instance: "VascularModel | CPIModel"
                if method == "vascular":
                    model_instance = VascularModel()
                else:  # cpi
                    n_cp = cpi_control_points
                    if n_cp == 0:
                        n_cp = select_n_control_points(
                            bolus_duration_frames, dt
                        )
                    model_instance = CPIModel(n_control_points=n_cp)

                # Fit model
                if bayesian == "none":
                    # Simple least squares via Bayesian MAP with flat priors
                    result = fit_dsc_model(
                        model=model_instance,
                        concentration=tissue_curve,
                        aif=aif,
                        time_points=time_points,
                        method="map",
                    )
                else:
                    result = fit_dsc_model(
                        model=model_instance,
                        concentration=tissue_curve,
                        aif=aif,
                        time_points=time_points,
                        method=bayesian,
                        mcmc_samples=mcmc_samples,
                    )

                # Extract parameters
                cbf[x, y, z] = result.parameters.get("cbf", result.parameters.get("CBF", 0.0))

                # For vascular model, compute derived parameters
                if method == "vascular":
                    alpha = result.parameters.get("alpha", 1.0)
                    beta = result.parameters.get("beta", 1.0)
                    mtt_val = alpha * beta
                    mtt[x, y, z] = mtt_val
                    cbv[x, y, z] = cbf[x, y, z] * mtt_val / 60.0  # Convert to consistent units
                    tmax[x, y, z] = result.parameters.get("delay", 0.0)
                else:  # cpi
                    # Extract from residue function
                    if result.fitted_residue is not None:
                        res = result.fitted_residue
                        cbv[x, y, z] = float(trapezoid(res, dx=dt))
                        if cbf[x, y, z] > EPSILON_CONCENTRATION:
                            mtt[x, y, z] = cbv[x, y, z] / cbf[x, y, z]
                        tmax[x, y, z] = float(np.argmax(res)) * dt

                # Track QC
                if result.r_squared is not None:
                    r_squared_list.append(result.r_squared)

                if bayesian != "none" and result.passes_convergence():
                    n_converged += 1

            except Exception:
                # Skip problematic voxels
                continue

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

        # Verify brain mask is in DSC space (guaranteed by extract_dsc_brain node)
        if mask.shape[:3] != conc_data.shape[:3]:
            raise RuntimeError(
                f"Brain mask shape {mask.shape[:3]} does not match "
                f"concentration data shape {conc_data.shape[:3]}. "
                "The brain mask must be in native DSC space. "
                "Ensure extract_dsc_brain node is connected upstream."
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

        spatial_shape = conc_data.shape[:3]

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
            psr_result = compute_psr(signal_data, baseline_frames, tail_frames)
            ph_result = compute_peak_height(signal_data, baseline_frames)
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

        # Compute TTP from concentration curves
        # TTP: time to peak concentration (seconds)
        ttp = np.argmax(conc_data, axis=-1).astype(np.float64) * dt

        # -----------------------------------------------------------
        # AUC decomposition: raw (pre-K2), corrected (post-K2), diff
        # -----------------------------------------------------------
        # auc_corr = integral of the (post-K2) concentration passed in
        try:
            auc_corr = np.trapezoid(conc_data, dx=dt, axis=-1)
        except AttributeError:
            auc_corr = np.trapz(conc_data, dx=dt, axis=-1)

        # auc_raw = integral of pre-K2 concentration (if available)
        if self.inputs.raw_concentration_file:
            raw_conc_img = nib.load(self.inputs.raw_concentration_file)
            raw_conc_data: npt.NDArray[Any] = raw_conc_img.get_fdata()  # type: ignore[attr-defined]
            try:
                auc_raw = np.trapezoid(raw_conc_data, dx=dt, axis=-1)
            except AttributeError:
                auc_raw = np.trapz(raw_conc_data, dx=dt, axis=-1)
        else:
            # No K2 correction was applied — raw == corrected
            auc_raw = auc_corr.copy()

        auc_diff = auc_raw - auc_corr

        # CNR: peak concentration / baseline std per voxel
        baseline_std = np.std(conc_data[..., :baseline_frames], axis=-1)
        peak_conc = np.max(conc_data, axis=-1)
        cnr = np.where(
            baseline_std > EPSILON_CONCENTRATION,
            peak_conc / baseline_std,
            0.0,
        )

        # -----------------------------------------------------------
        # SNR / tSNR (from raw signal, not concentration)
        # -----------------------------------------------------------
        if signal_data is not None:
            baseline_sig = signal_data[..., :baseline_frames]
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
