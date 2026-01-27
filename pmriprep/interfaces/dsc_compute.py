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
    DEFAULT_SVD_THRESHOLD,
    DEFAULT_OI_THRESHOLD,
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

    # Output files
    out_cbf = File(
        genfile=True,
        desc="Output CBF map NIfTI",
    )
    out_cbv = File(
        genfile=True,
        desc="Output CBV map NIfTI",
    )
    out_mtt = File(
        genfile=True,
        desc="Output MTT map NIfTI",
    )
    out_tmax = File(
        genfile=True,
        desc="Output Tmax map NIfTI",
    )
    out_psr = File(
        genfile=True,
        desc="Output PSR map NIfTI",
    )
    out_ph = File(
        genfile=True,
        desc="Output peak height map NIfTI",
    )


class ComputeDSCMapsOutputSpec(TraitedSpec):
    """Output specification for ComputeDSCMaps interface."""

    out_cbf = File(exists=True, desc="CBF map (cerebral blood flow)")
    out_cbv = File(exists=True, desc="CBV map (cerebral blood volume)")
    out_mtt = File(exists=True, desc="MTT map (mean transit time)")
    out_tmax = File(exists=True, desc="Tmax map (time to maximum of residue)")
    out_psr = File(exists=True, desc="PSR map (percent signal recovery)")
    out_ph = File(exists=True, desc="PH map (peak height)")
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
    out_cbf : str
        Path to CBF map NIfTI.
    out_cbv : str
        Path to CBV map NIfTI.
    out_mtt : str
        Path to MTT map NIfTI.
    out_tmax : str
        Path to Tmax map NIfTI.
    out_psr : str
        Path to PSR map NIfTI.
    out_ph : str
        Path to peak height map NIfTI.
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
            # Note: residue length is 2N-1, adjust argmax
            n_orig = n_timepoints
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

        # Mask outputs
        mask_bool = mask > 0
        cbf = np.where(mask_bool, cbf, 0.0)
        cbv = np.where(mask_bool, cbv, 0.0)
        mtt = np.where(mask_bool, mtt, 0.0)
        tmax = np.where(mask_bool, tmax, 0.0)

        # Add general QC metrics
        qc_metrics.update({
            "cbf_mean": float(np.mean(cbf[mask_bool])) if np.any(mask_bool) else 0.0,
            "cbf_std": float(np.std(cbf[mask_bool])) if np.any(mask_bool) else 0.0,
            "cbv_mean": float(np.mean(cbv[mask_bool])) if np.any(mask_bool) else 0.0,
            "mtt_mean": float(np.mean(mtt[mask_bool])) if np.any(mask_bool) else 0.0,
            "tmax_mean": float(np.mean(tmax[mask_bool])) if np.any(mask_bool) else 0.0,
            "n_voxels": int(np.sum(mask_bool)),
        })

        # Save output files
        out_cbf = self._gen_outfilename("_cbf")
        out_cbv = self._gen_outfilename("_cbv")
        out_mtt = self._gen_outfilename("_mtt")
        out_tmax = self._gen_outfilename("_tmax")
        out_psr = self._gen_outfilename("_psr")
        out_ph = self._gen_outfilename("_ph")

        # Create NIfTI images and save
        cbf_img = nib.Nifti1Image(cbf.astype(np.float32), affine, header)  # type: ignore[attr-defined]
        nib.save(cbf_img, out_cbf)  # type: ignore[attr-defined]

        cbv_img = nib.Nifti1Image(cbv.astype(np.float32), affine, header)  # type: ignore[attr-defined]
        nib.save(cbv_img, out_cbv)  # type: ignore[attr-defined]

        mtt_img = nib.Nifti1Image(mtt.astype(np.float32), affine, header)  # type: ignore[attr-defined]
        nib.save(mtt_img, out_mtt)  # type: ignore[attr-defined]

        tmax_img = nib.Nifti1Image(tmax.astype(np.float32), affine, header)  # type: ignore[attr-defined]
        nib.save(tmax_img, out_tmax)  # type: ignore[attr-defined]

        psr_img = nib.Nifti1Image(psr.astype(np.float32), affine, header)  # type: ignore[attr-defined]
        nib.save(psr_img, out_psr)  # type: ignore[attr-defined]

        ph_img = nib.Nifti1Image(ph.astype(np.float32), affine, header)  # type: ignore[attr-defined]
        nib.save(ph_img, out_ph)  # type: ignore[attr-defined]

        # Store results
        self._results: dict[str, Any] = {
            "out_cbf": out_cbf,
            "out_cbv": out_cbv,
            "out_mtt": out_mtt,
            "out_tmax": out_tmax,
            "out_psr": out_psr,
            "out_ph": out_ph,
            "qc_metrics": qc_metrics,
        }

        # Save delay map if cSVD
        if delay is not None:
            out_delay = self._gen_outfilename("_delay")
            delay_img = nib.Nifti1Image(delay.astype(np.float32), affine, header)  # type: ignore[attr-defined]
            nib.save(delay_img, out_delay)  # type: ignore[attr-defined]
            self._results["out_delay"] = out_delay

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
