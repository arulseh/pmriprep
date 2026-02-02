"""DCE map computation interface for pharmacokinetic modeling.

This module implements the main Nipype interface for computing DCE
pharmacokinetic parameter maps (Ktrans, ve, vp, kep) from concentration
curves and AIF using various PK models.
"""

from pathlib import Path
from typing import Any, Optional

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
)


class ComputeDCEMapsInputSpec(BaseInterfaceInputSpec):
    """Input specification for ComputeDCEMaps interface."""

    concentration_file = File(
        exists=True,
        mandatory=True,
        desc="4D NIfTI concentration curves (output from signal_to_concentration)",
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
        desc="Repetition time in seconds (required if model='auto')",
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
    dsc_aif_qc_file = File(
        desc="DSC AIF QC JSON for CBV quality assessment (optional)",
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

    # QC outputs
    out_r2 = File(desc="R² (goodness of fit) map")
    out_tsnr = File(desc="tSNR map")
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

    def _get_model_instance(self, model_name: str) -> Any:
        """Get model instance by name.

        Args:
            model_name: Model name (tofts, extended_tofts, patlak, 2cxm).

        Returns:
            PKModel instance.
        """
        from pmriprep.models.dce import (
            ToftsModel,
            ExtendedToftsModel,
            PatlakModel,
            TwoCompartmentExchangeModel,
        )

        if model_name == "tofts":
            return ToftsModel()
        elif model_name == "extended_tofts":
            return ExtendedToftsModel()
        elif model_name == "patlak":
            return PatlakModel()
        elif model_name == "2cxm":
            return TwoCompartmentExchangeModel()
        else:
            raise ValueError(f"Unknown model: {model_name}")

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

        from pmriprep.models.dce import (
            BayesianPKFitter,
            fit_pk_model,
        )
        from pmriprep.models.dce.bayesian import create_vp_prior_from_dsc
        from pmriprep.utils.dsc_cbv_constraint import (
            ConstraintMode,
            prepare_vp_constraint,
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
        elif self.inputs.tr:
            # TR is in seconds, convert to minutes for PK models
            tr_minutes = float(self.inputs.tr) / 60.0
            time_points = np.arange(n_timepoints, dtype=np.float64) * tr_minutes
        else:
            raise ValueError("Either tr or time_points_file must be provided")

        # Get TR for auto-model selection
        if self.inputs.tr:
            tr_seconds = float(self.inputs.tr)
        else:
            # Estimate TR from time points (minutes -> seconds)
            tr_seconds = float(time_points[1] - time_points[0]) * 60.0

        # Select model
        model_name = str(self.inputs.model)
        if model_name == "auto":
            model_name = self._select_model(tr_seconds)

        model = self._get_model_instance(model_name)

        # Baseline frames for tSNR
        baseline_frames = int(self.inputs.baseline_frames)
        if baseline_frames == 0:
            baseline_frames = self._detect_baseline_frames(conc_data, mask)

        # Compute tSNR if threshold is set
        tsnr_threshold = float(self.inputs.tsnr_threshold)
        tsnr_map: Optional[npt.NDArray[Any]] = None

        if tsnr_threshold > 0:
            tsnr_map = self._compute_tsnr(conc_data, mask, baseline_frames)

        # Create fitting mask (brain mask + tSNR threshold)
        mask_bool = mask > 0
        if tsnr_map is not None and tsnr_threshold > 0:
            fitting_mask = mask_bool & (tsnr_map >= tsnr_threshold)
        else:
            fitting_mask = mask_bool

        # Parameters
        bayesian = str(self.inputs.bayesian)
        mcmc_samples = int(self.inputs.mcmc_samples)
        use_gpu = bool(self.inputs.use_gpu)
        save_uncertainty = bool(self.inputs.save_uncertainty)
        save_fitted_curves = bool(self.inputs.save_fitted_curves)

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
            and isdefined(self.inputs.dsc_aif_qc_file)
            and self._has_vp(model_name)
        )

        if has_dsc_cbv:
            try:
                vp_result = prepare_vp_constraint(
                    cbv_file=self.inputs.dsc_cbv_file,
                    aif_qc_file=self.inputs.dsc_aif_qc_file,
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
        qc_metrics: dict[str, Any] = {
            "model": model_name,
            "bayesian": bayesian,
            "n_voxels_brain": int(np.sum(mask_bool)),
            "n_voxels_fitted": n_voxels,
            "tsnr_threshold": tsnr_threshold,
            "baseline_frames": baseline_frames,
            "tr_seconds": tr_seconds,
            "constraint_mode": constraint_mode.value,
        }

        n_converged = 0
        n_failed = 0
        n_constrained = 0
        n_unconstrained_fallback = 0
        r2_list: list[float] = []

        # Set up Bayesian fitter if needed (default, without vp override)
        fitter: Optional[BayesianPKFitter] = None
        if bayesian != "none":
            fitter = BayesianPKFitter(model)

        # Process voxels
        for i in range(n_voxels):
            x, y, z = (
                int(mask_coords[0][i]),
                int(mask_coords[1][i]),
                int(mask_coords[2][i]),
            )
            tissue_curve = conc_data[x, y, z, :]

            # Skip if curve is mostly zeros
            if np.max(np.abs(tissue_curve)) < EPSILON_VE:
                n_failed += 1
                continue

            # Determine per-voxel vp constraint
            voxel_vp: Optional[float] = None
            if vp_constraint_map is not None and constraint_mode != ConstraintMode.SKIP:
                vp_val = float(vp_constraint_map[x, y, z])
                if np.isfinite(vp_val) and vp_val > 0:
                    voxel_vp = vp_val

            try:
                result = None

                # HARD constraint: use fit_constrained with fixed vp
                if (
                    voxel_vp is not None
                    and constraint_mode == ConstraintMode.HARD
                    and hasattr(model, "fit_constrained")
                ):
                    result = model.fit_constrained(
                        tissue_curve, aif, time_points, fixed_vp=voxel_vp
                    )
                    n_constrained += 1

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
                        result = soft_fitter.fit_map(tissue_curve, aif, time_points)
                    else:
                        result = soft_fitter.fit_mcmc(
                            tissue_curve, aif, time_points, n_samples=mcmc_samples
                        )
                    n_constrained += 1

                # SOFT constraint with NLLS: fall back to hard constraint
                elif (
                    voxel_vp is not None
                    and constraint_mode == ConstraintMode.SOFT
                    and bayesian == "none"
                    and hasattr(model, "fit_constrained")
                ):
                    result = model.fit_constrained(
                        tissue_curve, aif, time_points, fixed_vp=voxel_vp
                    )
                    n_constrained += 1

                # Unconstrained fallback (no valid vp, skip mode, or no constraint)
                if result is None:
                    if voxel_vp is not None and constraint_mode != ConstraintMode.SKIP:
                        n_unconstrained_fallback += 1
                    if bayesian == "none":
                        result = model.fit(tissue_curve, aif, time_points)
                    elif bayesian == "map" and fitter is not None:
                        result = fitter.fit_map(tissue_curve, aif, time_points)
                    elif bayesian == "mcmc" and fitter is not None:
                        result = fitter.fit_mcmc(
                            tissue_curve, aif, time_points,
                            n_samples=mcmc_samples,
                        )
                    else:
                        result = model.fit(tissue_curve, aif, time_points)

                # Extract parameters
                ktrans[x, y, z] = result.parameters.get("Ktrans", 0.0)
                ve[x, y, z] = result.parameters.get("ve", 0.0)
                kep[x, y, z] = result.parameters.get("kep", 0.0)
                r2_map[x, y, z] = result.r_squared

                if self._has_vp(model_name):
                    vp[x, y, z] = result.parameters.get("vp", 0.0)

                # 2CXM-specific parameters
                if model_name == "2cxm":
                    fp_map[x, y, z] = result.parameters.get("Fp", 0.0)
                    ps_map[x, y, z] = result.parameters.get("PS", 0.0)
                    e_map[x, y, z] = result.parameters.get("E", 0.0)

                # Uncertainty (Bayesian only)
                if save_uncertainty and result.has_uncertainty():
                    ktrans_std[x, y, z] = result.get_uncertainty("Ktrans", 0.0)
                    ve_std[x, y, z] = result.get_uncertainty("ve", 0.0)
                    if self._has_vp(model_name):
                        vp_std[x, y, z] = result.get_uncertainty("vp", 0.0)

                # Fitted curve
                if save_fitted_curves and fitted_curves is not None:
                    if result.fitted_curve is not None:
                        fitted_curves[x, y, z, :] = result.fitted_curve.astype(np.float32)

                # Track convergence
                if result.converged:
                    n_converged += 1
                    r2_list.append(result.r_squared)
                else:
                    n_failed += 1

            except Exception:
                # Skip problematic voxels
                n_failed += 1
                continue

        # Compile QC metrics
        qc_metrics["n_converged"] = n_converged
        qc_metrics["n_failed"] = n_failed
        qc_metrics["convergence_rate"] = n_converged / max(1, n_voxels)
        qc_metrics["n_constrained"] = n_constrained
        qc_metrics["n_unconstrained_fallback"] = n_unconstrained_fallback

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

        # Initialize results
        self._results: dict[str, Any] = {
            "out_ktrans": out_ktrans,
            "out_ve": out_ve,
            "out_kep": out_kep,
            "out_r2": out_r2,
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

        # Save tSNR map
        if tsnr_map is not None and tsnr_threshold > 0:
            out_tsnr = self._gen_outfilename("_tSNR")
            tsnr_img = nib.Nifti1Image(tsnr_map.astype(np.float32), affine, header)  # type: ignore[attr-defined]
            nib.save(tsnr_img, out_tsnr)  # type: ignore[attr-defined]
            self._results["out_tsnr"] = out_tsnr

        # Save fitted curves
        if save_fitted_curves and fitted_curves is not None:
            out_fitted = self._gen_outfilename("_fitted_curves")
            fitted_img = nib.Nifti1Image(fitted_curves, affine)  # type: ignore[attr-defined]
            nib.save(fitted_img, out_fitted)  # type: ignore[attr-defined]
            self._results["out_fitted_curves"] = out_fitted

        return runtime

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List interface outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]

        if hasattr(self, "_results"):
            outputs.update(self._results)

        return outputs
