"""AIF (Arterial Input Function) detection interfaces for DSC-MRI.

This module implements Nipype interfaces for AIF detection in DSC perfusion
processing, with multi-level detection strategies (atlas-based, relaxed
criteria, synthetic fallback) and comprehensive quality metrics.
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
    QC_CNR_THRESHOLD,
)


def compute_aif_cnr(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
) -> float:
    """Compute contrast-to-noise ratio for an AIF curve.

    CNR = (peak - baseline_mean) / baseline_std

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course.
    baseline_frames : int
        Number of pre-bolus baseline frames (default: 5).

    Returns
    -------
    float
        Contrast-to-noise ratio. Higher is better (>4.0 typically good).
    """
    if len(aif_curve) < baseline_frames + 1:
        return 0.0

    baseline = aif_curve[:baseline_frames]
    baseline_mean = float(np.mean(baseline))
    baseline_std = float(np.std(baseline))

    if baseline_std < EPSILON_CONCENTRATION:
        # Perfect baseline (zero noise), return high CNR
        return 100.0

    peak = float(np.max(aif_curve))
    cnr = (peak - baseline_mean) / baseline_std

    return max(0.0, cnr)


def compute_aif_washout_fraction(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
    tail_frames: int = 5,
) -> float:
    """Compute washout fraction for an AIF curve.

    Washout fraction = (peak - tail_mean) / (peak - baseline_mean)
    A value of 1.0 indicates complete washout; 0.0 indicates no washout.

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course.
    baseline_frames : int
        Number of pre-bolus baseline frames (default: 5).
    tail_frames : int
        Number of tail frames to average (default: 5).

    Returns
    -------
    float
        Washout fraction (0-1). Arterial curves typically show >0.2 washout.
    """
    if len(aif_curve) < baseline_frames + tail_frames + 1:
        return 0.0

    baseline_mean = float(np.mean(aif_curve[:baseline_frames]))
    tail_mean = float(np.mean(aif_curve[-tail_frames:]))
    peak = float(np.max(aif_curve))

    # Denominator: signal drop from baseline to peak
    denom = peak - baseline_mean
    if denom < EPSILON_CONCENTRATION:
        return 0.0

    # Numerator: recovery from peak to tail
    washout = (peak - tail_mean) / denom

    return max(0.0, min(1.0, washout))


def compute_aif_fwhm_frames(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
) -> int:
    """Compute full width at half maximum (FWHM) of AIF peak in frames.

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course.
    baseline_frames : int
        Number of pre-bolus baseline frames (default: 5).

    Returns
    -------
    int
        FWHM in number of frames. Arterial curves typically 5-15 frames.
    """
    if len(aif_curve) < baseline_frames + 1:
        return 0

    baseline_mean = float(np.mean(aif_curve[:baseline_frames]))
    peak = float(np.max(aif_curve))
    peak_idx = int(np.argmax(aif_curve))

    half_max = baseline_mean + (peak - baseline_mean) / 2.0

    # Find first crossing above half max (rising edge)
    rising_idx = 0
    for i in range(baseline_frames, peak_idx):
        if aif_curve[i] >= half_max:
            rising_idx = i
            break

    # Find first crossing below half max after peak (falling edge)
    falling_idx = len(aif_curve) - 1
    for i in range(peak_idx + 1, len(aif_curve)):
        if aif_curve[i] <= half_max:
            falling_idx = i
            break

    fwhm = falling_idx - rising_idx
    return max(0, fwhm)


def compute_baseline_stability(
    aif_curve: npt.NDArray[Any],
    baseline_frames: int = 5,
) -> float:
    """Compute baseline stability as coefficient of variation.

    Parameters
    ----------
    aif_curve : np.ndarray
        1D AIF concentration time course.
    baseline_frames : int
        Number of pre-bolus baseline frames (default: 5).

    Returns
    -------
    float
        Coefficient of variation (std/mean) of baseline. Lower is better (<0.1 good).
    """
    if len(aif_curve) < baseline_frames:
        return 1.0

    baseline = aif_curve[:baseline_frames]
    baseline_mean = np.abs(np.mean(baseline))
    baseline_std = float(np.std(baseline))

    if baseline_mean < EPSILON_CONCENTRATION:
        return baseline_std / EPSILON_CONCENTRATION

    return baseline_std / float(baseline_mean)


class AIFDetectionInputSpec(BaseInterfaceInputSpec):
    """Input specification for AIFDetection interface."""

    concentration_file = File(
        exists=True,
        mandatory=True,
        desc="4D NIfTI concentration curves (output from signal_to_concentration)",
    )
    brain_mask_file = File(
        exists=True,
        mandatory=True,
        desc="3D brain mask NIfTI",
    )
    atlas_mask_file = File(
        exists=True,
        mandatory=False,
        desc="Optional 3D arterial atlas mask in DSC space",
    )
    lesion_mask_file = File(
        exists=True,
        mandatory=False,
        desc="Optional 3D lesion mask for exclusion",
    )

    # Detection parameters
    aif_method = traits.Enum(
        "auto",
        "population",
        usedefault=True,
        desc="AIF detection method: 'auto' (multi-level) or 'population' (synthetic)",
    )
    criteria = traits.Enum(
        "strict",
        "relaxed",
        usedefault=True,
        desc="AIF quality criteria: 'strict' or 'relaxed'",
    )
    n_candidates = traits.Int(
        10,
        usedefault=True,
        desc="Number of AIF candidate voxels to average",
    )
    lesion_distance = traits.Float(
        20.0,
        usedefault=True,
        desc="Minimum distance (mm) from lesion for AIF candidates",
    )
    prefer_contralateral = traits.Bool(
        True,
        usedefault=True,
        desc="Prefer contralateral AIF candidates when lesion present",
    )
    gamma_fit = traits.Bool(
        False,
        usedefault=True,
        desc="Apply gamma-variate fitting to AIF",
    )
    disable_fallback = traits.Bool(
        False,
        usedefault=True,
        desc="Disable synthetic AIF fallback when detection fails",
    )
    baseline_frames = traits.Int(
        5,
        usedefault=True,
        desc="Number of pre-bolus baseline frames",
    )

    # Output file
    out_aif_curve = File(
        genfile=True,
        desc="Output AIF curve as 1D text file",
    )
    out_aif_mask = File(
        genfile=True,
        desc="Output AIF voxel mask NIfTI",
    )


class AIFDetectionOutputSpec(TraitedSpec):
    """Output specification for AIFDetection interface."""

    out_aif_curve = File(exists=True, desc="AIF curve as 1D text file")
    out_aif_mask = File(exists=True, desc="AIF voxel mask NIfTI")
    aif_curve = traits.List(traits.Float, desc="AIF concentration curve values")
    detection_level = traits.Str(desc="Detection level: 'atlas', 'relaxed', 'synthetic'")
    quality_score = traits.Float(desc="Overall AIF quality score (0-1)")
    candidate_count = traits.Int(desc="Number of candidate voxels evaluated")
    gamma_fit_applied = traits.Bool(desc="Whether gamma-variate fitting was applied")
    qc_metrics = traits.Dict(desc="Detailed QC metrics dictionary")


class AIFDetection(BaseInterface):
    """Arterial Input Function (AIF) detection for DSC-MRI.

    This interface implements multi-level AIF detection with atlas-based
    search, relaxed criteria fallback, and synthetic AIF generation.

    The detection follows this hierarchy:
    1. Atlas-based search (strict criteria)
    2. Atlas-based search (relaxed criteria)
    3. Whole-brain search (relaxed criteria)
    4. Synthetic (population) AIF as fallback

    Parameters
    ----------
    concentration_file : str
        4D NIfTI with concentration time courses (voxels × time).
    brain_mask_file : str
        3D brain mask NIfTI.
    atlas_mask_file : str, optional
        3D arterial atlas mask in DSC space for constrained search.
    lesion_mask_file : str, optional
        3D lesion mask for excluding potentially abnormal regions.
    aif_method : str
        'auto' for multi-level detection, 'population' for synthetic.
    criteria : str
        Initial criteria level: 'strict' or 'relaxed'.
    n_candidates : int
        Number of top candidate voxels to average for final AIF.
    lesion_distance : float
        Minimum distance (mm) from lesion for candidate voxels.
    prefer_contralateral : bool
        Prefer contralateral hemisphere when lesion present.
    gamma_fit : bool
        Apply gamma-variate fitting to isolate first-pass bolus.
    disable_fallback : bool
        Disable synthetic AIF fallback (error if detection fails).
    baseline_frames : int
        Number of pre-bolus frames for baseline estimation.

    Returns
    -------
    out_aif_curve : str
        Path to AIF curve text file.
    out_aif_mask : str
        Path to AIF voxel mask NIfTI.
    aif_curve : list
        AIF concentration values.
    detection_level : str
        Method used: 'atlas', 'relaxed', or 'synthetic'.
    quality_score : float
        Overall quality score (0-1).
    candidate_count : int
        Number of candidate voxels used.
    gamma_fit_applied : bool
        Whether gamma fitting was applied.
    qc_metrics : dict
        Detailed quality metrics.

    Examples
    --------
    >>> aif = AIFDetection()
    >>> aif.inputs.concentration_file = 'concentration_4d.nii.gz'
    >>> aif.inputs.brain_mask_file = 'brain_mask.nii.gz'
    >>> aif.inputs.atlas_mask_file = 'arterial_atlas.nii.gz'
    >>> aif.inputs.n_candidates = 10
    >>> result = aif.run()  # doctest: +SKIP

    Notes
    -----
    AIF quality criteria:

    Strict thresholds:
    - Washout fraction ≥ 0.20
    - FWHM: 5-15 frames
    - CNR ≥ 4.0
    - Baseline CV < 0.10

    Relaxed thresholds:
    - Washout fraction ≥ 0.10
    - FWHM: 3-20 frames
    - CNR ≥ 2.0
    - Baseline CV < 0.20

    The final AIF is computed as the average of the top N candidate
    voxels ranked by a composite quality score.

    See Also
    --------
    pmriprep.utils.dsc_aif : Core AIF detection utilities (US-035, US-036)
    """

    input_spec = AIFDetectionInputSpec
    output_spec = AIFDetectionOutputSpec

    # AIF quality thresholds
    _STRICT_THRESHOLDS: dict[str, Any] = {
        "washout_min": 0.20,
        "fwhm_min": 5,
        "fwhm_max": 15,
        "cnr_min": 4.0,
        "baseline_cv_max": 0.10,
    }
    _RELAXED_THRESHOLDS: dict[str, Any] = {
        "washout_min": 0.10,
        "fwhm_min": 3,
        "fwhm_max": 20,
        "cnr_min": 2.0,
        "baseline_cv_max": 0.20,
    }

    def _gen_outfilename(self, suffix: str = "_aif_curve.txt") -> str:
        """Generate output filename based on input."""
        in_file = Path(self.inputs.concentration_file)
        if in_file.name.endswith(".nii.gz"):
            base = in_file.name[:-7]
        elif in_file.name.endswith(".nii"):
            base = in_file.stem
        else:
            base = in_file.stem
        return str(Path.cwd() / f"{base}{suffix}")

    def _get_thresholds(self, criteria: str) -> dict[str, Any]:
        """Get AIF quality thresholds for criteria level."""
        if criteria == "strict":
            return self._STRICT_THRESHOLDS.copy()
        return self._RELAXED_THRESHOLDS.copy()

    def _score_candidate(
        self,
        curve: npt.NDArray[Any],
        baseline_frames: int,
        thresholds: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
        """Score a candidate AIF curve.

        Returns a composite quality score and individual metrics.
        Score is 0 if thresholds are not met, otherwise 0-1.
        """
        metrics: dict[str, float] = {}

        # Compute individual metrics
        cnr = compute_aif_cnr(curve, baseline_frames)
        washout = compute_aif_washout_fraction(curve, baseline_frames)
        fwhm = compute_aif_fwhm_frames(curve, baseline_frames)
        baseline_cv = compute_baseline_stability(curve, baseline_frames)

        metrics["cnr"] = cnr
        metrics["washout"] = washout
        metrics["fwhm"] = float(fwhm)
        metrics["baseline_cv"] = baseline_cv

        # Check thresholds
        if cnr < thresholds["cnr_min"]:
            return 0.0, metrics
        if washout < thresholds["washout_min"]:
            return 0.0, metrics
        if fwhm < thresholds["fwhm_min"] or fwhm > thresholds["fwhm_max"]:
            return 0.0, metrics
        if baseline_cv > thresholds["baseline_cv_max"]:
            return 0.0, metrics

        # Compute composite score (all thresholds met)
        # Weight factors: CNR most important, then washout, then FWHM
        score = 0.0

        # CNR contribution (0-0.4): normalized to expected range 4-20
        cnr_score = min(1.0, (cnr - thresholds["cnr_min"]) / 16.0)
        score += 0.4 * cnr_score

        # Washout contribution (0-0.3): closer to 0.5 is ideal
        ideal_washout = 0.5
        washout_score = 1.0 - abs(washout - ideal_washout) / 0.5
        score += 0.3 * max(0, washout_score)

        # FWHM contribution (0-0.2): closer to 8-10 is ideal
        ideal_fwhm = 9
        fwhm_range = thresholds["fwhm_max"] - thresholds["fwhm_min"]
        fwhm_score = 1.0 - abs(fwhm - ideal_fwhm) / (fwhm_range / 2)
        score += 0.2 * max(0, fwhm_score)

        # Baseline stability contribution (0-0.1): lower is better
        baseline_score = 1.0 - (baseline_cv / thresholds["baseline_cv_max"])
        score += 0.1 * max(0, baseline_score)

        return min(1.0, score), metrics

    def _find_candidates_in_mask(
        self,
        concentration_data: npt.NDArray[Any],
        mask: npt.NDArray[Any],
        n_candidates: int,
        baseline_frames: int,
        thresholds: dict[str, Any],
    ) -> tuple[list[tuple[int, int, int]], list[float], list[dict[str, float]]]:
        """Find candidate AIF voxels within a mask.

        Returns voxel coordinates, scores, and metrics for candidates
        that meet the threshold criteria, sorted by score descending.
        """
        candidates: list[tuple[int, int, int]] = []
        scores: list[float] = []
        all_metrics: list[dict[str, float]] = []

        # Get mask coordinates
        mask_coords = np.where(mask > 0)
        n_voxels = len(mask_coords[0])

        for i in range(n_voxels):
            x, y, z = (
                int(mask_coords[0][i]),
                int(mask_coords[1][i]),
                int(mask_coords[2][i]),
            )
            curve = concentration_data[x, y, z, :]

            # Skip if curve is empty or all zeros
            if np.max(np.abs(curve)) < EPSILON_CONCENTRATION:
                continue

            score, metrics = self._score_candidate(curve, baseline_frames, thresholds)
            if score > 0:
                candidates.append((x, y, z))
                scores.append(score)
                all_metrics.append(metrics)

        # Sort by score descending
        if candidates:
            sorted_indices = np.argsort(scores)[::-1]
            candidates = [candidates[i] for i in sorted_indices[:n_candidates]]
            scores = [scores[i] for i in sorted_indices[:n_candidates]]
            all_metrics = [all_metrics[i] for i in sorted_indices[:n_candidates]]

        return candidates, scores, all_metrics

    def _generate_synthetic_aif(
        self,
        n_timepoints: int,
        tr: float = 1.5,
        baseline_frames: int = 5,
    ) -> npt.NDArray[Any]:
        """Generate synthetic (population) AIF using gamma-variate model.

        Uses population parameters from literature for a typical AIF.
        This provides relative quantification only.

        Parameters
        ----------
        n_timepoints : int
            Number of time points in the curve.
        tr : float
            Repetition time in seconds (default: 1.5s).
        baseline_frames : int
            Number of baseline frames before bolus arrival.

        Returns
        -------
        np.ndarray
            Synthetic AIF concentration curve.
        """
        # Population parameters for gamma-variate AIF
        # Based on typical DSC-MRI values
        t0 = baseline_frames * tr  # Bolus arrival time (seconds)
        alpha = 3.5  # Shape parameter
        beta = 1.5  # Scale parameter (seconds)
        amplitude = 10.0  # Peak concentration (arbitrary units for relative)

        # Time vector
        t = np.arange(n_timepoints) * tr

        # Gamma-variate: C(t) = A * (t-t0)^alpha * exp(-(t-t0)/beta)
        aif = np.zeros(n_timepoints, dtype=np.float64)
        post_arrival = t > t0

        if np.any(post_arrival):
            t_shifted = t[post_arrival] - t0
            # Normalize so max occurs at t = alpha * beta
            t_max = alpha * beta
            norm_factor = (t_max ** alpha) * np.exp(-alpha)
            aif[post_arrival] = (
                amplitude
                * (t_shifted ** alpha)
                * np.exp(-t_shifted / beta)
                / norm_factor
            )

        return aif

    def _run_interface(self, runtime: Any) -> Any:
        """Execute AIF detection."""
        import nibabel as nib  # type: ignore[import-untyped]

        # Load data
        conc_img = nib.load(self.inputs.concentration_file)
        conc_data: npt.NDArray[Any] = conc_img.get_fdata()  # type: ignore[attr-defined]
        affine = conc_img.affine  # type: ignore[attr-defined]

        brain_img = nib.load(self.inputs.brain_mask_file)
        brain_mask: npt.NDArray[Any] = brain_img.get_fdata()  # type: ignore[attr-defined]

        # Optional atlas mask
        atlas_mask: Optional[npt.NDArray[Any]] = None
        if self.inputs.atlas_mask_file:
            atlas_img = nib.load(self.inputs.atlas_mask_file)
            atlas_mask = atlas_img.get_fdata()  # type: ignore[attr-defined]

        # Optional lesion mask
        lesion_mask: Optional[npt.NDArray[Any]] = None
        if self.inputs.lesion_mask_file:
            lesion_img = nib.load(self.inputs.lesion_mask_file)
            lesion_mask = lesion_img.get_fdata()  # type: ignore[attr-defined]

        # Parameters
        aif_method = str(self.inputs.aif_method)
        criteria = str(self.inputs.criteria)
        n_candidates = int(self.inputs.n_candidates)
        baseline_frames = int(self.inputs.baseline_frames)
        gamma_fit = bool(self.inputs.gamma_fit)
        disable_fallback = bool(self.inputs.disable_fallback)

        n_timepoints = conc_data.shape[3]
        tr = 1.5  # Default TR, would ideally come from header

        # Initialize results
        detection_level: Literal["atlas", "relaxed", "synthetic"] = "synthetic"
        candidates: list[tuple[int, int, int]] = []
        scores: list[float] = []
        metrics_list: list[dict[str, float]] = []

        # Skip auto detection if population method requested
        if aif_method == "population":
            # Go directly to synthetic AIF
            pass
        else:
            # Multi-level AIF detection

            # Create search mask (brain minus lesion with distance buffer)
            search_mask = brain_mask.copy()
            if lesion_mask is not None:
                # Simple exclusion: remove lesion voxels
                # Full distance-based exclusion would be in utils/dsc_aif.py (US-035)
                search_mask[lesion_mask > 0] = 0

            # Level 1: Atlas-based with strict criteria
            if atlas_mask is not None:
                combined_mask = search_mask * atlas_mask
                thresholds = self._get_thresholds("strict")
                candidates, scores, metrics_list = self._find_candidates_in_mask(
                    conc_data, combined_mask, n_candidates, baseline_frames, thresholds
                )
                if candidates:
                    detection_level = "atlas"

            # Level 2: Atlas-based with relaxed criteria
            if not candidates and atlas_mask is not None:
                combined_mask = search_mask * atlas_mask
                thresholds = self._get_thresholds("relaxed")
                candidates, scores, metrics_list = self._find_candidates_in_mask(
                    conc_data, combined_mask, n_candidates, baseline_frames, thresholds
                )
                if candidates:
                    detection_level = "relaxed"

            # Level 3: Whole-brain with user-specified criteria
            if not candidates:
                thresholds = self._get_thresholds(criteria)
                candidates, scores, metrics_list = self._find_candidates_in_mask(
                    conc_data, search_mask, n_candidates, baseline_frames, thresholds
                )
                if candidates:
                    detection_level = "relaxed"

            # Level 4: Whole-brain with relaxed criteria (if not already relaxed)
            if not candidates and criteria == "strict":
                thresholds = self._get_thresholds("relaxed")
                candidates, scores, metrics_list = self._find_candidates_in_mask(
                    conc_data, search_mask, n_candidates, baseline_frames, thresholds
                )
                if candidates:
                    detection_level = "relaxed"

        # Generate AIF curve
        aif_curve: npt.NDArray[Any]
        aif_mask = np.zeros(conc_data.shape[:3], dtype=np.uint8)
        quality_score = 0.0
        qc_metrics: dict[str, Any] = {}

        if candidates:
            # Average curves from candidate voxels
            curves = []
            for x, y, z in candidates:
                curves.append(conc_data[x, y, z, :])
                aif_mask[x, y, z] = 1

            aif_curve = np.mean(curves, axis=0)
            quality_score = float(np.mean(scores))

            # Aggregate QC metrics
            if metrics_list:
                qc_metrics = {
                    "cnr_mean": float(np.mean([m["cnr"] for m in metrics_list])),
                    "washout_mean": float(np.mean([m["washout"] for m in metrics_list])),
                    "fwhm_mean": float(np.mean([m["fwhm"] for m in metrics_list])),
                    "baseline_cv_mean": float(
                        np.mean([m["baseline_cv"] for m in metrics_list])
                    ),
                    "n_candidates": len(candidates),
                }

        elif not disable_fallback:
            # Generate synthetic AIF as fallback
            aif_curve = self._generate_synthetic_aif(n_timepoints, tr, baseline_frames)
            detection_level = "synthetic"
            quality_score = 0.3  # Low score for synthetic
            qc_metrics = {
                "synthetic": True,
                "cnr_mean": 0.0,
                "washout_mean": 0.0,
                "fwhm_mean": 0.0,
                "baseline_cv_mean": 0.0,
                "n_candidates": 0,
            }
        else:
            # No fallback, raise error
            raise RuntimeError(
                "AIF detection failed and synthetic fallback is disabled"
            )

        # Gamma-variate fitting (placeholder, full implementation in US-037)
        gamma_fit_applied = False
        if gamma_fit and detection_level != "synthetic":
            # TODO: Implement gamma-variate fitting in utils/dsc_aif.py
            gamma_fit_applied = False  # Will be True when US-037 implemented

        # Save outputs
        out_aif_curve = self._gen_outfilename("_aif_curve.txt")
        np.savetxt(out_aif_curve, aif_curve)

        out_aif_mask = self._gen_outfilename("_aif_mask.nii.gz")
        mask_img = nib.Nifti1Image(aif_mask, affine)  # type: ignore[attr-defined]
        nib.save(mask_img, out_aif_mask)  # type: ignore[attr-defined]

        # Store results for _list_outputs
        self._results = {
            "out_aif_curve": out_aif_curve,
            "out_aif_mask": out_aif_mask,
            "aif_curve": [float(x) for x in aif_curve],
            "detection_level": detection_level,
            "quality_score": quality_score,
            "candidate_count": len(candidates),
            "gamma_fit_applied": gamma_fit_applied,
            "qc_metrics": qc_metrics,
        }

        return runtime

    def _list_outputs(self) -> dict[str, Any]:  # type: ignore[override]
        """List interface outputs."""
        outputs: dict[str, Any] = self.output_spec().get()  # type: ignore[misc]

        if hasattr(self, "_results"):
            outputs.update(self._results)

        return outputs
