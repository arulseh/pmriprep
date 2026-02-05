"""Integration test framework for PMRIPrep ground truth comparison.

This module provides infrastructure for comparing PMRIPrep outputs
against ground truth data from perf-plus derivatives.

Test subjects and configurations are defined by the PRD integration
test specifications (US-091 through US-097).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray


# =============================================================================
# Constants
# =============================================================================

# Ground truth directory location
PERF_PLUS_DERIVATIVES = Path("~/Developer/perf-plus/bidsroot/derivatives").expanduser()

# Acceptance thresholds
CORRELATION_THRESHOLD = 0.99
DICE_THRESHOLD = 0.95

# Test subjects configuration from PRD
TEST_SUBJECTS: List[Dict[str, Any]] = [
    {
        "subject": "sub-0001",
        "session": "ses-0002",
        "user_story": "US-091",
        "dsc_config": {
            "preload": False,
            "dose": "full",
            "method": "sSVD",
            "bayesian": "none",
        },
        "dce_config": None,  # DSC-only
        "expected_outputs": {
            "dsc": ["cbf", "cbv", "mtt", "tmax"],
        },
    },
    {
        "subject": "sub-0004",
        "session": "ses-0001",
        "user_story": "US-092",
        "dsc_config": {
            "preload": True,
            "dose": "half",
            "method": "oSVD",  # GBM preset
            "bayesian": "none",
            "gbm_preset": True,
        },
        "dce_config": {
            "model": "extended_tofts",
            "bayesian": "map",
            "scale_aif": True,
        },
        "expected_outputs": {
            "dsc": ["cbf", "cbv", "mtt", "tmax", "k2"],
            "dce": ["Ktrans", "ve", "vp", "kep"],
        },
    },
    {
        "subject": "sub-0010",
        "session": "ses-0001",
        "user_story": "US-093",
        "dsc_config": {
            "preload": True,
            "dose": "half",
            "method": "tikhonov",
            "bayesian": "none",
        },
        "dce_config": {
            "model": "tofts",
            "bayesian": "none",
            "scale_aif": True,
        },
        "expected_outputs": {
            "dsc": ["cbf", "cbv", "mtt", "tmax"],
            "dce": ["Ktrans", "ve", "kep"],
        },
    },
    {
        "subject": "sub-0011",
        "session": "ses-0001",
        "user_story": "US-094",
        "dsc_config": {
            "preload": True,
            "dose": "half",
            "method": "cpi",
            "bayesian": "none",
        },
        "dce_config": {
            "model": "extended_tofts",
            "bayesian": "none",
            "scale_aif": True,
        },
        "expected_outputs": {
            "dsc": ["cbf", "cbv", "mtt", "tmax"],
            "dce": ["Ktrans", "ve", "vp", "kep"],
        },
    },
    {
        "subject": "sub-0016",
        "session": "ses-0002",
        "user_story": "US-095",
        "dsc_config": {
            "preload": True,
            "dose": "half",
            "method": "oSVD",
            "bayesian": "vi",
        },
        "dce_config": {
            "model": "2cxm",
            "bayesian": "map",
            "scale_aif": True,
        },
        "expected_outputs": {
            "dsc": ["cbf", "cbv", "mtt", "tmax", "cbf_std", "cbv_std"],
            "dce": ["Ktrans", "ve", "vp", "Fp", "PS"],
        },
    },
    {
        "subject": "sub-0019",
        "session": "ses-0001",
        "user_story": "US-096",
        "dsc_config": {
            "preload": True,
            "dose": "half",
            "method": "cSVD",
            "bayesian": "none",
        },
        "dce_config": {
            "model": "patlak",
            "bayesian": "none",
            "scale_aif": True,
        },
        "expected_outputs": {
            "dsc": ["cbf", "cbv", "mtt", "tmax"],
            "dce": ["Ktrans", "vp"],
        },
    },
    {
        "subject": "sub-0020",
        "session": "ses-0001",
        "user_story": "US-097",
        "dsc_config": {
            "preload": True,
            "dose": "half",
            "method": "vascular",
            "bayesian": "map",
        },
        "dce_config": {
            "model": "tofts",
            "bayesian": "none",
            "scale_aif": True,
        },
        "expected_outputs": {
            "dsc": ["cbf", "cbv", "mtt", "tmax", "cbf_std", "cbv_std", "alpha", "beta"],
            "dce": ["Ktrans", "ve", "kep"],
        },
    },
]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ComparisonResult:
    """Result of comparing PMRIPrep output to ground truth."""

    output_name: str
    metric: str  # 'correlation' or 'dice'
    value: float
    threshold: float
    passed: bool
    pmriprep_path: Optional[Path] = None
    ground_truth_path: Optional[Path] = None
    error: Optional[str] = None


@dataclass
class SubjectTestResult:
    """Complete test result for a subject."""

    subject: str
    session: str
    user_story: str
    comparisons: List[ComparisonResult]
    all_passed: bool
    error: Optional[str] = None


# =============================================================================
# Ground Truth Loading
# =============================================================================


def find_ground_truth_directory(
    subject: str,
    session: str,
    pipeline: str = "dsc_pipeline",
    derivatives_base: Optional[Path] = None,
) -> Optional[Path]:
    """Find the ground truth directory for a subject/session/pipeline.

    Ground truth is expected at:
    ~/Developer/perf-plus/bidsroot/derivatives/{pipeline}/{subject}/{session}/

    Args:
        subject: Subject ID (e.g., 'sub-0001')
        session: Session ID (e.g., 'ses-0001')
        pipeline: Pipeline name ('dsc_pipeline', 'dce_pipeline', 'anat_pipeline', 'hdglio_pipeline')
        derivatives_base: Base directory for derivatives. Defaults to
            ~/Developer/perf-plus/bidsroot/derivatives/

    Returns:
        Path to ground truth directory, or None if not found.
    """
    if derivatives_base is None:
        derivatives_base = PERF_PLUS_DERIVATIVES

    gt_dir = derivatives_base / pipeline / subject / session
    if gt_dir.exists():
        return gt_dir
    return None


def find_ground_truth_file(
    subject: str,
    session: str,
    output_name: str,
    modality: str,
    derivatives_base: Optional[Path] = None,
) -> Optional[Path]:
    """Find a specific ground truth output file.

    Searches for files matching the BIDS naming pattern, preferring
    unsmoothed/unnormalized versions for accurate comparison.

    Args:
        subject: Subject ID
        session: Session ID
        output_name: Output name (e.g., 'cbf', 'Ktrans')
        modality: 'dsc' or 'dce'
        derivatives_base: Base directory for derivatives

    Returns:
        Path to ground truth file, or None if not found.
    """
    # Map modality to pipeline name
    pipeline = f"{modality}_pipeline"
    gt_dir = find_ground_truth_directory(
        subject, session, pipeline=pipeline, derivatives_base=derivatives_base
    )
    if gt_dir is None:
        return None

    # Look in perf subdirectory for perfusion maps
    perf_dir = gt_dir / "perf"
    if not perf_dir.exists():
        perf_dir = gt_dir  # Fall back to session directory

    # Define specific patterns for each output to get the correct version
    # Prefer unsmoothed (desc-dscproc) over smoothed (desc-fwhm3)
    # Prefer relative CBF/CBV over absolute (pmriprep uses relative)
    specific_patterns: Dict[str, List[str]] = {
        # DSC outputs - prefer unsmoothed, relative values
        "cbf": [
            f"*desc-rel_cbf.nii.gz",  # Relative CBF (unsmoothed, unnormalized)
            f"*desc-abs_cbf.nii.gz",  # Absolute CBF (unsmoothed, unnormalized)
        ],
        "cbv": [
            f"*desc-rel_cbv.nii.gz",  # Relative CBV (unsmoothed, unnormalized)
            f"*desc-abs_cbv.nii.gz",  # Absolute CBV (unsmoothed, unnormalized)
        ],
        "mtt": [
            f"*desc-dscproc_mtt.nii.gz",  # Unsmoothed MTT
        ],
        "tmax": [
            f"*desc-dscproc_tmax.nii.gz",  # Unsmoothed Tmax
        ],
        "ph": [
            f"*desc-dscproc_ph.nii.gz",  # Unsmoothed peak height
        ],
        "psr": [
            f"*desc-dscproc_psr.nii.gz",  # Unsmoothed PSR
        ],
        "k2": [
            f"*desc-dscproc_k2.nii.gz",  # Unsmoothed k2
        ],
        # Normalized outputs
        "ncbf": [
            f"*desc-dscproc_ncbf.nii.gz",  # Normalized CBF
        ],
        "ncbv": [
            f"*desc-dscproc_ncbv.nii.gz",  # Normalized CBV
        ],
        # Model-free maps (no AIF/deconvolution dependency)
        "ttp": [
            f"*desc-dscproc_ttp.nii.gz",  # Time to peak
        ],
        "cnr": [
            f"*desc-dscproc_cnr.nii.gz",  # Contrast-to-noise ratio
        ],
        "snr": [
            f"*desc-dscproc_snr.nii.gz",  # Signal-to-noise ratio
        ],
        "tsnr": [
            f"*desc-dscproc_tsnr.nii.gz",  # Temporal SNR
        ],
        # DCE outputs - prefer model-specific patterns with vpDSCmod or vpFit descriptor
        # IMPORTANT: Use specific patterns to avoid matching uncertainty maps (Std, CV, CI95)
        # Different models use different naming conventions:
        # - ExtTofts/Tofts: model-ExtTofts_desc-vpDSCmod_* or model-Tofts_desc-vpDSCmod_*
        # - 2CXM: model-2CXM_desc-vpFit_*
        # - Patlak: model-Patlak_desc-vpFit_*
        "ktrans": [
            f"*model-ExtTofts_desc-vpDSCmod_Ktrans.nii.gz",  # Extended Tofts
            f"*model-Tofts_desc-vpDSCmod_Ktrans.nii.gz",  # Standard Tofts
            f"*model-2CXM_desc-vpFit_Ktrans.nii.gz",  # 2CXM model
            f"*model-Patlak_desc-vpFit_Ktrans.nii.gz",  # Patlak model
        ],
        "ve": [
            f"*model-ExtTofts_desc-vpDSCmod_ve.nii.gz",  # Extended Tofts
            f"*model-Tofts_desc-vpDSCmod_ve.nii.gz",  # Standard Tofts
            f"*model-2CXM_desc-vpFit_ve.nii.gz",  # 2CXM model
        ],
        "vp": [
            f"*model-ExtTofts_desc-vpDSCmod_vp.nii.gz",  # Extended Tofts
            f"*model-2CXM_desc-vpFit_vp.nii.gz",  # 2CXM model
            f"*model-Patlak_desc-vpFit_vp.nii.gz",  # Patlak model
        ],
        "kep": [
            f"*model-ExtTofts_desc-vpDSCmod_kep.nii.gz",  # Extended Tofts
            f"*model-Tofts_desc-vpDSCmod_kep.nii.gz",  # Standard Tofts
        ],
        # 2CXM-specific parameters
        "fp": [
            f"*model-2CXM_desc-vpFit_Fp.nii.gz",  # 2CXM plasma flow
        ],
        "ps": [
            f"*model-2CXM_desc-vpFit_PS.nii.gz",  # 2CXM permeability-surface area product
        ],
    }

    # Get patterns for this output, or use generic patterns
    patterns = specific_patterns.get(output_name.lower(), [])

    # Add generic fallback patterns
    patterns.extend([
        f"{subject}_{session}*_{output_name}.nii.gz",
        f"{subject}_{session}*_{output_name}.nii",
        f"*{output_name}.nii.gz",
        f"*{output_name}.nii",
    ])

    for pattern in patterns:
        matches = list(perf_dir.glob(pattern))
        if matches:
            # If multiple matches, prefer the one with exact output name
            # (e.g., avoid ncbf when looking for cbf)
            for match in matches:
                # Check that we're not matching a longer suffix
                # e.g., avoid matching "ncbf" when looking for "cbf"
                stem = match.name.replace(".nii.gz", "").replace(".nii", "")
                if stem.endswith(f"_{output_name}") or stem.endswith(f"-{output_name}"):
                    return match
            # If no exact match, return first match
            return matches[0]

    return None


def load_ground_truth_map(
    subject: str,
    session: str,
    output_name: str,
    modality: str,
    derivatives_base: Optional[Path] = None,
) -> Tuple[Optional[NDArray[np.floating]], Optional[Path]]:
    """Load a ground truth map as a numpy array.

    Args:
        subject: Subject ID
        session: Session ID
        output_name: Output name (e.g., 'cbf', 'Ktrans')
        modality: 'dsc' or 'dce'
        derivatives_base: Base directory for derivatives

    Returns:
        Tuple of (data array, file path) or (None, None) if not found.
    """
    import nibabel as nib

    gt_path = find_ground_truth_file(
        subject, session, output_name, modality, derivatives_base
    )
    if gt_path is None:
        return None, None

    try:
        img = nib.load(gt_path)
        data: NDArray[np.floating] = np.asarray(
            img.get_fdata(),  # type: ignore[attr-defined]
            dtype=np.float64,
        )
        return data, gt_path
    except Exception:
        return None, gt_path


def load_brain_mask(
    subject: str,
    session: str,
    derivatives_base: Optional[Path] = None,
    pipeline: str = "dsc_pipeline",
) -> Tuple[Optional[NDArray[np.floating]], Optional[Path]]:
    """Load brain mask for comparison masking.

    Args:
        subject: Subject ID
        session: Session ID
        derivatives_base: Base directory for derivatives
        pipeline: Pipeline name to look for mask (default: dsc_pipeline)

    Returns:
        Tuple of (mask array, file path) or (None, None) if not found.
    """
    import nibabel as nib

    gt_dir = find_ground_truth_directory(
        subject, session, pipeline=pipeline, derivatives_base=derivatives_base
    )
    if gt_dir is None:
        return None, None

    # Look for brain mask in anat directory
    anat_dir = gt_dir / "anat"
    mask_patterns = [
        f"{subject}_{session}*brain_mask.nii.gz",
        f"{subject}*brain_mask.nii.gz",
        "*mask.nii.gz",
    ]

    mask_path = None
    for pattern in mask_patterns:
        search_dirs = [anat_dir, gt_dir]
        for search_dir in search_dirs:
            if search_dir.exists():
                matches = list(search_dir.glob(pattern))
                if matches:
                    mask_path = matches[0]
                    break
        if mask_path:
            break

    if mask_path is None:
        return None, None

    try:
        img = nib.load(mask_path)
        data: NDArray[np.floating] = np.asarray(
            img.get_fdata(),  # type: ignore[attr-defined]
            dtype=np.float64,
        )
        return data, mask_path
    except Exception:
        return None, mask_path


# =============================================================================
# Comparison Functions
# =============================================================================


def compare_numerical_maps(
    pmriprep_data: NDArray[np.floating],
    ground_truth_data: NDArray[np.floating],
    mask: Optional[NDArray[np.floating]] = None,
    threshold: float = CORRELATION_THRESHOLD,
) -> ComparisonResult:
    """Compare two numerical maps using Pearson correlation.

    Args:
        pmriprep_data: PMRIPrep output data array
        ground_truth_data: Ground truth data array
        mask: Optional brain mask for restricting comparison
        threshold: Minimum correlation for passing (default 0.99)

    Returns:
        ComparisonResult with correlation value and pass/fail status
    """
    from pmriprep.utils.registration import calculate_correlation

    try:
        correlation = calculate_correlation(
            pmriprep_data.astype(np.float64),
            ground_truth_data.astype(np.float64),
            mask=mask.astype(np.float64) if mask is not None else None,
        )

        return ComparisonResult(
            output_name="",  # To be set by caller
            metric="correlation",
            value=correlation,
            threshold=threshold,
            passed=correlation >= threshold,
        )
    except Exception as e:
        return ComparisonResult(
            output_name="",
            metric="correlation",
            value=0.0,
            threshold=threshold,
            passed=False,
            error=str(e),
        )


def compare_mask_maps(
    pmriprep_data: NDArray[np.floating],
    ground_truth_data: NDArray[np.floating],
    threshold: float = DICE_THRESHOLD,
) -> ComparisonResult:
    """Compare two binary masks using DICE coefficient.

    Args:
        pmriprep_data: PMRIPrep output mask array
        ground_truth_data: Ground truth mask array
        threshold: Minimum DICE for passing (default 0.95)

    Returns:
        ComparisonResult with DICE value and pass/fail status
    """
    from pmriprep.utils.registration import calculate_dice

    try:
        dice = calculate_dice(
            pmriprep_data.astype(np.float64),
            ground_truth_data.astype(np.float64),
        )

        return ComparisonResult(
            output_name="",  # To be set by caller
            metric="dice",
            value=dice,
            threshold=threshold,
            passed=dice >= threshold,
        )
    except Exception as e:
        return ComparisonResult(
            output_name="",
            metric="dice",
            value=0.0,
            threshold=threshold,
            passed=False,
            error=str(e),
        )


def compare_output_files(
    pmriprep_path: Path,
    ground_truth_path: Path,
    output_name: str,
    is_mask: bool = False,
    mask_path: Optional[Path] = None,
) -> ComparisonResult:
    """Compare PMRIPrep output file against ground truth.

    Args:
        pmriprep_path: Path to PMRIPrep output NIfTI
        ground_truth_path: Path to ground truth NIfTI
        output_name: Name of the output (for reporting)
        is_mask: If True, use DICE comparison; otherwise use correlation
        mask_path: Optional path to brain mask for numerical comparison

    Returns:
        ComparisonResult with comparison metrics
    """
    import nibabel as nib

    try:
        # Load PMRIPrep output
        pmriprep_img = nib.load(pmriprep_path)
        pmriprep_data: NDArray[np.floating] = np.asarray(
            pmriprep_img.get_fdata(),  # type: ignore[attr-defined]
            dtype=np.float64,
        )

        # Load ground truth
        gt_img = nib.load(ground_truth_path)
        gt_data: NDArray[np.floating] = np.asarray(
            gt_img.get_fdata(),  # type: ignore[attr-defined]
            dtype=np.float64,
        )

        # Load mask if provided
        mask_data: Optional[NDArray[np.floating]] = None
        if mask_path is not None and mask_path.exists():
            mask_img = nib.load(mask_path)
            mask_data = np.asarray(
                mask_img.get_fdata(),  # type: ignore[attr-defined]
                dtype=np.float64,
            )

        # Compare
        if is_mask:
            result = compare_mask_maps(pmriprep_data, gt_data)
        else:
            result = compare_numerical_maps(pmriprep_data, gt_data, mask=mask_data)

        result.output_name = output_name
        result.pmriprep_path = pmriprep_path
        result.ground_truth_path = ground_truth_path

        return result

    except Exception as e:
        return ComparisonResult(
            output_name=output_name,
            metric="dice" if is_mask else "correlation",
            value=0.0,
            threshold=DICE_THRESHOLD if is_mask else CORRELATION_THRESHOLD,
            passed=False,
            pmriprep_path=pmriprep_path,
            ground_truth_path=ground_truth_path,
            error=str(e),
        )


# =============================================================================
# Test Subject Execution
# =============================================================================


def run_subject_comparison(
    subject_config: Dict[str, Any],
    pmriprep_derivatives: Path,
    perf_plus_derivatives: Optional[Path] = None,
) -> SubjectTestResult:
    """Run all comparisons for a single test subject.

    Args:
        subject_config: Subject configuration dict from TEST_SUBJECTS
        pmriprep_derivatives: Path to PMRIPrep output derivatives
        perf_plus_derivatives: Path to perf-plus ground truth

    Returns:
        SubjectTestResult with all comparison results
    """
    subject = subject_config["subject"]
    session = subject_config["session"]
    user_story = subject_config["user_story"]
    expected_outputs = subject_config.get("expected_outputs", {})

    comparisons: List[ComparisonResult] = []

    # Load brain mask for numerical comparisons
    mask_data, mask_path = load_brain_mask(
        subject, session, perf_plus_derivatives
    )

    # Compare DSC outputs
    if "dsc" in expected_outputs:
        for output_name in expected_outputs["dsc"]:
            # Find ground truth
            gt_data, gt_path = load_ground_truth_map(
                subject, session, output_name, "dsc", perf_plus_derivatives
            )
            if gt_path is None:
                comparisons.append(
                    ComparisonResult(
                        output_name=f"dsc_{output_name}",
                        metric="correlation",
                        value=0.0,
                        threshold=CORRELATION_THRESHOLD,
                        passed=False,
                        error=f"Ground truth not found for {output_name}",
                    )
                )
                continue

            # Find PMRIPrep output
            pmriprep_dir = pmriprep_derivatives / subject / session / "perf"
            pmriprep_patterns = [
                f"{subject}_{session}*{output_name}.nii.gz",
                f"*{output_name}.nii.gz",
            ]

            pmriprep_path = None
            for pattern in pmriprep_patterns:
                matches = list(pmriprep_dir.glob(pattern)) if pmriprep_dir.exists() else []
                if matches:
                    pmriprep_path = matches[0]
                    break

            if pmriprep_path is None:
                comparisons.append(
                    ComparisonResult(
                        output_name=f"dsc_{output_name}",
                        metric="correlation",
                        value=0.0,
                        threshold=CORRELATION_THRESHOLD,
                        passed=False,
                        ground_truth_path=gt_path,
                        error=f"PMRIPrep output not found for {output_name}",
                    )
                )
                continue

            # Compare
            result = compare_output_files(
                pmriprep_path,
                gt_path,
                f"dsc_{output_name}",
                is_mask=False,
                mask_path=mask_path,
            )
            comparisons.append(result)

    # Compare DCE outputs
    if "dce" in expected_outputs:
        for output_name in expected_outputs["dce"]:
            # Find ground truth
            gt_data, gt_path = load_ground_truth_map(
                subject, session, output_name, "dce", perf_plus_derivatives
            )
            if gt_path is None:
                comparisons.append(
                    ComparisonResult(
                        output_name=f"dce_{output_name}",
                        metric="correlation",
                        value=0.0,
                        threshold=CORRELATION_THRESHOLD,
                        passed=False,
                        error=f"Ground truth not found for {output_name}",
                    )
                )
                continue

            # Find PMRIPrep output
            pmriprep_dir = pmriprep_derivatives / subject / session / "perf"
            pmriprep_patterns = [
                f"{subject}_{session}*{output_name}.nii.gz",
                f"*{output_name}.nii.gz",
            ]

            pmriprep_path = None
            for pattern in pmriprep_patterns:
                matches = list(pmriprep_dir.glob(pattern)) if pmriprep_dir.exists() else []
                if matches:
                    pmriprep_path = matches[0]
                    break

            if pmriprep_path is None:
                comparisons.append(
                    ComparisonResult(
                        output_name=f"dce_{output_name}",
                        metric="correlation",
                        value=0.0,
                        threshold=CORRELATION_THRESHOLD,
                        passed=False,
                        ground_truth_path=gt_path,
                        error=f"PMRIPrep output not found for {output_name}",
                    )
                )
                continue

            # Compare
            result = compare_output_files(
                pmriprep_path,
                gt_path,
                f"dce_{output_name}",
                is_mask=False,
                mask_path=mask_path,
            )
            comparisons.append(result)

    # Determine overall pass/fail
    all_passed = all(c.passed for c in comparisons) if comparisons else False

    return SubjectTestResult(
        subject=subject,
        session=session,
        user_story=user_story,
        comparisons=comparisons,
        all_passed=all_passed,
    )


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def perf_plus_derivatives_path() -> Path:
    """Fixture providing path to perf-plus derivatives."""
    return PERF_PLUS_DERIVATIVES


@pytest.fixture
def test_subjects_config() -> List[Dict[str, Any]]:
    """Fixture providing test subjects configuration."""
    return TEST_SUBJECTS


# =============================================================================
# Pytest Parametrized Tests
# =============================================================================


def get_test_subject_ids() -> List[str]:
    """Get list of test subject IDs for parametrization."""
    return [f"{s['subject']}_{s['session']}" for s in TEST_SUBJECTS]


def get_test_subject_by_id(subject_session: str) -> Optional[Dict[str, Any]]:
    """Get test subject config by combined ID."""
    for subject_config in TEST_SUBJECTS:
        if f"{subject_config['subject']}_{subject_config['session']}" == subject_session:
            return subject_config
    return None


@pytest.mark.integration
@pytest.mark.parametrize("subject_session", get_test_subject_ids())
def test_ground_truth_comparison(
    subject_session: str,
    tmp_path: Path,
) -> None:
    """Test PMRIPrep output against ground truth.

    This test is parameterized to run for each subject defined in TEST_SUBJECTS.
    It compares PMRIPrep outputs against perf-plus ground truth data.

    The test will skip if:
    - Ground truth data is not available
    - PMRIPrep has not been run for this subject

    Args:
        subject_session: Combined subject_session ID (e.g., 'sub-0001_ses-0002')
        tmp_path: Pytest temporary directory
    """
    subject_config = get_test_subject_by_id(subject_session)
    assert subject_config is not None, f"Unknown subject: {subject_session}"

    # Check if ground truth exists
    gt_dir = find_ground_truth_directory(
        subject_config["subject"],
        subject_config["session"],
    )
    if gt_dir is None:
        pytest.skip(f"Ground truth not found for {subject_session}")

    # For now, skip if PMRIPrep hasn't been run
    # In actual testing, this would point to real PMRIPrep outputs
    pmriprep_derivatives = tmp_path / "pmriprep"
    if not pmriprep_derivatives.exists():
        pytest.skip(f"PMRIPrep outputs not available for {subject_session}")

    # Run comparison
    result = run_subject_comparison(
        subject_config,
        pmriprep_derivatives,
        PERF_PLUS_DERIVATIVES,
    )

    # Report results
    for comparison in result.comparisons:
        if comparison.error:
            pytest.fail(
                f"{comparison.output_name}: {comparison.error}"
            )
        elif not comparison.passed:
            pytest.fail(
                f"{comparison.output_name}: {comparison.metric}={comparison.value:.4f} "
                f"< threshold {comparison.threshold}"
            )

    assert result.all_passed, f"Not all comparisons passed for {subject_session}"


# =============================================================================
# Utility Functions for Manual Testing
# =============================================================================


def print_comparison_report(result: SubjectTestResult) -> None:
    """Print a formatted comparison report for a subject.

    Args:
        result: SubjectTestResult to report
    """
    print(f"\n{'='*60}")
    print(f"Subject: {result.subject} / {result.session}")
    print(f"User Story: {result.user_story}")
    print(f"Overall: {'PASSED' if result.all_passed else 'FAILED'}")
    print(f"{'='*60}")

    for comp in result.comparisons:
        status = "PASS" if comp.passed else "FAIL"
        print(f"  [{status}] {comp.output_name}: {comp.metric}={comp.value:.4f}")
        if comp.error:
            print(f"         Error: {comp.error}")

    print()


def run_all_comparisons(
    pmriprep_derivatives: Path,
    perf_plus_derivatives: Optional[Path] = None,
) -> List[SubjectTestResult]:
    """Run comparisons for all test subjects.

    Args:
        pmriprep_derivatives: Path to PMRIPrep output derivatives
        perf_plus_derivatives: Path to perf-plus ground truth

    Returns:
        List of SubjectTestResult for all subjects
    """
    results = []
    for subject_config in TEST_SUBJECTS:
        result = run_subject_comparison(
            subject_config,
            pmriprep_derivatives,
            perf_plus_derivatives,
        )
        results.append(result)
        print_comparison_report(result)

    return results
