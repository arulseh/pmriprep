"""Integration test for sub-0001/ses-0002: DSC-only, no_preload, full_dose, sSVD.

US-091: Test DSC pipeline with:
- CLI: --dsc-no-preload --dsc-dose full
- DSC options: sSVD (default deconvolution method)
- No DCE data for this subject

Ground truth: ~/Developer/perf-plus/bidsroot/derivatives/sub-0001/ses-0002/
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from tests.integration.test_ground_truth import (
    CORRELATION_THRESHOLD,
    PERF_PLUS_DERIVATIVES,
    ComparisonResult,
    SubjectTestResult,
    compare_output_files,
    find_ground_truth_directory,
    get_correlation_threshold,
    load_brain_mask,
    load_ground_truth_map,
)

# PMRIPrep test outputs directory
PMRIPREP_TEST_OUTPUTS = Path(__file__).parent.parent.parent / "test_outputs"

# Subject-specific configuration
SUBJECT = "sub-0001"
SESSION = "ses-0002"
USER_STORY = "US-091"

# CLI arguments for this test
CLI_ARGS = [
    "--dsc-no-preload",
    "--dsc-dose", "full",
    # sSVD is default, no need to specify
]

# DSC configuration
DSC_CONFIG: Dict[str, Any] = {
    "preload_detected": False,
    "dose": "full",
    "method": "sSVD",
    "bayesian": "none",
}

# No DCE for this subject
DCE_CONFIG: Optional[Dict[str, Any]] = None

# Expected outputs
EXPECTED_DSC_OUTPUTS = ["cbf", "cbv", "mtt", "tmax"]
EXPECTED_DSC_MODEL_FREE = ["ph", "psr", "ttp", "cnr", "snr", "tsnr"]
EXPECTED_DCE_OUTPUTS: List[str] = []  # No DCE

# Specific file patterns for PMRIPrep outputs to avoid ambiguous matches
DSC_OUTPUT_PATTERNS: Dict[str, List[str]] = {
    "cbf": ["*desc-rel_cbf.nii*", "*_cbf.nii*"],
    "cbv": ["*desc-rel_cbv.nii*", "*_cbv.nii*"],
    "mtt": ["*desc-dscproc_mtt.nii*", "*_mtt.nii*"],
    "tmax": ["*desc-dscproc_tmax.nii*", "*_tmax.nii*"],
}


def find_pmriprep_output(
    output_dir: Path, output_name: str, patterns: Dict[str, List[str]]
) -> Optional[Path]:
    """Find PMRIPrep output file using specific patterns to avoid ambiguous matches."""
    if output_name in patterns:
        for pattern in patterns[output_name]:
            matches = list(output_dir.glob(pattern))
            if matches:
                return matches[0]
    # Fallback to generic pattern
    generic_matches = list(output_dir.glob(f"*_{output_name}.nii*"))
    if generic_matches:
        return generic_matches[0]
    return None


@pytest.fixture
def subject_ground_truth_dir() -> Optional[Path]:
    """Get the ground truth directory for this subject."""
    return find_ground_truth_directory(SUBJECT, SESSION, "dsc_pipeline", PERF_PLUS_DERIVATIVES)


@pytest.fixture
def subject_brain_mask(
    subject_ground_truth_dir: Optional[Path],
) -> Optional[Path]:
    """Get brain mask path for this subject."""
    if subject_ground_truth_dir is None:
        return None
    _, mask_path = load_brain_mask(SUBJECT, SESSION, PERF_PLUS_DERIVATIVES)
    return mask_path


@pytest.mark.integration
class TestSub0001Ses0002:
    """Integration tests for sub-0001/ses-0002."""

    def test_ground_truth_exists(
        self, subject_ground_truth_dir: Optional[Path]
    ) -> None:
        """Verify ground truth data exists for this subject."""
        if subject_ground_truth_dir is None:
            pytest.skip(
                f"Ground truth directory not found for {SUBJECT}/{SESSION}"
            )
        assert subject_ground_truth_dir.exists(), (
            f"Ground truth directory does not exist: {subject_ground_truth_dir}"
        )

    def test_cli_configuration(self) -> None:
        """Verify CLI arguments match expected configuration."""
        assert "--dsc-no-preload" in CLI_ARGS
        assert "--dsc-dose" in CLI_ARGS
        assert "full" in CLI_ARGS
        assert DSC_CONFIG["method"] == "sSVD"
        assert DCE_CONFIG is None  # DSC-only subject

    @pytest.mark.parametrize("output_name", EXPECTED_DSC_OUTPUTS)
    def test_dsc_output_ground_truth(
        self,
        output_name: str,
        subject_ground_truth_dir: Optional[Path],
        subject_brain_mask: Optional[Path],
    ) -> None:
        """Test DSC output against ground truth."""
        if subject_ground_truth_dir is None:
            pytest.skip(f"Ground truth not found for {SUBJECT}/{SESSION}")

        # Load ground truth
        gt_data, gt_path = load_ground_truth_map(
            SUBJECT, SESSION, output_name, "dsc", PERF_PLUS_DERIVATIVES
        )
        if gt_path is None:
            pytest.skip(f"Ground truth not found for DSC {output_name}")

        pmriprep_output_dir = PMRIPREP_TEST_OUTPUTS / "dsc_pipeline" / SUBJECT / SESSION / "perf"

        if not pmriprep_output_dir.exists():
            pytest.skip(
                f"PMRIPrep outputs not available for {SUBJECT}/{SESSION}"
            )

        pmriprep_file = find_pmriprep_output(
            pmriprep_output_dir, output_name, DSC_OUTPUT_PATTERNS
        )
        if pmriprep_file is None:
            pytest.fail(
                f"PMRIPrep output not found for {output_name} in {pmriprep_output_dir}"
            )

        # Compare
        result = compare_output_files(
            pmriprep_file,
            gt_path,
            f"dsc_{output_name}",
            is_mask=False,
            mask_path=subject_brain_mask,
        )

        threshold = get_correlation_threshold(output_name, "dsc")
        passed = result.value >= threshold

        assert passed, (
            f"DSC {output_name} correlation {result.value:.4f} < "
            f"threshold {threshold}"
        )

    @pytest.mark.parametrize("output_name", EXPECTED_DSC_MODEL_FREE)
    def test_dsc_model_free_ground_truth(
        self,
        output_name: str,
        subject_ground_truth_dir: Optional[Path],
        subject_brain_mask: Optional[Path],
    ) -> None:
        """Test model-free DSC maps against ground truth.

        These maps do not depend on AIF selection or deconvolution:
        - ph: peak height (signal-based)
        - psr: percent signal recovery (signal-based)
        - ttp: time to peak concentration
        - cnr: contrast-to-noise ratio
        - snr: signal-to-noise ratio (baseline)
        - tsnr: temporal signal-to-noise ratio
        """
        if subject_ground_truth_dir is None:
            pytest.skip(f"Ground truth not found for {SUBJECT}/{SESSION}")

        # Load ground truth
        gt_data, gt_path = load_ground_truth_map(
            SUBJECT, SESSION, output_name, "dsc", PERF_PLUS_DERIVATIVES
        )
        if gt_path is None:
            pytest.skip(f"Ground truth not found for DSC {output_name}")

        pmriprep_output_dir = PMRIPREP_TEST_OUTPUTS / "dsc_pipeline" / SUBJECT / SESSION / "perf"

        if not pmriprep_output_dir.exists():
            pytest.skip(
                f"PMRIPrep outputs not available for {SUBJECT}/{SESSION}"
            )

        # Use underscore-prefixed pattern to avoid ambiguity (e.g., snr vs tsnr)
        pmriprep_files = list(pmriprep_output_dir.glob(f"*_{output_name}.nii*"))
        if not pmriprep_files:
            pytest.fail(
                f"PMRIPrep output not found for {output_name} in {pmriprep_output_dir}"
            )

        # Compare
        result = compare_output_files(
            pmriprep_files[0],
            gt_path,
            f"dsc_{output_name}",
            is_mask=False,
            mask_path=subject_brain_mask,
        )

        assert result.passed, (
            f"DSC {output_name} (model-free) correlation {result.value:.4f} < "
            f"threshold {CORRELATION_THRESHOLD}"
        )

    def test_no_dce_outputs_expected(self) -> None:
        """Verify no DCE outputs are expected for this DSC-only subject."""
        assert len(EXPECTED_DCE_OUTPUTS) == 0, (
            "sub-0001 is DSC-only, no DCE outputs should be expected"
        )
        assert DCE_CONFIG is None, (
            "DCE config should be None for DSC-only subject"
        )


def get_expected_cli_command(
    bids_dir: Path,
    output_dir: Path,
    work_dir: Optional[Path] = None,
) -> List[str]:
    """Generate the expected CLI command for running PMRIPrep on this subject.

    Args:
        bids_dir: BIDS input directory
        output_dir: Output directory for derivatives
        work_dir: Optional working directory

    Returns:
        List of CLI arguments for subprocess.run()
    """
    cmd = [
        "pmriprep",
        str(bids_dir),
        str(output_dir),
        "participant",
        "--participant-label", SUBJECT.replace("sub-", ""),
        "--session-label", SESSION.replace("ses-", ""),
    ]
    cmd.extend(CLI_ARGS)

    if work_dir:
        cmd.extend(["-w", str(work_dir)])

    return cmd


def run_subject_test(
    pmriprep_derivatives: Path,
    perf_plus_derivatives: Optional[Path] = None,
) -> SubjectTestResult:
    """Run integration test for this subject.

    This function can be called programmatically to run the test
    outside of pytest.

    Args:
        pmriprep_derivatives: Path to PMRIPrep output derivatives
        perf_plus_derivatives: Path to perf-plus ground truth

    Returns:
        SubjectTestResult with all comparison results
    """
    if perf_plus_derivatives is None:
        perf_plus_derivatives = PERF_PLUS_DERIVATIVES

    comparisons: List[ComparisonResult] = []

    # Load brain mask
    _, mask_path = load_brain_mask(SUBJECT, SESSION, perf_plus_derivatives)

    # Compare DSC outputs
    for output_name in EXPECTED_DSC_OUTPUTS:
        gt_data, gt_path = load_ground_truth_map(
            SUBJECT, SESSION, output_name, "dsc", perf_plus_derivatives
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
        pmriprep_dir = pmriprep_derivatives / SUBJECT / SESSION / "perf"
        pmriprep_files = (
            list(pmriprep_dir.glob(f"*{output_name}*.nii*"))
            if pmriprep_dir.exists()
            else []
        )

        if not pmriprep_files:
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

        result = compare_output_files(
            pmriprep_files[0],
            gt_path,
            f"dsc_{output_name}",
            is_mask=False,
            mask_path=mask_path,
        )
        comparisons.append(result)

    all_passed = all(c.passed for c in comparisons) if comparisons else False

    return SubjectTestResult(
        subject=SUBJECT,
        session=SESSION,
        user_story=USER_STORY,
        comparisons=comparisons,
        all_passed=all_passed,
    )
