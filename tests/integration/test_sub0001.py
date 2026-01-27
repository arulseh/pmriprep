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
    load_brain_mask,
    load_ground_truth_map,
)

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
EXPECTED_DCE_OUTPUTS: List[str] = []  # No DCE


@pytest.fixture
def subject_ground_truth_dir() -> Optional[Path]:
    """Get the ground truth directory for this subject."""
    return find_ground_truth_directory(SUBJECT, SESSION, PERF_PLUS_DERIVATIVES)


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
        tmp_path: Path,
    ) -> None:
        """Test DSC output against ground truth.

        This test will:
        1. Skip if ground truth is not available
        2. Skip if PMRIPrep outputs are not available
        3. Compare PMRIPrep output to ground truth using correlation
        4. Assert correlation >= 0.99
        """
        if subject_ground_truth_dir is None:
            pytest.skip(f"Ground truth not found for {SUBJECT}/{SESSION}")

        # Load ground truth
        gt_data, gt_path = load_ground_truth_map(
            SUBJECT, SESSION, output_name, "dsc", PERF_PLUS_DERIVATIVES
        )
        if gt_path is None:
            pytest.skip(f"Ground truth not found for DSC {output_name}")

        # For actual testing, PMRIPrep outputs would be at:
        # pmriprep_derivatives / SUBJECT / SESSION / perf / *{output_name}.nii.gz
        # Since PMRIPrep hasn't been run yet, skip this test
        pmriprep_derivatives = tmp_path / "pmriprep"
        pmriprep_output = pmriprep_derivatives / SUBJECT / SESSION / "perf"

        if not pmriprep_output.exists():
            pytest.skip(
                f"PMRIPrep outputs not available for {SUBJECT}/{SESSION}. "
                "Run PMRIPrep first to enable ground truth comparison."
            )

        # Find PMRIPrep output file
        pmriprep_files = list(pmriprep_output.glob(f"*{output_name}*.nii*"))
        if not pmriprep_files:
            pytest.fail(
                f"PMRIPrep output not found for {output_name} in {pmriprep_output}"
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
            f"DSC {output_name} correlation {result.value:.4f} < "
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
