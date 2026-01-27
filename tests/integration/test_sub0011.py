"""Integration test for sub-0011/ses-0001: cpi, extended_tofts.

US-094: Test DSC + DCE pipeline with:
- CLI: --dsc-preload --dsc-dose half --dsc-deconv-method cpi --dce-model extended_tofts --dce-scale-aif
- DSC: CPI (Control Point Interpolation) deconvolution
- DCE: extended_tofts model

Config file: tests/fixtures/sub-0011_config.yaml
Ground truth: ~/Developer/perf-plus/bidsroot/derivatives/sub-0011/ses-0001/
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

SUBJECT = "sub-0011"
SESSION = "ses-0001"
USER_STORY = "US-094"

CLI_ARGS = [
    "--dsc-preload",
    "--dsc-dose", "half",
    "--dsc-deconv-method", "cpi",
    "--dce-model", "extended_tofts",
    "--dce-scale-aif",
]

CONFIG_FILE = Path(__file__).parent.parent / "fixtures" / "sub-0011_config.yaml"

DSC_CONFIG: Dict[str, Any] = {
    "preload_detected": True,
    "dose": "half",
    "method": "cpi",
    "bayesian": "none",
    "cpi_control_points": 0,  # Auto-select
}

DCE_CONFIG: Dict[str, Any] = {
    "model": "extended_tofts",
    "bayesian": "none",
    "scale_aif": True,
}

EXPECTED_DSC_OUTPUTS = ["cbf", "cbv", "mtt", "tmax"]
EXPECTED_DCE_OUTPUTS = ["Ktrans", "ve", "vp", "kep"]


@pytest.fixture
def subject_ground_truth_dir() -> Optional[Path]:
    return find_ground_truth_directory(SUBJECT, SESSION, PERF_PLUS_DERIVATIVES)


@pytest.fixture
def subject_brain_mask(subject_ground_truth_dir: Optional[Path]) -> Optional[Path]:
    if subject_ground_truth_dir is None:
        return None
    _, mask_path = load_brain_mask(SUBJECT, SESSION, PERF_PLUS_DERIVATIVES)
    return mask_path


@pytest.mark.integration
class TestSub0011Ses0001:
    """Integration tests for sub-0011/ses-0001."""

    def test_ground_truth_exists(self, subject_ground_truth_dir: Optional[Path]) -> None:
        if subject_ground_truth_dir is None:
            pytest.skip(f"Ground truth not found for {SUBJECT}/{SESSION}")
        assert subject_ground_truth_dir.exists()

    def test_config_file_exists(self) -> None:
        assert CONFIG_FILE.exists(), f"Config file not found: {CONFIG_FILE}"

    @pytest.mark.parametrize("output_name", EXPECTED_DSC_OUTPUTS)
    def test_dsc_output_ground_truth(
        self, output_name: str, subject_ground_truth_dir: Optional[Path],
        subject_brain_mask: Optional[Path], tmp_path: Path,
    ) -> None:
        if subject_ground_truth_dir is None:
            pytest.skip(f"Ground truth not found for {SUBJECT}/{SESSION}")

        gt_data, gt_path = load_ground_truth_map(
            SUBJECT, SESSION, output_name, "dsc", PERF_PLUS_DERIVATIVES
        )
        if gt_path is None:
            pytest.skip(f"Ground truth not found for DSC {output_name}")

        pmriprep_output = tmp_path / "pmriprep" / SUBJECT / SESSION / "perf"
        if not pmriprep_output.exists():
            pytest.skip(f"PMRIPrep outputs not available for {SUBJECT}/{SESSION}")

        pmriprep_files = list(pmriprep_output.glob(f"*{output_name}*.nii*"))
        if not pmriprep_files:
            pytest.fail(f"PMRIPrep output not found for {output_name}")

        result = compare_output_files(
            pmriprep_files[0], gt_path, f"dsc_{output_name}",
            is_mask=False, mask_path=subject_brain_mask,
        )
        assert result.passed, f"DSC {output_name} r={result.value:.4f} < {CORRELATION_THRESHOLD}"

    @pytest.mark.parametrize("output_name", EXPECTED_DCE_OUTPUTS)
    def test_dce_output_ground_truth(
        self, output_name: str, subject_ground_truth_dir: Optional[Path],
        subject_brain_mask: Optional[Path], tmp_path: Path,
    ) -> None:
        if subject_ground_truth_dir is None:
            pytest.skip(f"Ground truth not found for {SUBJECT}/{SESSION}")

        gt_data, gt_path = load_ground_truth_map(
            SUBJECT, SESSION, output_name, "dce", PERF_PLUS_DERIVATIVES
        )
        if gt_path is None:
            pytest.skip(f"Ground truth not found for DCE {output_name}")

        pmriprep_output = tmp_path / "pmriprep" / SUBJECT / SESSION / "perf"
        if not pmriprep_output.exists():
            pytest.skip(f"PMRIPrep outputs not available for {SUBJECT}/{SESSION}")

        pmriprep_files = list(pmriprep_output.glob(f"*{output_name}*.nii*"))
        if not pmriprep_files:
            pytest.fail(f"PMRIPrep output not found for {output_name}")

        result = compare_output_files(
            pmriprep_files[0], gt_path, f"dce_{output_name}",
            is_mask=False, mask_path=subject_brain_mask,
        )
        assert result.passed, f"DCE {output_name} r={result.value:.4f} < {CORRELATION_THRESHOLD}"


def get_expected_cli_command(bids_dir: Path, output_dir: Path, work_dir: Optional[Path] = None) -> List[str]:
    cmd = ["pmriprep", str(bids_dir), str(output_dir), "participant",
           "--participant-label", SUBJECT.replace("sub-", ""),
           "--session-label", SESSION.replace("ses-", ""),
           "--config-file", str(CONFIG_FILE)]
    cmd.extend(CLI_ARGS)
    if work_dir:
        cmd.extend(["-w", str(work_dir)])
    return cmd


def run_subject_test(pmriprep_derivatives: Path, perf_plus_derivatives: Optional[Path] = None) -> SubjectTestResult:
    if perf_plus_derivatives is None:
        perf_plus_derivatives = PERF_PLUS_DERIVATIVES

    comparisons: List[ComparisonResult] = []
    _, mask_path = load_brain_mask(SUBJECT, SESSION, perf_plus_derivatives)

    for output_name in EXPECTED_DSC_OUTPUTS:
        gt_data, gt_path = load_ground_truth_map(SUBJECT, SESSION, output_name, "dsc", perf_plus_derivatives)
        if gt_path is None:
            comparisons.append(ComparisonResult(output_name=f"dsc_{output_name}", metric="correlation",
                value=0.0, threshold=CORRELATION_THRESHOLD, passed=False, error=f"GT not found for {output_name}"))
            continue
        pmriprep_dir = pmriprep_derivatives / SUBJECT / SESSION / "perf"
        pmriprep_files = list(pmriprep_dir.glob(f"*{output_name}*.nii*")) if pmriprep_dir.exists() else []
        if not pmriprep_files:
            comparisons.append(ComparisonResult(output_name=f"dsc_{output_name}", metric="correlation",
                value=0.0, threshold=CORRELATION_THRESHOLD, passed=False, ground_truth_path=gt_path,
                error=f"PMRIPrep output not found for {output_name}"))
            continue
        result = compare_output_files(pmriprep_files[0], gt_path, f"dsc_{output_name}", is_mask=False, mask_path=mask_path)
        comparisons.append(result)

    for output_name in EXPECTED_DCE_OUTPUTS:
        gt_data, gt_path = load_ground_truth_map(SUBJECT, SESSION, output_name, "dce", perf_plus_derivatives)
        if gt_path is None:
            comparisons.append(ComparisonResult(output_name=f"dce_{output_name}", metric="correlation",
                value=0.0, threshold=CORRELATION_THRESHOLD, passed=False, error=f"GT not found for {output_name}"))
            continue
        pmriprep_dir = pmriprep_derivatives / SUBJECT / SESSION / "perf"
        pmriprep_files = list(pmriprep_dir.glob(f"*{output_name}*.nii*")) if pmriprep_dir.exists() else []
        if not pmriprep_files:
            comparisons.append(ComparisonResult(output_name=f"dce_{output_name}", metric="correlation",
                value=0.0, threshold=CORRELATION_THRESHOLD, passed=False, ground_truth_path=gt_path,
                error=f"PMRIPrep output not found for {output_name}"))
            continue
        result = compare_output_files(pmriprep_files[0], gt_path, f"dce_{output_name}", is_mask=False, mask_path=mask_path)
        comparisons.append(result)

    all_passed = all(c.passed for c in comparisons) if comparisons else False
    return SubjectTestResult(subject=SUBJECT, session=SESSION, user_story=USER_STORY, comparisons=comparisons, all_passed=all_passed)
