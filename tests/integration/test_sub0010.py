"""Integration test for sub-0010/ses-0001: tikhonov, tofts.

US-093: Test DSC + DCE pipeline with:
- CLI: --dsc-preload --dsc-dose half --dsc-deconv-method tikhonov --dce-model tofts --dce-scale-aif
- DSC: tikhonov deconvolution
- DCE: tofts model

Config file: tests/fixtures/sub-0010_config.yaml
Ground truth: ~/Developer/perf-plus/bidsroot/derivatives/{dsc,dce}_pipeline/sub-0010/ses-0001/
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

SUBJECT = "sub-0010"
SESSION = "ses-0001"
USER_STORY = "US-093"

CLI_ARGS = [
    "--dsc-preload",
    "--dsc-dose", "half",
    "--dsc-deconv-method", "tikhonov",
    "--dce-model", "tofts",
    "--dce-scale-aif",
]

CONFIG_FILE = Path(__file__).parent.parent / "fixtures" / "sub-0010_config.yaml"

DSC_CONFIG: Dict[str, Any] = {
    "preload_detected": True,
    "dose": "half",
    "method": "tikhonov",
    "bayesian": "none",
}

DCE_CONFIG: Dict[str, Any] = {
    "model": "tofts",
    "bayesian": "none",
    "scale_aif": True,
}

EXPECTED_DSC_OUTPUTS = ["cbf", "cbv", "mtt", "tmax"]
EXPECTED_DCE_OUTPUTS = ["Ktrans", "ve", "kep"]


@pytest.fixture
def dsc_ground_truth_dir() -> Optional[Path]:
    """Get the DSC ground truth directory for this subject."""
    return find_ground_truth_directory(SUBJECT, SESSION, "dsc_pipeline", PERF_PLUS_DERIVATIVES)


@pytest.fixture
def dce_ground_truth_dir() -> Optional[Path]:
    """Get the DCE ground truth directory for this subject."""
    return find_ground_truth_directory(SUBJECT, SESSION, "dce_pipeline", PERF_PLUS_DERIVATIVES)


@pytest.fixture
def subject_brain_mask(dsc_ground_truth_dir: Optional[Path]) -> Optional[Path]:
    """Get brain mask path for this subject."""
    if dsc_ground_truth_dir is None:
        return None
    _, mask_path = load_brain_mask(SUBJECT, SESSION, PERF_PLUS_DERIVATIVES)
    return mask_path


@pytest.mark.integration
class TestSub0010Ses0001:
    """Integration tests for sub-0010/ses-0001."""

    def test_ground_truth_exists(self, dsc_ground_truth_dir: Optional[Path]) -> None:
        """Verify ground truth data exists for this subject."""
        if dsc_ground_truth_dir is None:
            pytest.skip(f"Ground truth not found for {SUBJECT}/{SESSION}")
        assert dsc_ground_truth_dir.exists()

    def test_config_file_exists(self) -> None:
        assert CONFIG_FILE.exists(), f"Config file not found: {CONFIG_FILE}"

    @pytest.mark.parametrize("output_name", EXPECTED_DSC_OUTPUTS)
    def test_dsc_output_ground_truth(
        self,
        output_name: str,
        dsc_ground_truth_dir: Optional[Path],
        subject_brain_mask: Optional[Path],
    ) -> None:
        """Test DSC output against ground truth."""
        import os

        if dsc_ground_truth_dir is None:
            pytest.skip(f"Ground truth not found for {SUBJECT}/{SESSION}")

        gt_data, gt_path = load_ground_truth_map(
            SUBJECT, SESSION, output_name, "dsc", PERF_PLUS_DERIVATIVES
        )
        if gt_path is None:
            pytest.skip(f"Ground truth not found for DSC {output_name}")

        work_dir = os.environ.get("PMRIPREP_WORK_DIR")
        if work_dir is None:
            pytest.skip(
                "PMRIPREP_WORK_DIR not set. Set to work directory path to enable "
                "ground truth comparison."
            )

        work_path = Path(work_dir)
        subject_id = SUBJECT.replace("sub-", "")
        pmriprep_output = (
            work_path / f"pmriprep_sub{subject_id}_{SESSION}_wf" / "dsc_wf" / "compute_maps"
        )

        if not pmriprep_output.exists():
            pytest.skip(
                f"PMRIPrep outputs not available for {SUBJECT}/{SESSION} in {pmriprep_output}. "
                "Run PMRIPrep first."
            )

        if output_name in ("cbf", "cbv"):
            pmriprep_files = list(pmriprep_output.glob(f"*_rel_{output_name}.nii*"))
        else:
            pmriprep_files = list(pmriprep_output.glob(f"*_{output_name}.nii*"))
        if not pmriprep_files:
            pytest.fail(f"PMRIPrep output not found for {output_name} in {pmriprep_output}")

        result = compare_output_files(
            pmriprep_files[0], gt_path, f"dsc_{output_name}",
            is_mask=False, mask_path=subject_brain_mask,
        )
        assert result.passed, (
            f"DSC {output_name} correlation {result.value:.4f} < "
            f"threshold {CORRELATION_THRESHOLD}"
        )

    @pytest.mark.parametrize("output_name", EXPECTED_DCE_OUTPUTS)
    def test_dce_output_ground_truth(
        self,
        output_name: str,
        dce_ground_truth_dir: Optional[Path],
    ) -> None:
        """Test DCE output against ground truth.

        Note: No brain mask is used because DSC mask (128x128x19) doesn't
        match DCE native resolution (256x256x26). Correlation is computed
        over the full volume.
        """
        import os

        if dce_ground_truth_dir is None:
            pytest.skip(f"Ground truth not found for {SUBJECT}/{SESSION}")

        gt_data, gt_path = load_ground_truth_map(
            SUBJECT, SESSION, output_name, "dce", PERF_PLUS_DERIVATIVES
        )
        if gt_path is None:
            pytest.skip(f"Ground truth not found for DCE {output_name}")

        work_dir = os.environ.get("PMRIPREP_WORK_DIR")
        if work_dir is None:
            pytest.skip(
                "PMRIPREP_WORK_DIR not set. Set to work directory path to enable "
                "ground truth comparison."
            )

        work_path = Path(work_dir)
        subject_id = SUBJECT.replace("sub-", "")
        pmriprep_output = (
            work_path / f"pmriprep_sub{subject_id}_{SESSION}_wf" / "dce_wf" / "pk_modeling"
        )

        if not pmriprep_output.exists():
            pytest.skip(
                f"PMRIPrep outputs not available for {SUBJECT}/{SESSION} in {pmriprep_output}. "
                "Run PMRIPrep first."
            )

        pmriprep_files = list(pmriprep_output.glob(f"*_{output_name}.nii*"))
        if not pmriprep_files:
            pytest.fail(f"PMRIPrep output not found for {output_name} in {pmriprep_output}")

        result = compare_output_files(
            pmriprep_files[0], gt_path, f"dce_{output_name}",
            is_mask=False, mask_path=None,
        )
        assert result.passed, (
            f"DCE {output_name} correlation {result.value:.4f} < "
            f"threshold {CORRELATION_THRESHOLD}"
        )


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
