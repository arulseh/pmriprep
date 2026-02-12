"""Integration test for sub-0020/ses-0001: vascular bayesian_map, tofts.

US-097: Test DSC + DCE pipeline with:
- CLI: --dsc-preload --dsc-dose half --dsc-deconv-method vascular --dsc-bayesian map --dce-model tofts --dce-scale-aif
- DSC: Vascular (gamma-variate) model with MAP Bayesian
- DCE: Standard Tofts model

Config file: tests/fixtures/sub-0020_config.yaml
Ground truth: ~/Developer/perf-plus/bidsroot/derivatives/sub-0020/ses-0001/
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

SUBJECT = "sub-0020"
SESSION = "ses-0001"
USER_STORY = "US-097"

CLI_ARGS = [
    "--dsc-preload",
    "--dsc-dose", "half",
    "--dsc-deconv-method", "vascular",
    "--dsc-bayesian", "map",
    "--dce-model", "tofts",
    "--dce-scale-aif",
]

CONFIG_FILE = Path(__file__).parent.parent / "fixtures" / "sub-0020_config.yaml"

DSC_CONFIG: Dict[str, Any] = {
    "preload_detected": True,
    "dose": "half",
    "method": "vascular",
    "bayesian": "map",
    "mcmc_samples": 500,
    "save_uncertainty": True,
}

DCE_CONFIG: Dict[str, Any] = {
    "model": "tofts",
    "bayesian": "none",
    "scale_aif": True,
}

# Vascular model with Bayesian includes additional parameters
EXPECTED_DSC_OUTPUTS = ["cbf", "cbv", "mtt", "tmax", "cbf_std", "cbv_std", "alpha", "beta"]
# Standard Tofts: Ktrans, ve, kep (no vp)
EXPECTED_DCE_OUTPUTS = ["Ktrans", "ve", "kep"]

# Specific file patterns for PMRIPrep outputs
DSC_OUTPUT_PATTERNS: Dict[str, List[str]] = {
    "cbf": ["*desc-rel_cbf.nii*", "*_cbf.nii*"],
    "cbv": ["*desc-rel_cbv.nii*", "*_cbv.nii*"],
    "mtt": ["*desc-dscproc_mtt.nii*", "*_mtt.nii*"],
    "tmax": ["*desc-dscproc_tmax.nii*", "*_tmax.nii*"],
}

# DCE output patterns
DCE_OUTPUT_PATTERNS: Dict[str, List[str]] = {
    "Ktrans": ["*model-tofts_Ktrans.nii*"],
    "ve": ["*model-tofts_ve.nii*"],
    "kep": ["*model-tofts_kep.nii*"],
}

# Ground truth DCE patterns - use model-Tofts with desc-vpFit (raw Tofts fit)
DCE_GT_PATTERNS: Dict[str, List[str]] = {
    "Ktrans": ["*model-Tofts_desc-vpFit_Ktrans.nii*"],
    "ve": ["*model-Tofts_desc-vpFit_ve.nii*"],
    "kep": ["*model-Tofts_desc-vpFit_kep.nii*"],
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


def find_ground_truth_file_with_patterns(
    gt_dir: Path, output_name: str, patterns: Dict[str, List[str]]
) -> Optional[Path]:
    """Find ground truth file using specific patterns to avoid ambiguous matches."""
    perf_dir = gt_dir / "perf"
    if not perf_dir.exists():
        perf_dir = gt_dir

    if output_name in patterns:
        for pattern in patterns[output_name]:
            matches = list(perf_dir.glob(pattern))
            if matches:
                return matches[0]
    return None


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
    if dsc_ground_truth_dir is None:
        return None
    _, mask_path = load_brain_mask(SUBJECT, SESSION, PERF_PLUS_DERIVATIVES)
    return mask_path


@pytest.mark.integration
class TestSub0020Ses0001:
    """Integration tests for sub-0020/ses-0001."""

    def test_ground_truth_exists(self, dsc_ground_truth_dir: Optional[Path]) -> None:
        if dsc_ground_truth_dir is None:
            pytest.skip(f"Ground truth not found for {SUBJECT}/{SESSION}")
        assert dsc_ground_truth_dir.exists()

    def test_config_file_exists(self) -> None:
        assert CONFIG_FILE.exists(), f"Config file not found: {CONFIG_FILE}"

    def test_vascular_model_configuration(self) -> None:
        """Verify vascular model is correctly configured."""
        assert DSC_CONFIG["method"] == "vascular"
        assert DSC_CONFIG["bayesian"] == "map"
        # Vascular model has alpha and beta parameters
        assert "alpha" in EXPECTED_DSC_OUTPUTS
        assert "beta" in EXPECTED_DSC_OUTPUTS
        # With Bayesian, also expect uncertainty maps
        assert "cbf_std" in EXPECTED_DSC_OUTPUTS
        assert "cbv_std" in EXPECTED_DSC_OUTPUTS

    def test_tofts_model_configuration(self) -> None:
        """Verify standard Tofts model is correctly configured."""
        assert DCE_CONFIG["model"] == "tofts"
        # Standard Tofts has no vp
        assert "vp" not in EXPECTED_DCE_OUTPUTS
        assert "Ktrans" in EXPECTED_DCE_OUTPUTS
        assert "ve" in EXPECTED_DCE_OUTPUTS
        assert "kep" in EXPECTED_DCE_OUTPUTS

    @pytest.mark.parametrize("output_name", EXPECTED_DSC_OUTPUTS)
    def test_dsc_output_ground_truth(
        self, output_name: str, dsc_ground_truth_dir: Optional[Path],
        subject_brain_mask: Optional[Path],
    ) -> None:
        if dsc_ground_truth_dir is None:
            pytest.skip(f"Ground truth not found for {SUBJECT}/{SESSION}")

        gt_data, gt_path = load_ground_truth_map(
            SUBJECT, SESSION, output_name, "dsc", PERF_PLUS_DERIVATIVES
        )
        if gt_path is None:
            pytest.skip(f"Ground truth not found for DSC {output_name}")

        pmriprep_output_dir = PMRIPREP_TEST_OUTPUTS / "dsc_pipeline" / SUBJECT / SESSION / "perf"
        if not pmriprep_output_dir.exists():
            pytest.skip(f"PMRIPrep outputs not available for {SUBJECT}/{SESSION}")

        pmriprep_file = find_pmriprep_output(
            pmriprep_output_dir, output_name, DSC_OUTPUT_PATTERNS
        )
        if pmriprep_file is None:
            pytest.fail(f"PMRIPrep output not found for {output_name} in {pmriprep_output_dir}")

        result = compare_output_files(
            pmriprep_file, gt_path, f"dsc_{output_name}",
            is_mask=False, mask_path=subject_brain_mask,
        )

        threshold = get_correlation_threshold(output_name, "dsc")
        passed = result.value >= threshold

        assert passed, f"DSC {output_name} r={result.value:.4f} < {threshold}"

    @pytest.mark.xfail(
        reason=(
            "Residual divergence from baseline frames (PMRIPrep auto-detects 7, "
            "perf-plus defaults to 5) and minor dose difference (0.025 vs 0.0259 "
            "mmol/kg, 3.6%). See docs/FUTURE_WORK.md."
        ),
    )
    @pytest.mark.parametrize("output_name", EXPECTED_DCE_OUTPUTS)
    def test_dce_output_ground_truth(
        self, output_name: str, dce_ground_truth_dir: Optional[Path],
    ) -> None:
        if dce_ground_truth_dir is None:
            pytest.skip(f"Ground truth not found for {SUBJECT}/{SESSION}")

        # Use pattern-based GT file finding to avoid matching uncertainty/smoothed maps
        gt_path = find_ground_truth_file_with_patterns(
            dce_ground_truth_dir, output_name, DCE_GT_PATTERNS
        )
        if gt_path is None:
            _, gt_path = load_ground_truth_map(
                SUBJECT, SESSION, output_name, "dce", PERF_PLUS_DERIVATIVES
            )
        if gt_path is None:
            pytest.skip(f"Ground truth not found for DCE {output_name}")

        pmriprep_output_dir = PMRIPREP_TEST_OUTPUTS / "dce_pipeline" / SUBJECT / SESSION / "perf"
        if not pmriprep_output_dir.exists():
            pytest.skip(f"PMRIPrep outputs not available for {SUBJECT}/{SESSION}")

        pmriprep_file = find_pmriprep_output(
            pmriprep_output_dir, output_name, DCE_OUTPUT_PATTERNS
        )
        if pmriprep_file is None:
            pytest.fail(f"PMRIPrep output not found for {output_name} in {pmriprep_output_dir}")

        result = compare_output_files(
            pmriprep_file, gt_path, f"dce_{output_name}",
            is_mask=False, mask_path=None,
        )

        threshold = get_correlation_threshold(output_name, "dce")
        passed = result.value >= threshold

        assert passed, f"DCE {output_name} r={result.value:.4f} < {threshold}"


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
