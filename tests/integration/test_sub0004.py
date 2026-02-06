"""Integration test for sub-0004/ses-0001: GBM preset, extended_tofts bayesian_map.

US-092: Test DSC + DCE pipeline with:
- CLI: --dsc-preload --dsc-dose half --dsc-gbm-preset --dce-model extended_tofts --dce-bayesian map --dce-scale-aif
- DSC: oSVD with GBM preset (pre_bolus=10, aif_criteria=relaxed, n_aif_candidates=5, save_k2_map=true)
- DCE: extended_tofts with MAP bayesian, scale_aif

Config file: tests/fixtures/sub-0004_config.yaml
Ground truth: ~/Developer/perf-plus/bidsroot/derivatives/sub-0004/ses-0001/
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from tests.integration.test_ground_truth import (
    CORRELATION_THRESHOLD,
    DICE_THRESHOLD,
    DSC_CORRELATION_THRESHOLDS,
    DCE_CORRELATION_THRESHOLDS,
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
SUBJECT = "sub-0004"
SESSION = "ses-0001"
USER_STORY = "US-092"

# CLI arguments for this test
CLI_ARGS = [
    "--dsc-preload",
    "--dsc-dose", "half",
    "--dsc-gbm-preset",
    "--dce-model", "extended_tofts",
    "--dce-bayesian", "map",
    "--dce-scale-aif",
]

# Config file for advanced options
CONFIG_FILE = Path(__file__).parent.parent / "fixtures" / "sub-0004_config.yaml"

# DSC configuration (GBM preset)
DSC_CONFIG: Dict[str, Any] = {
    "preload_detected": True,
    "dose": "half",
    "method": "oSVD",  # GBM preset uses oSVD
    "bayesian": "none",  # DSC has no bayesian in GBM preset
    "pre_bolus": 10,
    "aif_criteria": "relaxed",
    "n_aif_candidates": 5,
    "save_k2_map": True,
}

# DCE configuration
DCE_CONFIG: Dict[str, Any] = {
    "model": "extended_tofts",
    "bayesian": "map",
    "scale_aif": True,
}

# Expected outputs
EXPECTED_DSC_OUTPUTS = ["cbf", "cbv", "mtt", "tmax", "k2"]
EXPECTED_DCE_OUTPUTS = ["Ktrans", "ve", "vp", "kep"]

# Specific file patterns for PMRIPrep outputs to avoid ambiguous matches
# (e.g., avoid matching ncbf when looking for cbf)
DSC_OUTPUT_PATTERNS: Dict[str, List[str]] = {
    "cbf": ["*desc-rel_cbf.nii*", "*_cbf.nii*"],
    "cbv": ["*desc-rel_cbv.nii*", "*_cbv.nii*"],
    "mtt": ["*desc-dscproc_mtt.nii*", "*_mtt.nii*"],
    "tmax": ["*desc-dscproc_tmax.nii*", "*_tmax.nii*"],
    "k2": ["*desc-dscproc_k2.nii*", "*_k2.nii*"],
}

# DCE output patterns - prefer model-specific with underscore separator
# IMPORTANT: Patterns must match base parameter maps, not uncertainty variants (Std, CV, CI95)
DCE_OUTPUT_PATTERNS: Dict[str, List[str]] = {
    "Ktrans": [
        "*model-extended_tofts_Ktrans.nii*",  # PMRIPrep output (model-extended_tofts)
    ],
    "ve": [
        "*model-extended_tofts_ve.nii*",  # PMRIPrep output
    ],
    "vp": [
        "*model-extended_tofts_vp.nii*",  # PMRIPrep output
    ],
    "kep": [
        "*model-extended_tofts_kep.nii*",  # PMRIPrep output
    ],
}

# Ground truth DCE patterns - use model-ExtTofts with desc-vpDSCmod
# CRITICAL: Match only the base maps, not uncertainty maps like Std, CV, CI95lower, CI95upper
DCE_GT_PATTERNS: Dict[str, List[str]] = {
    "Ktrans": [
        "*model-ExtTofts_desc-vpDSCmod_Ktrans.nii*",  # GT uses ExtTofts, not extended_tofts
    ],
    "ve": [
        "*model-ExtTofts_desc-vpDSCmod_ve.nii*",
    ],
    "vp": [
        "*model-ExtTofts_desc-vpDSCmod_vp.nii*",
    ],
    "kep": [
        "*model-ExtTofts_desc-vpDSCmod_kep.nii*",
    ],
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
    """Find ground truth file using specific patterns to avoid ambiguous matches.

    This is used for DCE outputs where generic patterns might match uncertainty
    maps (Std, CV, CI95lower, CI95upper) instead of the actual parameter maps.
    """
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
def subject_ground_truth_dir() -> Optional[Path]:
    """Get the ground truth directory for this subject."""
    return find_ground_truth_directory(
        SUBJECT, SESSION, pipeline="dsc_pipeline", derivatives_base=PERF_PLUS_DERIVATIVES
    )


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
class TestSub0004Ses0001:
    """Integration tests for sub-0004/ses-0001."""

    def test_ground_truth_exists(
        self, subject_ground_truth_dir: Optional[Path]
    ) -> None:
        """Verify ground truth data exists for this subject."""
        if subject_ground_truth_dir is None:
            pytest.skip(
                f"Ground truth directory not found for {SUBJECT}/{SESSION}"
            )
        assert subject_ground_truth_dir.exists()

    def test_config_file_exists(self) -> None:
        """Verify config file exists."""
        assert CONFIG_FILE.exists(), f"Config file not found: {CONFIG_FILE}"

    def test_cli_configuration(self) -> None:
        """Verify CLI arguments match expected configuration."""
        assert "--dsc-preload" in CLI_ARGS
        assert "--dsc-gbm-preset" in CLI_ARGS
        assert "--dce-model" in CLI_ARGS
        assert "extended_tofts" in CLI_ARGS
        assert "--dce-bayesian" in CLI_ARGS
        assert "map" in CLI_ARGS
        assert DSC_CONFIG["method"] == "oSVD"

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
            pytest.fail(f"PMRIPrep output not found for {output_name}")

        result = compare_output_files(
            pmriprep_file,
            gt_path,
            f"dsc_{output_name}",
            is_mask=False,
            mask_path=subject_brain_mask,
        )

        # Use map-specific threshold for DSC outputs
        threshold = get_correlation_threshold(output_name, "dsc")
        passed = result.value >= threshold

        assert passed, (
            f"DSC {output_name} correlation {result.value:.4f} < "
            f"threshold {threshold}"
        )

    @pytest.mark.parametrize("output_name", EXPECTED_DCE_OUTPUTS)
    def test_dce_output_ground_truth(
        self,
        output_name: str,
        subject_ground_truth_dir: Optional[Path],
        subject_brain_mask: Optional[Path],
    ) -> None:
        """Test DCE output against ground truth."""
        if subject_ground_truth_dir is None:
            pytest.skip(f"Ground truth not found for {SUBJECT}/{SESSION}")

        # Find DCE ground truth directory
        dce_gt_dir = find_ground_truth_directory(
            SUBJECT, SESSION, pipeline="dce_pipeline", derivatives_base=PERF_PLUS_DERIVATIVES
        )
        if dce_gt_dir is None:
            pytest.skip(f"DCE ground truth not found for {SUBJECT}/{SESSION}")

        # Use pattern-based GT file finding to avoid matching uncertainty maps
        gt_path = find_ground_truth_file_with_patterns(
            dce_gt_dir, output_name, DCE_GT_PATTERNS
        )
        if gt_path is None:
            # Fall back to generic load if specific patterns don't match
            _, gt_path = load_ground_truth_map(
                SUBJECT, SESSION, output_name, "dce", PERF_PLUS_DERIVATIVES
            )
        if gt_path is None:
            pytest.skip(f"Ground truth not found for DCE {output_name}")

        pmriprep_output_dir = PMRIPREP_TEST_OUTPUTS / "dce_pipeline" / SUBJECT / SESSION / "perf"

        if not pmriprep_output_dir.exists():
            pytest.skip(
                f"PMRIPrep outputs not available for {SUBJECT}/{SESSION}"
            )

        pmriprep_file = find_pmriprep_output(
            pmriprep_output_dir, output_name, DCE_OUTPUT_PATTERNS
        )
        if pmriprep_file is None:
            pytest.fail(f"PMRIPrep output not found for {output_name}")

        # DCE images may be in different space than DSC mask, skip masking if shapes mismatch
        result = compare_output_files(
            pmriprep_file,
            gt_path,
            f"dce_{output_name}",
            is_mask=False,
            mask_path=None,  # Skip masking for DCE (different resolution)
        )

        # Use map-specific threshold for DCE outputs
        threshold = get_correlation_threshold(output_name, "dce")
        passed = result.value >= threshold

        assert passed, (
            f"DCE {output_name} correlation {result.value:.4f} < "
            f"threshold {threshold}"
        )


def get_expected_cli_command(
    bids_dir: Path,
    output_dir: Path,
    work_dir: Optional[Path] = None,
) -> List[str]:
    """Generate the expected CLI command for running PMRIPrep on this subject."""
    cmd = [
        "pmriprep",
        str(bids_dir),
        str(output_dir),
        "participant",
        "--participant-label", SUBJECT.replace("sub-", ""),
        "--session-label", SESSION.replace("ses-", ""),
        "--config-file", str(CONFIG_FILE),
    ]
    cmd.extend(CLI_ARGS)

    if work_dir:
        cmd.extend(["-w", str(work_dir)])

    return cmd


def run_subject_test(
    pmriprep_derivatives: Path,
    perf_plus_derivatives: Optional[Path] = None,
) -> SubjectTestResult:
    """Run integration test for this subject."""
    if perf_plus_derivatives is None:
        perf_plus_derivatives = PERF_PLUS_DERIVATIVES

    comparisons: List[ComparisonResult] = []
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

    # Compare DCE outputs
    for output_name in EXPECTED_DCE_OUTPUTS:
        gt_data, gt_path = load_ground_truth_map(
            SUBJECT, SESSION, output_name, "dce", perf_plus_derivatives
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

        pmriprep_dir = pmriprep_derivatives / SUBJECT / SESSION / "perf"
        pmriprep_files = (
            list(pmriprep_dir.glob(f"*{output_name}*.nii*"))
            if pmriprep_dir.exists()
            else []
        )

        if not pmriprep_files:
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

        result = compare_output_files(
            pmriprep_files[0],
            gt_path,
            f"dce_{output_name}",
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
