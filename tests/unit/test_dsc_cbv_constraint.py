"""Unit tests for DSC CBV to vp constraint utilities."""

from typing import Any

import numpy as np
import pytest

from pmriprep.utils.dce_constants import (
    NAWM_VP_REFERENCE,
    VP_MEDIAN_MAX,
    VP_MEDIAN_MIN,
)
from pmriprep.utils.dsc_cbv_constraint import (
    CBVQualityAssessment,
    ConstraintMode,
    assess_dsc_cbv_quality,
    determine_constraint_mode,
    ncbv_to_vp_prior,
    prepare_vp_constraint_from_data,
)


# =============================================================================
# Quality Assessment
# =============================================================================




# =============================================================================
# Constraint Mode Determination
# =============================================================================


class TestDetermineConstraintMode:
    """Tests for determine_constraint_mode."""

    def test_high_quality_uses_fixed_prior(self) -> None:
        """High confidence and a valid K2 reference emit the fixed soft prior."""
        quality = CBVQualityAssessment(
            confidence="high",
            use_constraint=True,
            reasons=["All quality checks passed"],
        )
        k2 = {"k2_reference_available": True, "k2_signed": True}
        config = determine_constraint_mode(quality, k2_metrics=k2)
        assert config.mode == ConstraintMode.SOFT
        assert config.vp_uncertainty == pytest.approx(0.60)

    def test_moderate_quality_soft(self) -> None:
        """Moderate confidence and a valid reference use the fixed prior."""
        quality = CBVQualityAssessment(
            confidence="moderate",
            use_constraint=True,
            reasons=["CNR coverage 0.90 < 0.99"],
        )
        k2 = {"k2_reference_available": True, "k2_signed": True}
        config = determine_constraint_mode(quality, k2_metrics=k2)
        assert config.mode == ConstraintMode.SOFT
        assert config.vp_uncertainty == pytest.approx(0.60)

    def test_low_quality_skip(self) -> None:
        """Low confidence produces skip mode."""
        quality = CBVQualityAssessment(
            confidence="low",
            use_constraint=False,
            reasons=["CNR coverage 0.50 < 0.84"],
        )
        config = determine_constraint_mode(quality)
        assert config.mode == ConstraintMode.SKIP

    def test_user_override_skip(self) -> None:
        """``skip`` user override is honored regardless of quality."""
        quality = CBVQualityAssessment(
            confidence="high",
            use_constraint=True,
            reasons=["All quality checks passed"],
        )
        config = determine_constraint_mode(quality, user_override="skip")
        assert config.mode == ConstraintMode.SKIP

    def test_low_quality_auto_skips(self) -> None:
        """Low DSC confidence under ``auto`` → SKIP (no prior)."""
        quality = CBVQualityAssessment(
            confidence="low",
            use_constraint=False,
            reasons=["CNR coverage 0.50 < 0.84"],
        )
        config = determine_constraint_mode(quality, user_override="auto")
        assert config.mode == ConstraintMode.SKIP


# =============================================================================
# AIF QC Key Normalization
# =============================================================================




# =============================================================================
# QC Report Schema
# =============================================================================




# =============================================================================
# NAWM-normalized CBV → vp prior
# =============================================================================


class TestNcbvToVpPrior:
    """Tests for ncbv_to_vp_prior (NAWM-anchored vp prior)."""

    def test_nawm_value_yields_reference_vp(self) -> None:
        """nCBV=1 (NAWM reference) → vp = NAWM_VP_REFERENCE."""
        ncbv = np.array([1.0], dtype=np.float64)
        vp = ncbv_to_vp_prior(ncbv)
        assert vp[0] == pytest.approx(NAWM_VP_REFERENCE)

    def test_tumor_elevation_proportional(self) -> None:
        """nCBV=3 (hypervascular tumor) → vp = 3 × NAWM_VP_REFERENCE."""
        ncbv = np.array([3.0], dtype=np.float64)
        vp = ncbv_to_vp_prior(ncbv)
        assert vp[0] == pytest.approx(3.0 * NAWM_VP_REFERENCE)

    def test_negative_and_nan_return_nan(self) -> None:
        """Negative and NaN nCBV values produce NaN."""
        ncbv = np.array([-0.5, np.nan, 1.0], dtype=np.float64)
        vp = ncbv_to_vp_prior(ncbv)
        assert np.isnan(vp[0])
        assert np.isnan(vp[1])
        assert np.isfinite(vp[2])

    def test_above_physiological_max_clipped_to_nan(self) -> None:
        """nCBV large enough to exceed VP_MAX_PHYSIOLOGICAL → NaN."""
        # With NAWM_VP_REFERENCE = 0.010 and VP_MAX_PHYSIOLOGICAL
        # = 0.2, nCBV = 25 gives vp = 0.25 which is beyond the physiological
        # cap.
        ncbv = np.array([25.0], dtype=np.float64)
        vp = ncbv_to_vp_prior(ncbv)
        assert np.isnan(vp[0])

    def test_custom_reference(self) -> None:
        """Custom nawm_vp_reference scales the prior accordingly."""
        ncbv = np.array([1.0, 2.0], dtype=np.float64)
        vp = ncbv_to_vp_prior(ncbv, nawm_vp_reference=0.015)
        assert vp[0] == pytest.approx(0.015)
        assert vp[1] == pytest.approx(0.030)


# =============================================================================
# vp magnitude gate (undershoot / overshoot)
# =============================================================================


def _write_nifti(path: Any, data: np.ndarray) -> str:
    import nibabel as nib  # type: ignore[import-untyped]

    img = nib.Nifti1Image(data.astype(np.float32), affine=np.eye(4))
    nib.save(img, str(path))
    return str(path)


def _tissue_qc() -> dict[str, Any]:
    return {
        "cnr_whole_brain": {"fraction_cnr_gt_4": 0.995, "mean_cnr": 10.0},
    }


def _accepted_registration_qc() -> dict[str, dict[str, str]]:
    """Return accepted QC for both transforms in the spatial transfer."""
    return {
        "dsc_registration_qc": {"qc_status": "PASS"},
        "dce_registration_qc": {"qc_status": "PASS"},
    }


class TestVpMagnitudeGate:
    """Tests for the vp median undershoot/overshoot gate."""

    def test_undershoot_downgrades_to_skip(self, tmp_path: Any) -> None:
        """vp median below VP_MEDIAN_MIN triggers SKIP + vp_map=None."""
        # nCBV median far below 1 so vp = 0.02*nCBV falls under VP_MEDIAN_MIN.
        ncbv = np.full((8, 8, 8), 0.1, dtype=np.float64)
        mask = np.ones((8, 8, 8), dtype=np.float32)
        ncbv_file = _write_nifti(tmp_path / "ncbv.nii.gz", ncbv)
        mask_file = _write_nifti(tmp_path / "mask.nii.gz", mask)

        result = prepare_vp_constraint_from_data(
            ncbv_file=ncbv_file,
            dsc_qc_data=_tissue_qc(),
            brain_mask_file=mask_file,
            k2_metrics={"k2_reference_available": True},
            **_accepted_registration_qc(),
        )

        assert result.config.mode == ConstraintMode.SKIP
        assert result.vp_map is None
        assert "derived vp prior median" in result.config.reason
        assert result.qc_report.get("magnitude_gate_triggered") is True

    def test_overshoot_downgrades_to_skip(self, tmp_path: Any) -> None:
        """vp median above VP_MEDIAN_MAX triggers SKIP + vp_map=None."""
        # nCBV median ≈ 8 → vp = 0.16, above VP_MEDIAN_MAX = 0.10.
        ncbv = np.full((8, 8, 8), 8.0, dtype=np.float64)
        mask = np.ones((8, 8, 8), dtype=np.float32)
        ncbv_file = _write_nifti(tmp_path / "ncbv.nii.gz", ncbv)
        mask_file = _write_nifti(tmp_path / "mask.nii.gz", mask)

        result = prepare_vp_constraint_from_data(
            ncbv_file=ncbv_file,
            dsc_qc_data=_tissue_qc(),
            brain_mask_file=mask_file,
            k2_metrics={"k2_reference_available": True},
            **_accepted_registration_qc(),
        )

        assert result.config.mode == ConstraintMode.SKIP
        assert result.vp_map is None
        assert result.qc_report.get("magnitude_gate_triggered") is True

    def test_physiological_range_no_downgrade(self, tmp_path: Any) -> None:
        """vp median in physiological range preserves constraint mode."""
        # nCBV ≈ 1 → vp ≈ 0.02, squarely inside [VP_MEDIAN_MIN, VP_MEDIAN_MAX].
        ncbv = np.full((8, 8, 8), 1.0, dtype=np.float64)
        mask = np.ones((8, 8, 8), dtype=np.float32)
        ncbv_file = _write_nifti(tmp_path / "ncbv.nii.gz", ncbv)
        mask_file = _write_nifti(tmp_path / "mask.nii.gz", mask)

        result = prepare_vp_constraint_from_data(
            ncbv_file=ncbv_file,
            dsc_qc_data=_tissue_qc(),
            brain_mask_file=mask_file,
            k2_metrics={"k2_reference_available": True},
            **_accepted_registration_qc(),
        )

        assert result.config.mode != ConstraintMode.SKIP
        assert result.vp_map is not None
        assert result.qc_report.get("magnitude_gate_triggered") is not True
        assert VP_MEDIAN_MIN <= result.qc_report["vp_median"] <= VP_MEDIAN_MAX

    def test_signed_k2_preserves_qualified_bayesian_prior(self, tmp_path: Any) -> None:
        """Production signed K2 retains the fixed prior after gross-quality gates."""
        ncbv = np.full((8, 8, 8), 1.0, dtype=np.float64)
        mask = np.ones((8, 8, 8), dtype=np.float32)
        ncbv_file = _write_nifti(tmp_path / "ncbv.nii.gz", ncbv)
        mask_file = _write_nifti(tmp_path / "mask.nii.gz", mask)

        result = prepare_vp_constraint_from_data(
            ncbv_file=ncbv_file,
            dsc_qc_data=_tissue_qc(),
            brain_mask_file=mask_file,
            k2_metrics={
                "k2_signed": True,
                "k2_reference_available": True,
                "k2_median": 0.001,
                "percent_negative_k2": 35.0,
            },
            **_accepted_registration_qc(),
        )

        assert result.config.mode == ConstraintMode.SOFT
        assert result.config.vp_uncertainty == pytest.approx(0.60)
        assert result.vp_map is not None
        assert result.qc_report["vp_uncertainty"] == pytest.approx(0.60)


# =============================================================================
# nCBV integration path through prepare_vp_constraint_from_data
# =============================================================================


class TestNcbvIntegration:
    """End-to-end tests confirming the production path uses ncbv_to_vp_prior."""

    def test_prepare_vp_constraint_uses_ncbv_to_vp_prior(self, tmp_path: Any) -> None:
        """nCBV median 1.0 yields vp_map median ≈ NAWM_VP_REFERENCE (0.02)."""
        ncbv = np.full((6, 6, 6), 1.0, dtype=np.float64)
        mask = np.ones((6, 6, 6), dtype=np.float32)
        ncbv_file = _write_nifti(tmp_path / "ncbv.nii.gz", ncbv)
        mask_file = _write_nifti(tmp_path / "mask.nii.gz", mask)

        result = prepare_vp_constraint_from_data(
            ncbv_file=ncbv_file,
            dsc_qc_data=_tissue_qc(),
            brain_mask_file=mask_file,
            k2_metrics={"k2_reference_available": True},
            **_accepted_registration_qc(),
        )

        assert result.vp_map is not None
        vp_median = float(np.nanmedian(result.vp_map))
        assert vp_median == pytest.approx(NAWM_VP_REFERENCE, rel=1e-6)
        assert result.qc_report["nawm_vp_reference"] == pytest.approx(NAWM_VP_REFERENCE)

    @pytest.mark.parametrize(
        ("dsc_status", "dce_status"),
        [("POOR", "PASS"), ("PASS", "POOR"), ("UNKNOWN", "PASS"), ("PASS", None)],
    )
    def test_registration_failure_skips_spatial_prior(
        self,
        tmp_path: Any,
        dsc_status: str,
        dce_status: str | None,
    ) -> None:
        """A failed or unavailable transform makes a spatial prior unsafe."""
        ncbv_file = _write_nifti(tmp_path / "ncbv.nii.gz", np.ones((6, 6, 6)))
        mask_file = _write_nifti(tmp_path / "mask.nii.gz", np.ones((6, 6, 6)))

        result = prepare_vp_constraint_from_data(
            ncbv_file=ncbv_file,
            dsc_qc_data=_tissue_qc(),
            brain_mask_file=mask_file,
            dsc_registration_qc={"qc_status": dsc_status},
            dce_registration_qc=(
                {"qc_status": dce_status} if dce_status is not None else None
            ),
        )

        assert result.config.mode == ConstraintMode.SKIP
        assert result.vp_map is None
        assert result.qc_report["skip_reason"] == "cross_modal_registration_qc"

    def test_registration_warn_preserves_existing_width(self, tmp_path: Any) -> None:
        """Accepted WARN registration does not invent another prior-width tier."""
        ncbv_file = _write_nifti(tmp_path / "ncbv.nii.gz", np.ones((6, 6, 6)))
        mask_file = _write_nifti(tmp_path / "mask.nii.gz", np.ones((6, 6, 6)))
        common = {
            "ncbv_file": ncbv_file,
            "dsc_qc_data": _tissue_qc(),
            "brain_mask_file": mask_file,
            "k2_metrics": {"k2_reference_available": True},
        }

        passed = prepare_vp_constraint_from_data(
            **common,
            dsc_registration_qc={"qc_status": "PASS"},
            dce_registration_qc={"qc_status": "PASS"},
        )
        warned = prepare_vp_constraint_from_data(
            **common,
            dsc_registration_qc={"qc_status": "WARN"},
            dce_registration_qc={"qc_status": "PASS"},
        )

        assert warned.config.mode == passed.config.mode == ConstraintMode.SOFT
        assert warned.config.vp_uncertainty == passed.config.vp_uncertainty

    def test_missing_ncbv_file_emits_skip(self, tmp_path: Any) -> None:
        """ncbv_file=None yields SKIP with skip_reason='no_nawm_mask'."""
        mask = np.ones((4, 4, 4), dtype=np.float32)
        mask_file = _write_nifti(tmp_path / "mask.nii.gz", mask)

        result = prepare_vp_constraint_from_data(
            ncbv_file=None,
            dsc_qc_data=_tissue_qc(),
            brain_mask_file=mask_file,
        )

        assert result.config.mode == ConstraintMode.SKIP
        assert result.vp_map is None
        assert result.qc_report.get("skip_reason") == "no_nawm_mask"

    def test_missing_ncbv_reports_generated_nawm_exclusion(self, tmp_path: Any) -> None:
        """The vp-prior skip preserves the upstream auto-NAWM reason."""
        mask_file = _write_nifti(
            tmp_path / "mask.nii.gz", np.ones((4, 4, 4), dtype=np.float32)
        )
        reason = "insufficient_voxels_after_erosion_relaxation"

        result = prepare_vp_constraint_from_data(
            ncbv_file=None,
            dsc_qc_data=_tissue_qc(),
            brain_mask_file=mask_file,
            nawm_exclusion_reason=reason,
        )

        assert result.config.mode == ConstraintMode.SKIP
        assert result.qc_report["skip_reason"] == "no_nawm_mask"
        assert result.qc_report["nawm_exclusion_reason"] == reason
        assert "retained for QC" in str(result.qc_report["reason"])


# =============================================================================
# Dose-aware CNR coverage gate
# =============================================================================




class TestDetermineConstraintReferenceGate:
    """The K2 reference is a binary availability gate, not a classifier."""

    @pytest.mark.parametrize(
        "k2_metrics", [None, {}, {"k2_reference_available": False}]
    )
    def test_missing_or_unavailable_reference_skips(
        self,
        k2_metrics: dict[str, Any] | None,
    ) -> None:
        quality = CBVQualityAssessment(
            confidence="high",
            use_constraint=True,
            reasons=[],
        )
        config = determine_constraint_mode(
            quality,
            k2_metrics=k2_metrics,
        )

        assert config.mode == ConstraintMode.SKIP
        assert config.vp_uncertainty == 0.0
        assert "leakage-reference curve was unavailable" in config.reason


class TestBolusGdFractionValidation:
    """_bolus_gd_fraction rejects invalid dose."""

    def test_zero_dose_raises(self) -> None:
        from pmriprep.utils.dsc_cbv_constraint import _bolus_gd_fraction

        with pytest.raises(ValueError, match="dose must be positive"):
            _bolus_gd_fraction(0.0, 0.1)

    def test_negative_dose_raises(self) -> None:
        from pmriprep.utils.dsc_cbv_constraint import _bolus_gd_fraction

        with pytest.raises(ValueError, match="dose must be positive"):
            _bolus_gd_fraction(-0.1, 0.1)

    def test_zero_agent_raises(self) -> None:
        from pmriprep.utils.dsc_cbv_constraint import _bolus_gd_fraction

        with pytest.raises(ValueError, match="agent_standard_dose"):
            _bolus_gd_fraction(1.0, 0.0)




class TestFixedPriorWidth:
    """Qualified anchors use one fixed release-policy prior width."""

    def test_confidence_does_not_stratify_sigma(self) -> None:
        """High and moderate confidence use the same fixed width."""
        high = CBVQualityAssessment(confidence="high", use_constraint=True, reasons=[])
        mod = CBVQualityAssessment(
            confidence="moderate", use_constraint=True, reasons=[]
        )
        available = {"k2_reference_available": True, "k2_signed": True}

        cfg_high = determine_constraint_mode(high, k2_metrics=available)
        cfg_mod = determine_constraint_mode(mod, k2_metrics=available)

        assert cfg_high.mode == ConstraintMode.SOFT
        assert cfg_mod.mode == ConstraintMode.SOFT
        assert cfg_high.vp_uncertainty == pytest.approx(0.60)
        assert cfg_mod.vp_uncertainty == cfg_high.vp_uncertainty

    def test_signed_k2_uses_fixed_release_policy_prior_width(self) -> None:
        """Signed leakage metrics neither gate nor tune the Bayesian prior."""
        quality = CBVQualityAssessment(
            confidence="high",
            use_constraint=True,
            reasons=[],
        )
        signed = {
            "k2_signed": True,
            "k2_reference_available": True,
            "k2_median": -0.002,
            "percent_negative_k2": 60.0,
        }

        config = determine_constraint_mode(
            quality,
            k2_metrics=signed,
        )

        assert config.mode == ConstraintMode.SOFT
        assert config.vp_uncertainty == pytest.approx(0.60)
        assert "QC supported use of the NAWM-normalized CBV prior" in config.reason

    def test_disabled_leakage_correction_skips_prior(self) -> None:
        """An explicitly uncorrected CBV map cannot anchor the DCE prior."""
        quality = CBVQualityAssessment(
            confidence="high",
            use_constraint=True,
            reasons=[],
        )

        config = determine_constraint_mode(
            quality,
            k2_metrics={"leakage_correction_disabled": True},
        )

        assert config.mode == ConstraintMode.SKIP
        assert config.vp_uncertainty == 0.0
        assert "leakage correction was disabled" in config.reason.lower()


class TestNAWMVpReferenceAnchor:
    """Pin the configured NAWM plasma-volume prior anchor at 0.010."""

    def test_nawm_vp_reference_value(self) -> None:
        from pmriprep.utils.dce_constants import NAWM_VP_REFERENCE

        assert NAWM_VP_REFERENCE == 0.010

    def test_nawm_vp_reference_inside_median_bounds(self) -> None:
        """Anchor must sit inside [VP_MEDIAN_MIN, VP_MEDIAN_MAX] so normal
        NAWM-like nCBV inputs never trip the magnitude gate.
        """
        from pmriprep.utils.dce_constants import (
            NAWM_VP_REFERENCE,
            VP_MEDIAN_MIN,
            VP_MEDIAN_MAX,
        )

        assert VP_MEDIAN_MIN < NAWM_VP_REFERENCE < VP_MEDIAN_MAX




# =============================================================================
# Synthetic-AIF SKIP plumbing
# =============================================================================






@pytest.mark.parametrize("coverage, mean, dose, expected", [
    (0.99, 10.0, 1.0, "high"),
    (0.90, 10.0, 1.0, "moderate"),
    (0.80, 10.0, 1.0, "low"),
    (0.45, 10.0, 0.5, "moderate"),
    (0.99, 2.9, 1.0, "low"),
    (None, 10.0, 1.0, "low"),
    (0.99, np.nan, 1.0, "low"),
    (np.inf, 10.0, 1.0, "low"),
])
def test_tissue_cnr_gate(coverage, mean, dose, expected) -> None:
    qc = {"cnr_whole_brain": {"fraction_cnr_gt_4": coverage, "mean_cnr": mean}}
    result = assess_dsc_cbv_quality(qc, dose=dose)
    assert result.confidence == expected
    assert result.use_constraint == (expected != "low")


@pytest.mark.parametrize("source", ["atlas_strict", "external", "unavailable"])
def test_prior_uses_tissue_quality_with_available_spatial_support(tmp_path, source) -> None:
    import json

    ncbv = np.ones((3, 3, 3))
    ncbv[0] = np.nan
    qc = _tissue_qc()
    qc["detection_method"] = source
    result = prepare_vp_constraint_from_data(
        _write_nifti(tmp_path / "ncbv.nii.gz", ncbv), qc,
        _write_nifti(tmp_path / "mask.nii.gz", np.ones_like(ncbv)),
        k2_metrics={"k2_reference_available": True},
        **_accepted_registration_qc(),
    )
    assert result.config.mode == ConstraintMode.SOFT
    assert np.isnan(result.vp_map[0]).all()
    np.testing.assert_allclose(result.vp_map[1:], 0.010)
    json.dumps(result.qc_report, allow_nan=False)


def test_empty_transferred_prior_support_is_unavailable(tmp_path) -> None:
    import json

    result = prepare_vp_constraint_from_data(
        _write_nifti(tmp_path / "ncbv.nii.gz", np.full((3, 3, 3), np.nan)),
        _tissue_qc(),
        _write_nifti(tmp_path / "mask.nii.gz", np.ones((3, 3, 3))),
        k2_metrics={"k2_reference_available": True},
        **_accepted_registration_qc(),
    )
    assert result.config.mode == ConstraintMode.SKIP
    assert result.vp_map is None
    assert result.qc_report["skip_reason"] == "empty_prior_support"
    json.dumps(result.qc_report, allow_nan=False)
