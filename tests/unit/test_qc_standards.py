"""Unit tests for QC standards and colorblind-safe visualization palette."""

import pytest

from pmriprep.utils.qc_standards import (
    COLORBLIND_PALETTE,
    DICE_THRESHOLDS,
    DICE_THRESHOLDS_STRICT,
    MOTION_FD_MAX_THRESHOLDS,
    MOTION_FD_THRESHOLDS,
    MOTION_OUTLIER_PCT_THRESHOLDS,
    ColorPalette,
    QCRecommendation,
    QCThresholds,
    aggregate_qc_recommendations,
    build_qc_summary,
    get_composite_motion_status,
    get_qc_recommendation,
    get_qc_status,
    get_status_color,
)


class TestGetQCStatus:
    """Tests for get_qc_status()."""

    def test_higher_is_better_pass(self) -> None:
        """DICE = 0.9 with DICE_THRESHOLDS (pass=0.85) returns PASS."""
        status = get_qc_status(0.9, DICE_THRESHOLDS, higher_is_better=True)
        assert status == "PASS"

    def test_higher_is_better_warn(self) -> None:
        """DICE = 0.8 with DICE_THRESHOLDS (pass=0.85, warn=0.75) returns WARN."""
        status = get_qc_status(0.8, DICE_THRESHOLDS, higher_is_better=True)
        assert status == "WARN"

    def test_higher_is_better_fail(self) -> None:
        """DICE = 0.6 with DICE_THRESHOLDS (warn=0.75) returns POOR."""
        status = get_qc_status(0.6, DICE_THRESHOLDS, higher_is_better=True)
        assert status == "POOR"

    def test_lower_is_better_pass(self) -> None:
        """FD = 0.2 with MOTION_FD_THRESHOLDS (pass=0.3) returns PASS."""
        status = get_qc_status(0.2, MOTION_FD_THRESHOLDS, higher_is_better=False)
        assert status == "PASS"

    def test_lower_is_better_warn(self) -> None:
        """FD = 0.4 with MOTION_FD_THRESHOLDS (pass=0.3, warn=0.5) returns WARN."""
        status = get_qc_status(0.4, MOTION_FD_THRESHOLDS, higher_is_better=False)
        assert status == "WARN"

    def test_lower_is_better_fail(self) -> None:
        """FD = 0.6 with MOTION_FD_THRESHOLDS (warn=0.5) returns POOR."""
        status = get_qc_status(0.6, MOTION_FD_THRESHOLDS, higher_is_better=False)
        assert status == "POOR"


class TestGetStatusColor:
    """Tests for get_status_color()."""

    def test_pass_returns_success_color(self) -> None:
        """PASS status returns the success hex color."""
        assert get_status_color("PASS") == COLORBLIND_PALETTE.success

    def test_warn_returns_warning_color(self) -> None:
        """WARN status returns the warning hex color."""
        assert get_status_color("WARN") == COLORBLIND_PALETTE.warning

    def test_poor_returns_error_color(self) -> None:
        """POOR status returns the error hex color."""
        assert get_status_color("POOR") == COLORBLIND_PALETTE.error

    def test_unknown_returns_secondary_color(self) -> None:
        """UNKNOWN status returns the secondary hex color."""
        assert get_status_color("UNKNOWN") == COLORBLIND_PALETTE.secondary


class TestGetQCRecommendation:
    """Tests for get_qc_recommendation()."""

    def test_pass_recommendation(self) -> None:
        """PASS status yields action=include, severity=0."""
        rec = get_qc_recommendation(
            "PASS", "DICE", 0.9, DICE_THRESHOLDS, higher_is_better=True
        )
        assert rec.action == "include"
        assert rec.severity == 0

    def test_warn_recommendation(self) -> None:
        """WARN status yields action=review, severity=1."""
        rec = get_qc_recommendation(
            "WARN", "DICE", 0.8, DICE_THRESHOLDS, higher_is_better=True
        )
        assert rec.action == "review"
        assert rec.severity == 1

    def test_poor_higher_is_better(self) -> None:
        """POOR with higher_is_better yields action=exclude, reason contains 'below'."""
        rec = get_qc_recommendation(
            "POOR", "DICE", 0.6, DICE_THRESHOLDS, higher_is_better=True
        )
        assert rec.action == "exclude"
        assert "below" in rec.reason
        assert rec.severity == 3

    def test_poor_lower_is_better(self) -> None:
        """POOR with lower_is_better yields reason containing 'above'."""
        rec = get_qc_recommendation(
            "POOR", "FD mean", 0.6, MOTION_FD_THRESHOLDS, higher_is_better=False
        )
        assert rec.action == "exclude"
        assert "above" in rec.reason


class TestAggregateQCRecommendations:
    """Tests for aggregate_qc_recommendations()."""

    def test_all_pass(self) -> None:
        """All passing recommendations yield action=include, severity=0."""
        recs = [
            QCRecommendation(action="include", reason="OK", severity=0),
            QCRecommendation(action="include", reason="OK", severity=0),
        ]
        agg = aggregate_qc_recommendations(recs)
        assert agg.action == "include"
        assert agg.severity == 0
        assert agg.reason == "All QC metrics passed"

    def test_mixed_severity(self) -> None:
        """Mixed severities: action from worst; reasons joined with ';'."""
        recs = [
            QCRecommendation(action="include", reason="DICE OK", severity=0),
            QCRecommendation(action="review", reason="FD marginal (0.4)", severity=1),
            QCRecommendation(
                action="exclude", reason="NMI below threshold (0.1)", severity=3
            ),
        ]
        agg = aggregate_qc_recommendations(recs)
        assert agg.action == "exclude"
        assert agg.severity == 3
        assert "FD marginal" in agg.reason
        assert "NMI below threshold" in agg.reason
        assert ";" in agg.reason

    def test_empty_list(self) -> None:
        """Empty recommendation list yields action=include."""
        agg = aggregate_qc_recommendations([])
        assert agg.action == "include"


class TestBuildQCSummary:
    """Tests for build_qc_summary()."""

    def test_summary_structure(self) -> None:
        """Returns dict with overall_status, recommendation, and metrics keys."""
        rec = QCRecommendation(action="include", reason="All good", severity=0)
        metrics = {"DICE": (0.92, "PASS"), "FD": (0.15, "PASS")}

        summary = build_qc_summary(metrics, "PASS", rec)

        assert summary["overall_status"] == "PASS"
        assert "recommendation" in summary
        assert summary["recommendation"]["action"] == "include"
        assert summary["recommendation"]["reason"] == "All good"
        assert summary["recommendation"]["severity"] == 0
        assert "metrics" in summary
        assert summary["metrics"]["DICE"]["value"] == pytest.approx(0.92)
        assert summary["metrics"]["DICE"]["status"] == "PASS"
        assert summary["metrics"]["FD"]["value"] == pytest.approx(0.15)


class TestConstants:
    """Tests for QC standard constants."""

    def test_motion_fd_thresholds(self) -> None:
        """MOTION_FD_THRESHOLDS has expected pass=0.3 and warn=0.5."""
        assert MOTION_FD_THRESHOLDS.pass_threshold == 0.3
        assert MOTION_FD_THRESHOLDS.warn_threshold == 0.5

    def test_dice_thresholds_ordering(self) -> None:
        """DICE_THRESHOLDS pass > warn (stricter pass)."""
        assert DICE_THRESHOLDS.pass_threshold > DICE_THRESHOLDS.warn_threshold

    def test_dice_strict_vs_regular(self) -> None:
        """DICE_THRESHOLDS_STRICT pass > regular DICE pass."""
        assert DICE_THRESHOLDS_STRICT.pass_threshold > DICE_THRESHOLDS.pass_threshold


class TestCompositeMotionStatus:
    """Tests for get_composite_motion_status()."""

    def test_all_pass(self) -> None:
        """Low FD, low max, low outliers -> PASS."""
        assert get_composite_motion_status(0.1, 0.5, 2.0) == "PASS"

    def test_mean_fd_warns(self) -> None:
        """fd_mean=0.35 triggers WARN even with good max and outliers."""
        assert get_composite_motion_status(0.35, 0.5, 2.0) == "WARN"

    def test_pct_outliers_warns(self) -> None:
        """12% outliers triggers WARN even with good FD values."""
        assert get_composite_motion_status(0.1, 0.5, 12.0) == "WARN"

    def test_max_fd_warns(self) -> None:
        """fd_max=1.5 triggers WARN even with good mean and outliers."""
        assert get_composite_motion_status(0.1, 1.5, 2.0) == "WARN"

    def test_worst_wins(self) -> None:
        """fd_mean PASS + outliers WARN + fd_max POOR -> POOR."""
        assert get_composite_motion_status(0.1, 2.5, 12.0) == "POOR"

    def test_sub0005_scenario(self) -> None:
        """Sub-0005: fd_mean=0.297, fd_max=1.015, pct=11.7 -> WARN."""
        status = get_composite_motion_status(0.297, 1.015, 11.7)
        assert status == "WARN"


class TestNewMotionThresholds:
    """Tests for new motion threshold constants."""

    def test_outlier_pct_thresholds(self) -> None:
        """MOTION_OUTLIER_PCT_THRESHOLDS has expected values."""
        assert MOTION_OUTLIER_PCT_THRESHOLDS.pass_threshold == 5.0
        assert MOTION_OUTLIER_PCT_THRESHOLDS.warn_threshold == 20.0

    def test_fd_max_thresholds(self) -> None:
        """MOTION_FD_MAX_THRESHOLDS has expected values."""
        assert MOTION_FD_MAX_THRESHOLDS.pass_threshold == 1.0
        assert MOTION_FD_MAX_THRESHOLDS.warn_threshold == 2.0


class TestColorPalette:
    """Tests for ColorPalette fields."""

    def test_palette_fields_are_hex(self) -> None:
        """All ColorPalette fields are hex color strings starting with '#'."""
        for field_name in ("primary", "secondary", "success", "warning", "error"):
            value = getattr(COLORBLIND_PALETTE, field_name)
            assert isinstance(value, str)
            assert value.startswith("#"), f"{field_name} should start with '#'"
            assert len(value) == 7, f"{field_name} should be 7-char hex (e.g. #AABBCC)"


class TestCompoundDetectionLevelDispatch:
    """``recommend_dsc_qc`` must accept ``"+"``-joined ``detection_method``
    strings (emitted by the AIF cascade's union-pool fallback when voxels
    from multiple tiers contribute) and apply worst-tier-wins semantics.

    With only the AIF block populated (no motion / CNR), the aggregated
    overall recommendation IS the AIF recommendation, so we read
    ``out["recommendation"]["action"]`` directly.
    """

    @staticmethod
    def _aif_action(detection_method: str):
        from pmriprep.utils.qc_standards import recommend_dsc_qc
        out = recommend_dsc_qc({
            "aif_detection": {"detection_method": detection_method},
        })
        return out["recommendation"]["action"]

    def test_atlas_strict_atlas_relaxed_compound_includes(self) -> None:
        from pmriprep.utils.qc_standards import QCAction
        assert self._aif_action("atlas_strict+atlas_relaxed") == QCAction.INCLUDE

    def test_compound_with_permissive_reviews(self) -> None:
        """Worst-tier-wins: a single permissive part downgrades to REVIEW."""
        from pmriprep.utils.qc_standards import QCAction
        assert (
            self._aif_action("atlas_relaxed+atlas_permissive")
            == QCAction.REVIEW
        )

    def test_compound_with_synthetic_excludes(self) -> None:
        """Worst-tier-wins: a single synthetic part downgrades to EXCLUDE
        even when stricter tiers also contribute."""
        from pmriprep.utils.qc_standards import QCAction
        assert self._aif_action("atlas_relaxed+synthetic") == QCAction.EXCLUDE

    def test_full_atlas_permissive_reviews(self) -> None:
        """``full_atlas_permissive`` was previously missed by the
        chained-membership dispatch and would land in the catchall
        REVIEW branch. The set-membership dispatch now classifies it
        explicitly as REVIEW."""
        from pmriprep.utils.qc_standards import QCAction
        assert self._aif_action("full_atlas_permissive") == QCAction.REVIEW

    def test_unknown_part_in_compound_excludes(self) -> None:
        """An unknown part in a compound triggers the EXCLUDE branch
        (treated as ``unknown`` semantically)."""
        from pmriprep.utils.qc_standards import QCAction
        assert self._aif_action("atlas_strict+nonsense") == QCAction.EXCLUDE
