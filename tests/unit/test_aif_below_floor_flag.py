"""AIF retained below the gamma-fit floor when synthetic fallback is disabled.

With ``disable_aif_fallback=True`` a real AIF whose gamma fit is below the floor
(R² < 0) is retained rather than swapped for synthetic. These tests cover the
three surfaces that make that non-silent: the interface flag, the rendered
processing warning, and the QC recommendation. See the v0.78.1 plan.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Part A — the flag helper
# ---------------------------------------------------------------------------
class TestFlagHelper:
    def _flag(self, qc, **kw):
        from pmriprep.interfaces.dsc_aif import _flag_retained_below_floor

        defaults = dict(
            disable_fallback=True, detection_level="atlas_strict",
            gamma_fit_applied=False,
        )
        defaults.update(kw)
        _flag_retained_below_floor(qc, **defaults)
        return qc

    def test_flags_when_disabled_real_and_below_floor(self) -> None:
        qc = self._flag({"gamma_fit_r_squared": -0.4})
        assert qc["aif_gamma_fit_below_floor_retained"] is True
        assert qc["gamma_r2_retained"] == -0.4

    def test_noop_when_fallback_enabled(self) -> None:
        qc = self._flag({"gamma_fit_r_squared": -0.4}, disable_fallback=False)
        assert "aif_gamma_fit_below_floor_retained" not in qc

    def test_noop_when_synthetic(self) -> None:
        qc = self._flag({"gamma_fit_r_squared": -0.4}, detection_level="synthetic")
        assert "aif_gamma_fit_below_floor_retained" not in qc

    def test_noop_when_gamma_applied(self) -> None:
        qc = self._flag({"gamma_fit_r_squared": -0.4}, gamma_fit_applied=True)
        assert "aif_gamma_fit_below_floor_retained" not in qc

    def test_noop_when_r2_at_or_above_floor(self) -> None:
        assert "aif_gamma_fit_below_floor_retained" not in self._flag(
            {"gamma_fit_r_squared": 0.8})
        assert "aif_gamma_fit_below_floor_retained" not in self._flag(
            {"gamma_fit_r_squared": 0.0})

    def test_noop_when_r2_absent(self) -> None:
        # except / gamma_fit_mode=="off" paths leave no gamma_fit_r_squared.
        assert "aif_gamma_fit_below_floor_retained" not in self._flag({})


# ---------------------------------------------------------------------------
# Part B — the rendered processing warning
# ---------------------------------------------------------------------------
class TestFallbackWarning:
    def test_fires_on_retained_flag_non_synthetic(self) -> None:
        from pmriprep.utils.contrast_agent import collect_aif_fallback_warning

        w = collect_aif_fallback_warning(
            {"aif_detection": {
                "detection_method": "atlas_strict",
                "aif_gamma_fit_below_floor_retained": True,
                "gamma_r2_retained": -0.4,
            }},
            "DSC",
        )
        assert w is not None
        assert "retained despite a poor gamma-variate fit" in w
        # Physics wording: CBF/MTT/Tmax unreliable; TTP/PSR/AUC-nCBV survive.
        assert "Tmax may" in w and "AUC-nCBV" in w
        assert "Tmax are valid" not in w

    def test_none_for_normal_real_aif(self) -> None:
        from pmriprep.utils.contrast_agent import collect_aif_fallback_warning

        assert collect_aif_fallback_warning(
            {"aif_detection": {"detection_method": "atlas_strict"}}, "DSC",
        ) is None

    def test_synthetic_message_unchanged(self) -> None:
        from pmriprep.utils.contrast_agent import collect_aif_fallback_warning

        w = collect_aif_fallback_warning(
            {"aif_detection": {"detection_method": "synthetic"}}, "DSC",
        )
        assert w is not None and "population template" in w


# ---------------------------------------------------------------------------
# Part C — the QC recommendation
# ---------------------------------------------------------------------------
class TestRecommendation:
    def test_flag_adds_review(self) -> None:
        from pmriprep.utils.qc_standards import recommend_dsc_qc

        out = recommend_dsc_qc({"aif_detection": {
            "detection_method": "atlas_strict",
            "aif_gamma_fit_below_floor_retained": True,
        }})
        rec = out["recommendation"]
        assert rec["action"] == "review"
        assert rec["severity"] == 2

    def test_no_flag_normal_aif_unchanged(self) -> None:
        from pmriprep.utils.qc_standards import recommend_dsc_qc

        rec = recommend_dsc_qc(
            {"aif_detection": {"detection_method": "atlas_strict"}}
        )["recommendation"]
        assert rec["action"] == "include"
        assert rec["severity"] == 0
