"""Tests for DCE/DSC config validation rules."""

from typing import Any

import pytest

from pmriprep.config.dce import DCEConfig
from pmriprep.config.dsc import DSCConfig


class TestAIFPeakFloorConfig:
    """The automatic detected-AIF peak floor is explicitly configurable."""

    def test_defaults_on(self) -> None:
        assert DCEConfig().aif_peak_floor is True

    def test_can_be_disabled(self) -> None:
        assert DCEConfig(aif_peak_floor=False).aif_peak_floor is False


class TestCbvConstraintModeValidation:
    """``cbv_constraint_mode`` accepts only ``auto``, ``skip``."""

    def test_typo_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid cbv_constraint_mode"):
            DCEConfig(cbv_constraint_mode="sft")

    def test_auto_plus_2cxm_is_ok(self) -> None:
        cfg = DCEConfig(cbv_constraint_mode="auto", model="2cxm", inference="vi")
        assert cfg.cbv_constraint_mode == "auto"


class TestConcentrationParameterValidation:
    """Signal-conversion scaling parameters reject invalid domains."""

    @pytest.mark.parametrize("value", [0., -1., 99.9, float("nan"), float("inf"), True, "900"])
    def test_invalid_t1_fallback_is_rejected(self, value: Any) -> None:
        """Reject invalid units/config types instead of silently clipping T1."""
        with pytest.raises((ValueError, TypeError), match="t1_default_ms"):
            DCEConfig(t1_default_ms=value)

    def test_t1_fallback_config_checks_units_before_field_strength_is_known(self) -> None:
        """Reject seconds at startup, retain the lower endpoint and defer the upper limit."""
        with pytest.raises(ValueError, match="milliseconds; did you enter seconds"):
            DCEConfig(t1_default_ms=1.4)
        assert DCEConfig(t1_default_ms=100.).t1_default_ms == 100.
        assert DCEConfig(t1_default_ms=7000.).t1_default_ms == 7000.

    @pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
    def test_relaxivity_must_be_finite_positive(self, value: float) -> None:
        with pytest.raises(ValueError, match="r1_relaxivity"):
            DCEConfig(r1_relaxivity=value)

    @pytest.mark.parametrize("value", [-0.1, 1.0, float("nan")])
    def test_hematocrit_must_be_fraction(self, value: float) -> None:
        with pytest.raises(ValueError, match="hematocrit"):
            DCEConfig(hematocrit=value)


class TestSERConfigValidation:
    """Protocol-defined SER sample times must be finite and ordered."""

    def test_defaults_use_one_and_five_minute_samples(self) -> None:
        cfg = DCEConfig()
        assert cfg.ser_early_time_s == 60.0
        assert cfg.ser_late_time_s == 300.0

    def test_custom_protocol_is_accepted(self) -> None:
        cfg = DCEConfig(
            ser_early_time_s=90,
            ser_late_time_s=300,
        )
        assert cfg.ser_early_time_s == 90.0
        assert cfg.ser_late_time_s == 300.0

    def test_sample_times_must_be_ordered(self) -> None:
        with pytest.raises(ValueError, match="ser_late_time_s"):
            DCEConfig(ser_early_time_s=300, ser_late_time_s=90)

    @pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
    def test_sample_times_must_be_positive(self, value: float) -> None:
        with pytest.raises(ValueError, match="ser_early_time_s"):
            DCEConfig(ser_early_time_s=value)


@pytest.mark.parametrize("config_cls", [DCEConfig, DSCConfig])
class TestUnderlayValidation:
    """Underlays accept supported image names and reject unknown names."""

    def test_unknown_underlay_rejected(self, config_cls) -> None:
        with pytest.raises(ValueError, match="Invalid underlay"):
            config_cls(underlays=["flair_pre"])

    def test_flair_precon_and_postcon_accepted(self, config_cls) -> None:
        cfg = config_cls(underlays=["flair_precon", "flair_postcon"])
        assert cfg.underlays == ["flair_precon", "flair_postcon"]

    def test_t1w_subtraction_accepted(self, config_cls) -> None:
        cfg = config_cls(underlays=["t1w_subtraction"])
        assert cfg.underlays == ["t1w_subtraction"]
