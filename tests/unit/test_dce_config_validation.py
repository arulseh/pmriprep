"""Tests for DCE/DSC config validation rules."""

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

    def test_hard_is_not_a_valid_mode(self) -> None:
        with pytest.raises(ValueError, match="Invalid cbv_constraint_mode"):
            DCEConfig(cbv_constraint_mode="hard")

    def test_typo_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid cbv_constraint_mode"):
            DCEConfig(cbv_constraint_mode="sft")

    def test_soft_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid cbv_constraint_mode"):
            DCEConfig(cbv_constraint_mode="soft")

    def test_auto_plus_2cxm_is_ok(self) -> None:
        cfg = DCEConfig(cbv_constraint_mode="auto", model="2cxm", inference="vi")
        assert cfg.cbv_constraint_mode == "auto"


@pytest.mark.parametrize("config_cls", [DCEConfig, DSCConfig])
class TestUnderlayValidation:
    """``underlays`` takes the split FLAIR tokens; legacy ``flair`` is rejected."""

    def test_legacy_flair_token_rejected(self, config_cls) -> None:
        with pytest.raises(ValueError, match="Invalid underlay"):
            config_cls(underlays=["flair"])

    def test_unknown_underlay_rejected(self, config_cls) -> None:
        with pytest.raises(ValueError, match="Invalid underlay"):
            config_cls(underlays=["flair_pre"])

    def test_flair_precon_and_postcon_accepted(self, config_cls) -> None:
        cfg = config_cls(underlays=["flair_precon", "flair_postcon"])
        assert cfg.underlays == ["flair_precon", "flair_postcon"]

    def test_t1w_subtraction_accepted(self, config_cls) -> None:
        cfg = config_cls(underlays=["t1w_subtraction"])
        assert cfg.underlays == ["t1w_subtraction"]
