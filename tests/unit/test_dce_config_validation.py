"""Tests for DCEConfig validation rules."""

import pytest

from pmriprep.config.dce import DCEConfig


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
