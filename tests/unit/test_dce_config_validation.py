"""Tests for DCEConfig validation rules (v0.27.0+)."""

import pytest

from pmriprep.config.dce import DCEConfig


class TestHardConstraintRejectedForIncompatibleModels:
    """v0.27.0: HARD constraint is physically incompatible with 2CXM and Bayesian.

    Sourbron & Buckley MRM 2013: vp enters 2CXM nonlinearly (eigenvalues
    α± + coefficient mixing), so hard-fixing vp propagates DSC CBV bias
    into Fp and PS. For VI/MCMC, HARD bypasses the Bayesian fitter for
    bounded L-BFGS-B, triggering mass post-fit rejection on non-enhancing
    voxels. These combinations used to silently downgrade to SOFT —
    v0.27.0 explicitly rejects them so users pick a valid combination.
    """

    def test_hard_plus_2cxm_raises(self) -> None:
        with pytest.raises(ValueError, match="incompatible.*2cxm"):
            DCEConfig(cbv_constraint_mode="hard", model="2cxm")

    def test_hard_plus_vi_raises(self) -> None:
        with pytest.raises(ValueError, match="incompatible.*vi"):
            DCEConfig(cbv_constraint_mode="hard", bayesian="vi")

    def test_hard_plus_mcmc_raises(self) -> None:
        with pytest.raises(ValueError, match="incompatible.*mcmc"):
            DCEConfig(cbv_constraint_mode="hard", bayesian="mcmc")

    def test_hard_plus_tofts_is_ok(self) -> None:
        """HARD with a compatible model (tofts, NLLS) still succeeds."""
        cfg = DCEConfig(cbv_constraint_mode="hard", model="tofts", bayesian="none")
        assert cfg.cbv_constraint_mode == "hard"

    def test_soft_plus_2cxm_is_ok(self) -> None:
        """SOFT with any model/bayesian combination is the supported path."""
        cfg = DCEConfig(cbv_constraint_mode="soft", model="2cxm", bayesian="vi")
        assert cfg.cbv_constraint_mode == "soft"

    def test_auto_plus_2cxm_is_ok(self) -> None:
        """AUTO with 2CXM+VI is fine — auto never emits HARD."""
        cfg = DCEConfig(cbv_constraint_mode="auto", model="2cxm", bayesian="vi")
        assert cfg.cbv_constraint_mode == "auto"
