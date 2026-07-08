"""Unit tests for DCE pharmacokinetic model initialization utilities.

Tests cover uptake-slope Ktrans estimation, late-phase ve estimation,
and peak-ratio vp estimation, including fallback behavior.
"""

import numpy as np
import pytest

from pmriprep.models.dce.initialization import (
    estimate_ktrans_uptake,
    estimate_ve_late_phase,
    estimate_vp_peak_ratio,
)


# ---------------------------------------------------------------------------
# Test 1: uptake-slope Ktrans estimation with synthetic data
# ---------------------------------------------------------------------------

class TestEstimateKtransUptake:
    """Uptake-slope regression: Ct = Ktrans * integral(Cp) + vp * Cp."""

    def test_estimate_ktrans_uptake_synthetic(self) -> None:
        true_ktrans = 0.1
        true_vp = 0.02

        dt = 0.1
        n_points = 80
        time = np.arange(n_points) * dt

        # Create a simple AIF: Gaussian peak followed by decay
        aif = 3.0 * np.exp(-((time - 1.0) ** 2) / (2 * 0.3**2)) + 0.3 * np.exp(-0.2 * time)
        # Ensure AIF is strictly positive post-peak for valid Patlak
        aif = np.maximum(aif, 0.01)

        # Build tissue curve from Patlak model: Ct = vp*Cp + Ktrans*integral(Cp)
        cum_aif = np.cumsum(aif) * dt
        tissue_curve = true_vp * aif + true_ktrans * cum_aif

        peak_idx = int(np.argmax(aif))
        ktrans_est = estimate_ktrans_uptake(tissue_curve, aif, time, peak_idx)

        # Recovery within 20%
        np.testing.assert_allclose(
            ktrans_est,
            true_ktrans,
            rtol=0.20,
            err_msg=f"Uptake Ktrans estimate {ktrans_est:.4f} not within 20% of {true_ktrans}",
        )

    def test_estimate_ktrans_uptake_fallback(self) -> None:
        """With too short an uptake window (< 3 points), return 0.1."""
        # Very short data: 6 points total, so the AIF-peak-to-tissue-peak
        # uptake window has too few points to regress.
        time = np.arange(6) * 0.1
        aif = np.array([0.0, 0.1, 0.5, 1.0, 0.8, 0.6])
        tissue_curve = np.array([0.0, 0.01, 0.05, 0.1, 0.08, 0.06])
        peak_idx = 3  # uptake window from peak_idx+1 is too short

        ktrans_est = estimate_ktrans_uptake(tissue_curve, aif, time, peak_idx)
        assert ktrans_est == 0.1, f"Expected fallback 0.1, got {ktrans_est}"

    def test_estimate_ktrans_uptake_zero_aif(self) -> None:
        """All-zero AIF should return fallback."""
        time = np.arange(50) * 0.1
        aif = np.zeros(50)
        tissue_curve = np.zeros(50)
        peak_idx = 0

        ktrans_est = estimate_ktrans_uptake(tissue_curve, aif, time, peak_idx)
        assert ktrans_est == 0.1, f"Expected fallback 0.1 for zero AIF, got {ktrans_est}"


# ---------------------------------------------------------------------------
# Test 3: Late-phase ve estimation
# ---------------------------------------------------------------------------

class TestEstimateVeLatePahse:
    """At late times, Ct/Cp -> ve in equilibrium."""

    def test_estimate_ve_late_phase_equilibrium(self) -> None:
        true_ve = 0.3

        n_points = 60
        # Create AIF with peak at frame 5, then slow decay
        aif = np.zeros(n_points)
        for i in range(n_points):
            if i < 5:
                aif[i] = i * 0.5
            else:
                aif[i] = 2.5 * np.exp(-0.05 * (i - 5))
        # Ensure AIF stays meaningfully above zero in late phase
        aif = np.maximum(aif, 0.2)

        # At equilibrium: Ct = ve * Cp
        tissue_curve = true_ve * aif

        peak_idx = 5
        ve_est = estimate_ve_late_phase(tissue_curve, aif, peak_idx)

        np.testing.assert_allclose(
            ve_est,
            true_ve,
            rtol=0.10,
            err_msg=f"Late-phase ve estimate {ve_est:.4f} not within 10% of {true_ve}",
        )

    def test_estimate_ve_late_phase_fallback(self) -> None:
        """Very short data with late peak should return default 0.2.

        With n=8, peak_idx=5: late_start = min(5+12, 8-5) = min(17, 3) = 3.
        Then late_start >= n-3 -> 3 >= 5 is False. Need peak_idx such that
        late_start >= n-3. With n=6, peak_idx=4:
        late_start = min(4+12, 6-5) = min(16, 1) = 1.
        late_start >= n-3 -> 1 >= 3 is False. With n=5, peak_idx=4:
        late_start = min(16, 0) = 0. 0 >= 2 is False.
        We need late_start >= n-3, so need peak_idx + 12 >= n-3 AND n-5 >= n-3,
        the latter is impossible. So the first branch triggers when the late
        slice has too few points. With peak_idx large enough: n=10, peak_idx=8:
        late_start = min(20, 5) = 5. 5 >= 7 is False. With n=6, peak_idx=5:
        late_start = min(17, 1) = 1. 1 >= 3 is False.
        Actually the fallback at late_start >= n-3 is hard to trigger
        with min(peak_idx+12, n-5). The second fallback (low AIF) is easier.
        Test both conditions: (1) all-zero AIF triggers the low-AIF fallback.
        """
        # Fallback via mean_cp <= 0.05 * max(aif) when late-phase AIF is zero
        aif = np.zeros(20)
        aif[5] = 10.0  # One peak at frame 5
        tissue_curve = np.ones(20) * 0.3
        peak_idx = 5  # late_start = min(17, 15) = 15, cp_late all zeros

        ve_est = estimate_ve_late_phase(tissue_curve, aif, peak_idx)
        assert ve_est == 0.2, f"Expected fallback 0.2 for zero late-phase AIF, got {ve_est}"

    def test_estimate_ve_late_phase_low_aif(self) -> None:
        """When late-phase AIF is very low (< 5% of peak), return 0.2."""
        n_points = 60
        aif = np.zeros(n_points)
        aif[:10] = 5.0  # Peak in first 10 frames
        aif[10:] = 0.01  # Very low AIF in late phase (< 5% of 5.0 = 0.25)
        tissue_curve = 0.3 * aif
        peak_idx = 0

        ve_est = estimate_ve_late_phase(tissue_curve, aif, peak_idx)
        assert ve_est == 0.2, f"Expected fallback 0.2 for low late-phase AIF, got {ve_est}"


# ---------------------------------------------------------------------------
# Test 5: vp peak ratio estimation
# ---------------------------------------------------------------------------

class TestEstimateVpPeakRatio:
    """vp ~ tissue_peak / aif_peak at bolus peak."""

    def test_estimate_vp_peak_ratio(self) -> None:
        true_vp = 0.05
        aif_peak = 5.0

        # At peak, Ct ~ vp * Cp (vascular dominates)
        aif = np.array([0.0, 1.0, 3.0, aif_peak, 4.0, 2.0, 1.0])
        tissue_curve = np.array([0.0, 0.05, 0.15, true_vp * aif_peak, 0.2, 0.1, 0.05])

        vp_est = estimate_vp_peak_ratio(tissue_curve, aif)
        np.testing.assert_allclose(
            vp_est,
            true_vp,
            rtol=0.01,
            err_msg=f"vp estimate {vp_est:.4f} not within 1% of {true_vp}",
        )

    def test_estimate_vp_peak_ratio_zero_aif(self) -> None:
        """AIF peak near zero should return default 0.02."""
        aif = np.array([0.0, 1e-8, 1e-9, 0.0])
        tissue_curve = np.array([0.0, 0.01, 0.005, 0.0])

        vp_est = estimate_vp_peak_ratio(tissue_curve, aif)
        assert vp_est == 0.02, f"Expected fallback 0.02 for near-zero AIF, got {vp_est}"

    def test_estimate_vp_peak_ratio_exact_zero_aif(self) -> None:
        """Exactly zero AIF should return default 0.02."""
        aif = np.zeros(10)
        tissue_curve = np.zeros(10)

        vp_est = estimate_vp_peak_ratio(tissue_curve, aif)
        assert vp_est == 0.02, f"Expected fallback 0.02 for zero AIF, got {vp_est}"

    def test_estimate_vp_peak_ratio_different_ratios(self) -> None:
        """Varying tissue-to-AIF ratios should give correct vp."""
        for true_vp in [0.01, 0.05, 0.1, 0.2]:
            aif = np.array([0.0, 2.0, 5.0, 3.0, 1.0])
            tissue_curve = true_vp * aif

            vp_est = estimate_vp_peak_ratio(tissue_curve, aif)
            np.testing.assert_allclose(
                vp_est,
                true_vp,
                rtol=1e-10,
                err_msg=f"vp estimate {vp_est:.4f} wrong for true_vp={true_vp}",
            )
