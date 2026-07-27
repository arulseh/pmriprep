"""Unit tests for DSC signal processing utilities."""

import numpy as np
import pytest

from pmriprep.utils.dsc_signal import (
    compute_peak_height,
    compute_peak_height_legacy,
    compute_psr,
    compute_psr_legacy,
    compute_ttp_legacy,
    detect_bolus_arrival,
    find_bolus_arrival,
    get_baseline_signal,
    normalize_cbv,
    signal_to_concentration,
)


class TestSignalToConcentration:
    """Tests for signal_to_concentration (Delta-R2* conversion)."""

    def test_formula_1d(self) -> None:
        """Known S0=1000, signal drops to 800 with TE=0.03s."""
        n_timepoints = 20
        signal = np.ones(n_timepoints, dtype=np.float64) * 1000.0
        # After baseline (5 frames), signal drops to 800
        signal[5:] = 800.0
        te = 0.03

        conc = signal_to_concentration(signal, te=te, baseline_frames=5)

        # Baseline frames should have concentration ~0
        assert conc[0] == pytest.approx(0.0, abs=1e-6)
        assert conc[4] == pytest.approx(0.0, abs=1e-6)

        # Post-baseline: DR2* = -ln(800/1000) / 0.03 = -ln(0.8) / 0.03
        expected = -np.log(800.0 / 1000.0) / te
        assert conc[5] == pytest.approx(expected, rel=1e-6)
        assert conc[19] == pytest.approx(expected, rel=1e-6)

    def test_4d_input(self) -> None:
        """4D input (2,2,2,20) returns same shape, per-voxel baseline."""
        shape = (2, 2, 2, 20)
        signal = np.ones(shape, dtype=np.float64) * 500.0
        signal[..., 5:] = 300.0
        te = 0.025

        conc = signal_to_concentration(signal, te=te, baseline_frames=5)

        assert conc.shape == shape
        # All voxels identical => uniform concentration in post-baseline
        expected = -np.log(300.0 / 500.0) / te
        assert conc[0, 0, 0, 10] == pytest.approx(expected, rel=1e-6)
        assert conc[1, 1, 1, 15] == pytest.approx(expected, rel=1e-6)

    def test_signal_above_baseline_produces_negative_dr2(self) -> None:
        """Signal > S0 produces negative ΔR2* (no upper clipping).

        Negative ΔR2* preserves symmetric noise in post-bolus frames,
        avoiding half-rectified positive bias from clipping to zero.
        """
        signal = np.array([1000.0] * 5 + [1200.0] * 15, dtype=np.float64)
        te = 0.03

        conc = signal_to_concentration(signal, te=te, baseline_frames=5)

        # Signal above baseline: S/S0 = 1.2, ΔR2* = -ln(1.2)/0.03 < 0
        expected = -np.log(1.2) / te
        for i in range(5, 20):
            assert conc[i] == pytest.approx(expected, rel=1e-6)

    def test_validation_empty_array(self) -> None:
        """Empty array raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            signal_to_concentration(np.array([], dtype=np.float64), te=0.03)

    def test_validation_te_zero(self) -> None:
        """TE=0 raises ValueError."""
        signal = np.ones(10, dtype=np.float64)
        with pytest.raises(ValueError, match="TE must be positive"):
            signal_to_concentration(signal, te=0.0)

    def test_validation_baseline_exceeds_timepoints(self) -> None:
        """baseline_frames >= n_timepoints raises ValueError."""
        signal = np.ones(10, dtype=np.float64)
        with pytest.raises(ValueError, match="baseline_frames"):
            signal_to_concentration(signal, te=0.03, baseline_frames=10)


class TestDetectBolusArrival:
    """Tests for detect_bolus_arrival (z-score based detection)."""

    def test_synthetic_clear_bolus(self) -> None:
        """Baseline near 0 with jump at frame 10 detects arrival at ~10."""
        n = 30
        conc = np.zeros(n, dtype=np.float64)
        # Baseline: frames 0-9 are noise near 0
        rng = np.random.default_rng(42)
        conc[:10] = rng.normal(0.0, 0.01, 10)
        # Bolus: frames 10+ are high values
        conc[10:] = 5.0 + rng.normal(0.0, 0.1, 20)

        arrival_idx, n_baseline = detect_bolus_arrival(
            conc, baseline_frames=3, n_consecutive=1
        )

        assert arrival_idx == 10

    def test_consecutive_rejects_isolated_spike(self) -> None:
        """With n_consecutive=2, a single spike should not trigger arrival."""
        n = 40
        conc = np.zeros(n, dtype=np.float64)
        # Low noise baseline
        rng = np.random.default_rng(123)
        conc[:] = rng.normal(0.0, 0.01, n)
        # Single spike at frame 10
        conc[10] = 10.0
        # Actual bolus starting at frame 20
        conc[20:] = 5.0

        arrival_single, _ = detect_bolus_arrival(
            conc, baseline_frames=3, n_consecutive=1
        )
        arrival_consec, _ = detect_bolus_arrival(
            conc, baseline_frames=3, n_consecutive=2
        )

        # Single-frame detects the spike
        assert arrival_single == 10
        # Consecutive-2 should skip the spike and find the sustained bolus
        assert arrival_consec == 20


class TestComputePSR:
    """Tests for compute_psr (percent signal recovery)."""

    def test_1d_known_values(self) -> None:
        """Baseline=1000, S_min=400, tail=900 => PSR=83.33%."""
        n = 30
        signal = np.ones(n, dtype=np.float64) * 1000.0
        # Create a dip and recovery pattern
        signal[10:15] = 400.0  # Deep dip (S_min)
        signal[20:] = 900.0    # Recovery (tail)

        psr = compute_psr(signal, baseline_frames=10, tail_frames=10)

        # PSR = 100 * (S_tail - S_min) / (S_baseline - S_min)
        # S_min comes from global min = 400 (the compute_psr uses global min)
        expected = 100.0 * (900.0 - 400.0) / (1000.0 - 400.0)
        assert psr == pytest.approx(expected, rel=1e-4)


class TestComputePSRLegacy:
    """Tests for the legacy post-baseline PSR definition."""

    def test_1d_post_baseline_smin(self) -> None:
        """Legacy uses S_min from post-baseline only."""
        n = 30
        signal = np.ones(n, dtype=np.float64) * 1000.0
        # Global min in baseline region (frame 2)
        signal[2] = 300.0
        # Post-baseline min at frame 12
        signal[12] = 500.0
        # Recovery tail
        signal[25:] = 900.0

        psr_legacy = compute_psr_legacy(signal, baseline_frames=5, tail_frames=5)

        # Legacy: S_min from signal[5:] => 500
        # S0 = mean(signal[:5]) = (1000+1000+300+1000+1000)/5 = 860
        s0 = np.mean(signal[:5])
        expected = 100.0 * (900.0 - 500.0) / (s0 - 500.0)
        assert psr_legacy == pytest.approx(expected, rel=1e-4)


class TestComputePeakHeight:
    """Tests for compute_peak_height."""

    def test_1d_known_values(self) -> None:
        """Baseline=1000, S_min=400 => PH=60%."""
        n = 20
        signal = np.ones(n, dtype=np.float64) * 1000.0
        signal[10] = 400.0  # Minimum signal during bolus

        ph = compute_peak_height(signal, baseline_frames=5)

        # PH = 100 * (S_baseline - S_min) / S_baseline
        # S_min = global min = 400
        expected = 100.0 * (1000.0 - 400.0) / 1000.0
        assert ph == pytest.approx(expected, rel=1e-4)


class TestComputePeakHeightLegacy:
    """Tests for the legacy post-baseline peak-height definition."""

    def test_1d_post_baseline_smin(self) -> None:
        """Legacy uses S_min from post-baseline frames only."""
        n = 20
        signal = np.ones(n, dtype=np.float64) * 1000.0
        # Put global minimum in baseline
        signal[1] = 200.0
        # Post-baseline minimum
        signal[10] = 600.0

        ph_legacy = compute_peak_height_legacy(signal, baseline_frames=5)

        # Legacy: S_min from signal[5:] => 600
        # S0 = mean(signal[:5]) = (1000+200+1000+1000+1000)/5 = 840
        s0 = np.mean(signal[:5])
        expected = 100.0 * (s0 - 600.0) / s0
        assert ph_legacy == pytest.approx(expected, rel=1e-4)


class TestComputeTTPLegacy:
    """Tests for compute_ttp_legacy."""

    def test_1d_known_peak(self) -> None:
        """Concentration peak at frame 7, pre_bolus=5, dt=1.5 => TTP=10.5s."""
        n = 20
        conc = np.zeros(n, dtype=np.float64)
        # Peak at frame 7 (absolute index)
        conc[7] = 10.0
        conc[6] = 5.0
        conc[8] = 5.0

        ttp = compute_ttp_legacy(conc, pre_bolus_frames=5, dt=1.5)

        # post_bolus = conc[5:], peak_idx in post_bolus = 2 (frame 7 - 5)
        # ttp = (5 + 2) * 1.5 = 10.5
        assert float(ttp) == pytest.approx(10.5, rel=1e-6)

    def test_4d_unclipped(self) -> None:
        """TTP is returned unclipped (caller handles clipping)."""
        # Create 4D concentration with peak very late
        shape = (2, 2, 2, 100)
        conc = np.zeros(shape, dtype=np.float64)
        conc[..., 99] = 10.0  # Peak at last frame

        ttp = compute_ttp_legacy(conc, pre_bolus_frames=5, dt=1.0)

        assert ttp.shape == (2, 2, 2)
        # (5 + 94) * 1.0 = 99 — returned unclipped
        assert np.all(ttp == pytest.approx(99.0, rel=1e-6))

    def test_empty_window_raises(self) -> None:
        """Raises ValueError when pre_bolus_frames >= n_timepoints."""
        conc = np.zeros(10, dtype=np.float64)
        with pytest.raises(ValueError, match="pre_bolus_frames"):
            compute_ttp_legacy(conc, pre_bolus_frames=10, dt=1.0)


class TestNormalizeCBV:
    """Tests for normalize_cbv."""

    def test_nawm_mask_normalization(self) -> None:
        """nCBV = CBV / median(CBV[nawm_mask])."""
        cbv = np.ones((5, 5, 5), dtype=np.float64) * 4.0
        # Some voxels with different values
        cbv[0, 0, 0] = 8.0
        cbv[1, 1, 1] = 2.0

        nawm_mask = np.zeros((5, 5, 5), dtype=bool)
        # NAWM region with known values
        nawm_mask[2, 2, 2] = True
        nawm_mask[3, 3, 3] = True
        nawm_mask[4, 4, 4] = True
        # All NAWM voxels have CBV=4.0, so median=4.0

        ncbv = normalize_cbv(cbv, nawm_mask=nawm_mask)

        # nCBV = CBV / 4.0
        assert ncbv[0, 0, 0] == pytest.approx(2.0, rel=1e-6)
        assert ncbv[1, 1, 1] == pytest.approx(0.5, rel=1e-6)
        assert ncbv[2, 2, 2] == pytest.approx(1.0, rel=1e-6)

    def test_no_mask_returns_unchanged(self) -> None:
        """Without any mask, returns CBV unchanged (factor=1.0)."""
        cbv = np.ones((3, 3, 3), dtype=np.float64) * 5.0

        ncbv, factor = normalize_cbv(cbv, return_normalization_factor=True)

        assert factor == 1.0
        np.testing.assert_array_equal(ncbv, cbv)

    def test_zero_background_remains_zero(self) -> None:
        """NAWM normalization preserves a zero-valued image background."""
        cbv = np.zeros((5, 5, 5), dtype=np.float64)
        cbv[1:4, 1:4, 1:4] = 4.0
        nawm_mask = np.zeros_like(cbv, dtype=bool)
        nawm_mask[2, 2, 2] = True

        ncbv = normalize_cbv(cbv, nawm_mask=nawm_mask)

        np.testing.assert_array_equal(ncbv[cbv == 0], 0.0)
        np.testing.assert_array_equal(ncbv[cbv > 0], 1.0)

    def test_nawm_reference_uses_only_finite_positive_values(self) -> None:
        """Signed residuals and nonfinite sentinels cannot bias the median."""
        cbv = np.array(
            [-2.0, 0.0, np.nan, np.inf, 2.0, 4.0],
            dtype=np.float64,
        ).reshape((2, 3, 1))
        nawm_mask = np.ones_like(cbv, dtype=bool)

        _, factor = normalize_cbv(
            cbv,
            nawm_mask=nawm_mask,
            return_normalization_factor=True,
        )

        assert factor == pytest.approx(3.0)

    def test_brain_fallback_uses_only_finite_positive_values(self) -> None:
        """Whole-brain fallback applies the same reference admission rule."""
        cbv = np.array(
            [-1.0, np.nan, np.inf, 2.0, 6.0],
            dtype=np.float64,
        ).reshape((5, 1, 1))
        brain_mask = np.ones_like(cbv, dtype=bool)

        _, factor = normalize_cbv(
            cbv,
            brain_mask=brain_mask,
            return_normalization_factor=True,
        )

        assert factor == pytest.approx(4.0)


class TestGetBaselineSignal:
    """Tests for get_baseline_signal."""

    def test_1d_returns_scalar(self) -> None:
        """1D signal returns scalar float baseline."""
        signal = np.array([100.0, 200.0, 300.0, 400.0, 500.0,
                           0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        baseline = get_baseline_signal(signal, baseline_frames=5)

        assert isinstance(baseline, float)
        assert baseline == pytest.approx(300.0, rel=1e-6)

    def test_4d_returns_3d(self) -> None:
        """4D signal returns 3D baseline array."""
        shape = (2, 2, 2, 10)
        signal = np.ones(shape, dtype=np.float64) * 500.0
        signal[0, 0, 0, :3] = 100.0  # Different baseline for one voxel

        baseline = get_baseline_signal(signal, baseline_frames=3)

        assert baseline.shape == (2, 2, 2)
        assert baseline[0, 0, 0] == pytest.approx(100.0, rel=1e-6)
        assert baseline[1, 1, 1] == pytest.approx(500.0, rel=1e-6)


class TestFindBolusArrival:
    """Tests for find_bolus_arrival (signal-domain detection)."""

    def test_clear_drop_detected(self) -> None:
        """Signal with clear drop below threshold returns correct frame."""
        signal = np.ones(30, dtype=np.float64) * 1000.0
        # Stable baseline for 10 frames, then sharp drop
        signal[15:] = 400.0

        arrival = find_bolus_arrival(signal, baseline_frames=10, threshold_factor=3.0)

        assert arrival == 15

    def test_no_bolus_raises(self) -> None:
        """Flat signal with no drop raises ValueError."""
        signal = np.ones(30, dtype=np.float64) * 1000.0

        with pytest.raises(ValueError, match="No bolus arrival"):
            find_bolus_arrival(signal, baseline_frames=10)

    def test_non_1d_raises(self) -> None:
        """Non-1D input raises ValueError."""
        signal = np.ones((5, 5, 5, 20), dtype=np.float64)

        with pytest.raises(ValueError, match="1D"):
            find_bolus_arrival(signal, baseline_frames=5)
