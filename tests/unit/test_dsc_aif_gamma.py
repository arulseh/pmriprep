"""Unit tests for DSC AIF gamma-variate fitting functions."""

import numpy as np
import pytest

from pmriprep.utils.dsc_aif import (
    GammaVariateFitResult,
    _compute_auc,
    _estimate_initial_params,
    _find_recirculation_start,
    _gamma_variate_model,
    fit_gamma_variate,
)


class TestGammaVariateModel:
    """Tests for _gamma_variate_model."""

    def test_known_params(self) -> None:
        """Gamma-variate with known params produces positive peak at expected time."""
        t = np.arange(60, dtype=np.float64)
        amplitude = 10.0
        t0 = 5.0
        alpha = 2.5
        beta = 3.0

        curve = _gamma_variate_model(t, amplitude, t0, alpha, beta)

        # Peak should be positive
        assert float(np.max(curve)) > 0.0

        # Peak should occur near t0 + alpha*beta = 5 + 7.5 = 12.5
        peak_idx = int(np.argmax(curve))
        expected_peak_time = t0 + alpha * beta
        assert abs(peak_idx - expected_peak_time) <= 1.0

        # Pre-bolus frames (t < t0) should be zero
        assert np.all(curve[:int(t0)] == 0.0)

    def test_pre_bolus_zero(self) -> None:
        """All time points before t0 return exactly zero."""
        t = np.arange(60, dtype=np.float64)
        t0 = 10.0

        curve = _gamma_variate_model(t, amplitude=5.0, t0=t0, alpha=2.0, beta=3.0)

        # Frames 0 through 10 (t <= t0) should all be zero
        assert np.all(curve[:int(t0) + 1] == 0.0)

    def test_peak_time_analytical(self) -> None:
        """Peak time matches the analytical value t0 + alpha*beta within 1 frame."""
        t = np.arange(100, dtype=np.float64)
        t0 = 8.0
        alpha = 3.0
        beta = 4.0

        curve = _gamma_variate_model(t, amplitude=1.0, t0=t0, alpha=alpha, beta=beta)

        peak_idx = int(np.argmax(curve))
        analytical_peak = t0 + alpha * beta  # 8 + 12 = 20
        assert abs(peak_idx - analytical_peak) <= 1.0


class TestEstimateInitialParams:
    """Tests for _estimate_initial_params."""

    def test_synthetic_aif(self) -> None:
        """Initial param estimates are reasonable for a synthetic AIF with small noise."""
        rng = np.random.default_rng(42)
        t = np.arange(60, dtype=np.float64)
        true_amplitude = 8.0
        true_t0 = 5.0
        true_alpha = 2.5
        true_beta = 3.0

        clean_curve = _gamma_variate_model(t, true_amplitude, true_t0, true_alpha, true_beta)
        noise = rng.normal(0, 0.05 * float(np.max(clean_curve)), size=60)
        noisy_curve = clean_curve + noise
        noisy_curve[:5] = noise[:5] * 0.01  # Keep baseline near zero

        amp_est, t0_est, alpha_est, beta_est = _estimate_initial_params(
            noisy_curve, baseline_frames=5
        )

        # Amplitude within 50%
        assert amp_est > 0.0
        # t0 within 3 frames
        assert abs(t0_est - true_t0) <= 3.0
        # alpha within 50%
        assert 0.5 * true_alpha <= alpha_est <= 1.5 * true_alpha + 1.0
        # beta within 50%
        assert 0.5 * true_beta <= beta_est <= 1.5 * true_beta + 1.0


class TestComputeAUC:
    """Tests for _compute_auc."""

    def test_rectangular(self) -> None:
        """Constant curve of height 5 over 10 frames produces expected AUC."""
        curve = np.full(10, 5.0)
        tr = 1.5
        auc = _compute_auc(curve, tr=tr)
        # Trapezoidal rule for constant function: height * (n-1) * dx
        # np.trapezoid of constant 5.0 over 10 points with implicit dx=1 → 5.0 * 9 = 45
        # Then multiply by tr
        expected = 5.0 * 9.0 * tr
        assert auc == pytest.approx(expected, rel=1e-4)

    def test_triangular(self) -> None:
        """Triangle peak with height 10 and base 10 frames produces expected AUC."""
        # Triangle: rises from 0 to 10 over 5 frames, falls back to 0 over 5 frames
        curve = np.zeros(11)
        for i in range(6):
            curve[i] = 2.0 * i  # 0, 2, 4, 6, 8, 10
        for i in range(6, 11):
            curve[i] = 10.0 - 2.0 * (i - 5)  # 8, 6, 4, 2, 0

        tr = 1.0
        auc = _compute_auc(curve, tr=tr)
        # Triangle area = 0.5 * base * height = 0.5 * 10 * 10 = 50
        expected = 50.0
        assert auc == pytest.approx(expected, rel=1e-2)


class TestFitGammaVariate:
    """Tests for fit_gamma_variate."""

    def test_clean_data(self) -> None:
        """Clean synthetic AIF produces high R-squared and convergence."""
        t = np.arange(60, dtype=np.float64)
        aif = _gamma_variate_model(t, amplitude=10.0, t0=5.0, alpha=2.5, beta=3.0)

        result = fit_gamma_variate(aif, baseline_frames=5, remove_recirculation=False)

        assert result.fit_converged is True
        assert result.r_squared > 0.95

    def test_with_recirculation(self) -> None:
        """AIF with secondary peak triggers recirculation detection with good R²."""
        t = np.arange(60, dtype=np.float64)
        # First pass
        first_pass = _gamma_variate_model(t, amplitude=10.0, t0=5.0, alpha=2.5, beta=3.0)
        # Recirculation: delayed smaller peak
        recirc = _gamma_variate_model(t, amplitude=3.0, t0=25.0, alpha=2.0, beta=4.0)
        aif = first_pass + recirc

        result = fit_gamma_variate(aif, baseline_frames=5, remove_recirculation=True)

        assert result.fit_converged is True
        assert result.recirculation_removed is True
        assert result.r_squared > 0.5

    def test_preserve_auc(self) -> None:
        """With preserve_auc=True, AUC ratio is close to 1.0."""
        t = np.arange(60, dtype=np.float64)
        aif = _gamma_variate_model(t, amplitude=10.0, t0=5.0, alpha=2.5, beta=3.0)

        result = fit_gamma_variate(
            aif, baseline_frames=5, remove_recirculation=False, preserve_auc=True
        )

        assert result.auc_ratio == pytest.approx(1.0, abs=0.2)

    def test_noisy_data(self) -> None:
        """Noisy synthetic AIF still converges with reasonable R-squared."""
        rng = np.random.default_rng(123)
        t = np.arange(60, dtype=np.float64)
        clean = _gamma_variate_model(t, amplitude=10.0, t0=5.0, alpha=2.5, beta=3.0)
        peak = float(np.max(clean))
        noise_std = 0.5 * peak * 0.1  # 5% of peak
        aif = clean + rng.normal(0, noise_std, size=60)
        aif = np.maximum(aif, 0.0)

        result = fit_gamma_variate(aif, baseline_frames=5, remove_recirculation=False)

        assert result.fit_converged is True
        assert result.r_squared > 0.7

    def test_large_recirculation_r_squared(self) -> None:
        """Regression: large recirculation tail must not cause spurious R² rejection.

        Reproduces the sub-01 run2 scenario where a 120-frame AIF with
        baseline_frames=45, sharp first-pass peak (~28 mM), and persistent
        recirculation tail (~2-3 mM) caused R² = -0.554 because R² was
        evaluated after AUC-preservation scaling inflated the fitted curve.
        """
        n_frames = 120
        t = np.arange(n_frames, dtype=np.float64)
        baseline_frames = 45

        # First pass: sharp gamma-variate peaking at ~28 mM
        # Peak at t0 + alpha*beta = 46 + 2.5*1.5 = 49.75
        first_pass = _gamma_variate_model(t, amplitude=28.0, t0=46.0, alpha=2.5, beta=1.5)
        measured_peak = float(np.max(first_pass))

        # Persistent tail elevation (~2-3 mM) starting well after first-pass washout.
        # This models recirculation + slow washout adding ~60% extra full-curve AUC
        # without competing with the first-pass peak.
        tail = np.zeros(n_frames)
        tail_start = 60
        tail[tail_start:] = 3.0 * (1.0 - np.exp(-0.1 * (t[tail_start:] - tail_start)))

        aif = first_pass + tail

        result = fit_gamma_variate(
            aif, baseline_frames=baseline_frames, remove_recirculation=True, preserve_auc=True
        )

        # Fit must converge and not be spuriously rejected
        assert result.fit_converged is True
        assert result.r_squared > 0.5, (
            f"R² = {result.r_squared:.3f} — gamma fit should not be rejected "
            f"due to AUC-scaling inflation"
        )

        # Fitted peak should not be inflated beyond 20% of measured first-pass peak
        fitted_peak = float(np.max(result.fitted_curve))
        assert fitted_peak < 1.2 * measured_peak, (
            f"Fitted peak {fitted_peak:.1f} mM exceeds 1.2× measured "
            f"first-pass peak {measured_peak:.1f} mM — AUC scaling is inflating"
        )


class TestFindRecirculationStart:
    """Tests for _find_recirculation_start."""

    def test_clear_secondary_peak(self) -> None:
        """AIF with a clear secondary peak returns a positive frame index."""
        t = np.arange(60, dtype=np.float64)
        first_pass = _gamma_variate_model(t, amplitude=10.0, t0=5.0, alpha=2.5, beta=3.0)
        recirc = _gamma_variate_model(t, amplitude=4.0, t0=25.0, alpha=2.0, beta=4.0)
        aif = first_pass + recirc

        recirc_start = _find_recirculation_start(aif, first_pass, baseline_frames=5)

        assert recirc_start > 0

    def test_no_secondary_peak(self) -> None:
        """Clean gamma-variate without recirculation returns -1."""
        t = np.arange(60, dtype=np.float64)
        clean = _gamma_variate_model(t, amplitude=10.0, t0=5.0, alpha=2.5, beta=3.0)

        recirc_start = _find_recirculation_start(clean, clean, baseline_frames=5)

        assert recirc_start == -1


class TestGammaVariateFitResultDataclass:
    """Tests for GammaVariateFitResult dataclass."""

    def test_fields_accessible(self) -> None:
        """All fields of GammaVariateFitResult are accessible after construction."""
        curve = np.zeros(10, dtype=np.float64)
        result = GammaVariateFitResult(
            fitted_curve=curve,
            amplitude=1.0,
            t0=5.0,
            alpha=2.0,
            beta=3.0,
            auc_original=10.0,
            auc_fitted=9.5,
            auc_ratio=0.95,
            residual_rms=0.1,
            r_squared=0.99,
            fit_converged=True,
            recirculation_removed=False,
        )

        assert result.amplitude == 1.0
        assert result.t0 == 5.0
        assert result.alpha == 2.0
        assert result.beta == 3.0
        assert result.auc_original == 10.0
        assert result.auc_fitted == 9.5
        assert result.auc_ratio == pytest.approx(0.95)
        assert result.residual_rms == pytest.approx(0.1)
        assert result.r_squared == pytest.approx(0.99)
        assert result.fit_converged is True
        assert result.recirculation_removed is False
        assert result.fitted_curve.shape == (10,)
