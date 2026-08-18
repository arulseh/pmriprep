"""Unit tests for DSC AIF gamma-variate fitting functions."""

import numpy as np
import pytest

from pmriprep.config.dsc import DSCConfig
from pmriprep.utils.dsc_aif import (
    GAMMA_NEG_TAIL_MIN_FRAMES,
    GAMMA_NOISE_RATIO_THRESHOLD,
    GAMMA_SPR_THRESHOLD,
    GammaVariateFitResult,
    _compute_auc,
    _estimate_initial_params,
    _find_recirculation_start,
    _gamma_variate_model,
    compute_aif_fwhm_frames,
    fit_gamma_variate,
    get_aif_smoothing_window,
    should_apply_gamma_fit,
)


class TestAIFSmoothingWindow:
    """Temporal AIF smoothing should be acquisition-rate aware."""

    @pytest.mark.parametrize(
        ("tr", "expected"),
        [(0.5, 15), (1.5, 5), (3.0, 3)],
    )
    def test_window_tracks_duration(self, tr: float, expected: int) -> None:
        """The window remains near 7.5 seconds across representative TRs."""
        assert get_aif_smoothing_window(100, tr) == expected

    def test_window_is_odd_and_capped_to_curve(self) -> None:
        """Short curves receive the largest valid odd window."""
        assert get_aif_smoothing_window(10, 0.1) == 9

    @pytest.mark.parametrize("tr", [0.0, -1.0, float("nan")])
    def test_invalid_tr_uses_historical_fallback(self, tr: float) -> None:
        """Invalid timing retains the five-frame defensive fallback."""
        assert get_aif_smoothing_window(100, tr) == 5

    def test_empty_curve_is_rejected(self) -> None:
        """A smoothing window cannot be defined for an empty curve."""
        with pytest.raises(ValueError, match="positive"):
            get_aif_smoothing_window(0, 1.5)


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

    def test_initial_estimate_is_projected_into_solver_bounds(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A late noisy maximum cannot make the optimizer start infeasible."""
        captured: dict[str, np.ndarray] = {}

        def fake_least_squares(
            _function: object,
            x0: np.ndarray,
            *,
            args: object,
            bounds: tuple[tuple[float, ...], tuple[float, ...]],
            method: str,
            max_nfev: int,
        ) -> object:
            del args, method, max_nfev
            lower, upper = (np.asarray(bound, dtype=float) for bound in bounds)
            captured["x0"] = np.asarray(x0, dtype=float)
            assert np.all(captured["x0"] >= lower)
            assert np.all(captured["x0"] <= upper)
            return type(
                "Result",
                (),
                {"success": False, "x": captured["x0"]},
            )()

        monkeypatch.setattr(
            "pmriprep.utils.dsc_aif.optimize.least_squares",
            fake_least_squares,
        )
        t = np.arange(120, dtype=float)
        late_peak = 20.0 * np.exp(-0.5 * ((t - 80.0) / 5.0) ** 2)
        initial = _estimate_initial_params(late_peak, baseline_frames=40)
        assert initial[1] > 60.0

        result = fit_gamma_variate(late_peak, baseline_frames=40)

        assert captured["x0"][1] == 60.0
        assert result.fit_converged is False
        assert result.failure_reason == "optimizer_not_converged"

    def test_nonfinite_initial_estimate_returns_typed_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-finite heuristic estimate cannot escape as an exception."""
        monkeypatch.setattr(
            "pmriprep.utils.dsc_aif._estimate_initial_params",
            lambda *args: (float("inf"), 5.0, 2.0, 3.0),
        )
        t = np.arange(60, dtype=float)
        aif = _gamma_variate_model(t, amplitude=10.0, t0=5.0, alpha=2.5, beta=3.0)

        result = fit_gamma_variate(aif, baseline_frames=5)

        assert result.fit_converged is False
        assert result.failure_reason == "nonfinite_initialization"

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

    def test_symmetric_auc_window_preloaded(self) -> None:
        """Regression: preloaded-shape AIF (first-pass + persistent tail)
        produces auc_ratio ∈ [0.85, 1.15] with the symmetric-window fix.

        Pre-fix: ``auc_original`` was full-curve while ``auc_fitted`` was
        first-pass-window only, giving auc_ratio ~0.4-0.6 for any
        non-trivial tail (4 Gadovist 0.75+0.75 split-dose subjects all
        rejected on this asymmetric ratio despite R²>0.94).

        Uses ``preserve_auc=False`` so ``auc_ratio`` reflects pre-scaling
        shape mismatch — the metric we actually care about. With the
        symmetric window, ratio is near 1.0; the asymmetric (full-curve)
        denominator would have given ~0.5 here.
        """
        n_frames = 120
        t = np.arange(n_frames, dtype=np.float64)
        baseline_frames = 20

        # First-pass gamma-variate: peak at t0 + alpha*beta = 25 + 2.5*1.5 ≈ 28.75
        first_pass = _gamma_variate_model(
            t, amplitude=28.0, t0=25.0, alpha=2.5, beta=1.5,
        )

        # Persistent preload tail (~3 mM) AFTER first-pass washout — this is
        # the residual contrast still circulating from the DCE preload.
        tail = np.zeros(n_frames)
        tail_start = 50
        tail[tail_start:] = 3.0 * (
            1.0 - np.exp(-0.1 * (t[tail_start:] - tail_start))
        )

        aif = first_pass + tail

        result = fit_gamma_variate(
            aif, baseline_frames=baseline_frames,
            remove_recirculation=True, preserve_auc=False,
        )

        assert result.fit_converged is True
        # Symmetric window: numerator and denominator both first-pass.
        # Ratio should be near 1.0, inside the informational [0.85, 1.15]
        # range (not an enforced gate). An asymmetric full-curve
        # denominator would give ~0.5 here.
        assert 0.85 <= result.auc_ratio <= 1.15, (
            f"auc_ratio = {result.auc_ratio:.3f} — symmetric window should "
            f"yield ratio near 1.0 even with persistent post-bolus tail"
        )
        # first_pass_n must be populated for the eligibility prerequisite.
        assert result.first_pass_n >= 8

    def test_auc_ratio_independent_of_preserve_auc(self) -> None:
        """auc_ratio is computed BEFORE preservation scaling so it reflects
        actual shape-fit quality regardless of preserve_auc.

        Without this, with preserve_auc=True (production default) auc_ratio
        would be mechanically 1.0 by construction and the soft-warning
        machinery in should_apply_gamma_fit() would be dead code.
        """
        t = np.arange(60, dtype=np.float64)
        aif = _gamma_variate_model(t, amplitude=10.0, t0=5.0, alpha=2.5, beta=3.0)

        result_preserved = fit_gamma_variate(
            aif, baseline_frames=5, remove_recirculation=False, preserve_auc=True,
        )
        result_unscaled = fit_gamma_variate(
            aif, baseline_frames=5, remove_recirculation=False, preserve_auc=False,
        )

        # Both report the same pre-scaling auc_ratio (shape-fit quality).
        assert result_preserved.auc_ratio == pytest.approx(
            result_unscaled.auc_ratio, abs=0.01,
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


# =============================================================================
# Helpers for auto-decision tests
# =============================================================================


def _make_clean_aif(n: int = 60, baseline_frames: int = 5) -> np.ndarray:
    """Create a clean gamma-variate shaped AIF with no defects."""
    t = np.arange(n, dtype=float)
    aif = np.zeros(n)
    for i in range(baseline_frames, n):
        dt = t[i] - t[baseline_frames]
        aif[i] = 10.0 * (dt**2.0) * np.exp(-dt / 3.0)
    return aif


def _make_fit_result(
    original_aif: np.ndarray,
    r_squared: float = 0.95,
    fit_converged: bool = True,
    fitted_curve: np.ndarray | None = None,
    auc_ratio: float = 1.0,
    first_pass_n: int = 25,
) -> GammaVariateFitResult:
    """Create a GammaVariateFitResult for testing.

    Default ``first_pass_n=25`` is well above the
    ``GAMMA_MIN_FIRST_PASS_SAMPLES`` (8) eligibility threshold, so existing
    tests that don't care about sample count won't trip the prerequisite.
    """
    if fitted_curve is None:
        fitted_curve = original_aif.copy()
    return GammaVariateFitResult(
        fitted_curve=fitted_curve,
        amplitude=10.0,
        t0=5.0,
        alpha=2.0,
        beta=3.0,
        auc_original=100.0,
        auc_fitted=100.0 * auc_ratio,
        auc_ratio=auc_ratio,
        residual_rms=0.1,
        r_squared=r_squared,
        fit_converged=fit_converged,
        first_pass_n=first_pass_n,
    )


# =============================================================================
# TestShouldApplyGammaFit
# =============================================================================


class TestShouldApplyGammaFit:
    """Tests for should_apply_gamma_fit() auto-decision logic."""

    def test_reject_fit_not_converged(self) -> None:
        """Hard gate: reject when fit did not converge."""
        aif = _make_clean_aif()
        fit = _make_fit_result(aif, fit_converged=False)
        decision, _ = should_apply_gamma_fit(aif, fit)
        assert not decision.apply
        assert "converge" in decision.reason.lower()

    def test_reject_low_r_squared(self) -> None:
        """Hard gate: reject when R² is below threshold."""
        aif = _make_clean_aif()
        fit = _make_fit_result(aif, r_squared=0.84)
        decision, metrics = should_apply_gamma_fit(aif, fit)
        assert not decision.apply

    def test_reject_peak_ratio_too_high(self) -> None:
        """Hard gate: reject when peak ratio exceeds upper bound."""
        aif = _make_clean_aif()
        fitted = aif.copy() * 1.12
        fit = _make_fit_result(aif, fitted_curve=fitted)
        decision, metrics = should_apply_gamma_fit(aif, fit)
        assert not decision.apply
        assert "peak" in decision.reason.lower() or "Peak" in decision.reason

    def test_auc_ratio_no_longer_hard_gate_when_no_defect(self) -> None:
        """auc_ratio outside [0.85, 1.15] does not gate: fit is
        rejected for the usual 'no defect detected' reason, and auc_ratio
        is still reported in metrics for QC provenance.
        """
        aif = _make_clean_aif()
        fit = _make_fit_result(aif, auc_ratio=1.16)
        decision, metrics = should_apply_gamma_fit(aif, fit)
        assert not decision.apply
        # Reason is the terminal "no defect" branch, NOT an AUC-gate string.
        assert "AUC" not in decision.reason
        assert (
            "defect" in decision.reason.lower()
            or "clean" in decision.reason.lower()
        )
        # AUC ratio still reported for QC provenance.
        assert metrics["auc_ratio"] == pytest.approx(1.16)
        # 1.16 is inside the soft [0.7, 1.3] band; warning flag stays False.
        assert metrics["auc_ratio_warning"] is False

    def test_should_apply_gamma_fit_accepts_defect_with_bad_auc_ratio(self) -> None:
        """Defect-positive AIF + auc_ratio outside [0.85, 1.15] still accepts.

        Regression for the preloaded-DSC case: 4 Gadovist 0.75+0.75 subjects
        had R²>0.94 + recirculation defect, but the (asymmetric, pre-fix)
        auc_ratio was 0.39-0.59 and the gate rejected. With the gate
        demoted, defect detection drives acceptance regardless of auc_ratio.
        """
        # Build a defect-positive AIF (clean first pass + recirculation hump)
        aif = _make_clean_aif()
        peak_idx = int(np.argmax(aif))
        peak_val = float(np.max(aif))
        recirc_center = peak_idx + 15
        if recirc_center + 5 < len(aif):
            for i in range(recirc_center - 3, min(recirc_center + 4, len(aif))):
                offset = i - recirc_center
                aif[i] += peak_val * 0.35 * np.exp(-0.5 * offset**2)

        fit = _make_fit_result(aif, auc_ratio=0.5, r_squared=0.95)
        decision, metrics = should_apply_gamma_fit(aif, fit)
        assert decision.apply
        assert decision.defect_detected == "recirculation"
        # auc_ratio carried through; soft warning fires (outside [0.7, 1.3]).
        assert metrics["auc_ratio"] == pytest.approx(0.5)
        assert metrics["auc_ratio_warning"] is True

    def test_should_apply_gamma_fit_rejects_insufficient_first_pass_samples(
        self,
    ) -> None:
        """Hard prerequisite: < 8 first-pass samples → reject regardless of fit
        quality (Welker AJNR 2015, Calamante MRM 2010).
        """
        aif = _make_clean_aif()
        fit = _make_fit_result(
            aif, r_squared=0.99, auc_ratio=1.0, first_pass_n=5,
        )
        decision, metrics = should_apply_gamma_fit(aif, fit)
        assert not decision.apply
        assert decision.reason.startswith("insufficient first-pass samples")
        assert metrics["first_pass_n"] == 5

    def test_auc_ratio_soft_warning_inside_band(self) -> None:
        """auc_ratio inside [0.7, 1.3] does NOT trigger the soft warning."""
        aif = _make_clean_aif()
        fit = _make_fit_result(aif, auc_ratio=0.95)
        _, metrics = should_apply_gamma_fit(aif, fit)
        assert metrics["auc_ratio"] == pytest.approx(0.95)
        assert metrics["auc_ratio_warning"] is False

    def test_accept_recirculation_defect(self) -> None:
        """Recirculation defect: accept when SPR > threshold and local max."""
        aif = _make_clean_aif()
        peak_idx = int(np.argmax(aif))
        peak_val = float(np.max(aif))
        # Add a secondary hump well after first-pass washout
        recirc_center = peak_idx + 15
        if recirc_center + 5 < len(aif):
            for i in range(recirc_center - 3, min(recirc_center + 4, len(aif))):
                offset = i - recirc_center
                aif[i] += peak_val * 0.35 * np.exp(-0.5 * offset**2)

        fit = _make_fit_result(aif)
        decision, metrics = should_apply_gamma_fit(aif, fit)
        assert decision.apply
        assert decision.defect_detected == "recirculation"
        assert metrics["spr"] > GAMMA_SPR_THRESHOLD
        assert metrics["spr_is_local_max"]

    def test_reject_spr_above_threshold_but_monotonic(self) -> None:
        """Reject when SPR > threshold but tail is monotonic (no local max)."""
        aif = _make_clean_aif()
        peak_idx = int(np.argmax(aif))
        fwhm = compute_aif_fwhm_frames(aif, 5)
        recirc_start = peak_idx + max(fwhm, 5)
        if recirc_start < len(aif):
            peak_val = float(np.max(aif))
            # Monotonically increasing tail — no local max
            for i in range(recirc_start, len(aif)):
                aif[i] = peak_val * 0.3 * (i - recirc_start + 1) / (len(aif) - recirc_start)

        fit = _make_fit_result(aif)
        decision, metrics = should_apply_gamma_fit(aif, fit)
        assert not decision.apply or decision.defect_detected != "recirculation"

    def test_reject_spr_just_below_threshold(self) -> None:
        """Reject when SPR is just below threshold."""
        aif = _make_clean_aif()
        peak_idx = int(np.argmax(aif))
        peak_val = float(np.max(aif))
        fwhm = compute_aif_fwhm_frames(aif, 5)
        recirc_center = peak_idx + max(fwhm, 5) + 5
        if recirc_center + 3 < len(aif):
            for i in range(recirc_center - 2, min(recirc_center + 3, len(aif))):
                offset = i - recirc_center
                aif[i] += peak_val * 0.20 * np.exp(-0.5 * offset**2)

        fit = _make_fit_result(aif)
        decision, metrics = should_apply_gamma_fit(aif, fit)
        if metrics.get("spr", 0) < GAMMA_SPR_THRESHOLD:
            assert not decision.apply or decision.defect_detected != "recirculation"

    def test_accept_noise_and_negative_tail(self) -> None:
        """Accept when both noise and negative tail defects are present."""
        aif = _make_clean_aif()
        peak_idx = int(np.argmax(aif))
        peak_val = float(np.max(aif))

        # Add high-frequency noise
        rng = np.random.default_rng(42)
        noise = rng.normal(0, peak_val * 0.15, len(aif))
        noisy_aif = aif + noise
        noisy_aif[:5] = aif[:5]

        # Add negative tail
        tail_start = peak_idx + 10
        for i in range(tail_start, min(tail_start + 8, len(noisy_aif))):
            noisy_aif[i] = -peak_val * 0.08

        # Smooth fitted curve
        fit = _make_fit_result(noisy_aif, fitted_curve=aif.copy())
        decision, metrics = should_apply_gamma_fit(noisy_aif, fit, baseline_frames=5)

        if (
            metrics.get("noise_ratio", 0) > GAMMA_NOISE_RATIO_THRESHOLD
            and metrics.get("neg_tail_frames", 0) >= GAMMA_NEG_TAIL_MIN_FRAMES
        ):
            assert decision.apply
            assert decision.defect_detected == "noise_and_negative_tail"

    def test_reject_noise_alone(self) -> None:
        """Reject when only noise defect is present (no negative tail)."""
        aif = _make_clean_aif()
        peak_val = float(np.max(aif))

        rng = np.random.default_rng(42)
        noisy_aif = aif + rng.normal(0, peak_val * 0.15, len(aif))
        noisy_aif[:5] = aif[:5]
        noisy_aif = np.maximum(noisy_aif, 0)

        fit = _make_fit_result(noisy_aif, fitted_curve=aif.copy())
        decision, metrics = should_apply_gamma_fit(noisy_aif, fit, baseline_frames=5)
        if decision.defect_detected != "recirculation":
            assert not decision.apply or decision.defect_detected == "noise_and_negative_tail"

    def test_reject_negative_tail_alone(self) -> None:
        """Reject when only negative tail defect is present (no noise)."""
        aif = _make_clean_aif()
        peak_idx = int(np.argmax(aif))
        peak_val = float(np.max(aif))

        tail_start = peak_idx + 10
        for i in range(tail_start, min(tail_start + 8, len(aif))):
            aif[i] = -peak_val * 0.08

        fit = _make_fit_result(aif)
        decision, metrics = should_apply_gamma_fit(aif, fit, baseline_frames=5)
        if metrics.get("noise_ratio", 0) <= GAMMA_NOISE_RATIO_THRESHOLD:
            assert not decision.apply

    def test_reject_clean_aif(self) -> None:
        """Reject when AIF is clean — no defects detected."""
        aif = _make_clean_aif()
        fit = _make_fit_result(aif)
        decision, metrics = should_apply_gamma_fit(aif, fit)
        assert not decision.apply
        assert decision.defect_detected is None

    def test_metrics_populated(self) -> None:
        """Verify detail_metrics dict contains expected keys."""
        aif = _make_clean_aif()
        fit = _make_fit_result(aif)
        _, metrics = should_apply_gamma_fit(aif, fit)
        assert "spr" in metrics
        assert "spr_is_local_max" in metrics
        assert "noise_ratio" in metrics
        assert "neg_tail_frames" in metrics
        assert "neg_tail_min_fraction" in metrics


# =============================================================================
# TestDSCConfigGammaFitCoercion
# =============================================================================


class TestDSCConfigGammaFitRemoved:
    """Verify gamma_fit is not a DSCConfig field (hardcoded to auto)."""

    def test_no_gamma_fit_field(self) -> None:
        config = DSCConfig()
        assert not hasattr(config, "gamma_fit")
