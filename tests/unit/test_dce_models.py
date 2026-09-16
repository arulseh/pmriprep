"""Unit tests for DCE pharmacokinetic models (Tofts and Extended Tofts).

Tests cover analytical solutions for known AIFs, parameter recovery via
fitting, edge cases, and consistency between model variants.
"""

import numpy as np
import pytest

from pmriprep.models.dce.extended_tofts import ExtendedToftsModel
from pmriprep.models.dce.tofts import ToftsModel
from pmriprep.utils.dce_constants import EPSILON_VE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tofts_model() -> ToftsModel:
    return ToftsModel()


@pytest.fixture
def etofts_model() -> ExtendedToftsModel:
    return ExtendedToftsModel()


def _parker_like_aif(time: np.ndarray) -> np.ndarray:
    """Generate a Parker-like double-Gaussian AIF.

    Simplified version: two Gaussian peaks representing first-pass and
    recirculation, suitable for testing model consistency.
    """
    # First pass peak at t=0.5 min, sigma=0.15 min
    g1 = 5.0 * np.exp(-((time - 0.5) ** 2) / (2 * 0.15**2))
    # Recirculation peak at t=1.5 min, sigma=0.4 min
    g2 = 1.5 * np.exp(-((time - 1.5) ** 2) / (2 * 0.4**2))
    return g1 + g2


# ---------------------------------------------------------------------------
# Test 1: Step AIF analytical solution
# ---------------------------------------------------------------------------

class TestToftsStepAIF:
    """Verify Tofts model against closed-form step AIF solution.

    For a step AIF: Cp(t) = C0 * u(t), the analytical tissue curve is:
        Ct(t) = Ktrans * C0 / kep * (1 - exp(-kep * t))
    """

    def test_tofts_step_aif_analytical(self, tofts_model: ToftsModel) -> None:
        ktrans = 0.1
        ve = 0.2
        kep = ktrans / ve  # 0.5

        c0 = 1.0
        dt = 0.05  # minutes
        n_points = 200
        time = np.arange(n_points) * dt

        # Step AIF: zero for first few frames then constant C0
        aif = np.zeros(n_points)
        step_start = 5
        aif[step_start:] = c0

        ct_model = tofts_model.evaluate([ktrans, ve], aif, time)

        # The implementation uses np.convolve (left Riemann sum / rectangular
        # rule). For a step AIF starting at frame k, the discrete convolution
        # at frame i (i >= k) sums aif[k..i] * exp(-kep*(i-j)*dt) * dt * Ktrans.
        # This is equivalent to the analytical integral offset by +dt from a
        # continuous perspective: the first sample at the step onset contributes
        # a full dt, whereas the true integral from 0 to dt would be dt/2.
        # So the discrete model leads the analytical by ~dt * kep / 2.
        #
        # We verify the model output matches the analytical shape by comparing
        # the steady-state value (which is exact) and the overall shape.
        steady_state = ktrans * c0 / kep  # = ve * c0 = 0.2

        # Late-time values should converge to steady state
        np.testing.assert_allclose(
            ct_model[-10:],
            steady_state,
            rtol=0.01,
            err_msg="Tofts step-AIF steady state mismatch",
        )

        # The curve should be monotonically increasing after step onset
        ct_post_step = ct_model[step_start:]
        diffs = np.diff(ct_post_step)
        assert np.all(diffs >= -1e-12), "Tissue curve should be monotonically increasing"

        # Verify the initial slope matches the analytical expectation:
        # dCt/dt(0+) = Ktrans * C0 for the step AIF
        # Discrete approximation: (ct[step_start+1] - ct[step_start]) / dt
        initial_slope = (ct_model[step_start + 1] - ct_model[step_start]) / dt
        np.testing.assert_allclose(
            initial_slope,
            ktrans * c0,
            rtol=0.15,
            err_msg="Initial slope should approximate Ktrans * C0",
        )


# ---------------------------------------------------------------------------
# Test 2: Exponential AIF analytical solution
# ---------------------------------------------------------------------------

class TestToftsExponentialAIF:
    """Verify Tofts model against closed-form exponential AIF solution.

    For Cp(t) = A * exp(-k_aif * t), the analytical tissue curve is:
        Ct(t) = A * Ktrans / (kep - k_aif) * [exp(-k_aif*t) - exp(-kep*t)]
    (valid when kep != k_aif).
    """

    def test_tofts_exponential_aif_analytical(self, tofts_model: ToftsModel) -> None:
        ktrans = 0.1
        ve = 0.2
        kep = ktrans / ve  # 0.5
        a_aif = 1.0
        k_aif = 0.3

        dt = 0.01  # fine time steps for tight tolerance
        n_points = 1000
        time = np.arange(n_points) * dt

        aif = a_aif * np.exp(-k_aif * time)
        ct_model = tofts_model.evaluate([ktrans, ve], aif, time)

        # Analytical: Ct = A * Ktrans / (kep - k_aif) * [exp(-k_aif*t) - exp(-kep*t)]
        ct_analytical = (
            a_aif * ktrans / (kep - k_aif) * (np.exp(-k_aif * time) - np.exp(-kep * time))
        )

        # The rectangular rule (left Riemann sum) has O(dt) bias at early
        # times but converges for later times. Skip the early transient
        # (first ~50 samples) and use a 5% tolerance to accommodate the
        # discrete approximation error.
        np.testing.assert_allclose(
            ct_model[50:],
            ct_analytical[50:],
            rtol=0.05,
            err_msg="Tofts exponential-AIF mismatch exceeds 5% tolerance",
        )


# ---------------------------------------------------------------------------
# Test 3: Extended Tofts reduces to Standard Tofts when vp=0
# ---------------------------------------------------------------------------

class TestExtendedToftsReducesToTofts:
    """Extended Tofts with vp=0 must match Standard Tofts exactly."""

    def test_extended_tofts_reduces_to_tofts(
        self,
        tofts_model: ToftsModel,
        etofts_model: ExtendedToftsModel,
    ) -> None:
        ktrans = 0.15
        ve = 0.25

        dt = 0.05
        n_points = 100
        time = np.arange(n_points) * dt
        aif = _parker_like_aif(time)

        ct_tofts = tofts_model.evaluate([ktrans, ve], aif, time)
        ct_etofts = etofts_model.evaluate([ktrans, ve, 0.0], aif, time)

        np.testing.assert_allclose(
            ct_etofts,
            ct_tofts,
            atol=1e-10,
            err_msg="Extended Tofts with vp=0 does not match Standard Tofts",
        )


# ---------------------------------------------------------------------------
# Test 4: Zero AIF returns all-zero tissue curve
# ---------------------------------------------------------------------------

class TestZeroAIF:
    """All-zeros AIF should produce all-zeros tissue curve, no NaN/Inf."""

    def test_zero_aif_tofts(self, tofts_model: ToftsModel) -> None:
        time = np.arange(50) * 0.1
        aif = np.zeros(50)
        ct = tofts_model.evaluate([0.1, 0.2], aif, time)
        assert np.all(ct == 0.0), "Tofts should return zeros for zero AIF"
        assert np.all(np.isfinite(ct)), "Output must be finite"

    def test_zero_aif_etofts(self, etofts_model: ExtendedToftsModel) -> None:
        time = np.arange(50) * 0.1
        aif = np.zeros(50)
        ct = etofts_model.evaluate([0.1, 0.2, 0.05], aif, time)
        assert np.all(ct == 0.0), "Extended Tofts should return zeros for zero AIF"
        assert np.all(np.isfinite(ct)), "Output must be finite"


# ---------------------------------------------------------------------------
# Test 5: Extreme kep (very small ve)
# ---------------------------------------------------------------------------

class TestExtremeKep:
    """Very small ve produces large efflux rates with finite forward curves."""

    def test_extreme_kep(self, tofts_model: ToftsModel) -> None:
        ktrans = 0.5
        ve = 0.0001  # kep = 5000 min^-1

        time = np.arange(50) * 0.1
        aif = _parker_like_aif(time)
        ct = tofts_model.evaluate([ktrans, ve], aif, time)

        assert np.all(np.isfinite(ct)), "Output must be finite with extreme kep"
        assert ct.shape == time.shape, "Output shape must match time shape"

    def test_extreme_kep_etofts(self, etofts_model: ExtendedToftsModel) -> None:
        ktrans = 0.5
        ve = 0.0001

        time = np.arange(50) * 0.1
        aif = _parker_like_aif(time)
        ct = etofts_model.evaluate([ktrans, ve, 0.02], aif, time)

        assert np.all(np.isfinite(ct)), "Output must be finite with extreme kep"


# ---------------------------------------------------------------------------
# Test 6: Flat tissue curve fitting
# ---------------------------------------------------------------------------

class TestFlatTissueCurveFitting:
    """Fitting a flat (constant) tissue curve should yield near-zero Ktrans."""

    def test_flat_tissue_curve_fitting(self, tofts_model: ToftsModel) -> None:
        time = np.arange(60) * 0.1
        aif = _parker_like_aif(time)
        # Flat tissue curve (no enhancement)
        tissue_curve = np.ones(60) * 0.001

        result = tofts_model.fit(tissue_curve, aif, time)

        assert result.parameters["Ktrans"] < 0.01, (
            f"Ktrans should be near zero for flat curve, got {result.parameters['Ktrans']}"
        )
        assert isinstance(result.converged, bool)


# ---------------------------------------------------------------------------
# Test 7: Single time point
# ---------------------------------------------------------------------------

class TestSingleTimePoint:
    """evaluate() with length-1 arrays should return zeros gracefully."""

    def test_single_time_point_tofts(self, tofts_model: ToftsModel) -> None:
        time = np.array([0.0])
        aif = np.array([1.0])
        ct = tofts_model.evaluate([0.1, 0.2], aif, time)
        assert ct.shape == (1,), f"Expected shape (1,), got {ct.shape}"
        assert ct[0] == 0.0, "Single-point output should be zero"

    def test_single_time_point_etofts(self, etofts_model: ExtendedToftsModel) -> None:
        time = np.array([0.0])
        aif = np.array([1.0])
        ct = etofts_model.evaluate([0.1, 0.2, 0.05], aif, time)
        assert ct.shape == (1,), f"Expected shape (1,), got {ct.shape}"
        assert ct[0] == 0.0, "Single-point output should be zero"


# ---------------------------------------------------------------------------
# Test 8: Volume constraint (ve + vp <= 1.0)
# ---------------------------------------------------------------------------

class TestVolumeConstraint:
    """_apply_volume_constraint must scale ve+vp to <= 1.0."""

    def test_volume_constraint_scales_when_exceeded(
        self, etofts_model: ExtendedToftsModel
    ) -> None:
        # ve=0.95, vp=0.15 -> total=1.10, must be scaled down
        ktrans, ve, vp = etofts_model._apply_volume_constraint([0.1, 0.95, 0.15])

        assert ktrans == 0.1, "Ktrans should be unchanged"
        assert ve + vp <= 1.0 + 1e-12, f"ve+vp={ve + vp} exceeds 1.0"

        # Ratio must be preserved
        expected_ratio = 0.95 / 0.15
        actual_ratio = ve / vp
        np.testing.assert_allclose(
            actual_ratio,
            expected_ratio,
            rtol=1e-10,
            err_msg="ve/vp ratio not preserved after scaling",
        )

    def test_volume_constraint_unchanged_when_valid(
        self, etofts_model: ExtendedToftsModel
    ) -> None:
        # ve=0.3, vp=0.05 -> total=0.35, should be unchanged
        ktrans, ve, vp = etofts_model._apply_volume_constraint([0.1, 0.3, 0.05])

        assert ktrans == 0.1
        assert ve == 0.3
        assert vp == 0.05

    def test_volume_constraint_boundary(
        self, etofts_model: ExtendedToftsModel
    ) -> None:
        # Unit occupancy preserves both compartment volumes.
        from pmriprep.utils.dce_constants import VP_VE_TOTAL_MAX

        ktrans, ve, vp = etofts_model._apply_volume_constraint([0.1, 0.7, 0.3])

        assert ktrans == 0.1
        np.testing.assert_allclose(ve + vp, VP_VE_TOTAL_MAX, rtol=1e-12)
        np.testing.assert_allclose(ve / vp, 0.7 / 0.3, rtol=1e-12)


# ---------------------------------------------------------------------------
# Test 9: Tofts fit recovers known parameters
# ---------------------------------------------------------------------------

class TestToftsFitRecoversParameters:
    """Generate synthetic data with known params, verify fit recovery."""

    def test_tofts_fit_recovers_parameters(self, tofts_model: ToftsModel) -> None:
        true_ktrans = 0.1
        true_ve = 0.2

        dt = 0.05
        n_points = 120
        time = np.arange(n_points) * dt
        aif = _parker_like_aif(time)

        # Generate clean synthetic data
        ct_synthetic = tofts_model.evaluate([true_ktrans, true_ve], aif, time)

        # Add very small noise to avoid perfect-fit degeneracy
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.001, n_points)
        ct_noisy = ct_synthetic + noise

        result = tofts_model.fit(ct_noisy, aif, time)

        assert result.converged, "Fit should converge on clean synthetic data"
        np.testing.assert_allclose(
            result.parameters["Ktrans"],
            true_ktrans,
            rtol=0.1,
            err_msg="Ktrans not recovered within 10%",
        )
        np.testing.assert_allclose(
            result.parameters["ve"],
            true_ve,
            rtol=0.1,
            err_msg="ve not recovered within 10%",
        )


# ---------------------------------------------------------------------------
# Test 10: Derived parameters (kep)
# ---------------------------------------------------------------------------

class TestComputeDerivedParameters:
    """Kep equals Ktrans/ve only within its identifiable reporting range."""

    def test_kep_normal(self, tofts_model: ToftsModel) -> None:
        derived = tofts_model.compute_derived_parameters([0.1, 0.2])
        expected_kep = 0.1 / 0.2  # 0.5
        np.testing.assert_allclose(derived["kep"], expected_kep, rtol=1e-10)

    def test_kep_zero_ve(self, tofts_model: ToftsModel) -> None:
        derived = tofts_model.compute_derived_parameters([0.1, 0.0])
        assert np.isnan(derived["kep"])

    def test_kep_epsilon_ve(self, tofts_model: ToftsModel) -> None:
        derived = tofts_model.compute_derived_parameters([0.1, EPSILON_VE])
        assert derived["kep"] == pytest.approx(0.1 / EPSILON_VE)

    def test_kep_just_above_epsilon(self, tofts_model: ToftsModel) -> None:
        ve = 0.02
        derived = tofts_model.compute_derived_parameters([0.1, ve])
        assert derived["kep"] > 0.0, "kep should be positive when ve > EPSILON_VE"

    def test_high_finite_kep_is_reported(self, tofts_model: ToftsModel) -> None:
        ktrans = 5.0
        ve = EPSILON_VE + 1e-4
        derived = tofts_model.compute_derived_parameters([ktrans, ve])
        assert derived["kep"] == pytest.approx(ktrans / ve)

    def test_kep_etofts(self, etofts_model: ExtendedToftsModel) -> None:
        derived = etofts_model.compute_derived_parameters([0.15, 0.3, 0.05])
        expected_kep = 0.15 / 0.3  # 0.5
        np.testing.assert_allclose(derived["kep"], expected_kep, rtol=1e-10)
