"""Extended unit tests for DSC deconvolution methods.

Tests for oSVD, Tikhonov, block-circulant matrix, and batch methods.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from tests.dsc_accuracy import AccuracyTargetError, P2_NO_CORNER

from pmriprep.utils.dsc_deconvolution import (
    TikhonovSelection,
    _tikhonov_lcurve,
    _tikhonov_solve,
    build_block_circulant_matrix,
    build_toeplitz_matrix,
    build_volterra_matrix,
    compute_oscillation_index,
    svd_deconvolve_circulant,
    svd_deconvolve_circulant_batch,
    svd_deconvolve_oscillation,
    svd_deconvolve_oscillation_batch,
    svd_deconvolve_standard,
    svd_deconvolve_standard_batch,
    tikhonov_deconvolve,
    tikhonov_deconvolve_batch,
)


@pytest.mark.parametrize("scalar,batch", [
    (svd_deconvolve_standard, svd_deconvolve_standard_batch),
    (svd_deconvolve_oscillation, svd_deconvolve_oscillation_batch),
])
@pytest.mark.parametrize("fault", [
    "dt_zero", "dt_negative", "dt_nan", "dt_inf", "length", "empty", "one_sample",
    "aif_dimensions", "aif_zero", "aif_nan",
])
def test_svd_shared_input_validation(
    scalar: Callable[..., Any], batch: Callable[..., Any], fault: str,
) -> None:
    """Both entry points reject invalid sampling and selected curve data."""
    aif = np.array([0., 1., 2., 1.])
    tissue = aif.copy()
    dt = 1.0
    if fault.startswith("dt_"):
        dt = {"dt_zero": 0., "dt_negative": -1., "dt_nan": np.nan, "dt_inf": np.inf}[fault]
    elif fault == "length":
        tissue = tissue[:-1]
    elif fault == "empty":
        tissue = aif = np.array([])
    elif fault == "one_sample":
        tissue = aif = np.ones(1)
    elif fault == "aif_dimensions":
        aif = aif[:, None]
    elif fault == "aif_zero":
        aif[:] = 0
    elif fault == "aif_nan":
        aif[0] = np.nan
    with pytest.raises(ValueError):
        scalar(tissue, aif, dt)
    with pytest.raises(ValueError):
        batch(tissue[None], aif, dt)


@pytest.mark.parametrize("scalar,batch", [
    (svd_deconvolve_standard, svd_deconvolve_standard_batch),
    (svd_deconvolve_oscillation, svd_deconvolve_oscillation_batch),
])
def test_svd_shapes_and_masked_nonfinite_curves(
    scalar: Callable[..., Any], batch: Callable[..., Any],
) -> None:
    """A short or reshaped mask cannot silently select the wrong curves."""
    aif = np.array([0., 1., 2., 1.])
    curves = np.stack([aif, np.full_like(aif, np.nan)]).reshape(2, 1, 4)
    for mask in (np.ones(1, bool), np.ones(2, bool)):
        with pytest.raises(ValueError, match="mask shape"):
            batch(curves, aif, mask=mask)
    with pytest.raises(ValueError, match="spatial and time"):
        batch(aif, aif)
    with pytest.raises(ValueError, match="one-dimensional"):
        scalar(curves, aif)
    result = batch(curves, aif, mask=np.array([[True], [False]]))
    np.testing.assert_allclose(result[0][0, 0], scalar(aif, aif).residue)
    for values in result:
        np.testing.assert_array_equal(values[1], 0)


@pytest.mark.parametrize("scalar,batch", [
    (svd_deconvolve_standard, svd_deconvolve_standard_batch),
    (svd_deconvolve_oscillation, svd_deconvolve_oscillation_batch),
    (tikhonov_deconvolve, tikhonov_deconvolve_batch),
])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_tissue_is_confined_to_its_voxel(
    scalar: Callable[..., Any], batch: Callable[..., Any], bad: float,
) -> None:
    """Bad selected tissue is missing; finite neighbours and excluded zeros survive."""
    aif = np.array([0., 1., 2., 1.])
    curves = np.tile(aif, (3, 1, 1))
    curves[1:, 0, 2] = bad
    result = batch(curves, aif, mask=np.array([[True], [True], [False]]))
    np.testing.assert_allclose(result[0][0, 0], scalar(aif, aif).residue)
    numeric = result[:-1] if batch is tikhonov_deconvolve_batch else result
    for values in numeric:
        assert np.isnan(values[1]).all()
        np.testing.assert_array_equal(values[2], 0)
    if batch is tikhonov_deconvolve_batch:
        assert result[-1][1, 0] == TikhonovSelection.INVALID_INPUT
        assert result[-1][2, 0] == TikhonovSelection.INACTIVE
    with pytest.raises(ValueError, match="finite"):
        scalar(curves[1, 0], aif)


@pytest.mark.parametrize("threshold", [0., -0.1, 1., np.nan, np.inf])
def test_ssvd_shared_threshold_validation(threshold: float) -> None:
    aif = np.ones(4)
    for solver, curves in ((svd_deconvolve_standard, aif),
                           (svd_deconvolve_standard_batch, aif[None])):
        with pytest.raises(ValueError, match="threshold"):
            solver(curves, aif, threshold=threshold)


@pytest.mark.parametrize("settings", [
    {"oi_target": 0}, {"oi_target": np.nan}, {"oi_target": np.inf},
    {"threshold_min": 0}, {"threshold_min": 0.6}, {"threshold_max": 1},
    {"threshold_min": np.nan}, {"threshold_max": np.inf},
    {"max_iterations": 0}, {"max_iterations": -1},
    {"max_iterations": 2.5}, {"max_iterations": True},
])
def test_osvd_shared_search_validation(settings: dict[str, Any]) -> None:
    aif = np.ones(4)
    for solver, curves in ((svd_deconvolve_oscillation, aif),
                           (svd_deconvolve_oscillation_batch, aif[None])):
        with pytest.raises(ValueError):
            solver(curves, aif, **settings)


def _make_synthetic_data(
    n: int = 30,
    dt: float = 1.5,
    true_cbf: float = 2.0,
    true_mtt: float = 8.0,
    seed: int = 42,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Create synthetic AIF + tissue curve pair with known ground truth."""
    np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) * dt
    # Gamma-variate AIF
    aif = (t / 3.0) * np.exp(-t / 5.0)
    aif[0] = 0.0
    # Exponential residue
    true_residue = true_cbf * np.exp(-t / true_mtt)
    # Create tissue via the Volterra (product-integration) convolution, which
    # sSVD/Tikhonov use as the inverse operator (matched forward).
    matrix = build_volterra_matrix(aif, dt=dt)
    tissue = matrix @ true_residue
    return aif, tissue, t, true_residue


# =============================================================================
# Block-Circulant Matrix Tests
# =============================================================================


class TestBlockCirculantMatrix:
    """Tests for build_block_circulant_matrix."""

    def test_block_circulant_matrix_size(self) -> None:
        """AIF of length N produces a matrix of shape (2N, 2N)."""
        n = 10
        aif = np.arange(n, dtype=np.float64) * 0.5
        matrix = build_block_circulant_matrix(aif, dt=1.0)
        expected_size = 2 * n
        assert matrix.shape == (expected_size, expected_size)

    def test_block_circulant_matrix_circulant_structure(self) -> None:
        """Each row is a cyclic shift of the previous row."""
        n = 8
        aif = np.arange(1, n + 1, dtype=np.float64)
        dt = 1.0
        matrix = build_block_circulant_matrix(aif, dt=dt)
        m = 2 * n

        # For a circulant matrix C, C[i, j] = C[0, (j - i) mod m]
        # Check rows 0, 1, 2 against this pattern
        for i in range(3):
            for j in range(m):
                expected = matrix[0, (j - i) % m]
                assert matrix[i, j] == pytest.approx(expected, rel=1e-12), (
                    f"matrix[{i},{j}] = {matrix[i, j]}, "
                    f"expected {expected} (cyclic shift of row 0)"
                )


# =============================================================================
# oSVD Single Voxel Tests
# =============================================================================


class TestOSVDSingleVoxel:
    """Tests for svd_deconvolve_oscillation (oSVD)."""

    def test_osvd_converged(self) -> None:
        """Synthetic data produces a converged result with OI at or near target."""
        aif, tissue, _, _ = _make_synthetic_data()
        result = svd_deconvolve_oscillation(
            tissue, aif, dt=1.5, oi_target=0.065,
        )
        assert result.converged is True
        # OI should be at or below the target (with some tolerance for the
        # discrete threshold grid)
        assert result.oscillation_index <= result.oi_target + 0.01

    def test_osvd_stricter_oi_higher_threshold(self) -> None:
        """Stricter OI target requires equal or higher SVD threshold."""
        aif, tissue, _, _ = _make_synthetic_data()

        result_relaxed = svd_deconvolve_oscillation(
            tissue, aif, dt=1.5, oi_target=0.10,
        )
        result_strict = svd_deconvolve_oscillation(
            tissue, aif, dt=1.5, oi_target=0.03,
        )

        # A stricter OI target needs more regularization (higher threshold)
        assert result_strict.threshold_used >= result_relaxed.threshold_used

    def test_osvd_recovers_cbf(self) -> None:
        """oSVD recovers CBF within 50% of the known ground truth."""
        true_cbf = 2.0
        aif, tissue, _, _ = _make_synthetic_data(true_cbf=true_cbf)
        result = svd_deconvolve_oscillation(tissue, aif, dt=1.5)
        assert result.cbf == pytest.approx(true_cbf, rel=0.5)

    def test_osvd_validation(self) -> None:
        """Empty arrays and length mismatch raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            svd_deconvolve_oscillation(
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )
        with pytest.raises(ValueError, match="same length"):
            svd_deconvolve_oscillation(
                np.ones(10, dtype=np.float64),
                np.ones(15, dtype=np.float64),
            )


# =============================================================================
# Tikhonov Single Voxel Tests
# =============================================================================


class TestTikhonovSingleVoxel:
    """Tests for tikhonov_deconvolve."""

    @P2_NO_CORNER
    def test_tikhonov_recovers_cbf(self) -> None:
        """Tikhonov recovers CBF within 50% of the known ground truth."""
        true_cbf = 2.0
        aif, tissue, _, _ = _make_synthetic_data(true_cbf=true_cbf)
        result = tikhonov_deconvolve(tissue, aif, dt=1.5)
        assert result.selection_status == TikhonovSelection.UPPER_ENDPOINT
        assert np.isfinite(result.cbf) and result.cbf > 0
        if result.cbf != pytest.approx(true_cbf, rel=0.5):
            raise AccuracyTargetError(f"CBF {result.cbf} outside 50% of {true_cbf}")

    def test_tikhonov_lcurve_result_finite(self) -> None:
        """L-curve result fields are finite and consistent."""
        aif, tissue, _, _ = _make_synthetic_data()
        result = tikhonov_deconvolve(tissue, aif, dt=1.5)
        assert np.isfinite(result.residual_norm)
        assert np.isfinite(result.solution_norm)
        assert result.residual_norm >= 0
        assert result.solution_norm >= 0
        assert result.lambda_optimal > 0
        assert isinstance(result.lcurve_corner_found, bool)

    def test_tikhonov_validation(self) -> None:
        """Empty arrays, length mismatch, and invalid lambda raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            tikhonov_deconvolve(
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )
        with pytest.raises(ValueError, match="same length"):
            tikhonov_deconvolve(
                np.ones(10, dtype=np.float64),
                np.ones(15, dtype=np.float64),
            )
        with pytest.raises(ValueError, match="lambda_min"):
            tikhonov_deconvolve(
                np.ones(10, dtype=np.float64),
                np.ones(10, dtype=np.float64),
                lambda_min=1.0,
                lambda_max=0.5,
            )

    def test_tikhonov_higher_regularization_smoother(self) -> None:
        """Higher regularization (larger lambda) produces a smoother residue.

        Uses noisy data so the effect of regularization on oscillation is
        clearly visible.
        """
        n = 30
        dt = 1.5
        t = np.arange(n, dtype=np.float64) * dt
        aif = (t / 3.0) * np.exp(-t / 5.0)
        aif[0] = 0.0
        true_residue = 2.0 * np.exp(-t / 8.0)
        matrix = build_toeplitz_matrix(aif, dt=dt)
        rng = np.random.default_rng(99)
        tissue = matrix @ true_residue + rng.normal(0, 0.5, n)

        # Strong regularization: narrow range near lambda_max
        result_strong = tikhonov_deconvolve(
            tissue, aif, dt=dt, lambda_min=0.5, lambda_max=1.0,
        )
        # Weak regularization: narrow range near lambda_min
        result_weak = tikhonov_deconvolve(
            tissue, aif, dt=dt, lambda_min=1e-6, lambda_max=1e-4,
        )

        oi_strong = compute_oscillation_index(result_strong.residue)
        oi_weak = compute_oscillation_index(result_weak.residue)

        # With noisy data, higher regularization should give smoother residue
        assert oi_strong <= oi_weak

    @pytest.mark.parametrize("first_difference", [False, True])
    def test_full_residue_matches_augmented_least_squares(
        self, first_difference: bool,
    ) -> None:
        """All N lags, including the penalty-determined tail, enter the area."""
        n, dt = 50, 1.5
        aif = np.zeros(n)
        aif[10:20] = np.sin(np.linspace(0, np.pi, 12)[1:-1])
        matrix = build_volterra_matrix(aif, dt)
        supported = np.flatnonzero(np.any(matrix != 0, axis=0))[-1] + 1
        assert supported < n - 5
        tissue = matrix @ np.exp(-np.arange(n) * dt / 30)
        tissue += np.random.default_rng(51).normal(0, 0.001, n)
        if first_difference:
            penalty = np.diff(np.eye(n), axis=0)
            result = _tikhonov_solve(tissue, aif, dt, 0.001, 10, penalty)
        else:
            result = tikhonov_deconvolve(tissue, aif, dt)
            penalty = np.eye(n)
            penalty[np.arange(2, n), np.arange(1, n - 1)] = -1
        augmented = np.vstack((matrix, result.lambda_optimal * penalty))
        target = np.r_[tissue, np.zeros(penalty.shape[0])]
        reference = np.linalg.lstsq(augmented, target, rcond=None)[0]
        np.testing.assert_allclose(result.residue, reference, rtol=1e-10, atol=1e-12)
        assert np.all(result.residue[supported:] > 0)
        np.testing.assert_allclose(result.residue[supported:], result.residue[supported - 1])
        assert result.cbv == pytest.approx(np.trapezoid(reference, dx=dt), rel=1e-10)
        assert result.mtt == pytest.approx(result.cbv / np.max(reference))
        if not first_difference:
            batch = tikhonov_deconvolve_batch(tissue[None, :], aif, dt)
            np.testing.assert_allclose(batch[0][0], result.residue, rtol=1e-10, atol=1e-12)
            assert batch[2][0] == pytest.approx(result.cbv, rel=1e-10)

    def test_lcurve_curvature_against_norm_differences(self) -> None:
        """Analytic curvature and its derivative agree with independent differences."""
        singular = np.array([3.0, 0.5, 0.02])
        coefficients = np.array([[1.0], [0.4], [0.02]])
        perpendicular = np.array([0.001])
        log_lambdas = np.log(np.array([0.01, 0.1, 0.5, 2.0]))
        h = 1e-3

        def log_norms(offset: float) -> tuple[np.ndarray, np.ndarray]:
            lam = np.exp(log_lambdas + offset)[:, None]
            solution = singular / (singular ** 2 + lam ** 2) * coefficients.T
            residual = singular * solution - coefficients.T
            return (
                0.5 * np.log(np.sum(residual ** 2, axis=1) + perpendicular),
                0.5 * np.log(np.sum(solution ** 2, axis=1)),
            )

        x, y = log_norms(0)
        xp, yp = log_norms(h)
        xm, ym = log_norms(-h)
        dx, dy = (xp - xm) / (2 * h), (yp - ym) / (2 * h)
        ddx, ddy = (xp - 2 * x + xm) / h ** 2, (yp - 2 * y + ym) / h ** 2
        expected = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5
        _, _, curvature, derivative = _tikhonov_lcurve(
            singular, coefficients ** 2, perpendicular, np.exp(log_lambdas),
        )
        np.testing.assert_allclose(curvature[:, 0], expected, rtol=2e-5, atol=1e-7)
        kp = _tikhonov_lcurve(singular, coefficients ** 2, perpendicular, np.exp(log_lambdas + h))[2]
        km = _tikhonov_lcurve(singular, coefficients ** 2, perpendicular, np.exp(log_lambdas - h))[2]
        kp2 = _tikhonov_lcurve(singular, coefficients ** 2, perpendicular, np.exp(log_lambdas + 2 * h))[2]
        km2 = _tikhonov_lcurve(singular, coefficients ** 2, perpendicular, np.exp(log_lambdas - 2 * h))[2]
        np.testing.assert_allclose(derivative, (km2 - 8 * km + 8 * kp - kp2) / (12 * h), rtol=2e-5, atol=1e-7)

        # A single spectral mode has negative curvature, not a positive corner.
        curvature = _tikhonov_lcurve(np.ones(1), np.ones((1, 1)), np.zeros(1), np.ones(1))[2]
        assert curvature[0, 0] == pytest.approx(-1 / np.sqrt(2))

    @pytest.mark.parametrize("kwargs", [
        {"lambda_min": 1.0, "lambda_max": 0.5},
        {"lambda_min": 0.0}, {"lambda_max": np.inf},
        {"dt": 0.0}, {"dt": np.nan},
    ])
    def test_scalar_and_batch_reject_invalid_settings(self, kwargs: dict) -> None:
        """Both entry points reject invalid sampling and regularization bounds."""
        aif, tissue, _, _ = _make_synthetic_data()
        with pytest.raises(ValueError):
            tikhonov_deconvolve(tissue, aif, **kwargs)
        with pytest.raises(ValueError):
            tikhonov_deconvolve_batch(tissue[None, :], aif, **kwargs)

    def test_masked_nonfinite_curves_are_excluded(self) -> None:
        """Excluded NaNs stay zero; selected NaNs are missing without contaminating fits."""
        aif, tissue, _, _ = _make_synthetic_data()
        curves = np.stack((tissue, np.full_like(tissue, np.nan))).reshape(1, 2, -1)
        result = tikhonov_deconvolve_batch(curves, aif, mask=np.array([[True, False]]))
        for output in result:
            assert np.all(np.isfinite(output))
            assert np.all(output[0, 1] == 0)
        selected = tikhonov_deconvolve_batch(curves, aif)
        for output in selected[:-1]:
            assert np.isnan(output[0, 1]).all()
        np.testing.assert_allclose(selected[0][0, 0], result[0][0, 0])
        assert selected[-1][0, 1] == TikhonovSelection.INVALID_INPUT
        with pytest.raises(ValueError, match="finite"):
            tikhonov_deconvolve(curves[0, 1], aif)


# =============================================================================
# Batch Methods Tests
# =============================================================================


class TestBatchMethods:
    """Tests for batch deconvolution methods matching single-voxel results."""

    def _make_batch_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Create 3 different tissue curves with a common AIF."""
        n = 30
        dt = 1.5
        t = np.arange(n, dtype=np.float64) * dt
        aif = (t / 3.0) * np.exp(-t / 5.0)
        aif[0] = 0.0
        tissues = np.zeros((3, n), dtype=np.float64)
        matrix = build_toeplitz_matrix(aif, dt=dt)
        for i in range(3):
            residue = (1.0 + i * 0.5) * np.exp(-t / (5.0 + i * 2.0))
            tissues[i, :] = matrix @ residue
        return tissues, aif, dt

    def test_circulant_batch_matches_single(self) -> None:
        """Scalar and batch cSVD retain signed least-squares residues and readouts."""
        tissues, aif, dt = self._make_batch_data()
        threshold = 0.2
        matrix = build_block_circulant_matrix(aif, dt)
        padded = np.pad(tissues, ((0, 0), (0, len(aif))))
        reference = np.linalg.lstsq(matrix, padded.T, rcond=threshold)[0].T
        assert np.all(np.min(reference, axis=1) < -1e-3)

        single_cbf = np.zeros(3)
        single_cbv = np.zeros(3)
        single_mtt = np.zeros(3)
        for i in range(3):
            result = svd_deconvolve_circulant(
                tissues[i], aif, dt=dt, threshold=threshold,
            )
            np.testing.assert_allclose(result.residue, reference[i], atol=1e-12)
            peak = int(np.argmax(reference[i]))
            window = reference[i, (peak + np.arange(len(aif))) % (2 * len(aif))]
            assert result.cbv == pytest.approx(np.trapezoid(window, dx=dt))
            single_cbf[i] = result.cbf
            single_cbv[i] = result.cbv
            single_mtt[i] = result.mtt

        residues, batch_cbf, batch_cbv, batch_mtt, _ = svd_deconvolve_circulant_batch(
            tissues, aif, dt=dt, threshold=threshold,
        )

        np.testing.assert_allclose(residues, reference, atol=1e-12)
        np.testing.assert_allclose(batch_cbf, single_cbf, rtol=1e-8)
        np.testing.assert_allclose(batch_cbv, single_cbv, rtol=1e-8)
        np.testing.assert_allclose(batch_mtt, single_mtt, rtol=1e-8)

    def test_oscillation_batch_matches_single(self) -> None:
        """Batch oSVD produces the same CBF/CBV/MTT as looped single calls."""
        tissues, aif, dt = self._make_batch_data()

        single_cbf = np.zeros(3)
        single_cbv = np.zeros(3)
        single_mtt = np.zeros(3)
        for i in range(3):
            result = svd_deconvolve_oscillation(tissues[i], aif, dt=dt)
            single_cbf[i] = result.cbf
            single_cbv[i] = result.cbv
            single_mtt[i] = result.mtt

        _, batch_cbf, batch_cbv, batch_mtt, _, _, _ = (
            svd_deconvolve_oscillation_batch(tissues, aif, dt=dt)
        )

        np.testing.assert_allclose(batch_cbf, single_cbf, rtol=1e-8)
        np.testing.assert_allclose(batch_cbv, single_cbv, rtol=1e-8)
        np.testing.assert_allclose(batch_mtt, single_mtt, rtol=1e-8)

    def test_tikhonov_batch_matches_single(self) -> None:
        """Batch Tikhonov produces the same CBF/CBV/MTT as looped single calls."""
        tissues, aif, dt = self._make_batch_data()

        single_cbf = np.zeros(3)
        single_cbv = np.zeros(3)
        single_mtt = np.zeros(3)
        for i in range(3):
            result = tikhonov_deconvolve(tissues[i], aif, dt=dt)
            single_cbf[i] = result.cbf
            single_cbv[i] = result.cbv
            single_mtt[i] = result.mtt

        _, batch_cbf, batch_cbv, batch_mtt, _, _ = tikhonov_deconvolve_batch(
            tissues, aif, dt=dt,
        )

        np.testing.assert_allclose(batch_cbf, single_cbf, rtol=1e-8)
        np.testing.assert_allclose(batch_cbv, single_cbv, rtol=1e-8)
        np.testing.assert_allclose(batch_mtt, single_mtt, rtol=1e-8)

    def test_tikhonov_batch_matches_single_noisy(self) -> None:
        """Batch and single agree on noisy data and L-curve corner detection."""
        tissues, aif, dt = self._make_batch_data()
        rng = np.random.default_rng(7)
        noisy = tissues + rng.normal(
            0.0, 0.02 * float(np.max(np.abs(tissues))), size=tissues.shape
        )

        single_cbf = np.zeros(3)
        single_res = np.zeros_like(noisy)
        single_status = np.zeros(3, dtype=np.int8)
        for i in range(3):
            r = tikhonov_deconvolve(noisy[i], aif, dt=dt)
            single_cbf[i] = r.cbf
            single_res[i] = r.residue
            single_status[i] = r.selection_status

        batch_res, batch_cbf, _, _, _, batch_corner = (
            tikhonov_deconvolve_batch(noisy, aif, dt=dt)
        )

        np.testing.assert_array_equal(batch_corner, single_status)
        np.testing.assert_allclose(batch_res, single_res, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(batch_cbf, single_cbf, rtol=1e-8)

    def test_oscillation_batch_with_mask(self) -> None:
        """Masked voxels have zero CBF; unmasked voxels match single-voxel."""
        n = 30
        dt = 1.5
        t = np.arange(n, dtype=np.float64) * dt
        aif = (t / 3.0) * np.exp(-t / 5.0)
        aif[0] = 0.0

        tissues = np.zeros((5, n), dtype=np.float64)
        matrix = build_toeplitz_matrix(aif, dt=dt)
        for i in range(5):
            residue = (1.0 + i * 0.3) * np.exp(-t / (6.0 + i))
            tissues[i, :] = matrix @ residue

        mask = np.array([True, True, False, True, False])

        _, batch_cbf, _, _, _, _, _ = svd_deconvolve_oscillation_batch(
            tissues, aif, dt=dt, mask=mask,
        )

        # Masked-out voxels should have zero CBF
        assert batch_cbf[2] == 0.0
        assert batch_cbf[4] == 0.0

        # Unmasked voxels should match single-voxel results
        for i in [0, 1, 3]:
            single = svd_deconvolve_oscillation(tissues[i], aif, dt=dt)
            assert batch_cbf[i] == pytest.approx(single.cbf, rel=1e-8)


class TestOIComputationOrder:
    """OI uses the full signed residue."""

    def test_oi_from_raw_residue(self) -> None:
        """Returned OI should reflect raw negative oscillations, not clipped."""
        n = 40
        dt = 1.5
        t = np.arange(n, dtype=np.float64) * dt

        aif = (t / 3.0) * np.exp(-t / 5.0)
        aif[0] = 0.0

        true_residue = 2.0 * np.exp(-t / 8.0)
        matrix = build_block_circulant_matrix(aif, dt=dt)
        m = 2 * n
        residue_padded = np.zeros(m, dtype=np.float64)
        residue_padded[:n] = true_residue
        tissue = (matrix @ residue_padded)[:n]

        rng = np.random.default_rng(99)
        tissue_noisy = tissue + rng.normal(0, 0.15 * np.max(tissue), size=n)

        result = svd_deconvolve_oscillation(
            tissue_noisy, aif, dt=dt,
        )

        assert np.any(result.residue < 0.0)
        expected = np.sum(np.abs(np.diff(result.residue, n=2))) / (
            2 * n * np.max(result.residue)
        )
        assert result.oscillation_index == pytest.approx(expected)
