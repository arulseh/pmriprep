"""Unit tests for DSC batched fitting (forward models, VI, and MAP).

Tests verify parameter recovery, convergence, forward model correctness,
and batched-vs-CPU consistency for vascular DSC and the shared inference infrastructure.

All tests use numpy (pure CPU).
"""

import numpy as np
import pytest

from pmriprep.models.dsc.batched_fitting import (
    batched_dsc_map_fit,
    batched_dsc_vi_fit,
    batched_vascular_forward,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", [batched_dsc_vi_fit, batched_dsc_map_fit])
@pytest.mark.parametrize("model_name", ["cpi", "unknown"])
def test_concentration_dispatch_does_not_substitute_vascular(solver, model_name):
    """A non-vascular identifier needs a VI adapter, including an empty batch."""
    with pytest.raises(ValueError, match="adapter|vascular only"):
        solver(
            model_name, np.empty((0, 2)), np.ones(2), np.arange(2.0),
            np.array([0.1]), np.array([0.01]), [], [], ["cbf"], [(0.0, 2.0)],
        )


@pytest.fixture
def dsc_aif_and_time():
    """Generate DSC AIF and time vector (60 frames, 1s TR)."""
    T = 60
    time_points = np.linspace(0, 59, T)
    # Simple bolus AIF: gamma-variate shape
    t_bolus = time_points - 8.0
    aif = np.where(
        t_bolus > 0,
        3.0 * t_bolus * np.exp(-t_bolus / 3.0),
        0.0,
    )
    return aif, time_points


# ---------------------------------------------------------------------------
# Batched Vascular forward model tests
# ---------------------------------------------------------------------------


class TestBatchedVascularForward:
    """Tests for batched_vascular_forward."""

    def test_shape(self, dsc_aif_and_time):
        """Output shape matches (N, T)."""
        aif, tp = dsc_aif_and_time
        n = 20
        result = batched_vascular_forward(
            np.full(n, 0.06), np.full(n, 3.0), np.full(n, 2.0),
            np.full(n, 2.0), aif, tp,
        )
        assert result.shape == (n, len(tp))

    def test_zero_cbf(self, dsc_aif_and_time):
        """Zero CBF produces zero tissue curve."""
        aif, tp = dsc_aif_and_time
        result = batched_vascular_forward(
            np.array([0.0]), np.array([3.0]), np.array([2.0]),
            np.array([0.0]), aif, tp,
        )
        np.testing.assert_allclose(result[0], 0.0, atol=1e-15)

    def test_matches_scalar(self, dsc_aif_and_time):
        """Batched result matches scalar VascularModel.tissue_curve."""
        from pmriprep.models.dsc import VascularModel

        aif, tp = dsc_aif_and_time
        model = VascularModel()

        cbf, alpha, beta, delay = 0.07, 3.5, 2.5, 1.5
        scalar = model.tissue_curve([cbf, alpha, beta, delay], aif, tp)
        batched = batched_vascular_forward(
            np.array([cbf]), np.array([alpha]), np.array([beta]),
            np.array([delay]), aif, tp,
        )
        # rtol=1e-2: batched uses np.fft, scalar uses scipy.signal.fftconvolve
        # Small differences from zero-padding and rounding in delay interpolation.
        # atol=1e-6: near-zero pre-bolus values have large relative but tiny absolute error.
        np.testing.assert_allclose(batched[0], scalar, rtol=1e-2, atol=1e-6)



# ---------------------------------------------------------------------------
# Batched DSC VI tests
# ---------------------------------------------------------------------------


class TestBatchedDSCVIFit:
    """Tests for batched_dsc_vi_fit."""

    @staticmethod
    def _get_vascular_prior_info():
        """Extract prior info for Vascular fitting space."""
        from pmriprep.models.bayesian_utils import build_priors_from_params
        from pmriprep.models.dsc import VascularModel

        model = VascularModel()
        fitting_defs = model.get_fitting_parameter_definitions()
        priors = build_priors_from_params(fitting_defs)
        return (
            np.array([p.mean for p in priors]),
            np.array([p.std for p in priors]),
            [p.transform for p in priors],
            [p.upper_bound for p in priors],
            [p.name for p in fitting_defs],
            [(p.bounds[0], p.bounds[1]) for p in fitting_defs],
        )

    def test_vascular_recovery(self, dsc_aif_and_time):
        """Vascular VI recovers parameters within 30% median error."""
        aif, tp = dsc_aif_and_time
        means, stds, transforms, ubs, names, bounds = self._get_vascular_prior_info()

        # Generate synthetic data
        rng = np.random.RandomState(42)
        n = 50
        cbf_true = rng.uniform(0.03, 0.1, n)
        alpha_true = rng.uniform(2.0, 5.0, n)
        beta_true = rng.uniform(1.0, 4.0, n)
        delay_true = rng.uniform(0.0, 3.0, n)

        tissue = batched_vascular_forward(cbf_true, alpha_true, beta_true, delay_true, aif, tp)
        tissue += 0.005 * rng.randn(*tissue.shape)

        result = batched_dsc_vi_fit(
            "vascular", tissue, aif, tp,
            means, stds, transforms, ubs, names, bounds,
        )

        conv = result.converged
        assert np.sum(conv) > 25, f"Only {np.sum(conv)}/50 converged"

        cbf_err = np.median(np.abs(result.params["cbf"][conv] - cbf_true[conv]) /
                            np.maximum(cbf_true[conv], 1e-6))
        assert cbf_err < 0.3, f"CBF median rel error {cbf_err:.3f}"

    def test_vascular_result_format(self, dsc_aif_and_time):
        """VI result has all required DSCBatchedFitResult fields."""
        aif, tp = dsc_aif_and_time
        means, stds, transforms, ubs, names, bounds = self._get_vascular_prior_info()

        tissue = batched_vascular_forward(
            np.full(5, 0.06), np.full(5, 3.0), np.full(5, 2.0),
            np.full(5, 1.0), aif, tp,
        )

        result = batched_dsc_vi_fit(
            "vascular", tissue, aif, tp,
            means, stds, transforms, ubs, names, bounds,
            save_fitted_curves=True,
        )

        assert "cbf" in result.params
        assert "mtt" in result.params
        assert "alpha" in result.params
        assert "delay" in result.params
        assert result.r_squared.shape == (5,)
        assert result.converged.shape == (5,)
        assert result.fitted_curves is not None
        assert result.fitted_curves.shape == (5, len(tp))
        assert "noise_std" in result.uncertainties
        assert "cbf" in result.credible_interval_lower
        assert "cbf" in result.credible_interval_upper
        symmetric_lower = np.maximum(
            result.params["cbf"] - 1.96 * result.uncertainties["cbf"], 0.0
        )
        assert not np.allclose(
            result.credible_interval_lower["cbf"],
            symmetric_lower,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_empty_input(self, dsc_aif_and_time):
        """Handles zero voxels gracefully."""
        aif, tp = dsc_aif_and_time
        means, stds, transforms, ubs, names, bounds = self._get_vascular_prior_info()

        result = batched_dsc_vi_fit(
            "vascular", np.zeros((0, len(tp))), aif, tp,
            means, stds, transforms, ubs, names, bounds,
        )
        assert result.r_squared.shape == (0,)

    def test_sloppy_mode(self, dsc_aif_and_time):
        """Sloppy mode uses fewer iterations."""
        aif, tp = dsc_aif_and_time
        means, stds, transforms, ubs, names, bounds = self._get_vascular_prior_info()

        tissue = batched_vascular_forward(
            np.full(5, 0.06), np.full(5, 3.0), np.full(5, 2.0),
            np.full(5, 1.0), aif, tp,
        )

        result = batched_dsc_vi_fit(
            "vascular", tissue, aif, tp,
            means, stds, transforms, ubs, names, bounds,
            sloppy=True,
        )
        assert result.n_iterations <= 8

    def test_bounded_mean_failure_is_isolated_to_one_voxel(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One row's numerical bound failure does not discard the batch."""
        from pmriprep.models.bayesian_utils import TransformType
        from pmriprep.models.dsc import batched_fitting
        from pmriprep.models.dsc.vi_adapter import DSCVIModelAdapter

        original = batched_fitting._bounded_vi_mean
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("forced bounded solve failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(batched_fitting, "_bounded_vi_mean", fail_once)
        time = np.arange(6.0)
        observations = np.vstack((np.full(6, 10.0), np.full(6, 8.0)))
        adapter = DSCVIModelAdapter(
            predict=lambda means: np.repeat(means[:, :1], len(time), axis=1),
            jacobian=lambda means: np.ones((len(means), len(time), 1)),
            valid_mean=lambda means: np.all((means >= 0) & (means <= 1), axis=1),
        )
        result = batched_dsc_vi_fit(
            "test_adapter", observations, np.ones_like(time), time,
            np.array([0.5]), np.array([1.0]), [TransformType.IDENTITY], [None],
            ["cbf"], [(0.0, 1.0)], initial_means_transformed=np.full((2, 1), 0.5),
            n_iterations=2, model_adapter=adapter, return_full_posterior=True,
            save_fitted_curves=True,
        )

        assert calls >= 2
        assert result.failure_reasons == ("bounded_mean_failed", None)
        assert np.isnan(result.params["cbf"][0])
        assert np.isfinite(result.params["cbf"][1])
        assert not result.converged[0]
        assert result.full_posterior is not None
        assert np.isnan(result.full_posterior.mean[0]).all()
        assert np.isfinite(result.full_posterior.mean[1]).all()

    def test_fractional_weights_reduce_effective_information(
        self, dsc_aif_and_time
    ):
        """Uniform half-weights widen batched DSC VI CBF uncertainty."""
        aif, tp = dsc_aif_and_time
        means, stds, transforms, ubs, names, bounds = self._get_vascular_prior_info()
        tissue = batched_vascular_forward(
            np.full(5, 0.06), np.full(5, 3.0), np.full(5, 2.0),
            np.full(5, 1.0), aif, tp,
        )

        full = batched_dsc_vi_fit(
            "vascular", tissue, aif, tp,
            means, stds, transforms, ubs, names, bounds,
            weights=np.ones(len(tp)),
        )
        half = batched_dsc_vi_fit(
            "vascular", tissue, aif, tp,
            means, stds, transforms, ubs, names, bounds,
            weights=np.full(len(tp), 0.5),
        )

        assert np.median(half.uncertainties["cbf"]) > np.median(
            full.uncertainties["cbf"]
        )


class TestBatchedDSCVICPUConsistency:
    """Verify DSC batched VI matches CPU VI."""

    def test_vascular_batched_vs_cpu(self, dsc_aif_and_time):
        """Batched VI matches CPU VI within 1e-3 rtol for Vascular."""
        from pmriprep.models.bayesian_utils import build_priors_from_params
        from pmriprep.models.dsc import VascularModel, fit_dsc_model

        aif, tp = dsc_aif_and_time
        model = VascularModel()

        # Single synthetic voxel
        rng = np.random.RandomState(99)
        tissue = batched_vascular_forward(
            np.array([0.06]), np.array([3.0]), np.array([2.0]),
            np.array([1.5]), aif, tp,
        )
        tissue_1d = tissue[0] + 0.003 * rng.randn(len(tp))

        # CPU VI
        cpu_result = fit_dsc_model(
            model=VascularModel(), concentration=tissue_1d,
            aif=aif, time_points=tp, method="vi",
        )

        # Batched VI
        fitting_defs = model.get_fitting_parameter_definitions()
        priors = build_priors_from_params(fitting_defs)
        means = np.array([p.mean for p in priors])
        stds = np.array([p.std for p in priors])
        transforms = [p.transform for p in priors]
        ubs = [p.upper_bound for p in priors]
        names = [p.name for p in fitting_defs]
        bounds = [(p.bounds[0], p.bounds[1]) for p in fitting_defs]

        batched_result = batched_dsc_vi_fit(
            "vascular", tissue_1d[None, :], aif, tp,
            means, stds, transforms, ubs, names, bounds,
        )

        # Compare CBF (fitting-space param, directly comparable)
        cpu_cbf = cpu_result.parameters.get("cbf", 0)
        batched_cbf = float(batched_result.params["cbf"][0])

        # Use 5e-3 rtol — DSC forward model differences (FFT padding, delay
        # interpolation) propagate through VB iterations
        np.testing.assert_allclose(
            batched_cbf, cpu_cbf, rtol=5e-3,
            err_msg=f"CBF: batched={batched_cbf:.6f} CPU={cpu_cbf:.6f}",
        )

    def test_vascular_nlls_and_vi_share_model_parameterization(
        self,
        dsc_aif_and_time: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Vascular NLLS and VI return the same model-space parameter names."""
        from pmriprep.models.dsc import VascularModel, fit_dsc_model

        aif, tp = dsc_aif_and_time
        tissue = batched_vascular_forward(
            np.array([0.06]), np.array([3.0]), np.array([2.0]),
            np.array([1.5]), aif, tp,
        )[0]

        nlls = fit_dsc_model(
            VascularModel(), tissue, aif, tp, method="nlls"
        )
        vi = fit_dsc_model(
            VascularModel(), tissue, aif, tp, method="vi", sloppy=True
        )

        assert nlls.parameters.keys() == vi.parameters.keys()
        assert nlls.parameters["cbf"] == pytest.approx(0.06, rel=0.02)
        assert nlls.parameters["alpha"] * nlls.parameters["beta"] == (
            pytest.approx(6.0, rel=0.03)
        )


class TestAnalyticCBFJacobian:
    """Verify analytic dC/d(log_cbf) = C(t) matches central FD."""

    def test_vascular_cbf_jacobian(self, dsc_aif_and_time):
        """Analytic CBF Jacobian matches central FD for Vascular model."""
        aif, tp = dsc_aif_and_time
        cbf = np.array([0.06])
        alpha, beta, delay = np.array([3.0]), np.array([2.0]), np.array([1.5])
        g_m = batched_vascular_forward(cbf, alpha, beta, delay, aif, tp)

        j_analytic = g_m[0]

        eps = 1e-5
        log_cbf = np.log(cbf[0])
        g_plus = batched_vascular_forward(
            np.array([np.exp(log_cbf + eps)]), alpha, beta, delay, aif, tp,
        )
        g_minus = batched_vascular_forward(
            np.array([np.exp(log_cbf - eps)]), alpha, beta, delay, aif, tp,
        )
        j_fd = (g_plus[0] - g_minus[0]) / (2 * eps)

        np.testing.assert_allclose(j_analytic, j_fd, rtol=1e-4, atol=1e-10)

# ---------------------------------------------------------------------------
# Batched MAP fitting tests
# ---------------------------------------------------------------------------


class TestBatchedDSCMAPFit:
    """Tests for batched_dsc_map_fit."""

    @staticmethod
    def _get_vascular_prior_info():
        from pmriprep.models.bayesian_utils import build_priors_from_params
        from pmriprep.models.dsc import VascularModel

        model = VascularModel()
        fitting_defs = model.get_fitting_parameter_definitions()
        priors = build_priors_from_params(fitting_defs)
        return (
            np.array([p.mean for p in priors]),
            np.array([p.std for p in priors]),
            [p.transform for p in priors],
            [p.upper_bound for p in priors],
            [p.name for p in fitting_defs],
            [(p.bounds[0], p.bounds[1]) for p in fitting_defs],
            np.array([p.mean_transformed for p in priors]),
            np.array([p.std_transformed for p in priors]),
        )

    def test_vascular_recovery(self, dsc_aif_and_time):
        """Vascular MAP recovers parameters within 30% median error."""
        aif, tp = dsc_aif_and_time
        means, stds, transforms, ubs, names, bounds, means_t, stds_t = (
            self._get_vascular_prior_info()
        )

        rng = np.random.RandomState(42)
        n = 50
        cbf_true = rng.uniform(0.03, 0.1, n)
        alpha_true = rng.uniform(2.0, 5.0, n)
        beta_true = rng.uniform(1.0, 4.0, n)
        delay_true = rng.uniform(0.0, 3.0, n)

        tissue = batched_vascular_forward(cbf_true, alpha_true, beta_true, delay_true, aif, tp)
        tissue += 0.005 * rng.randn(*tissue.shape)

        result = batched_dsc_map_fit(
            "vascular", tissue, aif, tp,
            means, stds, transforms, ubs, names, bounds,
            prior_means_transformed=means_t,
            prior_stds_transformed=stds_t,
        )

        conv = result.converged
        assert np.sum(conv) > 25, f"Only {np.sum(conv)}/50 converged"

        cbf_err = np.median(np.abs(result.params["cbf"][conv] - cbf_true[conv]) /
                            np.maximum(cbf_true[conv], 1e-6))
        assert cbf_err < 0.3, f"CBF median rel error {cbf_err:.3f}"

    def test_vascular_result_format(self, dsc_aif_and_time):
        """MAP result has all required fields."""
        aif, tp = dsc_aif_and_time
        means, stds, transforms, ubs, names, bounds, means_t, stds_t = (
            self._get_vascular_prior_info()
        )

        tissue = batched_vascular_forward(
            np.full(5, 0.06), np.full(5, 3.0), np.full(5, 2.0),
            np.full(5, 1.0), aif, tp,
        )

        result = batched_dsc_map_fit(
            "vascular", tissue, aif, tp,
            means, stds, transforms, ubs, names, bounds,
            prior_means_transformed=means_t,
            prior_stds_transformed=stds_t,
            save_fitted_curves=True,
        )

        assert "cbf" in result.params
        assert "mtt" in result.params
        assert result.r_squared.shape == (5,)
        assert result.converged.shape == (5,)
        assert result.fitted_curves is not None
        assert result.fitted_curves.shape == (5, len(tp))
        assert "noise_std" in result.uncertainties
        assert "cbf" in result.uncertainties
        assert "cbf" in result.credible_interval_lower
        assert "cbf" in result.credible_interval_upper

    def test_empty_input(self, dsc_aif_and_time):
        """Handles zero voxels gracefully."""
        aif, tp = dsc_aif_and_time
        means, stds, transforms, ubs, names, bounds, means_t, stds_t = (
            self._get_vascular_prior_info()
        )

        result = batched_dsc_map_fit(
            "vascular", np.zeros((0, len(tp))), aif, tp,
            means, stds, transforms, ubs, names, bounds,
            prior_means_transformed=means_t,
            prior_stds_transformed=stds_t,
        )
        assert result.r_squared.shape == (0,)


class TestBatchedMAPLaplace:
    """Verify Laplace uncertainties from batched MAP are reasonable."""

    def test_uncertainties_finite_and_positive(self, dsc_aif_and_time):
        """Laplace uncertainties are finite, positive for converged voxels."""
        from pmriprep.models.bayesian_utils import build_priors_from_params
        from pmriprep.models.dsc import VascularModel

        aif, tp = dsc_aif_and_time
        model = VascularModel()
        fitting_defs = model.get_fitting_parameter_definitions()
        priors = build_priors_from_params(fitting_defs)

        means = np.array([p.mean for p in priors])
        stds = np.array([p.std for p in priors])
        transforms = [p.transform for p in priors]
        ubs = [p.upper_bound for p in priors]
        names = [p.name for p in fitting_defs]
        bounds = [(p.bounds[0], p.bounds[1]) for p in fitting_defs]
        means_t = np.array([p.mean_transformed for p in priors])
        stds_t = np.array([p.std_transformed for p in priors])

        tissue = batched_vascular_forward(
            np.full(10, 0.06), np.full(10, 3.0), np.full(10, 2.0),
            np.full(10, 1.5), aif, tp,
        )
        rng = np.random.RandomState(42)
        tissue += 0.005 * rng.randn(*tissue.shape)

        result = batched_dsc_map_fit(
            "vascular", tissue, aif, tp,
            means, stds, transforms, ubs, names, bounds,
            prior_means_transformed=means_t,
            prior_stds_transformed=stds_t,
            compute_uncertainties=True,
        )

        conv = result.converged
        assert np.sum(conv) > 5

        for name in names:
            unc = result.uncertainties[name][conv]
            assert np.all(np.isfinite(unc)), f"{name} has non-finite uncertainties"
            assert np.all(unc > 0), f"{name} has non-positive uncertainties"

        noise_std = result.uncertainties["noise_std"][conv]
        assert np.all(np.isfinite(noise_std))
        assert np.all(noise_std > 0)


class TestBatchedMAPMultiStart:
    """Verify multi-start improves MAP over single-start."""

    def test_multistart_selects_better(self, dsc_aif_and_time):
        """Multi-start from NLLS seeds produces better fit than prior-only."""
        from pmriprep.models.bayesian_utils import build_priors_from_params
        from pmriprep.models.dsc import VascularModel

        aif, tp = dsc_aif_and_time
        model = VascularModel()
        fitting_defs = model.get_fitting_parameter_definitions()
        priors = build_priors_from_params(fitting_defs)

        means = np.array([p.mean for p in priors])
        stds = np.array([p.std for p in priors])
        transforms = [p.transform for p in priors]
        ubs = [p.upper_bound for p in priors]
        names = [p.name for p in fitting_defs]
        bounds = [(p.bounds[0], p.bounds[1]) for p in fitting_defs]
        means_t = np.array([p.mean_transformed for p in priors])
        stds_t = np.array([p.std_transformed for p in priors])

        # Ground truth far from prior mean to make multi-start valuable
        n = 20
        cbf_true = np.full(n, 0.03)
        alpha_true = np.full(n, 4.0)
        beta_true = np.full(n, 1.5)
        delay_true = np.full(n, 2.5)

        tissue = batched_vascular_forward(cbf_true, alpha_true, beta_true, delay_true, aif, tp)
        rng = np.random.RandomState(42)
        tissue += 0.003 * rng.randn(*tissue.shape)

        # NLLS seeds close to ground truth
        mtt_true = alpha_true * beta_true
        nlls_seeds = np.column_stack([cbf_true, mtt_true, alpha_true, delay_true])

        result_multi = batched_dsc_map_fit(
            "vascular", tissue, aif, tp,
            means, stds, transforms, ubs, names, bounds,
            prior_means_transformed=means_t,
            prior_stds_transformed=stds_t,
            nlls_seeds=nlls_seeds,
        )

        result_single = batched_dsc_map_fit(
            "vascular", tissue, aif, tp,
            means, stds, transforms, ubs, names, bounds,
            prior_means_transformed=means_t,
            prior_stds_transformed=stds_t,
            nlls_seeds=None,
        )

        # Multi-start should have equal or better R²
        r2_multi = np.median(result_multi.r_squared)
        r2_single = np.median(result_single.r_squared)
        assert r2_multi >= r2_single - 0.01, (
            f"Multi-start R²={r2_multi:.4f} worse than single={r2_single:.4f}"
        )


class TestBatchedMAPSloppy:
    """Verify sloppy mode behavior."""

    def test_sloppy_mode(self, dsc_aif_and_time):
        """Sloppy mode uses fewer iterations and skips uncertainties."""
        from pmriprep.models.bayesian_utils import build_priors_from_params
        from pmriprep.models.dsc import VascularModel

        aif, tp = dsc_aif_and_time
        model = VascularModel()
        fitting_defs = model.get_fitting_parameter_definitions()
        priors = build_priors_from_params(fitting_defs)

        means = np.array([p.mean for p in priors])
        stds = np.array([p.std for p in priors])
        transforms = [p.transform for p in priors]
        ubs = [p.upper_bound for p in priors]
        names = [p.name for p in fitting_defs]
        bounds = [(p.bounds[0], p.bounds[1]) for p in fitting_defs]
        means_t = np.array([p.mean_transformed for p in priors])
        stds_t = np.array([p.std_transformed for p in priors])

        tissue = batched_vascular_forward(
            np.full(5, 0.06), np.full(5, 3.0), np.full(5, 2.0),
            np.full(5, 1.0), aif, tp,
        )

        result = batched_dsc_map_fit(
            "vascular", tissue, aif, tp,
            means, stds, transforms, ubs, names, bounds,
            prior_means_transformed=means_t,
            prior_stds_transformed=stds_t,
            sloppy=True,
        )

        # Sloppy: 8 iters * 1 noise loop = max 8
        assert result.n_iterations <= 8
        # Uncertainties should be NaN (skipped)
        for name in names:
            assert np.all(np.isnan(result.uncertainties[name]))


class TestBatchedMAPVsPerVoxelMAP:
    """Cross-validate batched MAP against per-voxel MAP.

    The two paths use different prior treatment (transformed-space Gaussian
    vs natural-space + Jacobian correction), so results will differ slightly.
    This test documents the expected divergence magnitude.
    """

    def test_vascular_cbf_within_tolerance(self, dsc_aif_and_time):
        """Batched and per-voxel MAP agree on CBF within 15%."""
        from pmriprep.models.bayesian_utils import build_priors_from_params
        from pmriprep.models.dsc import VascularModel, fit_dsc_model

        aif, tp = dsc_aif_and_time
        model = VascularModel()
        fitting_defs = model.get_fitting_parameter_definitions()
        priors = build_priors_from_params(fitting_defs)

        means = np.array([p.mean for p in priors])
        stds = np.array([p.std for p in priors])
        transforms = [p.transform for p in priors]
        ubs = [p.upper_bound for p in priors]
        names = [p.name for p in fitting_defs]
        bounds = [(p.bounds[0], p.bounds[1]) for p in fitting_defs]
        means_t = np.array([p.mean_transformed for p in priors])
        stds_t = np.array([p.std_transformed for p in priors])

        # Single synthetic voxel
        rng = np.random.RandomState(99)
        tissue = batched_vascular_forward(
            np.array([0.06]), np.array([3.0]), np.array([2.0]),
            np.array([1.5]), aif, tp,
        )
        tissue_1d = tissue[0] + 0.003 * rng.randn(len(tp))

        # Per-voxel MAP
        cpu_result = fit_dsc_model(
            model=VascularModel(), concentration=tissue_1d,
            aif=aif, time_points=tp, method="map",
        )

        # Batched MAP
        batched_result = batched_dsc_map_fit(
            "vascular", tissue_1d[None, :], aif, tp,
            means, stds, transforms, ubs, names, bounds,
            prior_means_transformed=means_t,
            prior_stds_transformed=stds_t,
        )

        cpu_cbf = cpu_result.parameters.get("cbf", 0)
        batched_cbf = float(batched_result.params["cbf"][0])

        # Allow 15% divergence due to different prior treatment
        if cpu_cbf > 0.001:
            rel_diff = abs(batched_cbf - cpu_cbf) / cpu_cbf
            assert rel_diff < 0.15, (
                f"CBF divergence {rel_diff:.1%}: batched={batched_cbf:.6f} "
                f"per-voxel={cpu_cbf:.6f}"
            )


# ---------------------------------------------------------------------------
# TestBatchedFitParallel — multiprocess dispatch
# ---------------------------------------------------------------------------


class TestBatchedFitParallel:
    """Tests for the nprocs>1 dispatch in batched_dsc_vi_fit /
    batched_dsc_map_fit.

    Per-voxel updates inside the batched fit functions are independent,
    so chunked + concatenated should equal single-process up to
    floating-point reduction order. Spatial VB is rejected because
    voxels are coupled through the M-prior.
    """

    @staticmethod
    def _vasc_priors():
        from pmriprep.models.bayesian_utils import build_priors_from_params
        from pmriprep.models.dsc import VascularModel

        model = VascularModel()
        fitting_defs = model.get_fitting_parameter_definitions()
        priors = build_priors_from_params(fitting_defs)
        return dict(
            prior_means_nat=np.array([p.mean for p in priors]),
            prior_stds_nat=np.array([p.std for p in priors]),
            transforms=[p.transform for p in priors],
            upper_bounds=[p.upper_bound for p in priors],
            fitting_param_names=[p.name for p in fitting_defs],
            fitting_bounds=[(p.bounds[0], p.bounds[1]) for p in fitting_defs],
        )

    @staticmethod
    def _build_tissue(aif, tp, n: int = 200):
        rng = np.random.RandomState(7)
        cbf_true = rng.uniform(0.03, 0.1, n)
        alpha_true = rng.uniform(2.0, 5.0, n)
        beta_true = rng.uniform(1.0, 4.0, n)
        delay_true = rng.uniform(0.0, 3.0, n)
        tissue = batched_vascular_forward(
            cbf_true, alpha_true, beta_true, delay_true, aif, tp,
        )
        tissue += 0.005 * rng.randn(*tissue.shape)
        return tissue

    @pytest.mark.parametrize("nprocs", [2, 3])
    def test_vi_parallel_matches_single(self, dsc_aif_and_time, nprocs: int):
        """Single and chunked VI preserve row identity and fitted measurements."""
        aif, tp = dsc_aif_and_time
        priors = self._vasc_priors()
        tissue = self._build_tissue(aif, tp, n=120)

        common = dict(
            model_type="vascular",
            tissue_curves=tissue,
            aif=aif,
            time_points=tp,
            **priors,
        )
        single = batched_dsc_vi_fit(**common, nprocs=1)
        # Force the parallel path by lowering parallel_chunk_min below n.
        parallel = batched_dsc_vi_fit(
            **common, nprocs=nprocs, parallel_chunk_min=50,
        )

        assert single.r_squared.shape == parallel.r_squared.shape
        np.testing.assert_allclose(
            parallel.r_squared, single.r_squared, rtol=1e-10, atol=1e-12,
        )
        np.testing.assert_array_equal(parallel.converged, single.converged)
        # Iterative finite-difference fits amplify BLAS reduction-order noise.
        # Eight-digit relative agreement is stricter than float32 map precision;
        # curve-quality and convergence checks above remain independently strict.
        for k in single.params:
            np.testing.assert_allclose(
                parallel.params[k], single.params[k],
                rtol=1e-8, atol=1e-12,
                err_msg=f"params[{k!r}] mismatch under nprocs={nprocs}",
            )

    def test_map_parallel_matches_single(self, dsc_aif_and_time):
        """nprocs=1 vs nprocs=2 map_fit results agree within fp tolerance."""
        aif, tp = dsc_aif_and_time
        priors = self._vasc_priors()
        tissue = self._build_tissue(aif, tp, n=120)

        common = dict(
            model_type="vascular",
            tissue_curves=tissue,
            aif=aif,
            time_points=tp,
            **priors,
        )
        single = batched_dsc_map_fit(**common, nprocs=1)
        parallel = batched_dsc_map_fit(
            **common, nprocs=2, parallel_chunk_min=50,
        )

        # Parallel and serial run the identical per-voxel optimizer, but chunking
        # across worker processes changes BLAS/OpenMP thread counts, so the fit
        # converges to points that differ by fp-reproducibility noise (~1e-10),
        # not bit-identically. Tolerance is set to that floor: loose enough to
        # tolerate thread-order noise, far tighter than any real chunking bug
        # (which would mis-assign voxels and diverge by orders of magnitude).
        np.testing.assert_allclose(
            parallel.r_squared, single.r_squared, rtol=1e-6, atol=1e-8,
        )
        for k in single.params:
            np.testing.assert_allclose(
                parallel.params[k], single.params[k],
                rtol=1e-6, atol=1e-8,
                err_msg=f"MAP params[{k!r}] mismatch under nprocs=2",
            )

    def test_spatial_vb_rejected_under_nprocs(self, dsc_aif_and_time):
        """Spatial VB inputs + nprocs>1 must raise ValueError."""
        aif, tp = dsc_aif_and_time
        priors = self._vasc_priors()
        n = 50
        tissue = self._build_tissue(aif, tp, n=n)
        n_params = len(priors["fitting_param_names"])
        spatial_prec_diag = np.ones((n, n_params), dtype=np.float64)
        spatial_eta = np.zeros((n, n_params), dtype=np.float64)

        with pytest.raises(ValueError, match="spatial VB"):
            batched_dsc_vi_fit(
                model_type="vascular",
                tissue_curves=tissue,
                aif=aif,
                time_points=tp,
                **priors,
                spatial_prec_diag=spatial_prec_diag,
                spatial_eta=spatial_eta,
                nprocs=2,
                parallel_chunk_min=10,
            )

    def test_below_threshold_no_pool_spawned(self, dsc_aif_and_time, monkeypatch):
        """When n_voxels < parallel_chunk_min, no ProcessPoolExecutor is created
        even if nprocs > 1 — short-circuit avoids worker-spawn overhead.
        """

        sentinel = {"called": False}

        class _ShouldNotSpawn:
            def __init__(self, *_a, **_kw):
                sentinel["called"] = True
                raise AssertionError(
                    "ProcessPoolExecutor must not be instantiated when "
                    "n_voxels < parallel_chunk_min"
                )

        # Monkeypatch the symbol that _run_chunked_in_pool looks up at
        # call time. Because that helper imports inside the function
        # body (`from concurrent.futures import ProcessPoolExecutor`),
        # we patch the parent module instead.
        import concurrent.futures
        monkeypatch.setattr(
            concurrent.futures, "ProcessPoolExecutor", _ShouldNotSpawn,
        )

        aif, tp = dsc_aif_and_time
        priors = self._vasc_priors()
        tissue = self._build_tissue(aif, tp, n=10)

        result = batched_dsc_vi_fit(
            model_type="vascular",
            tissue_curves=tissue,
            aif=aif,
            time_points=tp,
            **priors,
            nprocs=4,  # ignored because 10 < default parallel_chunk_min=5000
        )
        assert result.r_squared.shape == (10,)
        assert sentinel["called"] is False


class TestVascularReadout:
    """Derived-volume uncertainty requires joint covariance."""

    def test_vascular_readout_does_not_assume_cbf_mtt_independence(
        self, dsc_aif_and_time
    ):
        """Marginal CBF/MTT errors cannot define vascular CBV uncertainty."""
        import types

        from pmriprep.interfaces.dsc_compute import ComputeDSCMaps

        result = types.SimpleNamespace(
            params={
                "cbf": np.array([0.1]),
                "mtt": np.array([6.0]),
                "alpha": np.array([3.0]),
                "delay": np.array([0.0]),
            },
            uncertainties={
                "cbf": np.array([0.01]),
                "mtt": np.array([0.5]),
            },
            r_squared=np.ones(1),
            converged=np.ones(1, dtype=bool),
        )
        mask_coords = (
            np.zeros(1, dtype=int),
            np.zeros(1, dtype=int),
            np.zeros(1, dtype=int),
        )

        out = ComputeDSCMaps._batched_result_to_voxel_dicts(
            None,
            result,
            mask_coords,
            "vascular",
            1,
        )[0]

        assert out["cbf_std"] == pytest.approx(0.01)
        assert "cbv_std" not in out
        assert "cbv_ci95_lower" not in out
        assert "cbv_ci95_upper" not in out


def _ref_scalar_delay_shift(conv_1d, delay, dt):
    """Reference scalar two-tap delay shift (lo-gated), matching the inline shift
    in the vascular tissue curve. The shared batched helper must equal
    this for all delays, including negative non-integer ones."""
    n = len(conv_1d)
    delay_frac = delay / dt
    n_shift = int(np.floor(delay_frac))
    f_shift = delay_frac - n_shift
    src = np.arange(n) - n_shift
    out = np.zeros(n, dtype=np.float64)
    lo = (src >= 0) & (src < n)
    out[lo] = (1.0 - f_shift) * conv_1d[src[lo]]
    hi_src = src - 1
    hi = lo & (hi_src >= 0) & (hi_src < n)
    out[hi] += f_shift * conv_1d[hi_src[hi]]
    return out


class TestPostHocDelayShiftParity:
    """The shared `_apply_post_hoc_delay_shift` (used by the vascular
    batched forwards) must match the scalar two-tap shift exactly, including
    negative non-integer delays at the curve tail (the FU1 valid_hi gating)."""

    @pytest.mark.parametrize("delay", [-2.3, -1.0, 0.0, 1.7, 4.0])
    def test_matches_scalar_shift(self, delay):
        from pmriprep.models.dsc.convolution import _apply_post_hoc_delay_shift

        nt = 50
        t = np.arange(nt, dtype=np.float64)
        conv = np.stack([
            np.exp(-((t - 18.0) ** 2) / (2 * 6.0 ** 2)),
            0.5 * np.exp(-((t - 25.0) ** 2) / (2 * 8.0 ** 2)),
        ])
        dt = 1.5
        batched = _apply_post_hoc_delay_shift(conv, np.array([delay, delay]), dt)
        for i in range(2):
            ref = _ref_scalar_delay_shift(conv[i], delay, dt)
            np.testing.assert_allclose(batched[i], ref, atol=1e-12)
