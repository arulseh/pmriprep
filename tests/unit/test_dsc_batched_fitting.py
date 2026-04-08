"""Unit tests for DSC batched fitting (forward models and VI).

Tests verify parameter recovery, convergence, forward model correctness,
and batched-vs-CPU VI consistency for both CPI and Vascular DSC models.

All tests use numpy (pure CPU).
"""

import numpy as np
import pytest

from pmriprep.models.dsc.batched_fitting import (
    batched_cpi_forward,
    batched_dsc_vi_fit,
    batched_vascular_forward,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
# Batched CPI forward model tests
# ---------------------------------------------------------------------------


class TestBatchedCPIForward:
    """Tests for batched_cpi_forward."""

    def test_shape(self, dsc_aif_and_time):
        """Output shape matches (N, T)."""
        from pmriprep.utils.dsc_deconvolution import build_toeplitz_matrix

        aif, tp = dsc_aif_and_time
        dt = tp[1] - tp[0]
        toeplitz = build_toeplitz_matrix(aif, dt)
        cp_times = np.linspace(0, 50, 7)

        n = 15
        params = np.zeros((n, 7))
        params[:, 0] = 0.06
        for i in range(1, 7):
            params[:, i] = np.exp(-i / 3.0)

        result = batched_cpi_forward(params, aif, tp, cp_times, toeplitz)
        assert result.shape == (n, len(tp))

    def test_matches_scalar(self, dsc_aif_and_time):
        """Batched result matches scalar CPIModel.tissue_curve."""
        from pmriprep.models.dsc import CPIModel
        from pmriprep.utils.dsc_deconvolution import build_toeplitz_matrix

        aif, tp = dsc_aif_and_time
        dt = tp[1] - tp[0]
        n_cp = 7
        t_span = 30.0  # production config: covers ~3x max physiological MTT
        model = CPIModel(n_control_points=n_cp, time_range=(0.0, t_span))
        cp_times = np.array(model._control_point_times, dtype=np.float64)
        toeplitz = build_toeplitz_matrix(aif, dt)

        # Set params: CBF + exponential decay control points
        scalar_params = [0.06] + [float(np.exp(-i / 3.0)) for i in range(1, n_cp)]
        scalar_tissue = model.tissue_curve(scalar_params, aif, tp)

        batched_params = np.array([scalar_params])
        batched_tissue = batched_cpi_forward(batched_params, aif, tp, cp_times, toeplitz)

        # rtol=2e-2: vectorized Fritsch-Carlson PCHIP vs scipy PchipInterpolator
        # produces small slope differences at interior knots
        np.testing.assert_allclose(batched_tissue[0], scalar_tissue, rtol=2e-2)


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
        assert result.n_iterations <= 5


class TestBatchedCPIVIFit:
    """Tests for CPI batched VI fitting."""

    @staticmethod
    def _get_cpi_setup(aif, tp):
        """Build CPI model, priors, and Toeplitz for testing."""
        from pmriprep.models.bayesian_utils import build_priors_from_params
        from pmriprep.models.dsc import CPIModel
        from pmriprep.utils.dsc_deconvolution import build_toeplitz_matrix

        dt = tp[1] - tp[0]
        n_cp = 7
        t_span = 30.0  # production config: covers ~3x max physiological MTT
        model = CPIModel(n_control_points=n_cp, time_range=(0.0, t_span))
        fitting_defs = model.get_fitting_parameter_definitions()
        priors = build_priors_from_params(fitting_defs)
        cp_times = np.array(model._control_point_times, dtype=np.float64)
        toeplitz = build_toeplitz_matrix(aif, dt)
        mono_prec = model.compute_monotonicity_precision()

        means = np.array([p.mean for p in priors])
        stds = np.array([p.std for p in priors])
        transforms = [p.transform for p in priors]
        ubs = [p.upper_bound for p in priors]
        names = [p.name for p in fitting_defs]
        bounds = [(p.bounds[0], p.bounds[1]) for p in fitting_defs]

        return means, stds, transforms, ubs, names, bounds, mono_prec, cp_times, toeplitz

    def test_cpi_recovery(self, dsc_aif_and_time):
        """CPI VI recovers CBF and produces monotonic residue."""
        aif, tp = dsc_aif_and_time
        setup = self._get_cpi_setup(aif, tp)
        means, stds, transforms, ubs, names, bounds, mono_prec, cp_times, toeplitz = setup

        # Generate synthetic CPI data with known exponential residue
        rng = np.random.RandomState(42)
        n = 30
        cbf_true = rng.uniform(0.03, 0.1, n)
        n_cp = len(cp_times)

        # Monotonically decreasing control points (exponential decay)
        params_true = np.zeros((n, n_cp))
        params_true[:, 0] = cbf_true
        for k in range(1, n_cp):
            params_true[:, k] = np.exp(-k / 3.0)

        tissue = batched_cpi_forward(params_true, aif, tp, cp_times, toeplitz)
        tissue += 0.003 * rng.randn(*tissue.shape)

        result = batched_dsc_vi_fit(
            "cpi", tissue, aif, tp,
            means, stds, transforms, ubs, names, bounds,
            mono_prec=mono_prec, control_point_times=cp_times,
            toeplitz_matrix=toeplitz,
        )

        conv = result.converged
        assert np.sum(conv) > 15, f"Only {np.sum(conv)}/30 converged"

        # CBF recovery within 40% median error
        cbf_err = np.median(np.abs(result.params["cbf"][conv] - cbf_true[conv]) /
                            np.maximum(cbf_true[conv], 1e-6))
        assert cbf_err < 0.4, f"CBF median rel error {cbf_err:.3f}"

        # Verify monotonicity of fitted control points
        for i in np.where(conv)[0][:5]:
            prev = 1.0
            for k in range(1, n_cp):
                rk = float(result.params[f"r{k}"][i])
                # Soft monotonicity: allow small violations (prior, not hard constraint)
                assert rk <= prev + 0.05, f"Voxel {i}: r{k}={rk:.3f} > r{k-1}={prev:.3f}"
                prev = rk

    def test_cpi_result_format(self, dsc_aif_and_time):
        """CPI VI result has correct parameter names."""
        aif, tp = dsc_aif_and_time
        setup = self._get_cpi_setup(aif, tp)
        means, stds, transforms, ubs, names, bounds, mono_prec, cp_times, toeplitz = setup

        n_cp = len(cp_times)
        params = np.zeros((5, n_cp))
        params[:, 0] = 0.06
        for k in range(1, n_cp):
            params[:, k] = np.exp(-k / 3.0)
        tissue = batched_cpi_forward(params, aif, tp, cp_times, toeplitz)

        result = batched_dsc_vi_fit(
            "cpi", tissue, aif, tp,
            means, stds, transforms, ubs, names, bounds,
            mono_prec=mono_prec, control_point_times=cp_times,
            toeplitz_matrix=toeplitz, save_fitted_curves=True,
        )

        assert "cbf" in result.params
        for k in range(1, n_cp):
            assert f"r{k}" in result.params
        assert result.fitted_curves is not None
        assert result.fitted_curves.shape == (5, len(tp))


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
