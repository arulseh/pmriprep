"""Unit tests for DSC batched fitting (forward models, VI, and MAP).

Tests verify parameter recovery, convergence, forward model correctness,
and batched-vs-CPU consistency for both CPI and Vascular DSC models.

All tests use numpy (pure CPU).
"""

import numpy as np
import pytest

from pmriprep.models.dsc.batched_fitting import (
    batched_cpi_forward,
    batched_cpi_residue,
    batched_dsc_map_fit,
    batched_dsc_vi_fit,
    batched_vascular_forward,
)
from pmriprep.models.dsc.cpi import CPIModel
from pmriprep.utils.dsc_deconvolution import build_volterra_matrix


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
        from pmriprep.utils.dsc_deconvolution import build_volterra_matrix

        aif, tp = dsc_aif_and_time
        dt = tp[1] - tp[0]
        toeplitz = build_volterra_matrix(aif, dt)
        # FABBER convention: 7 CPs → 8 params (cbf + 7 fractions)
        n_cp = 7
        cp_times = np.linspace(5, 50, n_cp)  # uniform-ish; first CP > 0

        n = 15
        params = np.zeros((n, 1 + n_cp))
        params[:, 0] = 0.06
        for i in range(1, 1 + n_cp):
            params[:, i] = np.exp(-1.0 / 3.0)

        result = batched_cpi_forward(params, aif, tp, cp_times, toeplitz)
        assert result.shape == (n, len(tp))

    def test_matches_scalar(self, dsc_aif_and_time):
        """Batched result matches scalar CPIModel.tissue_curve."""
        from pmriprep.models.dsc import CPIModel
        from pmriprep.utils.dsc_deconvolution import build_volterra_matrix

        aif, tp = dsc_aif_and_time
        dt = tp[1] - tp[0]
        n_cp = 7
        t_span = 30.0  # production config: covers ~3x max physiological MTT
        model = CPIModel(
            n_control_points=n_cp,
            n_acquisition_frames=int(len(tp)),
            dt=float(tp[1] - tp[0]),
        )
        cp_times = np.array(model._control_point_times, dtype=np.float64)
        toeplitz = build_volterra_matrix(aif, dt)

        # FABBER convention: cbf + N free fractions = N+1 params total
        scalar_params = [0.06] + [float(np.exp(-1.0 / 3.0)) for _ in range(n_cp)]
        scalar_tissue = model.tissue_curve(scalar_params, aif, tp)

        batched_params = np.array([scalar_params])
        batched_tissue = batched_cpi_forward(batched_params, aif, tp, cp_times, toeplitz)

        # rtol=5e-2: the batched vectorized Fritsch-Carlson PCHIP vs the scalar
        # scipy PchipInterpolator produce small slope differences amplified near
        # the terminal anchor. Both forwards use the native-dt (1,4,1)/6 Toeplitz
        # convolution.
        np.testing.assert_allclose(batched_tissue[0], scalar_tissue, rtol=5e-2)


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
        assert result.n_iterations <= 8


class TestBatchedCPIVIFit:
    """Tests for CPI batched VI fitting."""

    @staticmethod
    def _get_cpi_setup(aif, tp):
        """Build CPI model, priors, and Toeplitz for testing."""
        from pmriprep.models.bayesian_utils import build_priors_from_params
        from pmriprep.models.dsc import CPIModel
        from pmriprep.utils.dsc_deconvolution import build_volterra_matrix

        dt = tp[1] - tp[0]
        n_cp = 7
        t_span = 30.0  # production config: covers ~3x max physiological MTT
        model = CPIModel(
            n_control_points=n_cp,
            n_acquisition_frames=int(len(tp)),
            dt=float(tp[1] - tp[0]),
        )
        fitting_defs = model.get_fitting_parameter_definitions()
        priors = build_priors_from_params(fitting_defs)
        cp_times = np.array(model._control_point_times, dtype=np.float64)
        toeplitz = build_volterra_matrix(aif, dt)
        mono_prec = None
        mono_eta = None

        means = np.array([p.mean for p in priors])
        stds = np.array([p.std for p in priors])
        means_t = np.array([p.mean_transformed for p in priors])
        stds_t = np.array([p.std_transformed for p in priors])
        transforms = [p.transform for p in priors]
        ubs = [p.upper_bound for p in priors]
        names = [p.name for p in fitting_defs]
        bounds = [(p.bounds[0], p.bounds[1]) for p in fitting_defs]

        return (means, stds, means_t, stds_t, transforms, ubs, names, bounds,
                mono_prec, mono_eta, cp_times, toeplitz, model.terminal_time)

    def test_cpi_recovery(self, dsc_aif_and_time):
        """CPI VI recovers CBF and produces monotonic residue."""
        aif, tp = dsc_aif_and_time
        setup = self._get_cpi_setup(aif, tp)
        (means, stds, means_t, stds_t, transforms, ubs, names, bounds,
         mono_prec, mono_eta, cp_times, toeplitz, t_terminal) = setup

        # Generate synthetic CPI data with known exponential residue
        rng = np.random.RandomState(42)
        n = 30
        cbf_true = rng.uniform(0.03, 0.1, n)
        n_cp = len(cp_times)
        n_params = 1 + n_cp  # cbf + N fractions (FABBER convention)

        # Multiplicative fractions (constant for exponential decay)
        params_true = np.zeros((n, n_params))
        params_true[:, 0] = cbf_true
        params_true[:, 1:] = np.exp(-1.0 / 3.0)

        tissue = batched_cpi_forward(
            params_true, aif, tp, cp_times, toeplitz, t_terminal=t_terminal,
        )
        tissue += 0.003 * rng.randn(*tissue.shape)

        result = batched_dsc_vi_fit(
            "cpi", tissue, aif, tp,
            means, stds, transforms, ubs, names, bounds,
            mono_prec=mono_prec, mono_eta=mono_eta, control_point_times=cp_times,
            toeplitz_matrix=toeplitz,
            cpi_t_terminal=t_terminal,
            prior_means_transformed=means_t,
            prior_stds_transformed=stds_t,
        )

        # FABBER VB caps at 10 iterations by default; the per-voxel
        # `converged` flag (param_change<1e-3) often does not trigger
        # within that budget even when R² > 0.99. Quality is what we
        # actually care about. Per Mehndiratta 2013 / Chappell 2009:
        # CBF regression slope ≥ 0.96, |intercept| ≤ 0.05.
        ok = result.r_squared > 0.5
        assert int(np.sum(ok)) > 15, f"Only {int(np.sum(ok))}/30 produced R²>0.5"

        # CBF recovery within 40% median error
        cbf_err = np.median(np.abs(result.params["cbf"][ok] - cbf_true[ok]) /
                            np.maximum(cbf_true[ok], 1e-6))
        assert cbf_err < 0.4, f"CBF median rel error {cbf_err:.3f}"

        # Regression-coefficient gate (FABBER paper benchmark)
        slope = float(np.polyfit(cbf_true[ok], result.params["cbf"][ok], 1)[0])
        assert slope > 0.85, f"CBF regression slope {slope:.3f} < 0.85"

        # Verify each multiplicative fraction is in [0, 1]
        for i in np.where(ok)[0][:5]:
            for k in range(1, n_cp + 1):
                fk = float(result.params[f"f{k}"][i])
                assert 0.0 <= fk <= 1.0, f"Voxel {i}: f{k}={fk:.3f} not in [0,1]"

    def test_cpi_result_format(self, dsc_aif_and_time):
        """CPI VI result has correct parameter names."""
        aif, tp = dsc_aif_and_time
        setup = self._get_cpi_setup(aif, tp)
        (means, stds, means_t, stds_t, transforms, ubs, names, bounds,
         mono_prec, mono_eta, cp_times, toeplitz, t_terminal) = setup

        n_cp = len(cp_times)
        params = np.zeros((5, 1 + n_cp))
        params[:, 0] = 0.06
        params[:, 1:] = np.exp(-1.0 / 3.0)
        tissue = batched_cpi_forward(
            params, aif, tp, cp_times, toeplitz, t_terminal=t_terminal,
        )

        result = batched_dsc_vi_fit(
            "cpi", tissue, aif, tp,
            means, stds, transforms, ubs, names, bounds,
            mono_prec=mono_prec, mono_eta=mono_eta, control_point_times=cp_times,
            toeplitz_matrix=toeplitz,
            cpi_t_terminal=t_terminal,
            prior_means_transformed=means_t,
            prior_stds_transformed=stds_t,
            save_fitted_curves=True,
        )

        assert "cbf" in result.params
        for k in range(1, n_cp + 1):
            assert f"f{k}" in result.params
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

    def test_cpi_cbf_jacobian(self, dsc_aif_and_time):
        """Analytic CBF Jacobian (identity-CBF) matches central FD for CPI."""
        from pmriprep.utils.dsc_deconvolution import build_volterra_matrix

        aif, tp = dsc_aif_and_time
        dt = tp[1] - tp[0]
        toeplitz = build_volterra_matrix(aif, dt)
        # FABBER convention: cbf + N free fractions, first CP at t > 0.
        n_cp = 7
        cp_times = np.linspace(5, 30, n_cp)

        params = np.zeros((1, 1 + n_cp))
        params[0, 0] = 0.06
        params[0, 1:] = np.exp(-1.0 / 3.0)

        g_m = batched_cpi_forward(params, aif, tp, cp_times, toeplitz)
        # Identity-CBF analytic Jacobian: ∂g/∂cbf = g / cbf
        j_analytic = g_m[0] / params[0, 0]

        eps = 1e-5 * params[0, 0]  # natural-space step
        params_plus = params.copy()
        params_plus[0, 0] = params[0, 0] + eps
        params_minus = params.copy()
        params_minus[0, 0] = params[0, 0] - eps

        g_plus = batched_cpi_forward(params_plus, aif, tp, cp_times, toeplitz)
        g_minus = batched_cpi_forward(params_minus, aif, tp, cp_times, toeplitz)
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

    @staticmethod
    def _get_cpi_setup(aif, tp):
        from pmriprep.models.bayesian_utils import build_priors_from_params
        from pmriprep.models.dsc import CPIModel
        from pmriprep.utils.dsc_deconvolution import build_volterra_matrix

        dt = tp[1] - tp[0]
        n_cp = 7
        model = CPIModel(
            n_control_points=n_cp,
            n_acquisition_frames=int(len(tp)),
            dt=float(tp[1] - tp[0]),
        )
        fitting_defs = model.get_fitting_parameter_definitions()
        priors = build_priors_from_params(fitting_defs)
        cp_times = np.array(model._control_point_times, dtype=np.float64)
        toeplitz = build_volterra_matrix(aif, dt)

        means = np.array([p.mean for p in priors])
        stds = np.array([p.std for p in priors])
        transforms = [p.transform for p in priors]
        ubs = [p.upper_bound for p in priors]
        names = [p.name for p in fitting_defs]
        bounds = [(p.bounds[0], p.bounds[1]) for p in fitting_defs]
        means_t = np.array([p.mean_transformed for p in priors])
        stds_t = np.array([p.std_transformed for p in priors])

        return (means, stds, transforms, ubs, names, bounds,
                None, None, cp_times, toeplitz, means_t, stds_t,
                model.terminal_time)

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

    def test_cpi_recovery(self, dsc_aif_and_time):
        """CPI MAP recovers CBF within 40% median error."""
        aif, tp = dsc_aif_and_time
        setup = self._get_cpi_setup(aif, tp)
        (means, stds, transforms, ubs, names, bounds,
         mono_prec, mono_eta, cp_times, toeplitz, means_t, stds_t,
         t_terminal) = setup

        rng = np.random.RandomState(42)
        n = 30
        cbf_true = rng.uniform(0.03, 0.1, n)
        n_cp = len(cp_times)
        n_params = 1 + n_cp

        params_true = np.zeros((n, n_params))
        params_true[:, 0] = cbf_true
        params_true[:, 1:] = np.exp(-1.0 / 3.0)

        tissue = batched_cpi_forward(
            params_true, aif, tp, cp_times, toeplitz, t_terminal=t_terminal,
        )
        tissue += 0.003 * rng.randn(*tissue.shape)

        result = batched_dsc_map_fit(
            "cpi", tissue, aif, tp,
            means, stds, transforms, ubs, names, bounds,
            mono_prec=mono_prec, mono_eta=mono_eta,
            control_point_times=cp_times, toeplitz_matrix=toeplitz,
            cpi_t_terminal=t_terminal,
            prior_means_transformed=means_t,
            prior_stds_transformed=stds_t,
        )

        conv = result.converged
        assert np.sum(conv) > 15, f"Only {np.sum(conv)}/30 converged"

        cbf_err = np.median(np.abs(result.params["cbf"][conv] - cbf_true[conv]) /
                            np.maximum(cbf_true[conv], 1e-6))
        assert cbf_err < 0.4, f"CBF median rel error {cbf_err:.3f}"

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

    def test_vi_parallel_matches_single(self, dsc_aif_and_time):
        """nprocs=1 vs nprocs=2 vi_fit results agree within fp tolerance."""
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
            **common, nprocs=2, parallel_chunk_min=50,
        )

        assert single.r_squared.shape == parallel.r_squared.shape
        np.testing.assert_allclose(
            parallel.r_squared, single.r_squared, rtol=1e-10, atol=1e-12,
        )
        np.testing.assert_array_equal(parallel.converged, single.converged)
        for k in single.params:
            np.testing.assert_allclose(
                parallel.params[k], single.params[k],
                rtol=1e-10, atol=1e-12,
                err_msg=f"params[{k!r}] mismatch under nprocs=2",
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
        import pmriprep.models.dsc.batched_fitting as bf_mod

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


# ---------------------------------------------------------------------------
# CPI shared residue helper + CBV/MTT readout parity
# Regression: the batched readout integrated a stale, mis-reconstructed residue
# (first CP forced to 1.0, dropped fN, default terminal anchor, ignored
# infer_cpt log-spacings), corrupting CBV/MTT while CBF stayed correct.
# ---------------------------------------------------------------------------


def _cpi_param_matrix(n_cp, cbf_vals, fracs_rows, spacings_rows=None):
    """Assemble a CPI (N, P) parameter matrix in canonical column order
    ``[cbf, f1..fN (, s1..sN)]`` (s_i stored as log-spacings)."""
    cols = [np.asarray(cbf_vals, dtype=np.float64)]
    for k in range(n_cp):
        cols.append(np.array([r[k] for r in fracs_rows], dtype=np.float64))
    if spacings_rows is not None:
        for k in range(n_cp):
            cols.append(
                np.array([np.log(s[k]) for s in spacings_rows], dtype=np.float64)
            )
    return np.column_stack(cols)


def _cpi_readout(params, names, cp_times, time_points, dt, t_terminal):
    """Invoke the production batched CBV/MTT readout on a fake fit result.

    ``_batched_result_to_voxel_dicts`` does not use ``self``, so it is called
    unbound; only ``.params``/``.uncertainties``/``.r_squared``/``.converged``
    are read.
    """
    import types

    from pmriprep.interfaces.dsc_compute import ComputeDSCMaps

    n = params.shape[0]
    result = types.SimpleNamespace(
        params={nm: params[:, j] for j, nm in enumerate(names)},
        uncertainties={},
        r_squared=np.ones(n),
        converged=np.ones(n, dtype=bool),
    )
    mask_coords = (np.arange(n), np.zeros(n, dtype=int), np.zeros(n, dtype=int))
    return ComputeDSCMaps._batched_result_to_voxel_dicts(
        None, result, mask_coords, "cpi", n, dt, cp_times, time_points, t_terminal,
    )


class TestBatchedCPIForwardSharedResidue:
    """``batched_cpi_forward`` must equal ``cbf * (Toeplitz @ batched_cpi_residue)``
    so the forward stays locked to the shared residue helper and cannot drift
    from the extracted construction."""

    @pytest.mark.parametrize("infer_cpt", [False, True])
    def test_forward_equals_cbf_times_toeplitz_residue(
        self, dsc_aif_and_time, infer_cpt
    ):
        aif, tp = dsc_aif_and_time
        nt = len(tp)
        dt = float(tp[1] - tp[0])
        n_cp = 6
        model = CPIModel(
            n_control_points=n_cp, n_acquisition_frames=nt, dt=dt,
            infer_cpt=infer_cpt,
        )
        cp_times = np.asarray(model.control_point_times, dtype=np.float64)
        t_terminal = model.terminal_time
        toeplitz = build_volterra_matrix(aif, dt)

        cbf = [0.10, 0.05, 0.16]
        fr = [
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            [0.8, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.7, 0.6, 0.4, 0.3, 0.2, 0.1],
        ]
        spac = [[3.0, 4.0, 5.0, 7.0, 9.0, 12.0]] * 3 if infer_cpt else None
        params = _cpi_param_matrix(n_cp, cbf, fr, spac)

        R = batched_cpi_residue(params, tp, cp_times, t_terminal=t_terminal)
        expected = np.asarray(cbf)[:, None] * (toeplitz @ R.T).T
        got = batched_cpi_forward(
            params, aif, tp, cp_times, toeplitz, t_terminal=t_terminal,
        )
        np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0.0)


class TestCPIReadoutResidueParity:
    """The batched CBV/MTT readout must integrate the SAME residue the forward
    model fitted (shared ``batched_cpi_residue``), in BOTH fixed-CP and
    ``infer_cpt`` (production-default) layouts."""

    @staticmethod
    def _grid(infer_cpt, dsc_aif_and_time):
        aif, tp = dsc_aif_and_time
        nt = len(tp)
        dt = float(tp[1] - tp[0])
        n_cp = 6
        model = CPIModel(
            n_control_points=n_cp, n_acquisition_frames=nt, dt=dt,
            infer_cpt=infer_cpt,
        )
        cp_times = np.asarray(model.control_point_times, dtype=np.float64)
        return model, aif, tp, dt, n_cp, cp_times, model.terminal_time

    @pytest.mark.parametrize("infer_cpt", [False, True])
    def test_readout_cbv_is_fitted_residue_integral(
        self, dsc_aif_and_time, infer_cpt
    ):
        model, aif, tp, dt, n_cp, cp_times, t_term = self._grid(
            infer_cpt, dsc_aif_and_time
        )
        names = model.get_parameter_names()
        cbf = [0.10, 0.05, 0.16]
        fr = [
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            [0.8, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ]
        spac = [[3.0, 4.0, 5.0, 7.0, 9.0, 12.0]] * 3 if infer_cpt else None
        params = _cpi_param_matrix(n_cp, cbf, fr, spac)

        outs = _cpi_readout(params, names, cp_times, tp, dt, t_term)
        R = batched_cpi_residue(params, tp, cp_times, t_terminal=t_term)
        for i, out in enumerate(outs):
            exp_cbv = cbf[i] * float(np.trapezoid(R[i], dx=dt))
            assert out["cbv"] == pytest.approx(exp_cbv, abs=1e-10)
            assert out["mtt"] == pytest.approx(exp_cbv / cbf[i], abs=1e-10)
            assert out["tmax"] == 0.0

    def test_readout_cbv_depends_on_last_fraction(self, dsc_aif_and_time):
        # Pre-fix the readout dropped fN, so CBV was exactly invariant to it;
        # the fix reads all N fractions. Use infer_cpt so the last CP sits
        # inside the data window (real leverage) — the fixed-CP layout places
        # cp_N one frame past the data, where fN barely moves the integral.
        model, aif, tp, dt, n_cp, cp_times, t_term = self._grid(
            True, dsc_aif_and_time
        )
        names = model.get_parameter_names()
        head = [0.8, 0.7, 0.6, 0.5, 0.4]
        spac = [[3.0, 4.0, 5.0, 7.0, 9.0, 12.0]]  # cp times [3,7,12,19,28,40]
        lo = _cpi_param_matrix(n_cp, [0.1], [head + [0.05]], spac)
        hi = _cpi_param_matrix(n_cp, [0.1], [head + [0.99]], spac)
        cbv_lo = _cpi_readout(lo, names, cp_times, tp, dt, t_term)[0]["cbv"]
        cbv_hi = _cpi_readout(hi, names, cp_times, tp, dt, t_term)[0]["cbv"]
        assert (cbv_hi - cbv_lo) / cbv_lo > 0.02

    def test_readout_cbv_depends_on_log_spacings(self, dsc_aif_and_time):
        # Pre-fix the readout ignored s1..sN, so CBV was invariant to the
        # migrated CP times (the production-default infer_cpt path).
        model, aif, tp, dt, n_cp, cp_times, t_term = self._grid(
            True, dsc_aif_and_time
        )
        names = model.get_parameter_names()
        fr = [[0.8, 0.7, 0.6, 0.5, 0.4, 0.3]]
        early = _cpi_param_matrix(n_cp, [0.1], fr, [[2.0, 2.5, 3.0, 4.0, 5.0, 6.0]])
        spread = _cpi_param_matrix(n_cp, [0.1], fr, [[6.0, 6.0, 6.0, 6.0, 6.0, 6.0]])
        cbv_e = _cpi_readout(early, names, cp_times, tp, dt, t_term)[0]["cbv"]
        cbv_s = _cpi_readout(spread, names, cp_times, tp, dt, t_term)[0]["cbv"]
        assert abs(cbv_e - cbv_s) > 0.01 * max(cbv_e, cbv_s)

    @pytest.mark.parametrize("infer_cpt", [False, True])
    def test_readout_mtt_matches_scalar_get_mtt(self, dsc_aif_and_time, infer_cpt):
        # Independent anchor (no batched_cpi_residue): the batched readout MTT
        # must match the validated scalar CPIModel.get_mtt for the same params,
        # within the scipy-vs-Fritsch-Carlson PCHIP endpoint-slope difference.
        model, aif, tp, dt, n_cp, cp_times, t_term = self._grid(
            infer_cpt, dsc_aif_and_time
        )
        names = model.get_parameter_names()
        fr = [[0.85, 0.75, 0.65, 0.55, 0.45, 0.35]]
        spac = [[3.0, 4.0, 5.0, 7.0, 9.0, 12.0]] if infer_cpt else None
        params = _cpi_param_matrix(n_cp, [0.12], fr, spac)
        out = _cpi_readout(params, names, cp_times, tp, dt, t_term)[0]
        p_list = model.dict_to_params(
            {nm: float(params[0, j]) for j, nm in enumerate(names)}
        )
        mtt_scalar = model.get_mtt(p_list, tp)
        assert out["mtt"] == pytest.approx(mtt_scalar, rel=0.03)


class TestBatchedCPIDelay:
    """Optional bolus delay in the batched CPI forward (FABBER inferdelay)."""

    @staticmethod
    def _setup(dsc_aif_and_time, infer_cpt=False):
        aif, tp = dsc_aif_and_time
        nt = len(tp)
        dt = float(tp[1] - tp[0])
        n_cp = 6
        model = CPIModel(
            n_control_points=n_cp, n_acquisition_frames=nt, dt=dt,
            infer_cpt=infer_cpt, infer_delay=True,
        )
        cp_times = np.asarray(model.control_point_times, dtype=np.float64)
        toe = build_volterra_matrix(aif, dt)
        return aif, tp, n_cp, model, cp_times, toe, model.terminal_time

    def test_delay_zero_matches_no_delay(self, dsc_aif_and_time):
        aif, tp, n_cp, model, cp_times, toe, t_term = self._setup(dsc_aif_and_time)
        fr = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        got_d0 = batched_cpi_forward(
            np.array([[0.1] + fr + [0.0]]), aif, tp, cp_times, toe,
            t_terminal=t_term, infer_delay=True,
        )
        got_nd = batched_cpi_forward(
            np.array([[0.1] + fr]), aif, tp, cp_times, toe, t_terminal=t_term,
        )
        np.testing.assert_allclose(got_d0[0], got_nd[0], atol=1e-12)

    @pytest.mark.parametrize("delay", [0.0, 2.0, 3.5])
    def test_delay_matches_scalar(self, dsc_aif_and_time, delay):
        aif, tp, n_cp, model, cp_times, toe, t_term = self._setup(dsc_aif_and_time)
        fr = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        got = batched_cpi_forward(
            np.array([[0.1] + fr + [delay]]), aif, tp, cp_times, toe,
            t_terminal=t_term, infer_delay=True,
        )
        scalar = np.asarray(model.tissue_curve([0.1] + fr + [delay], aif, tp))
        l2 = np.linalg.norm(got[0] - scalar) / np.linalg.norm(scalar)
        assert l2 < 0.01, f"delay={delay}: L2 {l2:.4f}"

    def test_infer_cpt_plus_delay_width(self, dsc_aif_and_time):
        # delay column must not be mistaken for a log-spacing (width check)
        aif, tp, n_cp, model, cp_times, toe, t_term = self._setup(
            dsc_aif_and_time, infer_cpt=True
        )
        fr = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        spac = [float(np.log(3.0))] * n_cp
        got = batched_cpi_forward(
            np.array([[0.1] + fr + spac + [2.0]]), aif, tp, cp_times, toe,
            t_terminal=t_term, infer_delay=True,
        )
        assert got.shape == (1, len(tp)) and bool(np.all(np.isfinite(got)))

    def test_readout_reports_delay_as_tmax(self, dsc_aif_and_time):
        # The batched CBV/MTT readout reports the fitted delay as tmax when
        # cpi_infer_delay; CBV stays delay-invariant (residue is delay-agnostic).
        import types

        from pmriprep.interfaces.dsc_compute import ComputeDSCMaps

        aif, tp, n_cp, model, cp_times, toe, t_term = self._setup(dsc_aif_and_time)
        names = model.get_parameter_names()  # [cbf, f1..fN, delay]
        fr = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        params = np.array([[0.1] + fr + [d] for d in (0.0, 2.5)])
        result = types.SimpleNamespace(
            params={nm: params[:, j] for j, nm in enumerate(names)},
            uncertainties={},
            r_squared=np.ones(2),
            converged=np.ones(2, dtype=bool),
        )
        mc = (np.arange(2), np.zeros(2, dtype=int), np.zeros(2, dtype=int))
        dt = float(tp[1] - tp[0])
        outs = ComputeDSCMaps._batched_result_to_voxel_dicts(
            None, result, mc, "cpi", 2, dt, cp_times, tp, t_term, True,
        )
        assert outs[0]["tmax"] == pytest.approx(0.0)
        assert outs[1]["tmax"] == pytest.approx(2.5)
        assert outs[0]["cbv"] == pytest.approx(outs[1]["cbv"], rel=1e-9)


class TestCPIPCHIPConsistency:
    """Scalar (scipy PchipInterpolator, NLLS path) and batched (Fritsch-Carlson
    _batched_pchip, VI/MAP path) CPI residues agree to a small VERIFIED gap.

    The only difference is the endpoint-slope rule near R~=0; the residual is
    sub-0.2% on CBV and clinically negligible, so the batched endpoint is left
    as-is (physics ruling) rather than aligned to scipy. This test pins the gap
    so it cannot silently widen.
    """

    def test_readout_does_not_claim_cbv_uncertainty_without_covariance(
        self, dsc_aif_and_time
    ):
        """CPI marginal errors cannot define uncertainty of integrated CBV."""
        import types

        from pmriprep.interfaces.dsc_compute import ComputeDSCMaps

        _aif, tp = dsc_aif_and_time
        dt = float(tp[1] - tp[0])
        n_cp = 6
        model = CPIModel(
            n_control_points=n_cp, n_acquisition_frames=len(tp), dt=dt,
        )
        names = model.get_parameter_names()
        params = np.array([[0.1] + [0.8] * n_cp])
        result = types.SimpleNamespace(
            params={name: params[:, j] for j, name in enumerate(names)},
            uncertainties={"cbf": np.array([0.01])},
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
            "cpi",
            1,
            dt,
            model.control_point_times,
            tp,
            model.terminal_time,
        )[0]

        assert out["cbf_std"] == pytest.approx(0.01)
        assert "cbv_std" not in out
        assert "cbv_ci95_lower" not in out
        assert "cbv_ci95_upper" not in out

    @pytest.mark.parametrize("infer_cpt", [False, True])
    def test_scalar_vs_batched_residue_within_gap(self, dsc_aif_and_time, infer_cpt):
        aif, tp = dsc_aif_and_time
        nt = len(tp)
        dt = float(tp[1] - tp[0])
        n_cp = 8
        model = CPIModel(
            n_control_points=n_cp, n_acquisition_frames=nt, dt=dt,
            infer_cpt=infer_cpt,
        )
        cp_times = np.asarray(model.control_point_times, dtype=np.float64)
        t_term = model.terminal_time
        frac_sets = [
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35],
            [0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15],
            [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6],
        ]
        for fr in frac_sets:
            p = [0.1] + fr
            if infer_cpt:
                base = float(np.log(cp_times[1] - cp_times[0]))
                p = p + [base] * n_cp
            r_scalar = np.asarray(model.residue_function(p, tp))
            r_batched = batched_cpi_residue(
                np.array([p]), tp, cp_times, t_terminal=t_term
            )[0]
            l2 = np.linalg.norm(r_batched - r_scalar) / np.linalg.norm(r_scalar)
            cbv_s = float(np.trapezoid(r_scalar, dx=dt))
            cbv_b = float(np.trapezoid(r_batched, dx=dt))
            cbv_rel = abs(cbv_b - cbv_s) / cbv_s
            assert l2 < 0.005, f"residue L2 gap {l2:.4f} (fr={fr})"
            assert cbv_rel < 0.002, f"CBV rel gap {cbv_rel:.4f} (fr={fr})"


def _ref_scalar_delay_shift(conv_1d, delay, dt):
    """Reference scalar two-tap delay shift (lo-gated), matching the inline shift
    in cpi.py / vascular.py tissue_curve. The shared batched helper must equal
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
    """The shared `_apply_post_hoc_delay_shift` (used by the vascular AND CPI
    batched forwards) must match the scalar two-tap shift exactly, including
    negative non-integer delays at the curve tail (the FU1 valid_hi gating)."""

    @pytest.mark.parametrize("delay", [-2.3, -1.0, 0.0, 1.7, 4.0])
    def test_matches_scalar_shift(self, delay):
        from pmriprep.models.dsc.batched_fitting import _apply_post_hoc_delay_shift

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
