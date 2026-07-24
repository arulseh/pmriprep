"""Unit tests for spatial variational Bayes (SVB) wrapper."""

import numpy as np
import pytest

from pmriprep.models.dsc.spatial_vb import (
    _build_neighborhood_graph,
    _compute_spatial_contributions,
    _exclude_inadmissible_neighbors,
    _update_aK_penny,
)


class TestBuildNeighborhoodGraph:
    """Tests for 3D neighborhood graph construction."""

    def test_3x3x1_center_has_4_neighbors(self) -> None:
        """Center voxel of 3x3x1 grid has 4 in-plane neighbors."""
        coords = np.array([
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        adj, degree = _build_neighborhood_graph(
            (coords[0], coords[1], coords[2]),
            spatial_shape=(3, 3, 1),
        )
        center_idx = 4  # (1,1,0)
        assert degree[center_idx] == 4

    def test_corner_has_2_neighbors(self) -> None:
        """Corner voxel of 3x3x1 grid has 2 neighbors."""
        coords = np.array([
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        adj, degree = _build_neighborhood_graph(
            (coords[0], coords[1], coords[2]),
            spatial_shape=(3, 3, 1),
        )
        corner_idx = 0  # (0,0,0)
        assert degree[corner_idx] == 2

    def test_edge_has_3_neighbors(self) -> None:
        """Edge voxel of 3x3x1 grid has 3 neighbors."""
        coords = np.array([
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        adj, degree = _build_neighborhood_graph(
            (coords[0], coords[1], coords[2]),
            spatial_shape=(3, 3, 1),
        )
        edge_idx = 1  # (0,1,0)
        assert degree[edge_idx] == 3

    def test_3d_center_has_6_neighbors(self) -> None:
        """Center voxel of 3x3x3 grid has 6 neighbors."""
        x, y, z = np.mgrid[0:3, 0:3, 0:3]
        coords = (x.ravel(), y.ravel(), z.ravel())
        adj, degree = _build_neighborhood_graph(coords, spatial_shape=(3, 3, 3))
        center_idx = 13  # (1,1,1) in flattened 3x3x3
        assert degree[center_idx] == 6

    def test_sparse_mask_skips_missing_neighbors(self) -> None:
        """Non-contiguous mask: neighbors outside mask are excluded."""
        coords = (
            np.array([0, 2]),
            np.array([0, 0]),
            np.array([0, 0]),
        )
        adj, degree = _build_neighborhood_graph(coords, spatial_shape=(3, 1, 1))
        assert degree[0] == 0
        assert degree[1] == 0

    def test_adjacency_symmetric(self) -> None:
        """Adjacency matrix is symmetric."""
        x, y, z = np.mgrid[0:4, 0:4, 0:2]
        coords = (x.ravel(), y.ravel(), z.ravel())
        adj, _ = _build_neighborhood_graph(coords, spatial_shape=(4, 4, 2))
        diff = adj - adj.T
        assert diff.nnz == 0

    def test_inadmissible_voxel_is_removed_from_neighbor_lists(self) -> None:
        """Rejected curves neither receive nor contribute spatial information."""
        coords = (
            np.array([0, 1, 2]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
        )
        adj, _ = _build_neighborhood_graph(coords, spatial_shape=(3, 1, 1))

        pruned, degree = _exclude_inadmissible_neighbors(
            adj, np.array([True, False, True])
        )

        assert pruned.nnz == 0
        np.testing.assert_array_equal(degree, np.zeros(3, dtype=np.int64))


class TestSpatialContributions:
    """Tests for spatial prior contributions."""

    def test_uniform_posteriors(self) -> None:
        """Uniform posteriors: eta pulls toward common value; prec ~ aK * (degree+eps)."""
        n_voxels = 9
        n_params = 2
        coords = np.array([
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        adj, degree = _build_neighborhood_graph(
            (coords[0], coords[1], coords[2]),
            spatial_shape=(3, 3, 1),
        )
        posteriors = np.full((n_voxels, n_params), 5.0)
        aK = np.array([1.0, 2.0])

        prec, eta = _compute_spatial_contributions(posteriors, adj, degree, aK)

        center_idx = 4
        # FABBER M-prior: prec = aK * (degree + 1e-8), so center has 4+1e-8 ≈ 4.
        assert prec[center_idx, 0] == pytest.approx(1.0 * 4, rel=1e-6)
        assert prec[center_idx, 1] == pytest.approx(2.0 * 4, rel=1e-6)
        assert eta[center_idx, 0] == pytest.approx(1.0 * 4 * 5.0)
        assert eta[center_idx, 1] == pytest.approx(2.0 * 4 * 5.0)

    def test_zero_aK_no_contribution(self) -> None:
        """Zero aK produces zero spatial contributions."""
        coords = (np.array([0, 1]), np.array([0, 0]), np.array([0, 0]))
        adj, degree = _build_neighborhood_graph(coords, spatial_shape=(2, 1, 1))
        posteriors = np.array([[1.0], [2.0]])
        aK = np.array([0.0])

        prec, eta = _compute_spatial_contributions(posteriors, adj, degree, aK)
        np.testing.assert_allclose(prec, 0.0)
        np.testing.assert_allclose(eta, 0.0)

    def test_nonuniform_boundary_matches_fabber_m_equation(self) -> None:
        """Boundary-corrected precision and information use actual neighbors."""
        coords = (
            np.array([0, 1, 2]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
        )
        adj, degree = _build_neighborhood_graph(coords, spatial_shape=(3, 1, 1))
        posteriors = np.array([[1.0], [4.0], [9.0]])

        prec, eta = _compute_spatial_contributions(
            posteriors, adj, degree, np.array([2.5])
        )

        np.testing.assert_allclose(prec[:, 0], 2.5 * (degree + 1e-8))
        np.testing.assert_allclose(eta[:, 0], 2.5 * np.array([4.0, 10.0, 4.0]))


class TestUpdateAKPenny:
    """Tests for FABBER M-prior aK closed-form update (Penny 2005)."""

    def test_homogeneous_field_high_aK(self) -> None:
        """Uniform posterior field produces high aK (strong coupling)."""
        n_voxels = 9
        coords = np.array([
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        adj, degree = _build_neighborhood_graph(
            (coords[0], coords[1], coords[2]),
            spatial_shape=(3, 3, 1),
        )
        posteriors = np.full((n_voxels, 1), 3.0)
        cov_diag = np.full((n_voxels, 1), 0.01)

        aK = _update_aK_penny(posteriors, cov_diag, adj, degree)
        # Uniform field: m^T L m ≈ 0; trace term tiny; aK ~ q1 * (N/2 + q2).
        assert aK[0] > 10.0

    def test_heterogeneous_field_low_aK(self) -> None:
        """Highly variable posterior field produces low aK (weak coupling)."""
        n_voxels = 9
        coords = np.array([
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        adj, degree = _build_neighborhood_graph(
            (coords[0], coords[1], coords[2]),
            spatial_shape=(3, 3, 1),
        )
        rng = np.random.default_rng(42)
        posteriors = rng.normal(0, 10, (n_voxels, 1))
        cov_diag = np.full((n_voxels, 1), 1.0)

        aK = _update_aK_penny(posteriors, cov_diag, adj, degree)
        # m^T L m large for noise-like fields → aK small.
        assert aK[0] < 1.0

    def test_isolated_voxels_return_finite(self) -> None:
        """No-neighbour voxels still yield finite aK (driven by hyperprior)."""
        coords = (np.array([0, 5]), np.array([0, 5]), np.array([0, 5]))
        adj, degree = _build_neighborhood_graph(coords, spatial_shape=(10, 10, 10))
        posteriors = np.array([[1.0], [2.0]])
        cov_diag = np.full((2, 1), 0.1)

        aK = _update_aK_penny(posteriors, cov_diag, adj, degree)
        assert np.isfinite(aK[0])
        assert aK[0] > 0.0

    def test_closed_form_matches_literal_fabber_loop(self) -> None:
        """Vectorized aK update matches FABBER's per-voxel M-prior loop."""
        coords = (
            np.array([0, 1, 2]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
        )
        adj, degree = _build_neighborhood_graph(coords, spatial_shape=(3, 1, 1))
        means = np.array([[1.0], [4.0], [9.0]])
        variances = np.array([[0.2], [0.3], [0.4]])

        trace_term = 0.0
        quadratic_term = 0.0
        for voxel in range(3):
            neighbors = adj.indices[adj.indptr[voxel]:adj.indptr[voxel + 1]]
            trace_term += variances[voxel, 0] * (len(neighbors) + 1e-8)
            difference_sum = sum(
                means[voxel, 0] - means[neighbor, 0]
                for neighbor in neighbors
            )
            quadratic_term += means[voxel, 0] * difference_sum
        expected = (
            (0.5 * len(means) + 1.0)
            / (0.5 * trace_term + 0.5 * quadratic_term + 0.1)
        )

        actual = _update_aK_penny(means, variances, adj, degree)
        assert actual[0] == pytest.approx(expected)


class TestFabberMPriorContract:
    """Characterize FABBER M-prior replacement and sweep state."""

    def test_m_prior_replaces_independent_prior(self) -> None:
        """M-prior retains structural constraints but drops ordinary shrinkage."""
        from pmriprep.models.dsc.batched_fitting import _build_vi_update_prior_terms

        prior_prec = np.diag([4.0, 9.0])
        prior_means = np.array([2.0, -3.0])
        mono_prec = np.array([[1.0, -1.0], [-1.0, 1.0]])
        mono_eta = np.array([0.25, -0.25])

        update_prec, update_eta = _build_vi_update_prior_terms(
            prior_prec + mono_prec,
            prior_means,
            mono_prec,
            mono_eta,
            spatial_replaces_parameter_prior=True,
        )

        np.testing.assert_allclose(update_prec, mono_prec)
        np.testing.assert_allclose(
            update_eta, mono_prec @ prior_means + mono_eta
        )

        selected_prec, selected_eta = _build_vi_update_prior_terms(
            prior_prec + mono_prec,
            prior_means,
            mono_prec,
            mono_eta,
            spatial_replaces_parameter_prior=True,
            spatial_parameter_mask=np.array([True, False]),
        )
        expected_prec = mono_prec + np.diag([0.0, 9.0])
        np.testing.assert_allclose(selected_prec, expected_prec)
        np.testing.assert_allclose(
            selected_eta, expected_prec @ prior_means + mono_eta
        )

    def test_default_schedule_uses_ten_fixed_sweeps(self) -> None:
        """SVB uses ten sweeps unless an explicit tolerance is requested."""
        import inspect

        from pmriprep.models.dsc.spatial_vb import batched_dsc_spatial_vi_fit

        parameters = inspect.signature(batched_dsc_spatial_vi_fit).parameters
        assert parameters["n_spatial_sweeps"].default == 10
        assert parameters["spatial_tol"].default is None

    def test_stateful_one_update_sweeps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each sweep receives the preceding posterior and noise precision."""
        from pmriprep.models.bayesian_utils import TransformType
        from pmriprep.models.dsc.batched_fitting import DSCBatchedFitResult
        from pmriprep.models.dsc import spatial_vb

        calls: list[dict] = []

        def fake_vi_fit(**kwargs: object) -> DSCBatchedFitResult:
            calls.append(dict(kwargs))
            initial = kwargs.get("initial_means_transformed")
            means = (
                np.zeros((2, 1), dtype=float)
                if initial is None else np.asarray(initial, dtype=float)
            ) + 1.0
            n_voxels = means.shape[0]
            return DSCBatchedFitResult(
                params={"cbf": means[:, 0]},
                r_squared=np.ones(n_voxels),
                converged=np.zeros(n_voxels, dtype=bool),
                fitted_curves=None,
                n_iterations=1,
                model_name="vascular",
                uncertainties={"noise_std": np.full(n_voxels, 0.5)},
                posterior_means_transformed=means,
                posterior_cov_diag=np.ones_like(means),
            )

        monkeypatch.setattr(spatial_vb, "batched_dsc_vi_fit", fake_vi_fit)
        spatial_result = spatial_vb.batched_dsc_spatial_vi_fit(
            model_type="vascular",
            tissue_curves=np.ones((2, 3)),
            aif=np.ones(3),
            time_points=np.arange(3, dtype=float),
            mask_coords=(
                np.array([0, 1]),
                np.array([0, 0]),
                np.array([0, 0]),
            ),
            spatial_shape=(2, 1, 1),
            prior_means_nat=np.array([0.0]),
            prior_stds_nat=np.array([1.0]),
            transforms=[TransformType.IDENTITY],
            upper_bounds=[None],
            fitting_param_names=["cbf"],
            fitting_bounds=[(-10.0, 10.0)],
            n_spatial_sweeps=3,
        )

        assert len(calls) == 4
        assert spatial_result.n_iterations == 3
        spatial_calls = calls[1:]
        assert all(call["n_iterations_override"] == 1 for call in spatial_calls)
        assert all(call["spatial_replaces_parameter_prior"] for call in spatial_calls)
        assert all(call["spatial_parameter_mask"][0] for call in spatial_calls)
        np.testing.assert_allclose(
            [call["initial_means_transformed"][0, 0] for call in spatial_calls],
            [1.0, 2.0, 3.0],
        )
        assert all(call["initial_cov_diag"][0, 0] == 1.0 for call in spatial_calls)
        assert all(
            call["initial_noise_precision"][0] == 4.0 for call in spatial_calls
        )
        np.testing.assert_allclose(spatial_result.params["cbf"], 4.0)


class TestSpatialVIIntegration:
    """Integration tests for spatial VB wrapper."""

    def _make_synthetic_data(
        self,
        n_side: int = 5,
        n_timepoints: int = 40,
        noise_std: float = 0.005,
        cbf: float = 0.1,
    ) -> tuple:
        """Create synthetic NxNx1 homogeneous phantom."""
        from pmriprep.models.dsc.cpi import CPIModel
        from pmriprep.utils.dsc_deconvolution import build_volterra_matrix

        n_voxels = n_side * n_side
        t = np.linspace(0, 40, n_timepoints)
        dt = float(t[1] - t[0])
        aif = (t / 3.0) * np.exp(-t / 5.0)
        aif[0] = 0.0

        model = CPIModel(
            n_control_points=6,
            n_acquisition_frames=n_timepoints,
            dt=dt,
        )
        toeplitz = build_volterra_matrix(aif, dt)

        rng = np.random.default_rng(42)
        params = [cbf, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]  # cbf + 6 fractions
        clean_curve = model.tissue_curve(params, aif, t)
        tissue_curves = np.tile(clean_curve, (n_voxels, 1))
        tissue_curves += rng.normal(0, noise_std, (n_voxels, n_timepoints))

        x, y = np.mgrid[0:n_side, 0:n_side]
        mask_coords = (x.ravel(), y.ravel(), np.zeros(n_voxels, dtype=int))

        return tissue_curves, aif, t, dt, mask_coords, model, toeplitz

    def test_spatial_vi_runs_without_error(self) -> None:
        """Spatial VI wrapper completes without errors."""
        from pmriprep.models.bayesian_utils import build_priors_from_params
        from pmriprep.models.dsc.spatial_vb import batched_dsc_spatial_vi_fit

        tissue_curves, aif, t, dt, mask_coords, model, toeplitz = (
            self._make_synthetic_data()
        )
        fitting_defs = list(model.get_fitting_parameter_definitions())
        priors = build_priors_from_params(fitting_defs)
        fitting_names = [p.name for p in fitting_defs]
        fitting_bounds = [(p.bounds[0], p.bounds[1]) for p in fitting_defs]
        cp_times = np.array(model._control_point_times, dtype=np.float64)
        means_nat = np.array([p.mean for p in priors], dtype=np.float64)
        stds_nat = np.array([p.std for p in priors], dtype=np.float64)
        means_t = np.array([p.mean_transformed for p in priors], dtype=np.float64)
        stds_t = np.array([p.std_transformed for p in priors], dtype=np.float64)
        transforms = [p.transform for p in priors]
        ubs: list[float | None] = [p.upper_bound for p in priors]

        result = batched_dsc_spatial_vi_fit(
            model_type="cpi",
            tissue_curves=tissue_curves,
            aif=aif,
            time_points=t,
            mask_coords=mask_coords,
            spatial_shape=(5, 5, 1),
            prior_means_nat=means_nat,
            prior_stds_nat=stds_nat,
            transforms=transforms,
            upper_bounds=ubs,
            fitting_param_names=fitting_names,
            fitting_bounds=fitting_bounds,
            mono_prec=None,
            mono_eta=None,
            control_point_times=cp_times,
            toeplitz_matrix=toeplitz,
            prior_means_transformed=means_t,
            prior_stds_transformed=stds_t,
            sloppy=True,
            save_fitted_curves=True,
            n_spatial_sweeps=3,
        )

        assert result.params["cbf"].shape == (25,)
        assert result.fitted_curves is not None
        assert np.all(np.isfinite(result.params["cbf"]))

    def test_spatial_vi_preserves_cpi_fit_quality(self) -> None:
        """CPI spatial updates remain finite and retain useful CBF recovery."""
        from pmriprep.models.bayesian_utils import build_priors_from_params
        from pmriprep.models.dsc.spatial_vb import batched_dsc_spatial_vi_fit

        tissue_curves, aif, t, dt, mask_coords, model, toeplitz = (
            self._make_synthetic_data(n_side=6, noise_std=0.01)
        )
        fitting_defs = list(model.get_fitting_parameter_definitions())
        priors = build_priors_from_params(fitting_defs)
        fitting_names = [p.name for p in fitting_defs]
        fitting_bounds = [(p.bounds[0], p.bounds[1]) for p in fitting_defs]
        cp_times = np.array(model._control_point_times, dtype=np.float64)
        means_nat = np.array([p.mean for p in priors], dtype=np.float64)
        stds_nat = np.array([p.std for p in priors], dtype=np.float64)
        means_t = np.array([p.mean_transformed for p in priors], dtype=np.float64)
        stds_t = np.array([p.std_transformed for p in priors], dtype=np.float64)
        transforms = [p.transform for p in priors]
        ubs: list[float | None] = [p.upper_bound for p in priors]

        common_kwargs: dict = dict(
            model_type="cpi",
            tissue_curves=tissue_curves,
            aif=aif,
            time_points=t,
            prior_means_nat=means_nat,
            prior_stds_nat=stds_nat,
            transforms=transforms,
            upper_bounds=ubs,
            fitting_param_names=fitting_names,
            fitting_bounds=fitting_bounds,
            mono_prec=None,
            mono_eta=None,
            control_point_times=cp_times,
            toeplitz_matrix=toeplitz,
            prior_means_transformed=means_t,
            prior_stds_transformed=stds_t,
        )

        result_svb = batched_dsc_spatial_vi_fit(
            **common_kwargs,
            mask_coords=mask_coords,
            spatial_shape=(6, 6, 1),
            save_fitted_curves=False,
        )

        cbf = np.asarray(result_svb.params["cbf"])
        assert np.all(np.isfinite(cbf))
        assert np.median(np.abs(cbf - 0.1) / 0.1) < 0.25
        assert np.median(result_svb.r_squared) > 0.5

    def test_cpi_synchronous_update_is_translation_invariant(self) -> None:
        """A one-voxel coordinate translation leaves CPI SVB unchanged."""
        from pmriprep.models.bayesian_utils import build_priors_from_params
        from pmriprep.models.dsc.batched_fitting import batched_cpi_forward
        from pmriprep.models.dsc.cpi import CPIModel
        from pmriprep.models.dsc.spatial_vb import batched_dsc_spatial_vi_fit
        from pmriprep.utils.dsc_deconvolution import build_volterra_matrix

        side = 4
        n_voxels = side**2
        time_points = np.linspace(0.0, 40.0, 40)
        dt = float(time_points[1] - time_points[0])
        aif = (time_points / 3.0) * np.exp(-time_points / 5.0)
        aif[0] = 0.0
        model = CPIModel(
            n_control_points=6,
            n_acquisition_frames=len(time_points),
            dt=dt,
        )
        convolution = build_volterra_matrix(aif, dt)
        cbf_truth = np.linspace(0.025, 0.14, n_voxels)
        fractions = np.linspace(0.65, 0.92, n_voxels)
        params = np.column_stack(
            [cbf_truth, np.repeat(fractions[:, None], 6, axis=1)]
        )
        tissue = batched_cpi_forward(
            params,
            aif,
            time_points,
            np.asarray(model._control_point_times),
            convolution,
            t_terminal=model.terminal_time,
        )
        tissue += np.random.default_rng(20260725).normal(
            0.0, 0.008, tissue.shape
        )

        definitions = list(model.get_fitting_parameter_definitions())
        priors = build_priors_from_params(definitions)
        common = dict(
            model_type="cpi",
            tissue_curves=tissue,
            aif=aif,
            time_points=time_points,
            spatial_shape=(side, side, 1),
            prior_means_nat=np.array([prior.mean for prior in priors]),
            prior_stds_nat=np.array([prior.std for prior in priors]),
            transforms=[prior.transform for prior in priors],
            upper_bounds=[prior.upper_bound for prior in priors],
            fitting_param_names=[definition.name for definition in definitions],
            fitting_bounds=[
                (definition.bounds[0], definition.bounds[1])
                for definition in definitions
            ],
            control_point_times=np.asarray(model._control_point_times),
            toeplitz_matrix=convolution,
            cpi_t_terminal=model.terminal_time,
            prior_means_transformed=np.array(
                [prior.mean_transformed for prior in priors]
            ),
            prior_stds_transformed=np.array(
                [prior.std_transformed for prior in priors]
            ),
        )
        x, y = np.mgrid[0:side, 0:side]
        coords = (x.ravel(), y.ravel(), np.zeros(n_voxels, dtype=int))
        result_a = batched_dsc_spatial_vi_fit(
            **common,
            mask_coords=coords,
        )
        shifted_common = dict(common)
        shifted_common["spatial_shape"] = (side + 1, side, 1)
        result_b = batched_dsc_spatial_vi_fit(
            **shifted_common,
            mask_coords=(coords[0] + 1, coords[1], coords[2]),
        )

        cbf_a = np.asarray(result_a.params["cbf"])
        cbf_b = np.asarray(result_b.params["cbf"])
        np.testing.assert_allclose(cbf_a, cbf_b, rtol=1e-12, atol=1e-12)

        def _mtt(result: object) -> np.ndarray:
            result_params = getattr(result, "params")
            values = np.column_stack(
                [
                    result_params["cbf"],
                    *[result_params[f"f{i}"] for i in range(1, 7)],
                ]
            )
            return np.array([model.get_mtt(row) for row in values])

        np.testing.assert_allclose(
            _mtt(result_a), _mtt(result_b), rtol=1e-12, atol=1e-12
        )

    def test_batched_ordering_agrees_with_sequential_reference(self) -> None:
        """Ten-sweep Jacobi and literal FABBER-order fits agree on a small DRO."""
        from pmriprep.models.bayesian_utils import build_priors_from_params
        from pmriprep.models.dsc import VascularModel
        from pmriprep.models.dsc.batched_fitting import (
            batched_dsc_vi_fit,
            batched_vascular_forward,
        )
        from pmriprep.models.dsc.spatial_vb import batched_dsc_spatial_vi_fit

        model = VascularModel()
        fitting_defs = list(model.get_fitting_parameter_definitions())
        priors = build_priors_from_params(fitting_defs)
        names = [parameter.name for parameter in fitting_defs]
        bounds = [(parameter.bounds[0], parameter.bounds[1]) for parameter in fitting_defs]
        means_nat = np.array([prior.mean for prior in priors])
        stds_nat = np.array([prior.std for prior in priors])
        means_t = np.array([prior.mean_transformed for prior in priors])
        stds_t = np.array([prior.std_transformed for prior in priors])
        transforms = [prior.transform for prior in priors]
        upper_bounds = [prior.upper_bound for prior in priors]

        time_points = np.linspace(0.0, 36.0, 37)
        aif = (time_points / 3.0) ** 3 * np.exp(-time_points / 3.0)
        aif[0] = 0.0
        cbf = np.full(9, 0.06)
        mtt = np.full(9, 5.0)
        cbf[4] = 0.11
        mtt[4] = 9.0
        alpha = np.full(9, 3.0)
        beta = mtt / alpha
        delay = np.zeros(9)
        tissue = batched_vascular_forward(
            cbf, alpha, beta, delay, aif, time_points
        )
        tissue += np.random.default_rng(42).normal(0.0, 0.002, tissue.shape)
        x, y = np.mgrid[0:3, 0:3]
        coords = (x.ravel(), y.ravel(), np.zeros(9, dtype=int))

        common_kwargs = dict(
            model_type="vascular",
            aif=aif,
            time_points=time_points,
            prior_means_nat=means_nat,
            prior_stds_nat=stds_nat,
            transforms=transforms,
            upper_bounds=upper_bounds,
            fitting_param_names=names,
            fitting_bounds=bounds,
            prior_means_transformed=means_t,
            prior_stds_transformed=stds_t,
        )
        jacobi = batched_dsc_spatial_vi_fit(
            **common_kwargs,
            tissue_curves=tissue,
            mask_coords=coords,
            spatial_shape=(3, 3, 1),
        )

        adj, degree = _build_neighborhood_graph(coords, (3, 3, 1))
        warm = batched_dsc_vi_fit(
            **common_kwargs,
            tissue_curves=tissue,
            return_posteriors=True,
        )
        assert warm.posterior_means_transformed is not None
        assert warm.posterior_cov_diag is not None
        posterior_means = warm.posterior_means_transformed.copy()
        posterior_cov = warm.posterior_cov_diag.copy()
        warm_noise_std = warm.uncertainties["noise_std"]
        noise_precision = 1.0 / warm_noise_std**2
        spatial_parameter_mask = np.array([name == "cbf" for name in names])
        a_k = np.zeros(len(names))
        a_k[spatial_parameter_mask] = 1e-8
        sequential_params = {name: np.zeros(9) for name in names}

        for sweep in range(10):
            if sweep > 0:
                updated_a_k = _update_aK_penny(
                    posterior_means, posterior_cov, adj, degree
                )
                a_k[spatial_parameter_mask] = updated_a_k[
                    spatial_parameter_mask
                ]
            for voxel in range(9):
                spatial_prec, spatial_eta = _compute_spatial_contributions(
                    posterior_means, adj, degree, a_k
                )
                result = batched_dsc_vi_fit(
                    **common_kwargs,
                    tissue_curves=tissue[voxel:voxel + 1],
                    spatial_prec_diag=spatial_prec[voxel:voxel + 1],
                    spatial_eta=spatial_eta[voxel:voxel + 1],
                    spatial_replaces_parameter_prior=True,
                    spatial_parameter_mask=spatial_parameter_mask,
                    initial_means_transformed=posterior_means[voxel:voxel + 1],
                    initial_noise_precision=noise_precision[voxel:voxel + 1],
                    return_posteriors=True,
                    n_iterations_override=1,
                )
                assert result.posterior_means_transformed is not None
                assert result.posterior_cov_diag is not None
                posterior_means[voxel] = result.posterior_means_transformed[0]
                posterior_cov[voxel] = result.posterior_cov_diag[0]
                noise_std = result.uncertainties["noise_std"][0]
                noise_precision[voxel] = 1.0 / noise_std**2
                for name in names:
                    sequential_params[name][voxel] = result.params[name][0]
        for name in ("cbf", "mtt"):
            reference = np.maximum(np.abs(sequential_params[name]), 1e-8)
            relative_difference = np.abs(
                jacobi.params[name] - sequential_params[name]
            ) / reference
            assert np.max(relative_difference) < 1e-6

        assert np.median(np.abs(jacobi.params["cbf"] - cbf) / cbf) < 0.25
        assert np.median(np.abs(jacobi.params["mtt"] - mtt) / mtt) < 0.35

    def test_fallback_single_voxel(self) -> None:
        """Single-voxel mask falls back to standard VI."""
        from pmriprep.models.bayesian_utils import build_priors_from_params
        from pmriprep.models.dsc.spatial_vb import batched_dsc_spatial_vi_fit

        t = np.linspace(0, 40, 40)
        aif = (t / 3.0) * np.exp(-t / 5.0)
        aif[0] = 0.0

        from pmriprep.models.dsc.cpi import CPIModel
        from pmriprep.utils.dsc_deconvolution import build_volterra_matrix

        dt = float(t[1] - t[0])
        model = CPIModel(
            n_control_points=6,
            n_acquisition_frames=int(len(t)),
            dt=dt,
        )
        toeplitz = build_volterra_matrix(aif, dt)
        params = [0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]  # cbf + 6 fractions
        tissue = model.tissue_curve(params, aif, t).reshape(1, -1)

        fitting_defs = list(model.get_fitting_parameter_definitions())
        priors = build_priors_from_params(fitting_defs)
        fitting_names = [p.name for p in fitting_defs]
        fitting_bounds = [(p.bounds[0], p.bounds[1]) for p in fitting_defs]
        cp_times = np.array(model._control_point_times, dtype=np.float64)

        result = batched_dsc_spatial_vi_fit(
            model_type="cpi",
            tissue_curves=tissue,
            aif=aif,
            time_points=t,
            mask_coords=(np.array([0]), np.array([0]), np.array([0])),
            spatial_shape=(1, 1, 1),
            prior_means_nat=np.array([p.mean for p in priors]),
            prior_stds_nat=np.array([p.std for p in priors]),
            transforms=[p.transform for p in priors],
            upper_bounds=[p.upper_bound for p in priors],
            fitting_param_names=fitting_names,
            fitting_bounds=fitting_bounds,
            control_point_times=cp_times,
            toeplitz_matrix=toeplitz,
            prior_means_transformed=np.array([p.mean_transformed for p in priors]),
            prior_stds_transformed=np.array([p.std_transformed for p in priors]),
            sloppy=True,
        )
        assert result.params["cbf"].shape == (1,)

    def test_non_sloppy_inner_iterations(self) -> None:
        """Non-sloppy spatial VI (4 inner iters) runs without error."""
        from pmriprep.models.bayesian_utils import build_priors_from_params
        from pmriprep.models.dsc.spatial_vb import batched_dsc_spatial_vi_fit

        tissue_curves, aif, t, dt, mask_coords, model, toeplitz = (
            self._make_synthetic_data(n_side=3)
        )
        fitting_defs = list(model.get_fitting_parameter_definitions())
        priors = build_priors_from_params(fitting_defs)
        fitting_names = [p.name for p in fitting_defs]
        fitting_bounds = [(p.bounds[0], p.bounds[1]) for p in fitting_defs]
        cp_times = np.array(model._control_point_times, dtype=np.float64)

        result = batched_dsc_spatial_vi_fit(
            model_type="cpi",
            tissue_curves=tissue_curves,
            aif=aif,
            time_points=t,
            mask_coords=mask_coords,
            spatial_shape=(3, 3, 1),
            prior_means_nat=np.array([p.mean for p in priors]),
            prior_stds_nat=np.array([p.std for p in priors]),
            transforms=[p.transform for p in priors],
            upper_bounds=[p.upper_bound for p in priors],
            fitting_param_names=fitting_names,
            fitting_bounds=fitting_bounds,
            control_point_times=cp_times,
            toeplitz_matrix=toeplitz,
            prior_means_transformed=np.array([p.mean_transformed for p in priors]),
            prior_stds_transformed=np.array([p.std_transformed for p in priors]),
            sloppy=False,
            n_spatial_sweeps=2,
        )
        assert result.params["cbf"].shape == (9,)
        assert np.all(np.isfinite(result.params["cbf"]))
