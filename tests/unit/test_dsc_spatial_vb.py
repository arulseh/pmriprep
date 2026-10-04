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
                model_name="linear",
                uncertainties={"noise_std": np.full(n_voxels, 0.5)},
                posterior_means_transformed=means,
                posterior_cov_diag=np.ones_like(means),
            )

        monkeypatch.setattr(spatial_vb, "batched_dsc_vi_fit", fake_vi_fit)
        spatial_result = spatial_vb.batched_dsc_spatial_vi_fit(
            model_type="linear",
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


@pytest.mark.parametrize("sloppy", [False, True])
def test_adapter_spatial_translation_and_permutation(sloppy: bool) -> None:
    """Graph translation and row ordering preserve the same complete field."""
    from tests.unit.test_dsc_batched_fitting import linear_arguments
    from pmriprep.models.dsc.spatial_vb import batched_dsc_spatial_vi_fit

    args = linear_arguments(9)
    args.pop("return_full_posterior")
    x, y = np.mgrid[0:3, 0:3]
    coords = (x.ravel(), y.ravel(), np.zeros(9, int))
    original = batched_dsc_spatial_vi_fit(**args, mask_coords=coords,
        spatial_shape=(3,3,1), sloppy=sloppy)
    order = np.array([8,3,2,5,4,7,1,0,6])
    args["tissue_curves"] = args["tissue_curves"][order]
    shifted = batched_dsc_spatial_vi_fit(**args,
        mask_coords=(coords[0][order]+1, coords[1][order], coords[2][order]),
        spatial_shape=(4,3,1), sloppy=sloppy)
    for key in ("cbf", "offset"):
        np.testing.assert_allclose(shifted.params[key][np.argsort(order)],
            original.params[key], rtol=1e-8, atol=1e-10)
    assert np.all(original.r_squared > .95)


def test_single_voxel_matches_vi_warm_start() -> None:
    """An isolated field preserves the ordinary local posterior and predictions."""
    from tests.unit.test_dsc_batched_fitting import linear_arguments
    from pmriprep.models.dsc.batched_fitting import batched_dsc_vi_fit
    from pmriprep.models.dsc.spatial_vb import batched_dsc_spatial_vi_fit

    args = linear_arguments(1)
    local = batched_dsc_vi_fit(**args, return_posteriors=True)
    args.pop("return_full_posterior")
    spatial = batched_dsc_spatial_vi_fit(**args,
        mask_coords=(np.zeros(1,int),)*3, spatial_shape=(1,1,1))
    for name in ("cbf", "offset"):
        np.testing.assert_array_equal(local.params[name], spatial.params[name])
        np.testing.assert_array_equal(local.uncertainties[name], spatial.uncertainties[name])
    np.testing.assert_array_equal(local.fitted_curves, spatial.fitted_curves)
