"""Unit tests for spatial variational Bayes (SVB) wrapper."""

import numpy as np
import pytest

from pmriprep.models.dsc.spatial_vb import (
    _build_neighborhood_graph,
    _compute_spatial_contributions,
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
        from pmriprep.utils.dsc_deconvolution import build_toeplitz_matrix

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
        toeplitz = build_toeplitz_matrix(aif, dt)

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

    def test_spatial_vi_reduces_variance(self) -> None:
        """Spatial VI produces lower voxel-wise CBF variance than standard VI."""
        from pmriprep.models.bayesian_utils import build_priors_from_params
        from pmriprep.models.dsc.batched_fitting import batched_dsc_vi_fit
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
            sloppy=True,
        )

        result_vi = batched_dsc_vi_fit(**common_kwargs, save_fitted_curves=False)
        result_svb = batched_dsc_spatial_vi_fit(
            **common_kwargs,
            mask_coords=mask_coords,
            spatial_shape=(6, 6, 1),
            save_fitted_curves=False,
            n_spatial_sweeps=6,
        )

        cbf_vi_var = np.var(result_vi.params["cbf"])
        cbf_svb_var = np.var(result_svb.params["cbf"])
        assert cbf_svb_var <= cbf_vi_var, (
            f"SVB variance ({cbf_svb_var:.6f}) should be <= VI variance ({cbf_vi_var:.6f})"
        )

    def test_fallback_single_voxel(self) -> None:
        """Single-voxel mask falls back to standard VI."""
        from pmriprep.models.bayesian_utils import build_priors_from_params
        from pmriprep.models.dsc.spatial_vb import batched_dsc_spatial_vi_fit

        t = np.linspace(0, 40, 40)
        aif = (t / 3.0) * np.exp(-t / 5.0)
        aif[0] = 0.0

        from pmriprep.models.dsc.cpi import CPIModel
        from pmriprep.utils.dsc_deconvolution import build_toeplitz_matrix

        dt = float(t[1] - t[0])
        model = CPIModel(
            n_control_points=6,
            n_acquisition_frames=int(len(t)),
            dt=dt,
        )
        toeplitz = build_toeplitz_matrix(aif, dt)
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
