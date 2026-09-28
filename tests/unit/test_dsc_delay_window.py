"""Finite-window delay checks against complete causal convolution."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaincc

from pmriprep.models.dsc import VascularModel
from pmriprep.models.dsc.batched_fitting import (
    batched_vascular_forward,
)
from pmriprep.models.dsc.convolution import (
    _apply_post_hoc_delay_shift,
    _volterra_tail_matrix,
)
from pmriprep.utils.dsc_deconvolution import build_volterra_matrix


def _aif(times: np.ndarray) -> np.ndarray:
    """Compact bolus with no unobserved future input."""
    return np.where(
        (times >= 5.0) & (times <= 15.0),
        np.sin(np.pi * (times - 5.0) / 10.0) ** 2,
        0.0,
    )


@pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("delay", [-3.0, -1.3])
def test_vascular_negative_delay_tail_matches_continuous_reference(
    dt: float, delay: float
) -> None:
    """Previously delivered tracer persists through the end of acquisition."""
    times = np.arange(0.0, 60.0, dt)
    aif = _aif(times)
    cbf, alpha, beta = 0.006, 3.0, 25.0 / 3.0
    scalar = VascularModel().tissue_curve([cbf, alpha, beta, delay], aif, times)
    batched = batched_vascular_forward(
        np.array([cbf]), np.array([alpha]), np.array([beta]), np.array([delay]),
        aif, times, volterra_matrix=build_volterra_matrix(aif, dt),
    )[0]
    np.testing.assert_allclose(batched, scalar, rtol=1e-12, atol=1e-14)
    reference = cbf * quad(
        lambda tau: np.sin(np.pi * (tau - 5.0) / 10.0) ** 2
        * gammaincc(alpha, (times[-1] - delay - tau) / beta),
        5.0, 15.0, epsabs=1e-12,
    )[0]
    assert scalar[-1] == pytest.approx(reference, rel=0.02)


@pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("delay", [-3.0, -1.3, 0.0, 2.7])
def test_forward_delay_matches_extended_convolution(
    dt: float, delay: float
) -> None:
    """Extending the causal response preserves the measured-window prediction."""
    times = np.arange(0.0, 60.0, dt)
    aif = _aif(times)
    extra = max(0, int(np.ceil(-delay / dt)))
    extended_times = np.arange(len(times) + extra) * dt
    operator = build_volterra_matrix(np.pad(aif, (0, extra)), dt)
    model = VascularModel()
    params = [0.006, 3.0, 25.0 / 3.0, delay]
    residue = model.residue_function(params, extended_times)
    batch = batched_vascular_forward(
        np.array([params[0]]), np.array([params[1]]), np.array([params[2]]),
        np.array([delay]), aif, times,
    )[0]
    batch_residue = residue
    complete = operator @ residue
    expected = np.interp(times - delay, extended_times, complete, left=0.0, right=0.0)
    scalar = model.tissue_curve(params, aif, times)
    np.testing.assert_allclose(scalar, expected, rtol=1e-11, atol=1e-13)
    expected_batch = np.interp(
        times - delay, extended_times, operator @ batch_residue, left=0.0, right=0.0
    )
    np.testing.assert_allclose(batch, expected_batch, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("n_times", [2, 10, 60])
@pytest.mark.parametrize("extra", [1, 3, 70])
def test_tail_operator_matches_complete_matrix(n_times: int, extra: int) -> None:
    """The extension uses the same quadrature even with nonzero endpoint AIF."""
    aif = np.random.default_rng(7).uniform(0.1, 2.0, n_times)
    complete = build_volterra_matrix(np.pad(aif, (0, extra)), 1.3)
    np.testing.assert_array_equal(
        _volterra_tail_matrix(aif, 1.3, extra), complete[n_times:]
    )
    np.testing.assert_array_equal(complete[:n_times, :n_times], build_volterra_matrix(aif, 1.3))


def test_mixed_delays_preserve_unaffected_samples() -> None:
    """Only response samples missing from the original grid need completion."""
    times = np.arange(60.0)
    aif = _aif(times)
    delays = np.array([-3.0, -1.3, 0.0, 2.7])
    cbf, alpha, beta = np.full(4, 0.006), np.full(4, 3.0), np.full(4, 25.0 / 3.0)
    native = batched_vascular_forward(cbf, alpha, beta, np.zeros(4), aif, times)
    expected = _apply_post_hoc_delay_shift(native, delays, 1.0)
    actual = batched_vascular_forward(cbf, alpha, beta, delays, aif, times)
    np.testing.assert_array_equal(actual[2:], expected[2:])
    np.testing.assert_array_equal(actual[0, :-3], expected[0, :-3])
    np.testing.assert_array_equal(actual[1, :-2], expected[1, :-2])
    assert np.all(actual[:2, -2:] > 0)


@pytest.mark.parametrize("delay", [0.0, -1.0, -2.0])
def test_tail_is_continuous_across_frame_boundaries(delay: float) -> None:
    """A sub-frame delay change must not make an endpoint vanish."""
    times = np.arange(60.0)
    aif = _aif(times)
    model = VascularModel()
    curves = [model.tissue_curve([0.006, 3.0, 25.0 / 3.0, delay + step], aif, times)
              for step in [-1e-8, 0.0, 1e-8]]
    np.testing.assert_allclose(curves[0], curves[1], atol=1e-10, rtol=0)
    np.testing.assert_allclose(curves[2], curves[1], atol=1e-10, rtol=0)


@pytest.mark.parametrize("method", ["nlls", "vi", "svb"])
@pytest.mark.parametrize("weighted", [False, True])
def test_negative_delay_fit_predicts_complete_tail(method: str, weighted: bool) -> None:
    """Production fits and map readout retain the independently completed tail."""
    from pmriprep.interfaces.dsc_compute import ComputeDSCMaps
    from pmriprep.models.bayesian_utils import build_priors_from_params
    from pmriprep.models.dsc.batched_fitting import batched_dsc_vi_fit
    from pmriprep.models.dsc.spatial_vb import batched_dsc_spatial_vi_fit
    from pmriprep.models.dsc.vascular import _fit_vascular_nlls

    times = np.arange(60.0)
    aif = _aif(times)
    extended_times = np.arange(62.0)
    truth = np.array([0.006, 25.0, 3.0, -2.0])  # CBF, MTT, alpha, delay
    residue = truth[0] * gammaincc(truth[2], extended_times / (truth[1] / truth[2]))
    tissue = (build_volterra_matrix(np.pad(aif, (0, 2)), 1.0) @ residue)[2:]
    weights = np.where((times >= 8) & (times <= 12), 0.3, 1.0) if weighted else None
    model = VascularModel()
    if method == "nlls":
        result = _fit_vascular_nlls(model, tissue, aif, times, weights=weights)
        assert result is not None
        params = np.array([result.parameters[p] for p in model.get_parameter_names()])
        fitted = model.model_to_fitting_params(params)
        prediction = model.tissue_curve(params, aif, times)
    else:
        definitions = model.get_fitting_parameter_definitions()
        priors = build_priors_from_params(definitions)
        kwargs = dict(
            model_type="vascular", tissue_curves=np.tile(tissue, (3, 1)), aif=aif,
            time_points=times, weights=weights,
            prior_means_nat=np.array([p.mean for p in priors]),
            prior_stds_nat=np.array([p.std for p in priors]),
            transforms=[p.transform for p in priors], upper_bounds=[p.upper_bound for p in priors],
            fitting_param_names=[p.name for p in definitions],
            fitting_bounds=[p.bounds for p in definitions],
            prior_means_transformed=np.array([p.mean_transformed for p in priors]),
            prior_stds_transformed=np.array([p.std_transformed for p in priors]),
            save_fitted_curves=True,
        )
        coords = (np.arange(3), np.zeros(3, dtype=int), np.zeros(3, dtype=int))
        result = (batched_dsc_vi_fit(**kwargs) if method == "vi" else
                  batched_dsc_spatial_vi_fit(**kwargs, mask_coords=coords, spatial_shape=(3, 1, 1)))
        fitted = np.array([result.params[p.name][0] for p in definitions])
        prediction = result.fitted_curves[0]
        voxel = ComputeDSCMaps()._batched_result_to_voxel_dicts(
            result, coords, "vascular", 3, inference=method,
        )[0]
        assert voxel["cbv"] == pytest.approx(fitted[0] * fitted[1])
        assert voxel["mtt"] == pytest.approx(fitted[1])
        assert not voxel["needs_fallback"]
    np.testing.assert_allclose(fitted, truth, rtol=0.002, atol=1e-5)
    # Recovery, not exact interpolation: default Bayesian priors remain active.
    assert np.sqrt(np.sum((prediction - tissue)**2) / np.sum(tissue**2)) < 0.001
    np.testing.assert_allclose(prediction[-2:], tissue[-2:], rtol=0.002)
