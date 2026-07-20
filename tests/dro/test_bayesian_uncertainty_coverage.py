"""Repeated-noise coverage checks for Bayesian perfusion uncertainty."""

import numpy as np
import pytest

from pmriprep.models.bayesian_utils import build_priors_from_params
from pmriprep.models.dce.bayesian import BayesianPKFitter
from pmriprep.models.dce.tofts import ToftsModel
from pmriprep.models.dsc import VascularModel
from pmriprep.models.dsc.batched_fitting import (
    batched_dsc_map_fit,
    batched_dsc_vi_fit,
    batched_vascular_forward,
)
from pmriprep.models.dsc.spatial_vb import batched_dsc_spatial_vi_fit


@pytest.mark.parametrize("inference", ["vi", "map"])
def test_dce_tofts_nominal_intervals_cover_repeated_noise(inference: str) -> None:
    """Interior Tofts truth is covered across deterministic noise repeats."""
    rng = np.random.default_rng(1847)
    time_points = np.linspace(0.0, 6.0, 60)
    aif = 5.0 * np.exp(-0.5 * time_points) + 2.0 * np.exp(-0.1 * time_points)
    aif[:3] = 0.0
    truth = {"Ktrans": 0.1, "ve": 0.2}
    model = ToftsModel()
    clean = model.evaluate(list(truth.values()), aif, time_points)
    covered = {name: 0 for name in truth}
    n_repeats = 24

    for _ in range(n_repeats):
        observed = clean + rng.normal(0.0, 0.01, len(time_points))
        fitter = BayesianPKFitter(ToftsModel(), noise_std=0.01)
        result = getattr(fitter, f"fit_{inference}")(
            observed, aif, time_points
        )
        for name, value in truth.items():
            lower, upper = result.credible_intervals[name]
            covered[name] += int(lower <= value <= upper)

    for name in truth:
        assert covered[name] / n_repeats >= 0.8


@pytest.mark.parametrize("inference", ["vi", "svb", "map"])
def test_dsc_vascular_cbf_intervals_cover_repeated_noise(inference: str) -> None:
    """Vascular CBF intervals cover an interior truth across noise repeats."""
    rng = np.random.default_rng(2718)
    time_points = np.arange(60, dtype=np.float64)
    bolus_time = time_points - 8.0
    aif = np.where(
        bolus_time > 0.0,
        3.0 * bolus_time * np.exp(-bolus_time / 3.0),
        0.0,
    )
    shape = (3, 3, 3)
    n_voxels = int(np.prod(shape))
    cbf_truth = 0.06
    clean = batched_vascular_forward(
        np.full(n_voxels, cbf_truth),
        np.full(n_voxels, 3.0),
        np.full(n_voxels, 2.0),
        np.full(n_voxels, 1.0),
        aif,
        time_points,
    )
    tissue = clean + rng.normal(0.0, 0.005, clean.shape)

    definitions = VascularModel().get_fitting_parameter_definitions()
    priors = build_priors_from_params(definitions)
    kwargs = dict(
        model_type="vascular",
        tissue_curves=tissue,
        aif=aif,
        time_points=time_points,
        prior_means_nat=np.asarray([prior.mean for prior in priors]),
        prior_stds_nat=np.asarray([prior.std for prior in priors]),
        transforms=[prior.transform for prior in priors],
        upper_bounds=[prior.upper_bound for prior in priors],
        fitting_param_names=[definition.name for definition in definitions],
        fitting_bounds=[definition.bounds for definition in definitions],
        prior_means_transformed=np.asarray([
            prior.mean_transformed for prior in priors
        ]),
        prior_stds_transformed=np.asarray([
            prior.std_transformed for prior in priors
        ]),
    )
    if inference == "svb":
        coords = np.indices(shape).reshape(3, -1)
        result = batched_dsc_spatial_vi_fit(
            **kwargs,
            mask_coords=(coords[0], coords[1], coords[2]),
            spatial_shape=shape,
        )
    elif inference == "map":
        result = batched_dsc_map_fit(**kwargs)
    else:
        result = batched_dsc_vi_fit(**kwargs)

    lower = result.credible_interval_lower["cbf"]
    upper = result.credible_interval_upper["cbf"]
    coverage = float(np.mean((lower <= cbf_truth) & (cbf_truth <= upper)))
    assert coverage >= 0.8
