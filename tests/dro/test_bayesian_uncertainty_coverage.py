"""Repeated-noise coverage checks for Bayesian perfusion uncertainty."""

import numpy as np
import pytest

from pmriprep.models.dce.bayesian import BayesianPKFitter
from pmriprep.models.dce.tofts import ToftsModel


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
