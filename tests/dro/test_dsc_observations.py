"""Independent dispersion checks and shared-measurement/noise contracts."""

from typing import Any, Literal

import numpy as np
import pytest
from scipy.integrate import quad

from .dsc_observations import paired_observations, signal_observation
from .dsc_reference import (
    analytic_residue,
    continuous_tissue,
    dispersed_tissue,
    recirculating_gamma_aif,
    spline_residue,
    transport_kernel,
)


def _step(time: Any) -> np.ndarray:
    """Continuous unit step at absolute time 10 seconds."""
    return np.asarray(np.asarray(time) >= 10.0, dtype=np.float64)


@pytest.mark.parametrize("family", ["gamma", "lognormal"])
def test_transport_full_mass_and_mean(family: Literal["gamma", "lognormal"]) -> None:
    """The kernel is normalized independently of the sampled acquisition."""
    kernel = transport_kernel(family, 2.3)
    mass, _ = quad(lambda t: float(kernel.density(t)), 0, np.inf)
    mean, _ = quad(lambda t: float(t * kernel.density(t)), 0, np.inf)
    assert mass == pytest.approx(1.0, abs=1e-9)
    assert mean == pytest.approx(2.3, abs=1e-8)
    assert 0 < float(kernel.cumulative(1.0)) < 1
    np.testing.assert_array_equal(kernel.density([-3.0, -0.1]), 0.0)


@pytest.mark.parametrize("delay", [-5.0, -0.37, 0.0, 0.37, 5.0])
def test_dispersed_step_has_closed_form_and_distinct_flow_truth(delay: float) -> None:
    """Equal exponential time constants give h(t)=F*(t/tau)*exp(-t/tau)."""
    tau, flow = 2.0, 0.01
    time = np.arange(0.0, 40.0, 1.5)
    result = dispersed_tissue(
        time, _step, analytic_residue("exponential", tau),
        transport_kernel("gamma", tau, gamma_shape=1),
        flow=flow, aif_onset_seconds=10.0, delay_seconds=delay,
    )
    x = np.maximum(time - 10.0 - delay, 0.0) / tau
    expected = flow * tau * (1 - (1 + x) * np.exp(-x))
    np.testing.assert_allclose(result.values, expected, atol=2e-7, rtol=1e-5)
    lag = result.response_lag_seconds
    np.testing.assert_allclose(
        result.effective_response, flow * lag / tau * np.exp(-lag / tau),
        atol=2e-8, rtol=1e-5,
    )
    assert result.intrinsic_flow == flow
    assert result.intrinsic_mtt_seconds == tau
    assert result.effective_peak == pytest.approx(flow / np.e, rel=1e-5)
    assert result.refinement_error <= 1e-5
    np.testing.assert_array_equal(result.values[time <= 10 + delay], 0.0)


@pytest.mark.parametrize("kernel_family", ["gamma", "lognormal"])
@pytest.mark.parametrize("residue_family", ["exponential", "gamma", "box", "triangle", "spline"])
def test_dispersed_reference_against_independent_adaptive_integral(
    kernel_family: Literal["gamma", "lognormal"],
    residue_family: Literal["exponential", "gamma", "box", "triangle", "spline"],
) -> None:
    """A step input reduces A*D*R to an independent adaptive integral of CDF*R."""
    kernel = transport_kernel(kernel_family, 1.7)
    residue = (spline_residue([0, 1, 3, 8, 20], [1, 0.97, 0.7, 0.1, 0])
               if residue_family == "spline" else analytic_residue(residue_family, 4.0))
    time = np.array([8.0, 10.0, 10.23, 11.93, 14.31, 21.7, 43.19])
    expected = continuous_tissue(
        time, lambda t: kernel.cumulative(np.asarray(t) - 10), residue,
        flow=0.01, aif_onset_seconds=10, delay_seconds=-0.27,
    ).values
    result = dispersed_tissue(
        time, _step, residue, kernel,
        flow=0.01, aif_onset_seconds=10, delay_seconds=-0.27,
    )
    assert np.max(np.abs(result.values - expected)) / np.max(expected) < 5e-5


def test_short_scan_does_not_renormalize_transport_or_mtt() -> None:
    """A short window retains the same measurements and reports missing support."""
    settings: dict[str, Any] = dict(
        aif=_step, residue=analytic_residue("exponential", 8),
        transport=transport_kernel("lognormal", 5), flow=0.02, aif_onset_seconds=10,
    )
    time = np.arange(0.0, 60, 1.5)
    short = dispersed_tissue(time[:10], **settings)
    long = dispersed_tissue(time, **settings)
    # Refinement is defined relative to curve peak, not each near-zero sample.
    assert np.max(np.abs(short.values - long.values[:10])) / np.max(short.values) < 5e-5
    assert short.kernel_mass_in_scan < long.kernel_mass_in_scan < 1
    assert short.residue_area_in_scan_seconds < long.residue_area_in_scan_seconds < 8
    assert short.intrinsic_mtt_seconds == long.intrinsic_mtt_seconds == 8
    assert short.effective_peak == pytest.approx(long.effective_peak, rel=1e-5)


def test_narrow_transport_requires_mass_resolution_and_two_accepted_grids() -> None:
    """An unresolved narrow kernel must refine, not pass as a zero response."""
    result = dispersed_tissue(
        np.array([9, 10, 10.1, 10.2, 11]), _step, analytic_residue("exponential", 0.1),
        transport_kernel("gamma", 0.02), flow=0.01, aif_onset_seconds=10,
        initial_step_seconds=0.04, max_refinements=12,
    )
    assert result.step_seconds < 0.001
    assert result.effective_peak > 0
    assert result.refinement_error <= 1e-5


def test_dispersion_refinement_failure_is_not_returned_as_verified() -> None:
    """An inadequate numerical budget fails explicitly rather than emitting bad truth."""
    with pytest.raises(RuntimeError, match="refinement tolerance"):
        dispersed_tissue(
            [9, 10, 11, 12], _step, analytic_residue("exponential", 4),
            transport_kernel("gamma", 1), flow=0.01, aif_onset_seconds=10,
            initial_step_seconds=0.5, max_refinements=1,
        )


def _observations(**overrides: Any) -> Any:
    """Generate paired observations from a shared eight-frame signal fixture."""
    settings = dict(
        time_seconds=np.arange(8) * 1.5,
        tissue_delta_r2=np.array([0, 0, 0, 2, 8, 5, 3, 1], dtype=float),
        aif_delta_r2=np.array([0, 0, 0, 15, 40, 20, 5, 1], dtype=float),
        reference_delta_r2=np.array([[0, 0, 0, 1, 4, 2, 1, 0.5]] * 4),
        baseline_mask=np.arange(8) < 3,
        te_seconds=0.03, s0=1000.0, noise_model="rician_signal",
        tissue_noise_sd=10.0, aif_noise_sd=10.0, reference_noise_sd=10.0, seed=123,
    )
    return paired_observations(**{**settings, **overrides})


@pytest.mark.parametrize("noise_model", ["rician_signal", "gaussian_delta_r2"])
def test_noiseless_signal_roundtrip_without_extra_baseline_frames(noise_model: str) -> None:
    """Baseline is already in the input; no prepending or sample shift occurs."""
    truth = np.array([0, 0, 0, 2, 8, 5, 3, 1], dtype=float)
    result = _observations(
        noise_model=noise_model, tissue_noise_sd=0, aif_noise_sd=0, reference_noise_sd=0,
    )
    np.testing.assert_allclose(result.tissue.delta_r2, truth, atol=1e-14)
    assert result.tissue.signal.shape == truth.shape
    assert result.references.signal.shape == (4, len(truth))
    assert result.tissue.baseline_signal == 1000


@pytest.mark.parametrize("noise_model", ["rician_signal", "gaussian_delta_r2"])
def test_measured_conversion_not_true_baseline_or_separate_noise(noise_model: str) -> None:
    """Signal and concentration methods see transforms of the same observations."""
    result = _observations(noise_model=noise_model, tissue_noise_sd=1)
    for observed in (result.tissue, result.aif, result.references):
        measured = np.mean(observed.signal[..., result.baseline_mask], axis=-1)
        np.testing.assert_array_equal(observed.baseline_signal, measured)
        assert np.all(measured != 1000)
        np.testing.assert_allclose(
            observed.delta_r2, -np.log(observed.signal / measured[..., None]) / result.te_seconds,
        )
        assert np.any(observed.delta_r2[..., result.baseline_mask] < 0)
    expected_snr = 1000.0 if noise_model == "rician_signal" else None
    assert result.tissue.baseline_snr == expected_snr


def test_rician_noise_uses_two_independent_complex_channels() -> None:
    """Check the stated absolute-noise convention against its defining equation."""
    truth = np.array([0, 0, 1, 3, -0.5], dtype=float)
    result = signal_observation(
        truth, [True, True, False, False, False], te_seconds=0.03, s0=1000,
        noise_model="rician_signal", noise_sd=20, seed=72,
    )
    rng = np.random.default_rng(72)
    real = 1000 * np.exp(-0.03 * truth) + 20 * rng.standard_normal(truth.shape)
    imaginary = 20 * rng.standard_normal(truth.shape)
    np.testing.assert_array_equal(result.signal, np.hypot(real, imaginary))
    assert result.baseline_snr == 50


def test_streams_replay_independently_of_reference_pool_and_seed_object_state() -> None:
    """Reusing an observation identity neither resamples data nor couples pools."""
    seed = np.random.SeedSequence(71, spawn_key=(4,))
    first = _observations(seed=seed)
    seed.spawn(3)
    second = _observations(seed=seed, reference_delta_r2=np.zeros((7, 8)))
    replay = _observations(seed=seed)
    for field in ("tissue", "aif"):
        np.testing.assert_array_equal(getattr(first, field).signal, getattr(second, field).signal)
    np.testing.assert_array_equal(first.references.signal, replay.references.signal)
    assert not np.array_equal(first.tissue.signal[:3], first.aif.signal[:3])
    assert not np.array_equal(first.references.signal[0], first.references.signal[1])


@pytest.mark.parametrize("overrides,match", [
    ({"baseline_mask": np.arange(8) < 4}, "precede"),
    ({"baseline_mask": np.arange(8) < 1}, "at least two"),
    ({"baseline_mask": np.ones(8, dtype=int)}, "boolean"),
    ({"te_seconds": 0}, "TE and S0"),
    ({"s0": np.nan}, "TE and S0"),
    ({"tissue_noise_sd": -1}, "noise_sd"),
    ({"noise_model": "percent"}, "Unknown noise"),
    ({"time_seconds": np.zeros(8)}, "increasing"),
    ({"aif_delta_r2": [0, 0]}, "share the time"),
    ({"reference_delta_r2": np.zeros((0, 8))}, "nonempty pool"),
])
def test_observation_contract_rejects_invalid_inputs(overrides: dict, match: str) -> None:
    """Bad baselines, noise units and shapes cannot quietly change method inputs."""
    with pytest.raises(ValueError, match=match):
        _observations(**overrides)


def test_signal_underflow_is_not_silently_floored() -> None:
    """Nonrepresentable signals are input failures, not valid noiseless measurements."""
    with pytest.raises(ValueError, match="no floor"):
        signal_observation(
            [0, 0, 1e9], [True, True, False], te_seconds=0.03, s0=1000,
            noise_model="rician_signal", noise_sd=0, seed=1,
        )


@pytest.mark.parametrize("tr", [1.5, 3.0])
def test_complete_shared_observation_with_recirculation_and_dispersion(tr: float) -> None:
    """Assemble continuous tissue, measured-site AIF and independent noisy references."""
    time = np.arange(0, 90, tr)
    onset, delay, te = 15.37, -2.43, 0.03

    def aif(t: Any) -> np.ndarray:
        return recirculating_gamma_aif(
            t, onset_seconds=onset, peak=40, alpha=2.5, beta_seconds=3,
            recirculation_fraction=0.35, recirculation_delay_seconds=15, broadening=2,
        )

    tissue = dispersed_tissue(
        time, aif, analytic_residue("gamma", 4), transport_kernel("lognormal", 1.7),
        flow=0.01, aif_onset_seconds=onset, delay_seconds=delay,
    )
    reference = continuous_tissue(
        time, aif, analytic_residue("gamma", 6), flow=0.005, aif_onset_seconds=onset,
    )
    result = paired_observations(
        time, tissue.values, aif(time), np.tile(reference.values, (4, 1)),
        time < onset + min(delay, 0), te_seconds=te, s0=1000,
        noise_model="rician_signal", tissue_noise_sd=10, aif_noise_sd=10,
        reference_noise_sd=10, seed=53,
    )
    np.testing.assert_array_equal(result.time_seconds, time)
    np.testing.assert_allclose(result.tissue.clean_signal, 1000 * np.exp(-te * tissue.values))
    np.testing.assert_allclose(result.aif.clean_signal, 1000 * np.exp(-te * aif(time)))
    assert result.tissue.signal.shape == result.aif.signal.shape == time.shape
    assert result.references.signal.shape == (4, len(time))
    assert tissue.effective_peak < tissue.intrinsic_flow == 0.01
