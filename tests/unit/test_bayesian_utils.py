"""Unit tests for shared Bayesian inference utilities."""

from dataclasses import dataclass

import numpy as np
import pytest

from pmriprep.models.bayesian_utils import (
    BayesianPrior,
    build_priors_from_params,
    compute_goodness_of_fit,
    compute_log_prior,
    compute_per_walker_ess,
    compute_split_rhat,
    run_ensemble_sampler,
    validate_bayesian_inputs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _MockParameter:
    """Minimal mock satisfying ParameterDef protocol."""

    log_transform: bool
    _mean: float
    _sigma: float

    def get_prior_mean(self) -> float:
        return self._mean

    def get_prior_sigma(self) -> float:
        return self._sigma


# ---------------------------------------------------------------------------
# BayesianPrior dataclass
# ---------------------------------------------------------------------------

class TestBayesianPrior:
    """Tests for the BayesianPrior dataclass."""

    def test_attributes_accessible(self) -> None:
        """BayesianPrior exposes mean, std, and log_transform attributes."""
        prior = BayesianPrior(mean=0.5, std=0.1, log_transform=True)
        assert prior.mean == 0.5
        assert prior.std == 0.1
        assert prior.log_transform is True

    def test_defaults(self) -> None:
        """log_transform defaults to False when not specified."""
        prior = BayesianPrior(mean=1.0, std=0.2)
        assert prior.log_transform is False


# ---------------------------------------------------------------------------
# validate_bayesian_inputs
# ---------------------------------------------------------------------------

class TestValidateBayesianInputs:
    """Tests for input validation utility."""

    def test_valid_inputs_no_error(self) -> None:
        """Finite arrays with matching lengths pass without error."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        validate_bayesian_inputs(("a", a), ("b", b))

    def test_nan_raises_value_error(self) -> None:
        """Array containing NaN raises ValueError."""
        a = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_bayesian_inputs(("a", a))

    def test_length_mismatch_raises_value_error(self) -> None:
        """Arrays with different lengths raise ValueError."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        with pytest.raises(ValueError, match="lengths must match"):
            validate_bayesian_inputs(("a", a), ("b", b))


# ---------------------------------------------------------------------------
# build_priors_from_params
# ---------------------------------------------------------------------------

class TestBuildPriorsFromParams:
    """Tests for building priors from parameter definitions."""

    def test_correct_prior_means_and_stds(self) -> None:
        """ModelParameter list produces priors with correct mean and std."""
        params = [
            _MockParameter(log_transform=False, _mean=0.1, _sigma=0.05),
            _MockParameter(log_transform=True, _mean=0.5, _sigma=0.1),
        ]
        priors = build_priors_from_params(params)

        assert len(priors) == 2
        assert priors[0].mean == pytest.approx(0.1)
        assert priors[0].std == pytest.approx(0.05)
        assert priors[0].log_transform is False
        assert priors[1].mean == pytest.approx(0.5)
        assert priors[1].std == pytest.approx(0.1)
        assert priors[1].log_transform is True


# ---------------------------------------------------------------------------
# compute_log_prior
# ---------------------------------------------------------------------------

class TestComputeLogPrior:
    """Tests for log prior computation."""

    def test_maximum_at_prior_mean(self) -> None:
        """Log prior is maximized (least negative) at the prior mean."""
        priors = [BayesianPrior(mean=1.0, std=0.2)]
        lp_at_mean = compute_log_prior(np.array([1.0]), priors)
        lp_away = compute_log_prior(np.array([1.5]), priors)
        assert lp_at_mean > lp_away

    def test_quadratic_decay_far_from_mean(self) -> None:
        """Log prior becomes very negative far from the mean."""
        priors = [BayesianPrior(mean=0.0, std=0.1)]
        lp_at_mean = compute_log_prior(np.array([0.0]), priors)
        lp_far = compute_log_prior(np.array([10.0]), priors)
        # 10 sigma away: -0.5 * (10/0.1)^2 = -5000
        assert lp_far < -1000.0
        assert lp_at_mean == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_goodness_of_fit
# ---------------------------------------------------------------------------

class TestComputeGoodnessOfFit:
    """Tests for goodness-of-fit metrics."""

    def test_perfect_fit(self) -> None:
        """Identical observed and predicted gives r_squared=1 and rss=0."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gof = compute_goodness_of_fit(obs, obs, n_params=2)
        assert gof["r_squared"] == pytest.approx(1.0)
        assert gof["rss"] == pytest.approx(0.0)

    def test_partial_fit(self) -> None:
        """Imperfect fit gives 0 < r_squared < 1 and finite AIC/BIC."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.array([1.1, 2.3, 2.8, 4.2, 4.7])
        gof = compute_goodness_of_fit(obs, pred, n_params=2)
        assert 0.0 < gof["r_squared"] < 1.0
        assert np.isfinite(gof["aic"])
        assert np.isfinite(gof["bic"])


# ---------------------------------------------------------------------------
# compute_split_rhat
# ---------------------------------------------------------------------------

class TestComputeSplitRhat:
    """Tests for split R-hat convergence diagnostic."""

    def test_converged_chains(self) -> None:
        """Well-mixed samples from one distribution give rhat close to 1."""
        rng = np.random.default_rng(42)
        n_iter, n_walkers, n_params = 2000, 10, 2
        samples = rng.normal(size=(n_iter, n_walkers, n_params))
        rhat = compute_split_rhat(samples)
        assert rhat.shape == (n_params,)
        for val in rhat:
            assert val == pytest.approx(1.0, abs=0.05)

    def test_non_converged_chains(self) -> None:
        """First half near 0, second half near 100 gives rhat >> 1."""
        rng = np.random.default_rng(42)
        n_iter, n_walkers, n_params = 200, 4, 1
        # Add small noise so within-chain variance is nonzero
        samples = rng.normal(0, 0.1, size=(n_iter, n_walkers, n_params))
        samples[n_iter // 2 :, :, :] += 100.0
        rhat = compute_split_rhat(samples)
        assert rhat[0] > 2.0


# ---------------------------------------------------------------------------
# compute_per_walker_ess
# ---------------------------------------------------------------------------

class TestComputePerWalkerESS:
    """Tests for effective sample size estimation."""

    def test_uncorrelated_chain_high_ess(self) -> None:
        """IID samples yield ESS near the total number of samples."""
        rng = np.random.default_rng(99)
        n_iter, n_walkers, n_params = 500, 4, 1
        samples = rng.normal(size=(n_iter, n_walkers, n_params))
        ess = compute_per_walker_ess(samples)
        total_samples = n_iter * n_walkers
        # ESS should be at least 40% of total for iid data
        assert ess[0] > total_samples * 0.4

    def test_correlated_chain_low_ess(self) -> None:
        """Highly autocorrelated chain yields ESS much less than n_samples."""
        n_iter, n_walkers, n_params = 500, 4, 1
        samples = np.zeros((n_iter, n_walkers, n_params))
        rng = np.random.default_rng(7)
        # Random walk: strong autocorrelation
        for w in range(n_walkers):
            for t in range(1, n_iter):
                samples[t, w, 0] = samples[t - 1, w, 0] + rng.normal(0, 0.01)
        ess = compute_per_walker_ess(samples)
        total_samples = n_iter * n_walkers
        assert ess[0] < total_samples * 0.3


# ---------------------------------------------------------------------------
# run_ensemble_sampler
# ---------------------------------------------------------------------------

class TestRunEnsembleSampler:
    """Tests for the affine-invariant ensemble sampler."""

    def test_samples_around_known_mode(self) -> None:
        """Sampler recovers the mode of a known Gaussian posterior."""
        np.random.seed(123)
        true_mean = np.array([3.0, -1.0])
        true_std = np.array([0.5, 0.3])

        def log_prob(params: np.ndarray) -> float:
            return float(-0.5 * np.sum(((params - true_mean) / true_std) ** 2))

        n_walkers = 20
        n_params = 2
        init = true_mean + 0.1 * np.random.randn(n_walkers, n_params)
        bounds = [(-10.0, 10.0), (-10.0, 10.0)]

        samples = run_ensemble_sampler(
            log_prob=log_prob,
            init_positions=init,
            n_iterations=500,
            bounds=bounds,
        )
        assert samples.shape == (500, n_walkers, n_params)

        # Discard first half as burn-in and check posterior mean
        burned = samples[250:]
        post_mean = burned.reshape(-1, n_params).mean(axis=0)
        assert post_mean[0] == pytest.approx(true_mean[0], abs=0.3)
        assert post_mean[1] == pytest.approx(true_mean[1], abs=0.3)
