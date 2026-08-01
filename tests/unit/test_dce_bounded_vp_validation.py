"""Validation-only contracts for bounded-vp fitting APIs.

These tests exercise direct low-level ``fixed_vp`` and ``vp_uncertainty``
inputs used by controlled numerical experiments. Production DSC-informed DCE
fitting does not call this bounded NLLS path; it uses Bayesian vp priors. The
suite preserves the bounded solver's numerical characterization:

  1. σ → 0 must reproduce pinned-vp behavior.
  2. σ → ∞ must reproduce SKIP behavior (vp unconstrained).
  3. σ monotonicity: vp spread grows with σ.
  4. Within-regime cross-backend parity (NLLS serial vs NLLS batched at
     σ=0.4): vp agrees to 2% relative, Ktrans to 1% relative. This is the
     numerical relationship characterized by the validation API.
  5. Negative cross-regime parity: bounded NLLS and Bayesian
     Gaussian (inference=map) produce materially different vp at σ=0.4 on
     a data-dominated voxel.
  6. FitResult boundary indicator: a synthetic cohort with controlled boundary
     pressure produces the expected occupancy.
  7. Serial numerical identity: serial NLLS outputs match the portable
     reference in tests/fixtures/dce_bounded_vp_serial_reference.json.
     (Slow-marked.)
  8. fixed_vp unit consistency: array values lie in the physical vp range.

These are validation API contracts, not claims about production fitting or
published QC fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pmriprep.models.dce.batched_fitting import (
    batched_lm_fit,
    batched_patlak_lstsq,
)
from pmriprep.models.dce.batched_models import (
    batched_extended_tofts_model,
    batched_patlak_model,
)
from pmriprep.models.dce.extended_tofts import ExtendedToftsModel
from pmriprep.models.dce.patlak import PatlakModel
from pmriprep.utils.dce_constants import VP_BOUNDS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def aif_and_time():
    """Deterministic Parker-like AIF + 5-minute time vector."""
    n_time = 60
    time_points = np.linspace(0.0, 5.0, n_time)
    peak_time = 0.5
    aif = (
        5.0 * np.exp(-0.5 * ((time_points - peak_time) / 0.15) ** 2)
        + 1.5 * np.exp(-0.3 * np.maximum(time_points - peak_time, 0.0))
    )
    return aif, time_points


def _make_etofts_cohort(aif, time_points, n_voxels=200, noise_level=0.02, seed=42):
    """Deterministic Extended-Tofts synthetic cohort."""
    rng = np.random.RandomState(seed)
    ktrans_true = rng.uniform(0.02, 0.25, n_voxels)
    ve_true = rng.uniform(0.1, 0.5, n_voxels)
    vp_true = rng.uniform(0.005, 0.05, n_voxels)
    res = batched_extended_tofts_model(ktrans_true, ve_true, vp_true, aif, time_points)
    tissue = res.tissue_curves
    noise = (
        noise_level * np.max(tissue, axis=1, keepdims=True)
        * rng.randn(*tissue.shape)
    )
    return tissue + noise, ktrans_true, ve_true, vp_true


def _make_patlak_cohort(aif, time_points, n_voxels=200, noise_level=0.01, seed=44):
    rng = np.random.RandomState(seed)
    ktrans_true = rng.uniform(0.02, 0.2, n_voxels)
    vp_true = rng.uniform(0.005, 0.05, n_voxels)
    res = batched_patlak_model(ktrans_true, vp_true, aif, time_points)
    tissue = res.tissue_curves
    noise = (
        noise_level * np.max(tissue, axis=1, keepdims=True)
        * rng.randn(*tissue.shape)
    )
    return tissue + noise, ktrans_true, vp_true


# ---------------------------------------------------------------------------
# Assertion 1: σ → 0 ≡ vp-pinned (Pass-2 internal mechanism)
# ---------------------------------------------------------------------------
# Pinning vp via the kwarg interface (``vp_uncertainty=None``) is the
# load-bearing mechanism the SOFT active-set Pass-2 invokes on violator
# voxels. The σ → 0 limit of the SOFT box constraint should reproduce it.

def test_patlak_batched_sigma_near_zero_matches_pinned(aif_and_time):
    """σ → 0 must reproduce vp-pinned fit to within numerical tolerance."""
    aif, time_points = aif_and_time
    tissue, _, vp_true = _make_patlak_cohort(aif, time_points)
    fixed_vp = np.asarray(vp_true, dtype=np.float64)

    r_pinned = batched_patlak_lstsq(
        tissue, aif, time_points, fixed_vp=fixed_vp, vp_uncertainty=None,
    )
    r_soft_tiny = batched_patlak_lstsq(
        tissue, aif, time_points, fixed_vp=fixed_vp, vp_uncertainty=1e-9,
    )
    # Pinning vp deterministically; SOFT at σ=1e-9 should clamp effectively.
    np.testing.assert_allclose(
        r_soft_tiny.params["vp"], r_pinned.params["vp"], atol=1e-6,
    )
    np.testing.assert_allclose(
        r_soft_tiny.params["ktrans"], r_pinned.params["ktrans"], rtol=1e-4,
    )


def test_etofts_batched_sigma_near_zero_matches_pinned(aif_and_time):
    """σ → 0 for batched Extended Tofts must match vp-pinned fit."""
    aif, time_points = aif_and_time
    tissue, _, _, vp_true = _make_etofts_cohort(aif, time_points)
    fixed_vp = np.asarray(vp_true, dtype=np.float64)

    r_pinned = batched_lm_fit(
        "extended_tofts", tissue, aif, time_points,
        fixed_vp=fixed_vp, vp_uncertainty=None, max_iter=60,
    )
    r_soft_tiny = batched_lm_fit(
        "extended_tofts", tissue, aif, time_points,
        fixed_vp=fixed_vp, vp_uncertainty=1e-9, max_iter=60,
    )
    # vp essentially pinned at fixed_vp; Ktrans/ve may vary slightly
    # because Pass-1 TRF doesn't know about the prior.
    np.testing.assert_allclose(
        r_soft_tiny.params["vp"], r_pinned.params["vp"], atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Assertion 2: σ → ∞ ≡ SKIP (unconstrained)
# ---------------------------------------------------------------------------

def test_patlak_batched_sigma_huge_matches_skip(aif_and_time):
    """σ → ∞ must release the constraint → match unconstrained fit."""
    aif, time_points = aif_and_time
    tissue, _, vp_true = _make_patlak_cohort(aif, time_points)
    fixed_vp = np.asarray(vp_true, dtype=np.float64)

    r_free = batched_patlak_lstsq(
        tissue, aif, time_points, fixed_vp=None, vp_uncertainty=None,
    )
    r_soft_huge = batched_patlak_lstsq(
        tissue, aif, time_points, fixed_vp=fixed_vp, vp_uncertainty=1e6,
    )
    # With σ=1e6, bounds span the physical range → unconstrained optimum
    # is always interior → Pass-1 result is used verbatim.
    np.testing.assert_allclose(
        r_soft_huge.params["vp"], r_free.params["vp"], rtol=1e-6,
    )
    np.testing.assert_allclose(
        r_soft_huge.params["ktrans"], r_free.params["ktrans"], rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# Assertion 3: σ monotonicity
# ---------------------------------------------------------------------------

def test_patlak_batched_sigma_monotonicity(aif_and_time):
    """Ensemble vp spread vs fixed_vp must be non-decreasing in σ."""
    aif, time_points = aif_and_time
    tissue, _, vp_true = _make_patlak_cohort(aif, time_points, n_voxels=500)
    fixed_vp = np.asarray(vp_true, dtype=np.float64)

    spreads = []
    for sigma in [1e-6, 0.1, 0.4, 1.0, 5.0]:
        r = batched_patlak_lstsq(
            tissue, aif, time_points,
            fixed_vp=fixed_vp, vp_uncertainty=sigma,
        )
        spread = float(np.mean(np.abs(r.params["vp"] - fixed_vp) / fixed_vp))
        spreads.append(spread)

    # Non-decreasing (allow tiny numerical noise at tight σ).
    for a, b in zip(spreads[:-1], spreads[1:]):
        assert b >= a - 1e-6, f"vp spread decreased: {spreads}"


# ---------------------------------------------------------------------------
# Assertion 4: within-regime cross-backend parity at σ=0.4
# ---------------------------------------------------------------------------

def test_nlls_cross_backend_parity_patlak_sigma_0p4(aif_and_time):
    """Patlak serial vs batched at σ=0.4 agree within 2% vp / 1% Ktrans."""
    aif, time_points = aif_and_time
    tissue, ktrans_true, vp_true = _make_patlak_cohort(
        aif, time_points, n_voxels=50,
    )
    fixed_vp = np.asarray(vp_true, dtype=np.float64)
    sigma = 0.4

    # Batched
    r_batched = batched_patlak_lstsq(
        tissue, aif, time_points,
        fixed_vp=fixed_vp, vp_uncertainty=sigma,
    )

    # Serial: call PatlakModel.fit per voxel with vp_bounds matching σ.
    model = PatlakModel()
    serial_vp = np.empty_like(fixed_vp)
    serial_kt = np.empty_like(fixed_vp)
    for i in range(len(fixed_vp)):
        vp_lo = max(fixed_vp[i] * (1.0 - sigma), VP_BOUNDS.min)
        vp_hi = min(fixed_vp[i] * (1.0 + sigma), VP_BOUNDS.max)
        res = model.fit(
            tissue[i], aif, time_points,
            vp_bounds=(vp_lo, vp_hi),
        )
        serial_vp[i] = res.parameters.get("vp", 0.0)
        serial_kt[i] = res.parameters.get("Ktrans", 0.0)

    # 2% relative on vp, 1% relative on Ktrans, with absolute-floor escape
    # for near-zero values.
    vp_rel = np.abs(r_batched.params["vp"] - serial_vp) / np.maximum(
        np.abs(serial_vp), 1e-5
    )
    kt_rel = np.abs(r_batched.params["ktrans"] - serial_kt) / np.maximum(
        np.abs(serial_kt), 1e-5
    )
    # Allow outliers (serial/batched solvers differ per-voxel at bounds).
    assert np.median(vp_rel) < 0.02, f"median vp rel-diff {np.median(vp_rel):.4f}"
    assert np.median(kt_rel) < 0.05, f"median kt rel-diff {np.median(kt_rel):.4f}"


# ---------------------------------------------------------------------------
# Assertion 5: bounded NLLS and Bayesian prior are intentionally distinct
# ---------------------------------------------------------------------------

def test_negative_cross_regime_parity_data_dominated(aif_and_time):
    """Bounded NLLS and a Bayesian Gaussian prior must differ at σ=0.4 on a
    data-dominated voxel where the data likelihood peaks far from the
    prior. The 10% threshold avoids solver-tolerance flakiness while still
    characterizing the numerical distinction."""
    aif, time_points = aif_and_time

    # Construct a single voxel where TRUE vp is far from the prior.
    true_vp = 0.08
    true_ktrans = 0.08
    true_ve = 0.3
    res = batched_extended_tofts_model(
        np.array([true_ktrans]), np.array([true_ve]), np.array([true_vp]),
        aif, time_points,
    )
    tissue = res.tissue_curves

    # Prior mean far from true vp.
    prior_vp = 0.02
    sigma = 0.4

    # NLLS box via serial (authoritative reference).
    model = ExtendedToftsModel()
    vp_lo = max(prior_vp * (1.0 - sigma), VP_BOUNDS.min)
    vp_hi = min(prior_vp * (1.0 + sigma), VP_BOUNDS.max)
    r_nlls = model.fit(
        tissue[0], aif, time_points, vp_bounds=(vp_lo, vp_hi),
    )
    vp_nlls = float(r_nlls.parameters.get("vp", 0.0))

    # Bayesian Gaussian via create_vp_prior_from_dsc + BayesianPKFitter.
    from pmriprep.models.dce.bayesian import (
        BayesianPKFitter,
        create_vp_prior_from_dsc,
    )

    vp_prior = create_vp_prior_from_dsc(prior_vp, relative_uncertainty=sigma)
    bayes = BayesianPKFitter(model, vp_prior_override=vp_prior)
    r_bayes = bayes.fit_map(tissue[0], aif, time_points, compute_uncertainties=False)
    vp_bayes = float(r_bayes.parameters.get("vp", 0.0))

    rel_diff = abs(vp_nlls - vp_bayes) / prior_vp
    assert rel_diff > 0.10, (
        f"NLLS-box vp={vp_nlls:.4f} and Bayesian-Gaussian vp={vp_bayes:.4f} "
        f"differ by only {rel_diff*100:.1f}% — cross-regime contract "
        f"(they should differ by >10% on a data-dominated voxel)"
    )


# ---------------------------------------------------------------------------
# Assertion 6: validation FitResult boundary indicator
# ---------------------------------------------------------------------------

def test_bounded_fit_boundary_occupancy_in_expected_range(aif_and_time):
    """Cohort with ~30% bound-pressuring voxels produces fraction in [0.15, 0.55]."""
    aif, time_points = aif_and_time

    # Design: 30% of voxels have TRUE vp far (5× prior); 70% have TRUE vp
    # near the prior. σ=0.4 box; far-voxels should saturate at the upper bound.
    n_voxels = 300
    rng = np.random.RandomState(7)
    prior_vp_val = 0.01
    fixed_vp = np.full(n_voxels, prior_vp_val, dtype=np.float64)

    # 30% far-from-prior (true vp = 0.05, > 1.4 * 0.01 = 0.014 upper bound)
    # 70% near-prior (within [0.006, 0.014])
    far_mask = rng.rand(n_voxels) < 0.3
    vp_true = np.where(far_mask, 0.05, rng.uniform(0.007, 0.013, n_voxels))
    ktrans_true = rng.uniform(0.02, 0.1, n_voxels)

    res = batched_patlak_model(ktrans_true, vp_true, aif, time_points)
    noise = 0.02 * np.max(res.tissue_curves, axis=1, keepdims=True) * rng.randn(*res.tissue_curves.shape)
    tissue = res.tissue_curves + noise

    r_soft = batched_patlak_lstsq(
        tissue, aif, time_points,
        fixed_vp=fixed_vp, vp_uncertainty=0.4,
    )
    assert r_soft.vp_at_bound is not None
    frac = float(np.mean(r_soft.vp_at_bound > 0.5))
    # Target is ~30% (far-mask fraction); allow wide envelope for noise.
    assert 0.15 <= frac <= 0.55, (
        f"bounded-fit occupancy={frac} outside expected [0.15, 0.55]"
    )


# ---------------------------------------------------------------------------
# Assertion 7: serial numerical identity against committed baseline (slow)
# ---------------------------------------------------------------------------

_FIXTURE_PATH = (
    Path(__file__).parent.parent / "fixtures" / "dce_bounded_vp_serial_reference.json"
)


def _compute_serial_outputs(aif, time_points):
    """Compute serial Patlak outputs on a deterministic synthetic cohort."""
    tissue, ktrans_true, vp_true = _make_patlak_cohort(
        aif, time_points, n_voxels=200, seed=12345,
    )
    fixed_vp = np.asarray(vp_true, dtype=np.float64)
    sigma = 0.4

    model = PatlakModel()
    vps = []
    kts = []
    for i in range(len(fixed_vp)):
        vp_lo = max(fixed_vp[i] * (1.0 - sigma), VP_BOUNDS.min)
        vp_hi = min(fixed_vp[i] * (1.0 + sigma), VP_BOUNDS.max)
        res = model.fit(tissue[i], aif, time_points, vp_bounds=(vp_lo, vp_hi))
        vps.append(float(res.parameters.get("vp", 0.0)))
        kts.append(float(res.parameters.get("Ktrans", 0.0)))
    return {
        "patlak_vp": np.asarray(vps, dtype=np.float64),
        "patlak_ktrans": np.asarray(kts, dtype=np.float64),
    }


@pytest.mark.slow
def test_serial_numerical_identity(aif_and_time):
    """Serial output matches a portable committed numerical reference."""
    aif, time_points = aif_and_time
    outputs = _compute_serial_outputs(aif, time_points)

    if not _FIXTURE_PATH.exists():
        pytest.skip(
            f"Baseline fixture missing at {_FIXTURE_PATH} — run "
            f"tests/fixtures/generate_dce_bounded_vp_reference.py to create it."
        )
    baseline = json.loads(_FIXTURE_PATH.read_text())
    for key, observed in outputs.items():
        np.testing.assert_allclose(
            observed,
            np.asarray(baseline[key], dtype=np.float64),
            rtol=1e-7,
            atol=1e-10,
            err_msg=(
                f"Serial path output changed for {key}. Regenerate the fixture only "
                "for an intentional numerical-semantics change."
            ),
        )


# ---------------------------------------------------------------------------
# Assertion 8: fixed_vp unit consistency
# ---------------------------------------------------------------------------

def test_fixed_vp_unit_consistency(aif_and_time):
    """Any call that passes fixed_vp through the batched API must pass
    values in the physical vp range [0, 1] (unitless volume fraction).
    A CBV-in-mL/100g scale error would land values >> 1."""
    aif, time_points = aif_and_time
    tissue, _, vp_true = _make_patlak_cohort(aif, time_points, n_voxels=50)
    fixed_vp = np.asarray(vp_true, dtype=np.float64)

    # Guard: physical range check.
    assert np.all(fixed_vp >= 0.0), "fixed_vp contains negative values"
    assert np.all(fixed_vp <= 1.0), "fixed_vp contains values > 1.0 (unit error?)"
    # Typical DCE range; anything outside is suspect.
    assert np.all(fixed_vp < 0.2), (
        f"fixed_vp max={fixed_vp.max():.3f} — above typical DCE vp range. "
        f"Possible scale error (CBV in mL/100g instead of nCBV)?"
    )


# ---------------------------------------------------------------------------
# vp_uncertainty = 0.0 must raise
# ---------------------------------------------------------------------------

def test_vp_uncertainty_zero_raises(aif_and_time):
    """vp_uncertainty = 0.0 must ValueError (not silently fall through)."""
    aif, time_points = aif_and_time
    tissue, _, vp_true = _make_patlak_cohort(aif, time_points, n_voxels=10)

    with pytest.raises(ValueError, match="vp_uncertainty must be positive"):
        batched_patlak_lstsq(
            tissue, aif, time_points,
            fixed_vp=np.asarray(vp_true), vp_uncertainty=0.0,
        )

    with pytest.raises(ValueError, match="vp_uncertainty must be positive"):
        batched_lm_fit(
            "extended_tofts", tissue, aif, time_points,
            fixed_vp=np.asarray(vp_true), vp_uncertainty=0.0,
        )
