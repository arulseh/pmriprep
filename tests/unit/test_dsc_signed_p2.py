"""Selection and signed-integration contracts for DSC deconvolution."""

from collections.abc import Callable
import json

import numpy as np
import pytest

from pmriprep.interfaces.dsc_compute import ComputeDSCMaps
from pmriprep.utils.provenance import enrich_provenance_from_qc
from pmriprep.utils.sidecar_builders import dump_sidecar_json
from pmriprep.utils.dsc_constants import LCURVE_N_LAMBDAS
from pmriprep.utils.dsc_deconvolution import (
    DeconvolutionResultBase,
    TikhonovSelection,
    _prepare_tikhonov_system,
    _select_lcurve_corner,
    build_block_circulant_matrix,
    build_volterra_matrix,
    svd_deconvolve_oscillation,
    svd_deconvolve_oscillation_batch,
    svd_deconvolve_standard,
    svd_deconvolve_standard_batch,
    tikhonov_deconvolve,
    tikhonov_deconvolve_batch,
)


@pytest.mark.parametrize("values,index,status", [
    ([0, 8, 0, 2, 0], 3, TikhonovSelection.POSITIVE_CORNER),
    ([-8, -2, -7, -1, -3], 3, TikhonovSelection.NONPOSITIVE_CORNER),
    ([-2, 0, -1], 1, TikhonovSelection.NONPOSITIVE_CORNER),
    ([0, 2, 0, 2, 0], 3, TikhonovSelection.POSITIVE_CORNER),
    ([0, 2, 2, 0], 3, TikhonovSelection.UPPER_ENDPOINT),
    ([0, 1, 2, 3], 3, TikhonovSelection.UPPER_ENDPOINT),
    ([3, 2, 1, 0], 3, TikhonovSelection.UPPER_ENDPOINT),
    ([1, 1, 1], 2, TikhonovSelection.UPPER_ENDPOINT),
    ([np.nan, 2, 0], 2, TikhonovSelection.UPPER_ENDPOINT),
    ([-np.inf, -np.inf, -np.inf], 2, TikhonovSelection.UPPER_ENDPOINT),
])
def test_p2_strict_interior_selection(
    values: list[float], index: int, status: TikhonovSelection,
) -> None:
    """Only strict finite interior maxima compete, in descending grid order."""
    indices, states = _select_lcurve_corner(np.asarray(values, dtype=float)[:, None])
    assert indices[0] == index
    assert states[0] == status


def _curves() -> tuple[np.ndarray, np.ndarray, float]:
    """Return a delayed arterial curve and a noisy causal tissue observation."""
    dt = 1.5
    t = np.arange(40) * dt
    aif = np.maximum(t - 9, 0) ** 3 * np.exp(-np.maximum(t - 9, 0) / 2)
    aif /= aif.max()
    tissue = build_volterra_matrix(aif, dt) @ np.exp(-t / 14)
    tissue += np.random.default_rng(23).normal(0, 0.025 * tissue.max(), t.size)
    return aif, tissue, dt


def test_p2_grid_and_inactive_mask_contract() -> None:
    """Zero observations and masked nonfinite observations have distinct accounting."""
    aif, tissue, dt = _curves()
    curves = np.stack([tissue, np.zeros_like(tissue), np.full_like(tissue, np.nan)])
    batch = tikhonov_deconvolve_batch(curves, aif, dt, mask=np.array([True, True, False]))
    system = _prepare_tikhonov_system(aif, dt, 0.001, 10)
    scale = np.linalg.svd(build_volterra_matrix(aif, dt), compute_uv=False)[0]
    assert LCURVE_N_LAMBDAS == 100
    np.testing.assert_allclose(system.lambdas, scale * np.logspace(-3, 1, 100))
    assert np.any(batch[4][0] == system.lambdas)
    np.testing.assert_array_equal(batch[0][1:], 0)
    np.testing.assert_array_equal(batch[4][1:], 0)
    np.testing.assert_array_equal(batch[5][1:], TikhonovSelection.INACTIVE)
    assert np.isnan(batch[3][1])
    assert batch[3][2] == 0
    scalar = tikhonov_deconvolve(np.zeros_like(tissue), aif, dt)
    assert scalar.selection_status == TikhonovSelection.INACTIVE
    assert not scalar.lcurve_corner_found
    assert np.isnan(scalar.mtt)


@pytest.mark.parametrize("valid_neighbour", [True, False])
def test_tikhonov_invalid_input_has_separate_qc(valid_neighbour: bool) -> None:
    """Invalid tissue is missing and excluded from active lambda/corner statistics."""
    aif, tissue, dt = _curves()
    missing = np.full_like(tissue, np.nan)
    curves = np.stack([tissue if valid_neighbour else missing,
                       np.zeros_like(tissue) if valid_neighbour else missing,
                       missing, missing])
    mask = np.array([True, True, True, False])
    cbf, cbv, mtt, tmax, qc = ComputeDSCMaps()._deconvolve_svd(
        curves, aif, mask, "tikhonov", dt, 0.2, 0.095,
    )
    for values in (cbf, cbv, mtt, tmax):
        assert np.isnan(values[2])
        assert values[3] == 0
    counts = qc["lcurve_selection_counts"]
    assert counts["invalid_input"] == (1 if valid_neighbour else 3)
    assert counts["inactive"] == (1 if valid_neighbour else 0)
    assert sum(counts.values()) == 3
    if valid_neighbour:
        scalar = tikhonov_deconvolve(tissue, aif, dt)
        assert counts[scalar.selection_status.name.lower()] == 1
        assert qc["lambda_mean"] == pytest.approx(scalar.lambda_optimal)
        assert qc["lcurve_corner_fraction"] == float(scalar.lcurve_corner_found)
        assert qc["lcurve_upper_endpoint_fraction"] == float(not scalar.lcurve_corner_found)
    else:
        assert np.isnan(cbf[0])
        assert qc["lambda_mean"] is None
        assert qc["lcurve_corner_fraction"] is None
        assert qc["lcurve_upper_endpoint_fraction"] is None
    sidecar = enrich_provenance_from_qc(
        {"PerfusionProcessing": {"Deconvolution": {"method": "tikhonov"}}},
        {"deconvolution": qc}, "DSC",
    )
    assert sidecar["PerfusionProcessing"]["Deconvolution"]["lcurve_selection_counts"] == counts


@pytest.mark.parametrize("case", ["empty_mask", "zero_tissue", "invalid_tissue"])
def test_tikhonov_no_active_fits_serializes_null_statistics(case: str) -> None:
    """Undefined selection statistics retain explicit nulls in strict JSON."""
    aif, tissue, dt = _curves()
    curves = np.zeros((1, tissue.size))
    if case == "invalid_tissue":
        curves[:] = np.nan
    mask = np.array([case != "empty_mask"])
    qc = ComputeDSCMaps()._deconvolve_svd(
        curves, aif, mask, "tikhonov", dt, 0.2, 0.095,
    )[-1]
    fields = ("lambda_mean", "lcurve_corner_fraction", "lcurve_upper_endpoint_fraction")
    assert {field: qc[field] for field in fields} == dict.fromkeys(fields)
    sidecar = enrich_provenance_from_qc(
        {"PerfusionProcessing": {"Deconvolution": {"method": "tikhonov"}}},
        {"deconvolution": qc}, "DSC",
    )
    serialized = dump_sidecar_json(sidecar)

    def reject_nonfinite(value: str) -> None:
        pytest.fail(f"Non-standard JSON token: {value}")

    decoded = json.loads(serialized, parse_constant=reject_nonfinite)
    deconvolution = decoded["PerfusionProcessing"]["Deconvolution"]
    assert {field: deconvolution[field] for field in fields} == dict.fromkeys(fields)
    json.dumps(decoded, allow_nan=False)


def test_p2_matches_independent_finite_difference_selection() -> None:
    """Direct augmented least squares and norm differences recover the selected grid point."""
    aif, tissue, dt = _curves()
    n = aif.size
    matrix = build_volterra_matrix(aif, dt)
    penalty = np.eye(n)
    penalty[np.arange(2, n), np.arange(1, n - 1)] = -1
    scale = np.linalg.svd(matrix, compute_uv=False)[0]
    grid = scale * np.logspace(-3, 1, 100)
    h = 1e-3

    def norms(offset: float) -> np.ndarray:
        result = []
        for lam in grid * np.exp(offset):
            r = np.linalg.lstsq(
                np.vstack([matrix, lam * penalty]),
                np.r_[tissue, np.zeros(n)], rcond=None,
            )[0]
            result.append([np.log(np.linalg.norm(matrix @ r - tissue)),
                           np.log(np.linalg.norm(penalty @ r))])
        return np.array(result)

    mid, plus, minus = norms(0), norms(h), norms(-h)
    first = (plus - minus) / (2 * h)
    second = (plus - 2 * mid + minus) / h ** 2
    curvature = (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / (
        np.sum(first ** 2, axis=1) ** 1.5
    )
    pick = len(grid) - 1
    for i in range(len(grid) - 2, 0, -1):
        if curvature[i] > curvature[i - 1] and curvature[i] > curvature[i + 1]:
            pick = i
            break
    fit = tikhonov_deconvolve(tissue, aif, dt)
    assert fit.lambda_optimal == pytest.approx(grid[pick], rel=1e-12)
    reference = np.linalg.lstsq(
        np.vstack([matrix, grid[pick] * penalty]), np.r_[tissue, np.zeros(n)], rcond=None,
    )[0]
    np.testing.assert_allclose(fit.residue, reference, rtol=1e-10, atol=1e-12)


def test_no_corner_fit_matches_upper_endpoint_least_squares() -> None:
    """A no-corner fit must solve the selected problem despite poor flow recovery."""
    n, dt = 30, 1.5
    time = np.arange(n) * dt
    aif = (time / 3) * np.exp(-time / 5)
    matrix = build_volterra_matrix(aif, dt)
    tissue = matrix @ (2 * np.exp(-time / 8))
    penalty = np.eye(n)
    penalty[np.arange(2, n), np.arange(1, n - 1)] = -1
    upper_lambda = 10 * np.linalg.svd(matrix, compute_uv=False)[0]
    reference = np.linalg.lstsq(
        np.vstack([matrix, upper_lambda * penalty]),
        np.r_[tissue, np.zeros(n)], rcond=None,
    )[0]
    fit = tikhonov_deconvolve(tissue, aif, dt)
    assert fit.selection_status == TikhonovSelection.UPPER_ENDPOINT
    assert fit.lambda_optimal == pytest.approx(upper_lambda, rel=1e-12)
    np.testing.assert_allclose(fit.residue, reference, rtol=1e-10, atol=1e-12)
    assert fit.cbf == pytest.approx(reference.max(), rel=1e-12)
    assert fit.cbv == pytest.approx(np.trapezoid(reference, dx=dt), rel=1e-12)


@pytest.mark.parametrize("noise", [0, 1e-8, 0.03])
def test_lambda_always_uses_discrete_grid_and_is_scale_invariant(noise: float) -> None:
    """Near-noiseless inputs use the same grid rule as noisy inputs."""
    aif, _, dt = _curves()
    n = aif.size
    tissue = build_volterra_matrix(aif, dt) @ np.exp(-np.arange(n) * dt / 10)
    tissue += np.random.default_rng(41).normal(0, noise * tissue.max(), n)
    reference = tikhonov_deconvolve(tissue, aif, dt)
    scaled = tikhonov_deconvolve(tissue * 100, aif, dt)
    grid = _prepare_tikhonov_system(aif, dt, 0.001, 10).lambdas
    assert np.any(grid == reference.lambda_optimal)
    assert scaled.lambda_optimal == reference.lambda_optimal
    np.testing.assert_allclose(scaled.residue, reference.residue * 100, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("dt", [0.5, 1.5, 3.0])
def test_osvd_periodic_area_shift_and_sampling(dt: float) -> None:
    """Circular closure conserves signed area under shifts and time-step changes."""
    aif = np.r_[1.0, np.zeros(23)]
    tissue = np.zeros(24)
    tissue[6:10] = [1.0, 2.0, -0.8, 0.4]
    curves = np.stack([np.roll(tissue, shift) for shift in [-4, 0, 7]])
    batch = svd_deconvolve_oscillation_batch(
        curves, aif, dt, threshold_min=0.2, threshold_max=0.2,
    )
    expected_area = np.sum(tissue) / (4 / 6 + 1 / 6)
    for i in range(3):
        result = svd_deconvolve_oscillation(
            curves[i], aif, dt, threshold_min=0.2, threshold_max=0.2,
        )
        assert np.any(result.residue < 0)
        assert result.cbv == pytest.approx(expected_area, rel=1e-12)
        assert result.cbv == pytest.approx(dt * result.residue.sum(), rel=1e-12)
        np.testing.assert_allclose(result.residue, batch[0][i], atol=1e-12)
        assert result.mtt == pytest.approx(batch[3][i])
    unit = svd_deconvolve_oscillation(tissue, aif, 1, threshold_min=0.2, threshold_max=0.2)
    np.testing.assert_allclose(batch[1] * dt, unit.cbf, rtol=1e-12)
    np.testing.assert_allclose(batch[3] / dt, unit.mtt, rtol=1e-12)


@pytest.mark.parametrize("method,scalar,batch", [
    ("sSVD", svd_deconvolve_standard, svd_deconvolve_standard_batch),
    ("oSVD", svd_deconvolve_oscillation, svd_deconvolve_oscillation_batch),
    ("tikhonov", tikhonov_deconvolve, tikhonov_deconvolve_batch),
])
def test_production_dispatch_preserves_signed_area(
    method: str,
    scalar: Callable[..., DeconvolutionResultBase],
    batch: Callable[..., tuple[np.ndarray, ...]],
) -> None:
    """Actual dispatch matches signed scalar/batch solutions and accounts for invalid areas."""
    aif, tissue, dt = _curves()
    curves = np.stack([tissue, -tissue, np.zeros_like(tissue), tissue])
    mask = np.array([True, True, True, False])
    direct = batch(curves, aif, dt, mask=mask)
    cbf, cbv, mtt, _, qc = ComputeDSCMaps()._deconvolve_svd(
        curves, aif, mask, method, dt, 0.2, 0.095,
    )
    for i in range(3):
        result = scalar(curves[i], aif, dt)
        np.testing.assert_allclose(direct[0][i], result.residue, atol=1e-10)
        assert cbv[i] == pytest.approx(result.cbv, abs=1e-10)
    np.testing.assert_allclose(cbf, direct[1])
    np.testing.assert_allclose(mtt, direct[3])
    assert cbv[1] < 0
    assert np.isnan(mtt[1]) and np.isnan(mtt[2])
    assert cbv[3] == 0 and mtt[3] == 0
    assert qc["residue_post_fit_clipping"] is False
    diagnostics = qc["residue_diagnostics"]
    assert diagnostics["in_mask_voxels"] == 3
    assert diagnostics["negative_cbv_voxels"] == 1
    assert diagnostics["zero_cbv_voxels"] == 1
    assert diagnostics["invalid_residue_mtt_voxels"] == 2
    residue = direct[0][mask]
    negative = -np.minimum(residue, 0)
    neg_area = (dt * negative.sum(axis=-1) if method == "oSVD"
                else np.trapezoid(negative, dx=dt, axis=-1))
    absolute_area = (dt * np.abs(residue).sum(axis=-1) if method == "oSVD"
                     else np.trapezoid(np.abs(residue), dx=dt, axis=-1))
    valid = absolute_area > 0
    assert diagnostics["negative_area_fraction_mean"] == pytest.approx(
        np.mean(neg_area[valid] / absolute_area[valid])
    )
    if method == "tikhonov":
        assert qc["lambda_grid_size"] == 100
        assert qc["regularization_selection"] == "lcurve_upper_extremum"
        assert sum(qc["lcurve_selection_counts"].values()) == 3
        assert qc["lcurve_selection_counts"]["inactive"] == 1
    primary = ComputeDSCMaps._finalize_mtt(method, mtt, cbf, np.ones_like(cbf))
    positive = cbf > 1e-6
    np.testing.assert_allclose(primary[positive], 1 / cbf[positive])
    aux = np.zeros_like(cbf)
    maps = ComputeDSCMaps._finalize_dsc_maps(
        cbf=cbf, cbv=cbv, cbv_auc=np.ones_like(cbv), mtt=primary,
        tmax=aux, ttp=aux, auc_raw=aux, auc_corrected=aux,
        auc_difference=aux, cnr=aux, tsnr=aux, psr=aux, cbf_std=None,
        mask_bool=mask,
    )
    assert np.isnan(maps.rel_cbv[1])
    assert maps.rel_cbv[2] == 0
    assert maps.rel_cbv[3] == 0
    np.testing.assert_allclose(maps.rel_cbv_auc[mask], 1)
    np.testing.assert_allclose(maps.mtt[mask], primary[mask])
    sidecar = enrich_provenance_from_qc(
        {"PerfusionProcessing": {"Deconvolution": {"method": method}}},
        {"deconvolution": qc}, "DSC",
    )
    assert sidecar["PerfusionProcessing"]["Deconvolution"]["residue_diagnostics"] == diagnostics


def test_selection_qc_separates_active_and_excluded_voxels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All P2 outcomes are counted, with fractions over active in-mask fits only."""
    def fits(*args: object, **kwargs: object) -> tuple[np.ndarray, ...]:
        residues = np.ones((5, 4))
        ones = np.ones(5)
        return (residues, ones, ones, ones, np.array([0, 2, 4, 6, 999]),
                np.array([0, 1, 2, 3, 3], dtype=np.int8))

    monkeypatch.setattr(
        "pmriprep.utils.dsc_deconvolution.tikhonov_deconvolve_batch", fits,
    )
    mask = np.array([True, True, True, True, False])
    qc = ComputeDSCMaps()._deconvolve_svd(
        np.ones((5, 4)), np.ones(4), mask, "tikhonov", 1, 0.2, 0.095,
    )[-1]
    assert qc["lcurve_selection_counts"] == {
        "inactive": 1, "positive_corner": 1, "nonpositive_corner": 1, "upper_endpoint": 1,
        "invalid_input": 0,
    }
    assert qc["lambda_mean"] == 4
    assert qc["lcurve_corner_fraction"] == pytest.approx(2 / 3)
    assert qc["lcurve_upper_endpoint_fraction"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("case", ["mixed", "empty", "invalid"])
def test_osvd_target_status_reaches_qc_and_sidecar(case: str) -> None:
    """An unmet target retains the fit and is counted only inside the mask."""
    x = np.maximum(np.arange(32) - 4, 0)
    aif = x ** 3 * np.exp(-x / 0.25)
    aif /= aif.max()
    clean = (build_block_circulant_matrix(aif)
             @ np.r_[0.1 * np.exp(-np.arange(32) / 4), np.zeros(32)])[:32]
    noisy = clean + np.random.default_rng(0).normal(0, 0.1, 32)
    curves = np.stack([clean, noisy, np.full_like(clean, np.nan), np.full_like(clean, np.nan)])
    mask = np.array([case == "mixed", case == "mixed", case != "empty", False])
    cbf, cbv, mtt, _, qc = ComputeDSCMaps()._deconvolve_svd(
        curves, aif, mask, "oSVD", 1, 0.2, 0.095,
    )
    status = qc["oscillation_target"]
    assert status == {
        "target": 0.095, "met_voxels": 1 if case == "mixed" else 0,
        "unmet_voxels": 1 if case == "mixed" else 0,
        "nonfinite_oi_voxels": 0 if case == "empty" else 1,
        "unmet_fraction": 0.5 if case == "mixed" else None,
        "fraction_denominator": "finite OI in-mask voxels", "unmet_policy": "retain_and_flag",
    }
    if case == "mixed":
        for i, converged in enumerate([True, False]):
            fit = svd_deconvolve_oscillation(curves[i], aif)
            assert fit.converged is converged
            np.testing.assert_allclose([cbf[i], cbv[i], mtt[i]], [fit.cbf, fit.cbv, fit.mtt])
        assert np.all(np.isfinite([cbf[1], cbv[1], mtt[1]]))
        assert np.isfinite(qc["oi_mean"]) and np.isfinite(qc["threshold_mean"])
    else:
        assert qc["oi_mean"] is qc["threshold_mean"] is None
    sidecar = enrich_provenance_from_qc(
        {"PerfusionProcessing": {"Deconvolution": {"method": "oSVD"}}},
        {"deconvolution": qc}, "DSC",
    )
    assert sidecar["PerfusionProcessing"]["Deconvolution"]["oscillation_target"] == status
    if case != "mixed":
        deconvolution = sidecar["PerfusionProcessing"]["Deconvolution"]
        assert deconvolution["oscillation_index_mean"] is None
        assert deconvolution["threshold_mean"] is None
