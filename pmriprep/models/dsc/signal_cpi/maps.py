"""Native-unit image assembly for the single signal-CPI production route."""

from collections import Counter
from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .acquisition import CPIAcquisition, fit_selected_voxels
from .fallback import resolve_fallbacks
from .field import fit_selected_field
from .execution import CPIExecutor, plan_resources

logger = logging.getLogger("nipype.interface")


@dataclass(frozen=True)
class CPIMaps:
    """Measurements, selected-fit predictions and distinct replacement provenance."""

    cbf: np.ndarray
    cbv: np.ndarray
    mtt: np.ndarray
    delay: np.ndarray
    cbf_std: np.ndarray
    r_squared: np.ndarray
    fallback_mask: np.ndarray
    predictions: np.ndarray
    fitted_k2: np.ndarray
    dispersion: np.ndarray
    converged: np.ndarray
    qc: dict[str, Any]


def fit_cpi_maps(
    acquisition: CPIAcquisition,
    raw_concentration: np.ndarray,
    corrected_concentration: np.ndarray,
    *,
    oi_target: float,
    prediction_file: str | Path,
    n_procs: int = 1,
    memory_gb: float | None = None,
) -> CPIMaps:
    """Fit the selected family and assemble outputs without a separate QC refit.

    Parameters
    ----------
    acquisition : CPIAcquisition
        Complete prepared image with ordinary or spatial inference selected.
    raw_concentration, corrected_concentration : ndarray
        Baseline-centred inputs for optional oSVD, on the original signal grid.
    oi_target : float
        Configured oSVD oscillation threshold.
    prediction_file : str or Path
        Node-local memory-mapped storage for selected CPI signal predictions.
        This avoids retaining full posterior/covariance arrays across the image.
    n_procs : int, optional
        Maximum fitting processes; shared VI/SVB and branch selection are unchanged.
    memory_gb : float, optional
        Scheduler's node allocation, used to cap workers and queued work.

    Returns
    -------
    CPIMaps
        Native flow/volume units and seconds; rejected values remain missing.
        Primary AUC-based CBV is calculated separately by the compute interface.
    """
    a = acquisition.branches[0]
    points = max((b.decision.selected for b in acquisition.branches if b.decision), default=5)
    # Maps plus image-sized spatial marginal state/row bookkeeping allowance.
    extra = len(a.signals) * (80 + 8 * a.time.size)
    if a.inference_method == "svb":
        extra += int(np.sum(a.fit_support)) * (4096 + 8 * (8 * points + 4 * a.time.size))
    resources = plan_resources(n_procs, int(np.sum(a.fit_support)), a.time.size, points,
                               memory_gb=memory_gb, extra_parent_bytes=extra)
    logger.info("CPI fitting uses %d/%d processes, one numerical thread each; sloppy=%s",
                resources.processes, resources.requested_processes, a.sloppy)
    with CPIExecutor(resources) as executor:
        result = _assemble_maps(acquisition, raw_concentration, corrected_concentration,
                                oi_target, prediction_file, executor)
    result.qc["execution"] = asdict(resources) | {"sloppy": a.sloppy}
    return result


def _assemble_maps(
    acquisition: CPIAcquisition, raw_concentration: np.ndarray,
    corrected_concentration: np.ndarray, oi_target: float,
    prediction_file: str | Path, executor: CPIExecutor,
) -> CPIMaps:
    """Assemble image-indexed fits with the pool lifetime owned by the caller."""
    a = acquisition.branches[0]
    shape = a.spatial_shape
    maps = [np.full(shape, np.nan) for _ in range(8)]
    cbf, cbv, mtt, delay, std, r2, k2, dispersion = maps
    replaced = np.zeros(shape, dtype=np.uint8)
    converged = np.zeros(shape, dtype=bool)
    predictions = np.lib.format.open_memmap(
        prediction_file, mode="w+", dtype=np.float64, shape=(*shape, a.time.size)
    )
    predictions[:] = np.nan
    spatial_sweeps = 0
    if a.inference_method == "svb":
        field = fit_selected_field(acquisition, executor=executor)
        rows = iter(field.voxels)
        spatial_sweeps = field.spatial_sweeps
    else:
        rows = fit_selected_voxels(acquisition, executor=executor, ordered=False)
    sources: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    failed: Counter[str] = Counter()
    requested = attempted = n_converged = accepted = gamma = 0
    delay_bound = spacing_bound = 0
    for output in resolve_fallbacks(
        acquisition,
        rows,
        raw_concentration=raw_concentration,
        corrected_concentration=corrected_concentration,
        oi_target=oi_target,
    ):
        row = output.fit
        i = row.flat_index
        sources[output.source] += 1
        reasons.update(row.reasons)
        requested += row.status == "fallback_requested"
        attempted += output.fallback_attempted
        if output.fallback_reason is not None:
            failed[output.fallback_reason] += 1
        for target, value in zip(
            (cbf, cbv, mtt, delay, std, k2),
            (
                output.cbf,
                output.cbv,
                output.mtt,
                output.delay,
                output.cbf_std,
                output.fitted_k2,
            ),
            strict=True,
        ):
            target.flat[i] = value
        replaced.flat[i] = output.source == "osvd"
        if row.fitted_signal is not None:
            predictions.reshape(a.signals.shape)[i] = row.fitted_signal
        if row.assessment is not None:
            r2.flat[i] = (
                row.assessment.r_squared
                if row.assessment.r_squared is not None
                else np.nan
            )
            converged.flat[i] = row.assessment.converged
            n_converged += row.assessment.converged
        if output.source == "cpi":
            accepted += 1
            gamma += row.family == "cpi_vtf"
            dispersion.flat[i] = float(row.family == "cpi_vtf")
            if row.assessment is not None:
                delay_bound += "arrival_at_bound" in row.assessment.warnings
                spacing_bound += "spacing_at_bound" in row.assessment.warnings
    predictions.flush()
    n_mask = int(np.sum(a.fit_support))
    eligible = a.fit_support & a.valid_signal
    if a.guidance_status is not None:
        eligible &= a.guidance_status == 0
    admissible = int(np.sum(eligible))
    qc = dict(
        method="cpi",
        inference=a.inference_method,
        fit_domain="signal",
        cpi_dispersion=acquisition.options.dispersion,
        cpi_infer_delay=a.infer_delay,
        cpi_control_points={
            b.family: asdict(b.decision) if b.decision else None
            for b in acquisition.branches
        },
        cpi_leakage_method=a.leakage_method,
        cpi_fallback=acquisition.options.fallback,
        n_mask_voxels=n_mask,
        n_fit_admissible=admissible,
        n_fit_inadmissible=n_mask - admissible,
        n_converged=n_converged,
        fe_plateau_rate=n_converged / max(1, admissible),
        n_cpi_retained=accepted,
        n_cpi_gamma=gamma,
        cpi_gamma_fraction=gamma / max(1, accepted),
        n_fallback_requested=requested,
        n_fallback_attempted=attempted,
        n_fallback_osvd=sources["osvd"],
        n_fallback_failed=attempted - sources["osvd"],
        n_missing=sources["missing"],
        fallback_requested_rate=requested / max(1, n_mask),
        fallback_rate=sources["osvd"] / max(1, n_mask),
        fallback_fraction_denominator="fitting support",
        rejection_reasons=dict(reasons),
        fallback_outcomes=dict(failed),
        cpi_spacing_at_bound_n=spacing_bound,
        cpi_spacing_at_bound_fraction=spacing_bound / max(1, accepted),
        tmax_method="fitted_arrival_delay" if a.infer_delay else "not_estimated",
        fallback_tmax_method="osvd_residue_peak_delay"
        if acquisition.options.fallback == "osvd"
        else "disabled",
        spatial_sweeps=spatial_sweeps,
        cpi_selection_stage="ordinary_vi_before_spatial_refinement",
    )
    if a.infer_delay:
        qc.update(
            delay_at_bound_n=delay_bound,
            delay_at_bound_fraction=delay_bound / max(1, accepted),
        )
    finite_r2 = r2[np.isfinite(r2)]
    if finite_r2.size:
        qc.update(
            r_squared_mean=float(finite_r2.mean()), r_squared_std=float(finite_r2.std())
        )
    return CPIMaps(
        cbf,
        cbv,
        mtt,
        delay,
        std,
        r2,
        replaced,
        predictions,
        k2,
        dispersion,
        converged,
        qc,
    )
