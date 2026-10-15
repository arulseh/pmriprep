"""Golden characterization harness for ``AIFDetection._run_interface``.

This pins the end-to-end behaviour of the DCE AIF detection interface across a
config matrix BEFORE the ~790-line ``_run_interface`` is decomposed into phase
methods, and is the byte-identity gate for every extraction step. It also closes
a real coverage gap: the interface was previously exercised end-to-end only by
``TestDetectedAIFEmissionInSitu``.

Each matrix cell runs ``_run_interface`` (via ``iface.run()``), canonicalises
``self._results`` at FULL float precision (``float.hex``) plus the on-disk AIF
mask (nonzero count + coords — the mask is not in ``self._results``), and diffs
against a committed baseline. Set ``PMRIPREP_REGEN_DCE_AIF_GOLDEN=1`` to
regenerate the baseline after an INTENTIONAL change.

Determinism: the emit/finalize path has no RNG; detection is deterministic on
fixed input. Cascade-dependent cells mock ``_run_two_pass_cascade`` with a fixed
``CascadeResult`` so the real emit runs on fixed synthetic data.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from pmriprep.interfaces.dce_aif import (
    AIFDetection,
    CascadeResult,
    GateQC,
    TierResult,
)

_REGEN_ENV = "PMRIPREP_REGEN_DCE_AIF_GOLDEN"
_BASELINE = Path(__file__).parent / "data" / "dce_aif_run_interface_golden.json"


# ---------------------------------------------------------------------------
# Canonicalisation — exact, JSON-stable
# ---------------------------------------------------------------------------
def _canon(obj: Any) -> Any:
    """Recursively canonicalise to a JSON-stable, full-precision structure.

    Floats -> ``float.hex()`` (exact); numpy scalars/arrays -> python; enums ->
    their string; dicts -> key-sorted. This makes a verbatim block-hoist compare
    equal while catching any low-order-digit reimplementation.
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return {"__f__": float(obj).hex()}
    if isinstance(obj, (np.floating,)):
        return {"__f__": float(obj).hex()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_canon(x) for x in obj.tolist()]
    if isinstance(obj, (set, frozenset)):
        # Sort canonicalised elements so set iteration order can't diff.
        return {"__set__": sorted(json.dumps(_canon(x)) for x in obj)}
    if isinstance(obj, (list, tuple)):
        return [_canon(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _canon(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, str):
        # Redact volatile absolute paths (e.g. qc_metrics["external_aif_file"])
        # to their stable basename so per-run tmp dirs don't spuriously diff.
        return os.path.basename(obj) if obj.startswith("/") else obj
    if isinstance(obj, int) or obj is None:
        return obj
    # Enums (e.g. QCStatus) and anything else -> stable string.
    if hasattr(obj, "name") and hasattr(obj, "value"):
        return f"<enum:{obj.__class__.__name__}.{obj.name}>"
    return str(obj)


def _snapshot(iface: AIFDetection) -> dict[str, Any]:
    """Canonical snapshot of a completed interface run.

    Captures ``self._results`` (minus the volatile output file paths) plus the
    on-disk AIF mask's nonzero-voxel count and coords (the mask is built by
    distinct emit logic and is absent from ``self._results``).
    """
    res = dict(iface._results)  # type: ignore[attr-defined]
    mask_path = res.pop("out_aif_mask", None)
    res.pop("out_aif_curve", None)  # volatile path; curve values kept via "aif_curve"

    mask_summary: dict[str, Any] = {"present": False}
    if mask_path and Path(mask_path).exists():
        mask = np.asarray(nib.load(mask_path).get_fdata())
        nz = np.argwhere(mask != 0)
        mask_summary = {
            "present": True,
            "nonzero_count": int(nz.shape[0]),
            "nonzero_coords": sorted([list(map(int, c)) for c in nz]),
            "dtype": str(nib.load(mask_path).get_data_dtype()),
        }
    return {"results": _canon(res), "aif_mask": mask_summary}


# ---------------------------------------------------------------------------
# Synthetic data + mock builders
# ---------------------------------------------------------------------------
def _make_conc(
    shape: tuple[int, int, int, int] = (20, 20, 20, 40),
    baseline_frames: int = 5,
    arterial_coords: list[tuple[int, int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 4D DCE concentration (sharp arterial + broad tissue) + mask."""
    nx, ny, nz, nt = shape
    conc = np.zeros(shape, dtype=np.float64)
    mask = np.ones((nx, ny, nz), dtype=np.uint8)
    t = np.arange(nt, dtype=np.float64)

    tissue = 0.5 * np.exp(-0.5 * ((t - (baseline_frames + 8)) / 6.0) ** 2)
    tissue[:baseline_frames] = 0.0
    conc[:] = tissue[np.newaxis, np.newaxis, np.newaxis, :]

    art = 5.0 * np.exp(-0.5 * ((t - (baseline_frames + 2)) / 1.5) ** 2)
    art[:baseline_frames] = 0.0
    for x, y, z in arterial_coords or [(10, 10, 10), (10, 11, 10), (11, 10, 10)]:
        conc[x, y, z, :] = art
    return conc, mask


def _bilateral_affine() -> np.ndarray:
    aff = np.eye(4, dtype=np.float64)
    aff[0, 3] = -10.0  # world_x = i - 10 -> i<10 left, i>=10 right
    return aff


def _write_niftis(tmp_path: Path, conc: np.ndarray, mask: np.ndarray) -> tuple[str, str]:
    aff = _bilateral_affine()
    conc_f = tmp_path / "conc.nii.gz"
    mask_f = tmp_path / "brain.nii.gz"
    nib.save(nib.Nifti1Image(conc, aff), str(conc_f))
    nib.save(nib.Nifti1Image(mask, aff), str(mask_f))
    return str(conc_f), str(mask_f)


def _full_metric(peak_mm: float = 4.0, venous: bool = False) -> dict[str, Any]:
    return {
        "peak_mm": peak_mm, "fwhm_seconds": 12.0, "fwhm_frames": 3.0,
        "washout": 0.8, "cnr": 20.0, "auc": 60.0, "upstroke_slope": 1.0,
        "tier_used": 1, "is_likely_venous": venous, "peak_time_frame": 7,
    }


def _bolus(n: int = 40, peak: float = 6.0) -> np.ndarray:
    """A non-trivial arterial-shaped curve so the reuse branch emits real values."""
    t = np.arange(float(n))
    c = peak * np.exp(-0.5 * ((t - 8.0) / 2.0) ** 2)
    c[:5] = 0.0
    return c


def _tier_result(cands: list[tuple[int, int, int]], status: str = "PASS") -> TierResult:
    return TierResult(
        candidates=list(cands), scores=[0.9] * len(cands),
        metrics=[_full_metric()] * len(cands), curve_scaled=_bolus(),
        scale_factor=1.5, required_scale_factor=1.5,
        qc=_gate_qc(), atlas_tier="MCA_ACA_dce_strict",
        hemisphere="bilateral", n_slabs=2, status=status, gate_decisions=[],
    )


def _gate_qc() -> GateQC:
    return GateQC(
        peak_mm=8.0, fwhm_seconds=14.0, auc_mm_s=800.0, washout=0.8,
        multi_bump_count=0, late_tail_per_peak=0.20, ratio_30=0.30,
    )


def _cascade(
    candidates: list[tuple[int, int, int]],
    scores: list[float],
    tier_result: TierResult | None,
    detection_method: str = "atlas",
) -> CascadeResult:
    return CascadeResult(
        candidates=list(candidates), scores=list(scores),
        metrics_list=[_full_metric() for _ in candidates],
        detection_method=detection_method, atlas_tier_used="MCA_ACA_dce_strict",
        cascade_tiers_attempted=1, anchor_outcomes=[], tier_result=tier_result,
        tier_gate_decisions=[],
        atlas_search_outcome=(
            "selected" if detection_method == "atlas" else "exhausted"
        ),
        atlas_search_support_voxels=125,
    )


def _base_iface(conc_f: str, mask_f: str, **overrides: Any) -> AIFDetection:
    iface = AIFDetection()
    iface.inputs.concentration_file = conc_f
    iface.inputs.brain_mask_file = mask_f
    iface.inputs.tr = 4.0
    iface.inputs.hematocrit = 0.42
    for k, v in overrides.items():
        setattr(iface.inputs, k, v)
    return iface


_IDENTITY_BALANCE = lambda cands, scs, mets, *a, **k: (cands, scs, mets)  # noqa: E731


# ---------------------------------------------------------------------------
# Matrix cells — each returns a canonical snapshot
# ---------------------------------------------------------------------------
def _cell_population(tmp_path: Path) -> dict[str, Any]:
    """Cell 1: aif_method='population' — population_default + population emit."""
    conc, mask = _make_conc()
    conc_f, mask_f = _write_niftis(tmp_path, conc, mask)
    iface = _base_iface(conc_f, mask_f, aif_method="population")
    iface.run()
    return _snapshot(iface)


def test_population_aif_preserves_arrival_at_coarse_sampling(tmp_path: Path) -> None:
    """An undersampled population peak retains its configured arrival time."""
    baseline_frames = 4
    tr_seconds = 20.0
    conc, mask = _make_conc(baseline_frames=baseline_frames)
    conc_f, mask_f = _write_niftis(tmp_path, conc, mask)
    result = _base_iface(
        conc_f, mask_f, aif_method="population",
        tr=tr_seconds, baseline_frames=baseline_frames,
    ).run(cwd=str(tmp_path))

    curve = np.loadtxt(result.outputs.out_aif_curve)
    qc = result.outputs.qc_metrics
    assert np.max(curve) * 0.58 < 2.5
    np.testing.assert_allclose(curve[:baseline_frames], 1e-6 / 0.58)
    assert int(np.argmax(curve)) == baseline_frames + 1
    assert qc["bat_seconds"] == baseline_frames * tr_seconds
    assert qc["bat_method"] == "baseline_frames"
    assert np.argmax(curve) * tr_seconds == qc["bat_seconds"] + tr_seconds


def _cell_external(tmp_path: Path) -> dict[str, Any]:
    """Cell 7: external AIF file — P1 early-return path."""
    conc, mask = _make_conc()
    conc_f, mask_f = _write_niftis(tmp_path, conc, mask)
    ext = tmp_path / "external_aif.txt"
    t = np.arange(40.0)
    curve = 6.0 * np.exp(-0.5 * ((t - 8.0) / 2.0) ** 2)
    curve[:5] = 0.0
    np.savetxt(str(ext), curve)
    iface = _base_iface(conc_f, mask_f, external_aif_file=str(ext))
    iface.run()
    return _snapshot(iface)


def test_external_aif_accepts_emitted_dce_tsv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The named DCE derivative TSV can be supplied as an external AIF."""
    from pmriprep.interfaces.qc_writers import _write_aif_timeseries_tsv

    monkeypatch.chdir(tmp_path)
    conc, mask = _make_conc()
    conc_f, mask_f = _write_niftis(tmp_path, conc, mask)
    ext = tmp_path / "sub-01_desc-dceAif_timeseries.tsv"
    t = np.arange(40.0)
    curve = 6.0 * np.exp(-0.5 * ((t - 8.0) / 2.0) ** 2)
    curve[:5] = 0.0
    _write_aif_timeseries_tsv(ext, curve, "DCE")

    iface = _base_iface(conc_f, mask_f, external_aif_file=str(ext))
    result = iface.run()
    np.testing.assert_allclose(result.outputs.aif_curve, curve)


def test_external_aif_retains_signed_baseline_noise(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Negative external-AIF noise is not rectified into positive AUC bias."""
    conc, mask = _make_conc()
    conc_f, mask_f = _write_niftis(tmp_path, conc, mask)
    ext = tmp_path / "external_aif_signed.txt"
    t = np.arange(40.0)
    curve = 6.0 * np.exp(-0.5 * ((t - 8.0) / 2.0) ** 2)
    curve[:5] = np.array([-0.08, 0.03, -0.04, 0.02, -0.01])
    np.savetxt(ext, curve)

    with caplog.at_level("INFO"):
        result = _base_iface(
            conc_f, mask_f, external_aif_file=str(ext)
        ).run(cwd=str(tmp_path))

    np.testing.assert_allclose(result.outputs.aif_curve, curve)
    qc = result.outputs.qc_metrics
    assert qc["external_aif_negative_samples"] == 3
    assert qc["external_aif_negative_fraction"] == pytest.approx(3 / 40)
    assert qc["external_aif_minimum_mm"] == pytest.approx(-0.08)
    assert "retained as signed baseline-subtracted noise" in caplog.text
    assert all(record.levelname != "WARNING" for record in caplog.records)


def _cell_c2_divergence(tmp_path: Path) -> dict[str, Any]:
    """Cell 2: mocked cascade, C2 divergence — detected emit C2 branch."""
    a, b, c = (10, 10, 10), (10, 11, 10), (11, 10, 10)
    conc, mask = _make_conc(arterial_coords=[a, b, c])
    conc_f, mask_f = _write_niftis(tmp_path, conc, mask)
    casc = _cascade([a, b], [0.9, 0.8], _tier_result([a, b, c]))
    iface = _base_iface(conc_f, mask_f)
    with patch.object(AIFDetection, "_run_two_pass_cascade", return_value=casc), \
         patch.object(AIFDetection, "_select_top_candidates", side_effect=_IDENTITY_BALANCE):
        iface.run()
    return _snapshot(iface)


def _cell_c2_nondivergent(tmp_path: Path) -> dict[str, Any]:
    """Cell 3: mocked cascade, candidates == tier_result.candidates — reuse branch."""
    a, b, c = (10, 10, 10), (10, 11, 10), (11, 10, 10)
    conc, mask = _make_conc(arterial_coords=[a, b, c])
    conc_f, mask_f = _write_niftis(tmp_path, conc, mask)
    casc = _cascade([a, b, c], [0.9, 0.8, 0.7], _tier_result([a, b, c]))
    iface = _base_iface(conc_f, mask_f)
    with patch.object(AIFDetection, "_run_two_pass_cascade", return_value=casc), \
         patch.object(AIFDetection, "_select_top_candidates", side_effect=_IDENTITY_BALANCE):
        iface.run()
    return _snapshot(iface)


def _cell_final_timing_marginal(tmp_path: Path) -> dict[str, Any]:
    """Cell 9: final emitted-curve timing downgrades otherwise standard confidence."""
    a, b, c = (10, 10, 10), (10, 11, 10), (11, 10, 10)
    conc, mask = _make_conc(arterial_coords=[a, b, c])
    # Make the brain-wide tissue peak coincide with the emitted AIF peak while
    # retaining enough amplitude variation for the tissue-IQR estimator.
    amplitude = np.linspace(0.5, 1.5, conc.shape[0])[:, None, None, None]
    conc[..., :] = amplitude * _bolus()[np.newaxis, np.newaxis, np.newaxis, :]
    conc_f, mask_f = _write_niftis(tmp_path, conc, mask)
    casc = _cascade([a, b, c], [0.9, 0.8, 0.7], _tier_result([a, b, c]))
    iface = _base_iface(conc_f, mask_f)
    with patch.object(AIFDetection, "_run_two_pass_cascade", return_value=casc), \
         patch.object(AIFDetection, "_select_top_candidates", side_effect=_IDENTITY_BALANCE):
        iface.run()
    snapshot = _snapshot(iface)
    assert snapshot["results"]["aif_confidence"] == "reduced"
    return snapshot


def _cell_hs2_emptied(tmp_path: Path) -> dict[str, Any]:
    """Cell 4: HS-2 — quality filter empties non-empty low-score candidates.

    A non-None tier_result is forwarded, so population emit runs while
    ``state.tier_result`` is not None (pins the ``_emit_population_aif``
    pass-through).
    """
    a, b, c = (10, 10, 10), (10, 11, 10), (11, 10, 10)
    conc, mask = _make_conc(arterial_coords=[a, b, c])
    conc_f, mask_f = _write_niftis(tmp_path, conc, mask)
    # scores all < _MIN_CANDIDATE_SCORE (0.5) so the quality filter empties them
    casc = _cascade([a, b, c], [0.2, 0.15, 0.1], _tier_result([a, b, c]))
    iface = _base_iface(conc_f, mask_f)
    with patch.object(AIFDetection, "_run_two_pass_cascade", return_value=casc), \
         patch.object(AIFDetection, "_select_top_candidates", side_effect=_IDENTITY_BALANCE):
        iface.run()
    return _snapshot(iface)


def _cell_filtered_nonzero(tmp_path: Path) -> dict[str, Any]:
    """Cell 5: detected emit with n_filtered_out > 0 (drop 1, keep >= 2)."""
    a, b, c = (10, 10, 10), (10, 11, 10), (11, 10, 10)
    conc, mask = _make_conc(arterial_coords=[a, b, c])
    conc_f, mask_f = _write_niftis(tmp_path, conc, mask)
    # a,b survive (>= 0.5); c dropped (< 0.5) -> n_filtered_out == 1
    casc = _cascade([a, b, c], [0.9, 0.8, 0.2], _tier_result([a, b, c]))
    iface = _base_iface(conc_f, mask_f)
    with patch.object(AIFDetection, "_run_two_pass_cascade", return_value=casc), \
         patch.object(AIFDetection, "_select_top_candidates", side_effect=_IDENTITY_BALANCE):
        iface.run()
    return _snapshot(iface)


def _cell_real_detection(tmp_path: Path) -> dict[str, Any]:
    """Cell 6: real (unmocked) detection cascade on synthetic data, no atlas.

    Exercises the real ``_run_detection`` cascade + exhaustion-provenance path.
    On this synthetic data the cascade finds no accepted tier and falls back to
    population emit (detected emit is covered by the mocked cells 2/3/5).
    """
    conc, mask = _make_conc()
    conc_f, mask_f = _write_niftis(tmp_path, conc, mask)
    iface = _base_iface(conc_f, mask_f)
    iface.run()
    return _snapshot(iface)


def _cell_lesion_mask(tmp_path: Path) -> dict[str, Any]:
    """Cell 8: real (unmocked) detection WITH a lesion mask.

    Exercises the lesion-load branch of ``_load_atlas_and_lesion_masks`` +
    lesion-aware search-mask prep inside ``_run_detection``. As with cell 6 the
    real cascade falls back to population emit on this synthetic data.
    """
    conc, mask = _make_conc()
    conc_f, mask_f = _write_niftis(tmp_path, conc, mask)
    lesion = np.zeros((20, 20, 20), dtype=np.uint8)
    lesion[3:6, 3:6, 9:11] = 1  # small left-hemisphere lesion
    lesion_f = tmp_path / "lesion.nii.gz"
    nib.save(nib.Nifti1Image(lesion, _bilateral_affine()), str(lesion_f))
    iface = _base_iface(conc_f, mask_f, lesion_mask_file=str(lesion_f))
    iface.run()
    return _snapshot(iface)


_CELLS = {
    "population": _cell_population,
    "external": _cell_external,
    "c2_divergence": _cell_c2_divergence,
    "c2_nondivergent": _cell_c2_nondivergent,
    "final_timing_marginal": _cell_final_timing_marginal,
    "hs2_emptied": _cell_hs2_emptied,
    "filtered_nonzero": _cell_filtered_nonzero,
    "real_detection": _cell_real_detection,
    "lesion_mask": _cell_lesion_mask,
}


def _run_all(tmp_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, fn in _CELLS.items():
        cell_dir = tmp_root / name
        cell_dir.mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(cell_dir)
        try:
            out[name] = fn(cell_dir)
        finally:
            os.chdir(cwd)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _golden_close(a: Any, b: Any, *, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    """Structural equality with float leaves compared within a tolerance.

    The canonical snapshot stores floats as exact ``float.hex()`` under an
    ``{"__f__": ...}`` leaf. Exact hex is bit-reproducible on one machine but
    the low mantissa bits differ across libm/BLAS builds (the baseline is
    regenerated on macOS-ARM; CI runs on the Linux-x86 image), so float leaves
    are compared with a relative tolerance while everything else — ints,
    strings, enums, structure, mask coords — stays exact. The tolerance is
    far above the ~1e-15 platform noise yet far below any real reimplementation
    delta, so it does not weaken the regression guard.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        if len(a) == 1 and len(b) == 1 and "__f__" in a and "__f__" in b:
            fa, fb = float.fromhex(a["__f__"]), float.fromhex(b["__f__"])
            if math.isnan(fa) and math.isnan(fb):
                return True
            return math.isclose(fa, fb, rel_tol=rel_tol, abs_tol=abs_tol)
        if a.keys() != b.keys():
            return False
        return all(_golden_close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_golden_close(x, y) for x, y in zip(a, b))
    return bool(a == b)


def test_run_interface_golden(tmp_path: Path) -> None:
    """Every matrix cell's canonical snapshot matches the committed baseline."""
    current = _run_all(tmp_path)

    if os.environ.get(_REGEN_ENV):
        _BASELINE.parent.mkdir(parents=True, exist_ok=True)
        _BASELINE.write_text(json.dumps(current, indent=2, sort_keys=True))
        pytest.skip(f"regenerated baseline at {_BASELINE}")

    assert _BASELINE.exists(), (
        f"baseline missing; regenerate with {_REGEN_ENV}=1"
    )
    baseline = json.loads(_BASELINE.read_text())
    assert set(current) == set(baseline), "cell set changed"
    for name in baseline:
        if current[name] == baseline[name]:
            continue  # bit-identical (same platform as the committed baseline)
        if _golden_close(current[name], baseline[name]):
            continue  # differs only within float tolerance (cross-platform noise)
        # Real divergence — re-assert for pytest's rich structural diff.
        assert current[name] == baseline[name], f"golden mismatch in cell '{name}'"


def test_regen_flag_not_set_in_normal_run() -> None:
    """Anti-leak guard: the regen env var must not be set in a normal run."""
    assert not os.environ.get(_REGEN_ENV), (
        f"{_REGEN_ENV} must not be set during a normal test run"
    )
