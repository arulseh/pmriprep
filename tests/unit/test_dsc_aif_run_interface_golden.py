"""Golden characterization harness for DSC ``AIFDetection._run_interface``.

The fixed phase results keep this test independent of candidate-search details
while exercising post-cascade gamma handling, fallback policy, QC assembly, and
serialization through the public Nipype interface runner.
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

from pmriprep.interfaces import dsc_aif

_REGEN_ENV = "PMRIPREP_REGEN_DSC_AIF_GOLDEN"
_BASELINE = Path(__file__).parent / "data" / "dsc_aif_run_interface_golden.json"


def _canon(value: Any) -> Any:
    """Convert interface state to an exact, JSON-stable representation."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (float, np.floating)):
        return {"__f__": float(value).hex()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_canon(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_canon(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canon(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, str):
        return os.path.basename(value) if value.startswith("/") else value
    if isinstance(value, int) or value is None:
        return value
    return str(value)


def _golden_close(
    actual: Any,
    expected: Any,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> bool:
    """Compare canonical state while tolerating platform-level float noise."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) == set(expected) == {"__f__"}:
            left = float.fromhex(actual["__f__"])
            right = float.fromhex(expected["__f__"])
            if math.isnan(left) and math.isnan(right):
                return True
            return math.isclose(left, right, rel_tol=rel_tol, abs_tol=abs_tol)
        return actual.keys() == expected.keys() and all(
            _golden_close(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _golden_close(left, right) for left, right in zip(actual, expected)
        )
    return bool(actual == expected)


def _write_inputs(tmp_path: Path) -> tuple[str, str]:
    """Write the minimal files required by Nipype input validation."""
    concentration = np.zeros((2, 2, 2, 6), dtype=np.float32)
    brain_mask = np.ones((2, 2, 2), dtype=np.uint8)
    concentration_file = tmp_path / "dsc_concentration.nii.gz"
    brain_mask_file = tmp_path / "brain_mask.nii.gz"
    nib.save(nib.Nifti1Image(concentration, np.eye(4)), concentration_file)
    nib.save(nib.Nifti1Image(brain_mask, np.eye(4)), brain_mask_file)
    return str(concentration_file), str(brain_mask_file)


def _context(*, disable_fallback: bool = False) -> dsc_aif._DSCAIFDetectionContext:
    concentration = np.zeros((2, 2, 2, 6), dtype=np.float64)
    return dsc_aif._DSCAIFDetectionContext(
        concentration=concentration,
        affine=np.eye(4),
        brain_mask=np.ones(concentration.shape[:3], dtype=bool),
        atlas_mask=None,
        atlas_is_mra=False,
        full_atlas_mask=None,
        lesion_mask=None,
        enhancing_mask=None,
        signal=None,
        aif_method="auto",
        initial_criteria="strict",
        n_candidates=10,
        baseline_frames=2,
        gamma_fit_mode="auto",
        disable_fallback=disable_fallback,
        lesion_distance=15.0,
        n_timepoints=6,
        field_strength=3.0,
        synthetic_effective_dose=1.0,
        voxel_sizes=(1.0, 1.0, 1.0),
        tr=1.5,
    )


def _cascade(*, accepted: bool = True) -> dsc_aif._CascadeDetectionResult:
    curve = np.array([0.0, 0.0, 1.0, 4.0, 2.0, 0.5])
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    candidates = [(0.1, (0, 0, 0), {"snr": 10.0})] if accepted else []
    return dsc_aif._CascadeDetectionResult(
        accepted_aif=(curve, mask, 1) if accepted else None,
        candidates=candidates,
        n_available=len(candidates),
        detection_level="atlas_strict" if accepted else "none",
        filter_metadata={},
        initial_level="atlas_strict",
        tiers_attempted=["atlas_strict"],
        tier_provenance={"atlas_strict": 1} if accepted else {},
        postfilter_diagnostics={"atlas_strict": {"aif_n_after_filters": 1}},
        spatial_filter_diagnostics={},
    )


def _detected() -> dsc_aif._FinalizedDetectedAIF:
    curve = np.array([0.0, 0.0, 1.0, 4.0, 2.0, 0.5])
    mask = np.zeros((2, 2, 2), dtype=np.uint8)
    mask[0, 0, 0] = 1
    return dsc_aif._FinalizedDetectedAIF(
        curve=curve,
        mask=mask,
        quality_score=0.8,
        qc_metrics={
            "aif_quality": "detected",
            "aif_voxel_coord": [0, 0, 0],
            "cascade_downgraded": False,
        },
        baseline_frames=2,
    )


def _snapshot(interface: dsc_aif.AIFDetection) -> dict[str, Any]:
    """Capture public results and the contents of every emitted artifact."""
    results = dict(interface._results)
    artifacts: dict[str, Any] = {}
    for key in (
        "out_aif_curve",
        "out_raw_aif_curve",
        "out_gamma_aif_curve",
        "out_measured_aif_curve",
    ):
        path = results.pop(key)
        artifacts[key] = None if not path else _canon(np.loadtxt(path))

    mask_path = results.pop("out_aif_mask")
    mask_image = nib.load(mask_path)
    mask = np.asarray(mask_image.get_fdata())
    artifacts["out_aif_mask"] = {
        "dtype": str(mask_image.get_data_dtype()),
        "nonzero_coords": np.argwhere(mask != 0).tolist(),
    }
    return {"results": _canon(results), "artifacts": artifacts}


def _run_cell(tmp_path: Path, mode: str) -> dict[str, Any]:
    """Run one fixed full-interface orchestration cell."""
    concentration_file, brain_mask_file = _write_inputs(tmp_path)
    disable_fallback = mode == "retained_below_floor"
    interface = dsc_aif.AIFDetection()
    interface.inputs.concentration_file = concentration_file
    interface.inputs.brain_mask_file = brain_mask_file
    interface.inputs.disable_fallback = disable_fallback
    context = _context(disable_fallback=disable_fallback)
    cascade = _cascade(accepted=mode != "cascade_exhausted")
    synthetic = np.array([0.0, 0.0, 2.0, 8.0, 3.0, 1.0])

    def gamma_fit(
        curve: np.ndarray,
        gamma_fit_mode: str,
        baseline_frames: int,
        tr: float,
        qc_metrics: dict[str, Any],
    ) -> tuple[bool, np.ndarray | None, np.ndarray, np.ndarray | None]:
        del gamma_fit_mode, baseline_frames, tr
        fitted = curve * 0.9
        qc_metrics["gamma_fit_r_squared"] = 0.95 if mode == "gamma_applied" else -0.25
        if mode == "gamma_applied":
            qc_metrics["gamma_fit_applied"] = True
            return True, curve.copy(), fitted, fitted
        qc_metrics["gamma_fit_applied"] = False
        qc_metrics["gamma_fit_rejected_reason"] = "low_r_squared"
        return False, None, curve, fitted

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with (
            patch.object(interface, "_load_detection_context", return_value=context),
            patch.object(interface, "_prepare_detection_baseline", return_value=3),
            patch.object(interface, "_run_candidate_cascade", return_value=cascade),
            patch.object(interface, "_finalize_detected_aif", return_value=_detected()),
            patch.object(interface, "_apply_gamma_fit_if_appropriate", side_effect=gamma_fit),
            patch.object(
                interface,
                "_prepare_cached_synthetic_fallback",
                return_value=(synthetic, {"synthetic_peak": 8.0}),
            ),
        ):
            interface.run()
    finally:
        os.chdir(cwd)
    return _snapshot(interface)


_MODES = (
    "gamma_applied",
    "gamma_r2_fallback",
    "retained_below_floor",
    "cascade_exhausted",
)


def test_run_interface_golden(tmp_path: Path) -> None:
    """The post-cascade interface matrix matches the committed baseline."""
    current: dict[str, Any] = {}
    for mode in _MODES:
        cell_path = tmp_path / mode
        cell_path.mkdir()
        current[mode] = _run_cell(cell_path, mode)

    if os.environ.get(_REGEN_ENV):
        _BASELINE.parent.mkdir(parents=True, exist_ok=True)
        _BASELINE.write_text(json.dumps(current, indent=2, sort_keys=True))
        pytest.skip(f"regenerated baseline at {_BASELINE}")

    assert _BASELINE.exists(), f"baseline missing; regenerate with {_REGEN_ENV}=1"
    baseline = json.loads(_BASELINE.read_text())
    assert current.keys() == baseline.keys(), "cell set changed"
    for mode in baseline:
        if _golden_close(current[mode], baseline[mode]):
            continue
        assert current[mode] == baseline[mode], f"golden mismatch in cell {mode!r}"


def test_regen_flag_not_set_in_normal_run() -> None:
    """The baseline regeneration switch must not leak into normal test runs."""
    assert not os.environ.get(_REGEN_ENV)
