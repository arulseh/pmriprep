"""Tests for v0.25.4 scale-mismatch safeguard removal + QC diagnostic.

The legacy v0.25.0 safeguard at ``dce_compute.py:1516-1607`` widened SOFT
σ → 0.5 (or downgraded HARD → SKIP) when ``vp_prior × Cp(peak) >
C_tissue(peak)`` in > 30% of voxels. Post-v0.25.1 NAWM-anchored prior
and v0.25.3 plasma-correct AIF, that test statistic no longer measures
scale — it primarily captures first-pass temporal undersampling
(Buckley MRM 2002). Widening σ at long TR is backward.

v0.25.4:
- Deletes the sigma-widening / SKIP-downgrade behavior.
- Keeps the arithmetic as a QC-only diagnostic (``aif_tissue_overshoot_fraction``,
  ``peak_alignment_lag_s``, ``acquisition_limited_note``).
- Tightens ``VP_MEDIAN_MIN/MAX`` to ``[0.008, 0.06]`` for the NAWM-anchored path.

These tests validate the arithmetic of the new diagnostic at unit level
(reproducing the code path without instantiating the full ComputeDCEMaps
interface) plus regression guards against re-introducing the removed
behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from pmriprep.utils.dce_constants import (
    VP_MEDIAN_MAX,
    VP_MEDIAN_MIN,
)


# =============================================================================
# VP_MEDIAN bounds tightening (Part 2)
# =============================================================================


class TestVpMedianBoundsTightened:
    """v0.27.0: VP_MEDIAN_MIN/MAX re-tuned to [0.005, 0.03] around the
    Larsson-2009-anchored NAWM_VP_REFERENCE = 0.010. Factor-of-3-either-side
    band preserved to keep the gross-miscalibration gate intact.
    """

    def test_vp_median_min_retuned(self) -> None:
        assert VP_MEDIAN_MIN == pytest.approx(0.005)

    def test_vp_median_max_retuned(self) -> None:
        assert VP_MEDIAN_MAX == pytest.approx(0.03)

    def test_nawm_anchor_inside_retuned_bounds(self) -> None:
        """NAWM_VP_REFERENCE=0.010 should sit inside [0.005, 0.03] so
        normal NAWM-like nCBV inputs never trip the gate.
        """
        from pmriprep.utils.dce_constants import NAWM_VP_REFERENCE
        assert VP_MEDIAN_MIN < NAWM_VP_REFERENCE < VP_MEDIAN_MAX


# =============================================================================
# Acquisition-limited diagnostic arithmetic (replaces safeguard)
# =============================================================================


def _compute_acquisition_limited_qc(
    vp_prior_map: np.ndarray,
    aif_curve: np.ndarray,
    conc_data: np.ndarray,
    fitting_mask: np.ndarray,
    tr: float,
) -> dict[str, float | str | None]:
    """Re-implementation of the v0.25.4 diagnostic for unit testing.

    Mirrors the code path in ``dce_compute.py`` (post-v0.25.4 scale-
    mismatch safeguard removal) so we can exercise the arithmetic
    without running the full ComputeDCEMaps interface.
    """
    aif_peak = float(np.max(aif_curve))
    if aif_peak <= 0:
        return {}

    valid = (
        fitting_mask
        & np.isfinite(vp_prior_map)
        & (vp_prior_map > 0)
    )
    n_valid = int(np.sum(valid))
    if n_valid <= 100:
        return {}

    vp_cp_peak = vp_prior_map[valid] * aif_peak
    ct_peak_arr = np.max(conc_data[valid], axis=-1)
    overshoot_frac = float(np.mean(vp_cp_peak > ct_peak_arr))

    aif_peak_frame = int(np.argmax(aif_curve))
    tissue_peak_frames = np.argmax(conc_data[valid], axis=-1)
    median_tissue_peak_frame = float(np.median(tissue_peak_frames))
    peak_alignment_lag_s = (
        (median_tissue_peak_frame - aif_peak_frame) * float(tr)
    )

    out: dict[str, float | str | None] = {
        "aif_tissue_overshoot_fraction": overshoot_frac,
        "peak_alignment_lag_s": peak_alignment_lag_s,
    }
    if overshoot_frac > 0.7:
        out["acquisition_limited_note"] = (
            "High AIF-to-tissue overshoot (>70%): observed tissue "
            "peak under-samples the first pass (Buckley MRM 2002 "
            "recommends TR ≲ 2 s for vp identifiability). Posterior "
            "vp is prior-dominated by design at this acquisition; "
            "informational only — no action taken."
        )
    return out


class TestAcquisitionLimitedDiagnostic:
    """v0.25.4 diagnostic arithmetic — pure-function tests of the code
    path that replaced the removed scale-mismatch safeguard."""

    @staticmethod
    def _synthesize(
        n_voxels: int,
        n_frames: int,
        vp: float,
        aif_peak: float,
        tissue_peak_ratio: float,
        aif_peak_frame: int = 10,
        tissue_peak_frame: int = 12,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build vp_prior_map, aif_curve, 4D conc_data, fitting_mask."""
        side = int(np.ceil(n_voxels ** (1 / 3))) + 1
        shape = (side, side, side)
        vp_prior_map = np.full(shape, vp, dtype=np.float64)
        aif = np.zeros(n_frames, dtype=np.float64)
        aif[aif_peak_frame] = aif_peak
        aif[aif_peak_frame - 1] = aif_peak * 0.5
        aif[aif_peak_frame + 1] = aif_peak * 0.7

        target_tissue_peak = vp * aif_peak * tissue_peak_ratio
        conc = np.zeros(shape + (n_frames,), dtype=np.float64)
        conc[..., tissue_peak_frame] = target_tissue_peak
        conc[..., tissue_peak_frame - 1] = target_tissue_peak * 0.7
        conc[..., tissue_peak_frame + 1] = target_tissue_peak * 0.8

        fitting_mask = np.ones(shape, dtype=bool)
        return vp_prior_map, aif, conc, fitting_mask

    def test_overshoot_fraction_high_when_tissue_under_samples(self) -> None:
        """Tissue peak measured at 0.4× expected → 100% overshoot (all voxels)."""
        vp_map, aif, conc, mask = self._synthesize(
            n_voxels=200, n_frames=40, vp=0.02, aif_peak=8.0,
            tissue_peak_ratio=0.4,
        )
        out = _compute_acquisition_limited_qc(vp_map, aif, conc, mask, tr=4.0)
        assert out["aif_tissue_overshoot_fraction"] == pytest.approx(1.0)
        assert "acquisition_limited_note" in out
        assert "prior-dominated" in out["acquisition_limited_note"]

    def test_overshoot_fraction_low_when_tissue_matches_prior(self) -> None:
        """Tissue peak ~matches vp × Cp (1.2× — above prior) → 0% overshoot."""
        vp_map, aif, conc, mask = self._synthesize(
            n_voxels=200, n_frames=40, vp=0.02, aif_peak=8.0,
            tissue_peak_ratio=1.2,
        )
        out = _compute_acquisition_limited_qc(vp_map, aif, conc, mask, tr=4.0)
        assert out["aif_tissue_overshoot_fraction"] == pytest.approx(0.0)
        assert "acquisition_limited_note" not in out

    def test_peak_alignment_lag_positive_when_tissue_lags(self) -> None:
        """Tissue peaks 2 frames after AIF at TR=4 s → lag 8 s."""
        vp_map, aif, conc, mask = self._synthesize(
            n_voxels=200, n_frames=40, vp=0.02, aif_peak=8.0,
            tissue_peak_ratio=0.4,
            aif_peak_frame=10, tissue_peak_frame=12,
        )
        out = _compute_acquisition_limited_qc(vp_map, aif, conc, mask, tr=4.0)
        assert out["peak_alignment_lag_s"] == pytest.approx(8.0)

    def test_empty_output_when_no_valid_voxels(self) -> None:
        vp_map = np.zeros((3, 3, 3))
        aif = np.ones(20)
        conc = np.zeros((3, 3, 3, 20))
        mask = np.ones((3, 3, 3), dtype=bool)
        out = _compute_acquisition_limited_qc(vp_map, aif, conc, mask, tr=4.0)
        assert out == {}

    def test_empty_output_when_aif_peak_zero(self) -> None:
        vp_map = np.full((6, 6, 6), 0.02)
        aif = np.zeros(20)
        conc = np.full((6, 6, 6, 20), 0.01)
        mask = np.ones((6, 6, 6), dtype=bool)
        out = _compute_acquisition_limited_qc(vp_map, aif, conc, mask, tr=4.0)
        assert out == {}


# =============================================================================
# Regression: safeguard text/action no longer in source
# =============================================================================


class TestSafeguardRemovalRegression:
    """Confirm the v0.25.0 scale-mismatch safeguard's mutating behavior is
    gone from the live source (sigma widening, HARD→SKIP downgrade)."""

    def test_no_sigma_widening_to_05_in_dce_compute(self) -> None:
        """The removed safeguard wrote constraint_vp_uncertainty = 0.5.
        The legitimate SOFT default is 0.40 (SIGMA_SOFT_DEFAULT). If
        another 0.5 appears in dce_compute.py's SOFT-path logic, the
        safeguard came back."""
        import inspect

        from pmriprep.interfaces.dce_compute import ComputeDCEMaps

        src = inspect.getsource(ComputeDCEMaps)
        # Allow "0.5" to appear in comments / docstrings. The key symptom
        # of regression is the literal assignment pattern.
        assert "constraint_vp_uncertainty = 0.5" not in src
        assert "widened soft prior uncertainty to 0.5" not in src

    def test_no_scale_mismatch_skip_downgrade(self) -> None:
        """The removed safeguard wrote constraint_qc['reason'] with
        'Scale mismatch' strings. Confirm those strings are gone."""
        import inspect

        from pmriprep.interfaces.dce_compute import ComputeDCEMaps

        src = inspect.getsource(ComputeDCEMaps)
        assert "Scale mismatch: vp*Cp overshoots" not in src
        assert "Scale mismatch: widened soft prior" not in src
        assert "widen_soft_uncertainty" not in src
        assert "downgrade_to_skip" not in src
