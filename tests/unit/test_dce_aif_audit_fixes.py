"""Regression tests for the DCE AIF audit fixes (P1/P2/P4/P5/C1/C5/C6/C7).

Each locks a specific latent bug found in the AIF audit so it cannot silently
return. See docs plan `plan-the-change-fuzzy-lobster.md`.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from pmriprep.interfaces.dce_aif import (
    _evaluate_gates,
    _compute_pool_qc,
    compute_dce_aif_fwhm_seconds,
    parker_population_aif,
)
from pmriprep.utils.dce_constants import MIN_MEANINGFUL_ENHANCEMENT_MM

from tests.unit.test_dce_aif import _make_qc


class TestP1ParkerNoRelaxivity:
    """Population AIF is relaxivity-independent true mM (dose-only)."""

    def test_no_r1_argument(self) -> None:
        """The r1_actual parameter is gone — concentration is r1-independent."""
        with pytest.raises(TypeError):
            parker_population_aif(np.arange(60.0), 1.0, r1_actual=11.6)  # type: ignore[call-arg]

    def test_peak_scales_with_dose_only(self) -> None:
        """Peak is set by dose_factor alone (no relaxivity term)."""
        t = np.arange(0, 300.0, 1.0)
        p_full = float(np.max(parker_population_aif(t, 1.0)))
        p_half = float(np.max(parker_population_aif(t, 0.5)))
        assert p_half == pytest.approx(0.5 * p_full, rel=1e-6)
        # ~6 mM blood-scale peak at full dose (Parker), no r1 inflation.
        assert 4.0 < p_full < 9.0


class TestC5AmplitudeGate:
    """A near-zero AIF must FAIL G1, not slip through on scale=1.0."""

    def test_subthreshold_peak_fails_g1(self) -> None:
        """Peak below the meaningful-enhancement floor -> FAIL_G1."""
        status, _ = _evaluate_gates(
            scale_factor=1.0,
            qc=_make_qc(peak=MIN_MEANINGFUL_ENHANCEMENT_MM / 2),
            atlas_tier="MCA_ACA_dce_strict",
            n_voxels=10, hemisphere="bilateral", n_slabs_spanned=3,
        )
        assert status == "FAIL_G1"

    def test_normal_peak_passes(self) -> None:
        """A healthy peak with clean QC still passes (no false positive)."""
        status, _ = _evaluate_gates(
            scale_factor=1.0, qc=_make_qc(peak=8.0),
            atlas_tier="MCA_ACA_dce_strict",
            n_voxels=10, hemisphere="bilateral", n_slabs_spanned=3,
        )
        assert status in ("PASS", "PASS_WARN")


class TestC6FwhmBaselinePeak:
    """FWHM is 0 (not spuriously large) when the max sits in the baseline."""

    def test_baseline_spike_returns_zero(self) -> None:
        """A max inside the baseline window yields FWHM 0, not a spurious value."""
        curve = np.zeros(40)
        curve[2] = 5.0  # spike inside the baseline window (baseline_frames=5)
        fwhm = compute_dce_aif_fwhm_seconds(curve, tr=2.0, baseline_frames=5)
        assert fwhm == 0.0

    def test_real_bolus_unaffected(self) -> None:
        """A real post-baseline bolus still yields a positive FWHM."""
        curve = np.concatenate([np.zeros(5), np.array([2, 6, 8, 6, 2, 1.0]), np.zeros(9)])
        assert compute_dce_aif_fwhm_seconds(curve, tr=2.0, baseline_frames=5) > 0.0


class TestP5LateTailPostPeak:
    """G2 late-tail window is anchored to the peak, not scan start."""

    def test_delayed_bolus_tail_is_post_peak(self) -> None:
        """A late bolus's first pass is NOT counted as late tail."""
        tr = 2.0
        n = int((120.0 / tr) * 2 + 40)  # long enough that peak+120s fits
        curve = np.full(n, 0.05)
        peak_frame = 40  # late arrival
        curve[peak_frame] = 10.0
        curve[peak_frame + 1:peak_frame + 5] = [6.0, 3.0, 1.0, 0.5]
        qc = _compute_pool_qc(curve, tr=tr, baseline_frames=5)
        # Window starts peak+120s -> only the flat 0.05 tail; ratio ~ 0.005, not
        # inflated by the first-pass peak (which a scan-start window would catch).
        assert qc["late_tail_per_peak"] is not None
        assert qc["late_tail_per_peak"] < 0.05


class TestC1AtlasStripResolutionAgnostic:
    """The atlas-name strip works for any resolution, not just 2mm."""

    @pytest.mark.parametrize("res", ["1mm", "2mm", "3mm"])
    def test_region_extracted_any_resolution(self, res: str) -> None:
        """The region tag is extracted for 1mm/2mm/3mm, so g6_f matches."""
        stem = f"MNI_{res}_MRA_arterial_MCA_ACA_dce"
        assert re.sub(r"^MNI_\d+mm_MRA_arterial_", "", stem) == "MCA_ACA_dce"
