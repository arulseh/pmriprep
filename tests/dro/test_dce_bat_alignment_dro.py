"""DRO accuracy tests for DCE AIF->tissue BAT pre-alignment.

A known arterial-to-tissue delay is injected into a synthetic enhancing voxel
(the tissue sees the AIF arriving later than the sampling artery). Because the
Tofts convolution is delay-free, fitting the unshifted AIF biases vp; BAT
pre-alignment ('auto') is expected to remove most of that bias at fine temporal
resolution, and to leave the fit unchanged when the delay is unresolvable
(coarse TR / same arrival frame) — the no-op that keeps default behavior intact.
"""
from __future__ import annotations

import numpy as np

from pmriprep.models.dce.extended_tofts import ExtendedToftsModel
from pmriprep.utils.dce_bat import (
    compute_aif_tissue_delay,
    detect_bolus_arrival_time,
    shift_aif,
)


def _gamma_aif(
    t_min: np.ndarray, t0_min: float, amp: float = 5.0,
    alpha: float = 3.0, beta: float = 0.06,
) -> np.ndarray:
    """Gamma-variate bolus with arrival ``t0_min`` (minutes), peak ``amp`` (mM)."""
    x = np.maximum(t_min - t0_min, 0.0)
    a = amp * (x ** alpha) * np.exp(-x / beta)
    m = float(a.max())
    return (a / m * amp) if m > 0 else a


def _fit(model, tissue, aif, t):
    r = model.fit(tissue, aif, t)
    return r.parameters["vp"], r.parameters["Ktrans"]


def test_bat_alignment_reduces_vp_bias_fine_tr():
    tr_s = 2.0
    dt = tr_s / 60.0
    n = 90
    t = np.arange(n, dtype=np.float64) * dt
    aif = _gamma_aif(t, t0_min=8 * dt)        # AIF arrives ~frame 8

    kt_true, ve_true, vp_true = 0.10, 0.30, 0.05
    model = ExtendedToftsModel()

    # Tissue sees the AIF delayed by a known transit delay.
    delay_true_s = 6.0
    aif_delayed = shift_aif(aif, t, delay_true_s)
    tissue = model.evaluate([kt_true, ve_true, vp_true], aif_delayed, t)

    # OFF: fit with the unshifted (measured) AIF -> biased.
    vp_off, kt_off = _fit(model, tissue, aif, t)

    # AUTO: detect the delay and align the AIF before fitting.
    d_est, prov = compute_aif_tissue_delay(aif, tissue, t, tr_s, baseline_frames=5)
    assert prov["applied"], prov
    assert abs(d_est - delay_true_s) <= tr_s   # recovered within ~one frame
    vp_auto, kt_auto = _fit(model, tissue, shift_aif(aif, t, d_est), t)

    bias_off = abs(vp_off - vp_true)
    bias_auto = abs(vp_auto - vp_true)
    assert bias_off > 0.01, f"no meaningful vp bias to correct (vp_off={vp_off:.4f})"
    assert bias_auto <= 0.5 * bias_off, (
        f"vp bias not at least halved: off={bias_off:.4f} auto={bias_auto:.4f}"
    )
    # Ktrans must not be made worse by alignment.
    assert abs(kt_auto - kt_true) <= abs(kt_off - kt_true) + 0.01


def test_bat_alignment_noop_when_delay_unresolvable():
    # Coarse TR (penn-like); AIF and tissue share the same arrival frame, so the
    # delay is below temporal resolution -> auto must apply no shift, leaving the
    # fit byte-identical to the off path.
    tr_s = 7.0
    dt = tr_s / 60.0
    n = 60
    t = np.arange(n, dtype=np.float64) * dt
    aif = _gamma_aif(t, t0_min=6 * dt)
    model = ExtendedToftsModel()
    tissue = model.evaluate([0.10, 0.30, 0.05], aif, t)  # no injected delay

    d_est, prov = compute_aif_tissue_delay(aif, tissue, t, tr_s, baseline_frames=5)
    assert d_est == 0.0
    assert prov["reason"] in ("delay_below_resolution", "low_bat_confidence")

    vp_off, kt_off = _fit(model, tissue, aif, t)
    vp_auto, kt_auto = _fit(model, tissue, shift_aif(aif, t, d_est), t)
    assert vp_auto == vp_off and kt_auto == kt_off   # identical fit (no-op)


def test_subframe_visit_consistency_near_frame_boundary():
    # Two "visits" whose true AIF->tissue delay differs only by sub-frame jitter,
    # placed so the tissue arrival straddles a frame-grid boundary. A grid-snapped
    # (pre-0.63 frame-quantized) detector jumps a full frame between visits -- the
    # test-retest hazard that motivated sub-frame detection; sub-frame keeps the
    # recovered delays within a small fraction of a frame. The consistency
    # assertion below fails if the detector is reverted to frame quantization
    # (then both recovered delays become grid values and flip by a full frame).
    tr_s = 4.0
    dt = tr_s / 60.0
    n = 80
    t = np.arange(n, dtype=np.float64) * dt
    aif = _gamma_aif(t, t0_min=8 * dt)

    aif_bat = detect_bolus_arrival_time(aif, t, baseline_frames=5).bat_seconds
    # A true delay near 1.5 frames places the tissue arrival astride a grid
    # boundary, so the two jittered visits snap to different frames.
    base = 1.5 * tr_s
    jit = 0.15 * tr_s

    def recovered(delay_s):
        tissue = shift_aif(aif, t, delay_s)
        _, prov = compute_aif_tissue_delay(aif, tissue, t, tr_s, baseline_frames=5)
        tissue_bat = detect_bolus_arrival_time(tissue, t, baseline_frames=5).bat_seconds
        return prov["raw_delay_seconds"], tissue_bat

    d_sub_lo, bat_lo = recovered(base - jit)
    d_sub_hi, bat_hi = recovered(base + jit)

    # Grid-quantized arrivals (round to nearest frame), representative of the
    # pre-0.63 detector that returned a frame time.
    def grid(bat: float) -> float:
        return round(bat / tr_s) * tr_s

    d_grid_lo = grid(bat_lo) - grid(aif_bat)
    d_grid_hi = grid(bat_hi) - grid(aif_bat)

    # Construction check: grid quantization flips ~one whole frame between visits.
    assert abs(d_grid_hi - d_grid_lo) >= 0.9 * tr_s, (d_grid_lo, d_grid_hi)
    # Regression guard: sub-frame detection stays consistent, well under a frame
    # (this fails if detection regresses to frame quantization).
    assert abs(d_sub_hi - d_sub_lo) <= 0.5 * tr_s, (d_sub_lo, d_sub_hi)
    # The recovered delay is genuinely non-integer -- a grid detector could only
    # return integer multiples of TR.
    assert not np.isclose((d_sub_lo / tr_s) % 1.0, 0.0, atol=0.05)
