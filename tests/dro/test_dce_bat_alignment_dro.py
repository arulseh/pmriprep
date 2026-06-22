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
from pmriprep.utils.dce_bat import compute_aif_tissue_delay, shift_aif


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
