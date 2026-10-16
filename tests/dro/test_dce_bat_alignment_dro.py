"""DRO accuracy tests for DCE AIF->tissue BAT pre-alignment.

A known arterial-to-tissue delay is injected into synthetic tissue (the tissue
sees the AIF arriving later than the sampling artery). Because the Tofts
convolution is delay-free, fitting the unshifted AIF biases the parameters; BAT
pre-alignment ('auto') detects the delay, snaps it to a whole number of
acquisition frames, and shifts the AIF before fitting. These tests cover the
accuracy side of the documented accuracy/repeatability trade: BAT removes the
delay component of arterial-input bias (lower Ktrans/vp RMSE vs ground truth),
including at coarse (clinical) TR, and is a no-op when no real delay is present.
"""
from __future__ import annotations

import numpy as np

from pmriprep.models.dce.extended_tofts import ExtendedToftsModel
from pmriprep.utils.dce_bat import (
    compute_aif_tissue_delay,
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


def _bat_montecarlo(
    model, tr_s, delay_true_s, *, kt=0.05, ve=0.30, vp=0.04,
    n_real=50, noise_frac=0.04, seed=20260623,
):
    """Monte-Carlo off-vs-BAT(snapped) fit over noise realizations at one TR/delay.

    Generates tissue with known ground-truth parameters fed by a ``delay_true_s``
    delayed AIF, then fits with the undelayed (measured) AIF both ways: 'off'
    carries the uncorrected delay; BAT detects it, snaps to whole frames, shifts
    the AIF, and refits. Reuses the pipeline's ExtendedTofts model and dce_bat
    helpers. Returns the mean uncorrected Ktrans bias, off/BAT Ktrans and vp
    RMSE vs ground truth, and the fraction of realizations BAT applied a shift.
    """
    rng = np.random.default_rng(seed)
    dt = tr_s / 60.0
    n = int(round(300.0 / tr_s))                  # ~5 min acquisition
    t = np.arange(n, dtype=np.float64) * dt
    aif = _gamma_aif(t, t0_min=30.0 / 60.0)       # arterial bolus arrives ~30 s
    aif_peak = float(aif.max())
    clean = model.evaluate([kt, ve, vp], shift_aif(aif, t, delay_true_s), t)
    tissue_peak = max(float(clean.max()), 1e-6)
    baseline = max(2, int(round(30.0 / tr_s)))
    bias_off, sq_off, sq_bat, sqv_off, sqv_bat = [], [], [], [], []
    n_applied = 0
    for _ in range(n_real):
        tissue = clean + rng.normal(0.0, noise_frac * tissue_peak, n)
        aif_noisy = aif + rng.normal(0.0, noise_frac * aif_peak, n)
        r_off = model.fit(tissue, aif_noisy, t)
        d_est, prov = compute_aif_tissue_delay(
            aif_noisy, tissue, t, tr_s, baseline_frames=baseline,
        )
        n_applied += int(bool(prov.get("applied")))
        aif_sh = shift_aif(aif_noisy, t, d_est) if d_est != 0.0 else aif_noisy
        r_bat = model.fit(tissue, aif_sh, t)
        if not (r_off.converged and r_bat.converged):
            continue
        ko, kb = r_off.parameters["Ktrans"], r_bat.parameters["Ktrans"]
        vo, vb = r_off.parameters["vp"], r_bat.parameters["vp"]
        bias_off.append(ko - kt)
        sq_off.append((ko - kt) ** 2)
        sq_bat.append((kb - kt) ** 2)
        sqv_off.append((vo - vp) ** 2)
        sqv_bat.append((vb - vp) ** 2)
    return {
        "bias_off": float(np.mean(bias_off)),
        "ktrans_rmse_off": float(np.sqrt(np.mean(sq_off))),
        "ktrans_rmse_bat": float(np.sqrt(np.mean(sq_bat))),
        "vp_rmse_off": float(np.sqrt(np.mean(sqv_off))),
        "vp_rmse_bat": float(np.sqrt(np.mean(sqv_bat))),
        "frac_applied": n_applied / n_real,
    }


def test_bat_reduces_ktrans_bias_with_real_delay_coarse_tr():
    # Compare Ktrans bias and Ktrans/vp RMSE with and without a six-second delay
    # at 4.76-second sampling. Applied BAT shifts are snapped to whole frames.
    model = ExtendedToftsModel()
    tr_s, kt_true = 4.76, 0.05
    delayed = _bat_montecarlo(model, tr_s, delay_true_s=6.0, kt=kt_true)
    nodelay = _bat_montecarlo(model, tr_s, delay_true_s=0.0, kt=kt_true)

    # (i) a real delay biases the uncorrected Ktrans (> 10% of GT), growing from
    #     the no-delay case.
    assert abs(delayed["bias_off"]) > 0.1 * kt_true, delayed
    assert abs(delayed["bias_off"]) > abs(nodelay["bias_off"]), (delayed, nodelay)
    # (ii) BAT substantially reduces Ktrans and vp RMSE vs ground truth.
    assert delayed["ktrans_rmse_bat"] < 0.75 * delayed["ktrans_rmse_off"], delayed
    assert delayed["vp_rmse_bat"] < 0.75 * delayed["vp_rmse_off"], delayed
    # (iii) no injected delay -> BAT no-ops at coarse TR (mostly) and does no harm.
    assert nodelay["frac_applied"] <= 0.2, nodelay
    assert nodelay["ktrans_rmse_bat"] <= 1.1 * nodelay["ktrans_rmse_off"] + 1e-9, nodelay


def test_detector_subframe_but_applied_shift_is_frame_snapped():
    # The 10%-of-peak detector returns a sub-frame raw delay. The applied shift
    # is snapped to a whole number of frames on the uniform grid.
    tr_s = 4.0
    dt = tr_s / 60.0
    n = 80
    t = np.arange(n, dtype=np.float64) * dt
    aif = _gamma_aif(t, t0_min=8 * dt)
    # A true delay near 1.5 frames: the raw estimate is sub-frame, the applied
    # shift snaps to a whole frame.
    tissue = shift_aif(aif, t, 1.5 * tr_s)
    applied, prov = compute_aif_tissue_delay(aif, tissue, t, tr_s, baseline_frames=5)

    # Detector raw delay is genuinely sub-frame (non-integer multiple of TR).
    raw_frames = prov["raw_delay_seconds"] / tr_s
    assert not np.isclose(raw_frames % 1.0, 0.0, atol=0.05), raw_frames

    # Applied shift is an exact integer number of frames ...
    assert prov["applied"] is True, prov
    assert prov["delay_frames"] == float(int(prov["delay_frames"])), prov
    assert abs(applied - prov["delay_frames"] * tr_s) < 1e-9
    # ... so shift_aif is an exact frame translation on the uniform grid
    # (baseline left-fill, no sub-frame regridding).
    n_shift = int(round(applied / tr_s))
    shifted = shift_aif(aif, t, applied)
    assert np.allclose(shifted[n_shift:], aif[:n - n_shift])
