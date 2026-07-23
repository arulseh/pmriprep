"""Unit tests for DCE AIF->tissue BAT pre-alignment.

Covers ``pmriprep.utils.dce_bat`` -- the ``shift_aif`` time-shift helper, every
guard branch of ``compute_aif_tissue_delay`` (incl. the integer-frame snap), and
the minute-vs-second units contract of the BAT detector (the 60x conversion
trap) -- plus the enhancing-tumor tissue-reference selector
``pmriprep.interfaces.dce_compute._resolve_bat_tissue_reference``.
"""
from __future__ import annotations

import numpy as np

from pmriprep.interfaces.dce_compute import (
    BAT_TUMOR_REFERENCE_MIN_VOXELS,
    _resolve_bat_tissue_reference,
)
from pmriprep.utils.dce_bat import (
    MAX_AIF_TISSUE_DELAY_S,
    compute_aif_tissue_delay,
    detect_bolus_arrival_time,
    shift_aif,
)


def _step(n: int, arrival: int, val: float = 5.0) -> np.ndarray:
    """Step curve: 0 before ``arrival``, ``val`` from ``arrival`` onward."""
    c = np.zeros(n, dtype=np.float64)
    c[arrival:] = val
    return c


def _minutes(n: int, tr_seconds: float) -> np.ndarray:
    """Uniform time axis in minutes for ``n`` frames spaced ``tr_seconds``."""
    return np.arange(n, dtype=np.float64) * (tr_seconds / 60.0)


class TestShiftAif:
    def test_zero_delay_is_identity(self):
        t = _minutes(20, 6.0)
        aif = np.linspace(0.0, 5.0, 20)
        np.testing.assert_array_equal(shift_aif(aif, t, 0.0), aif)

    def test_one_frame_shift_aligns(self):
        # 6 s == 0.1 min == one frame at this spacing; aif(t - 1 frame).
        t = _minutes(20, 6.0)
        aif = np.arange(20, dtype=np.float64)
        s = shift_aif(aif, t, 6.0)
        assert s[5] == aif[4]          # shifted later by one frame
        assert s[0] == aif[0]          # baseline left-fill, not zero/negative

    def test_baseline_left_fill_no_spurious_negative(self):
        t = _minutes(20, 6.0)
        aif = np.concatenate([np.zeros(3), np.linspace(0.0, 5.0, 17)])
        s = shift_aif(aif, t, 12.0)    # +2 frames
        assert s.min() >= aif.min()    # no acausal negative ramp introduced
        assert s[0] == aif[0]

    def test_non_uniform_axis(self):
        t = np.array([0.0, 0.05, 0.12, 0.2, 0.31, 0.45, 0.6, 0.8])
        aif = np.array([0.0, 1, 3, 5, 4, 3, 2, 1])
        s = shift_aif(aif, t, 3.0)
        assert s.shape == aif.shape
        assert np.all(np.isfinite(s))


class TestComputeAifTissueDelay:
    def test_applied_fine_tr(self):
        n, tr = 30, 2.0
        d, prov = compute_aif_tissue_delay(
            _step(n, 6), _step(n, 9), _minutes(n, tr), tr, baseline_frames=3,
        )
        assert prov["applied"] is True
        assert prov["reason"] == "applied"
        assert prov["methods_match"] is True     # both arrivals via threshold
        assert abs(d - 6.0) < 1e-6      # (9-6) frames * 2 s
        assert prov["delay_frames"] == 3.0       # snapped frame count
        assert prov["raw_delay_frames"] == 3.0
        assert "granularity" not in prov  # added by the caller, not the helper

    def test_subresolution_noop(self):
        n, tr = 30, 8.0                 # coarse TR; same arrival frame -> 0 delay
        d, prov = compute_aif_tissue_delay(
            _step(n, 6), _step(n, 6), _minutes(n, tr), tr, baseline_frames=3,
        )
        assert d == 0.0
        assert prov["reason"] == "delay_below_resolution"
        assert prov["raw_delay_frames"] == 0.0   # frames breadcrumb present even on no-op

    def test_positive_delay_below_one_frame_is_unresolved(
        self, monkeypatch
    ) -> None:
        """A fractional-frame tissue lag must not round into a false shift."""
        from pmriprep.utils import dce_bat
        from pmriprep.utils.dce_bat import BATResult

        detections = iter(
            [
                BATResult(10.0, 20.0, "high", "threshold"),
                BATResult(11.6, 22.0, "high", "threshold"),
            ]
        )
        monkeypatch.setattr(
            dce_bat,
            "detect_bolus_arrival_time",
            lambda *_args, **_kwargs: next(detections),
        )
        n, tr = 30, 2.0
        delay, provenance = compute_aif_tissue_delay(
            _step(n, 6),
            _step(n, 7),
            _minutes(n, tr),
            tr,
            baseline_frames=3,
        )
        assert delay == 0.0
        assert provenance["raw_delay_frames"] == 0.8
        assert provenance["reason"] == "delay_below_resolution"

    def test_ceiling_clamp(self):
        n, tr = 40, 2.0                 # (25-6)*2 = 38 s > 12 s ceiling
        d, prov = compute_aif_tissue_delay(
            _step(n, 6), _step(n, 25), _minutes(n, tr), tr, baseline_frames=3,
        )
        assert d == 0.0
        assert prov["reason"] == "delay_clamped"
        assert prov["raw_delay_seconds"] > MAX_AIF_TISSUE_DELAY_S

    def test_negative_delay_rejected(self):
        n, tr = 30, 2.0                 # tissue arrives before the AIF
        d, prov = compute_aif_tissue_delay(
            _step(n, 9), _step(n, 6), _minutes(n, tr), tr, baseline_frames=3,
        )
        assert d == 0.0
        assert prov["reason"] == "negative_delay_rejected"

    def test_low_confidence_gate(self):
        n, tr = 30, 2.0                 # flat tissue -> default-method BAT
        d, prov = compute_aif_tissue_delay(
            _step(n, 6), np.zeros(n), _minutes(n, tr), tr, baseline_frames=3,
        )
        assert d == 0.0
        assert prov["reason"] == "low_bat_confidence"


class TestBatUnitsContract:
    def test_bat_seconds_in_seconds_regime(self):
        # An AIF arriving at frame 6 with TR=2 s reports BAT in SECONDS. The
        # 10%-of-peak crossing on the 0->5 step interpolates to ~5.1*tr (just
        # after frame 5), bracketed by the straddling frame times. If the
        # minute->second conversion were dropped this would be ~0.2 s, far below
        # the band -- the 60x guard.
        n, tr = 30, 2.0
        r = detect_bolus_arrival_time(_step(n, 6), _minutes(n, tr), baseline_frames=3)
        assert r.method == "threshold"
        assert 5 * tr <= r.bat_seconds <= 6 * tr


class TestSubFrameDetection:
    def test_peak_fraction_anchor_is_sub_frame(self):
        # A gentle rise: the 10%-of-peak level crosses strictly between two
        # frames, so the arrival is sub-frame (not snapped to the grid).
        n, tr = 30, 2.0
        c = np.zeros(n, dtype=np.float64)
        c[6:] = np.linspace(1.0, 5.0, n - 6)   # jumps to 1.0 at frame 6, peak 5.0
        r = detect_bolus_arrival_time(c, _minutes(n, tr), baseline_frames=3)
        assert r.method == "threshold"
        # level 0.5 crosses between frame 5 (0.0) and frame 6 (1.0) at frac 0.5
        assert abs(r.bat_seconds - 5.5 * tr) < 1e-9
        assert 0.01 < (r.bat_seconds / tr) % 1.0 < 0.99   # genuinely sub-frame

    def test_peak_inside_baseline_falls_through(self):
        # A peak within the baseline window must not be read as the arrival;
        # the detector falls through (no index error, finite result).
        n, tr = 30, 2.0
        c = np.zeros(n, dtype=np.float64)
        c[1] = 10.0
        r = detect_bolus_arrival_time(c, _minutes(n, tr), baseline_frames=5)
        assert r.method != "threshold"
        assert np.isfinite(r.bat_seconds)

    def test_interp_crossing_interpolates_and_clamps(self):
        from pmriprep.utils.dce_bat import _interp_crossing
        time_sec = np.arange(10, dtype=np.float64) * 2.0
        series = np.array([0, 0, 0, 0, 0, 0, 2.0, 4, 4, 4], dtype=np.float64)
        # level 1.0 between frame 5 (0.0) and frame 6 (2.0): frac 0.5 -> 11.0 s
        assert _interp_crossing(time_sec, series, 6, 1.0) == 11.0
        # level above series[i]: frac clamped to 1 -> the frame time
        assert _interp_crossing(time_sec, series, 6, 5.0) == time_sec[6]

    def test_interp_crossing_degenerate_falls_back(self):
        from pmriprep.utils.dce_bat import _interp_crossing
        time_sec = np.arange(10, dtype=np.float64) * 2.0
        flat = np.array([0, 0, 0, 5, 5, 5, 5, 5, 5, 5], dtype=np.float64)
        assert _interp_crossing(time_sec, flat, 4, 5.0) == time_sec[4]  # denom 0
        assert _interp_crossing(time_sec, flat, 0, 1.0) == time_sec[0]  # i <= 0

    def test_integer_frame_snap(self):
        # The applied shift is snapped to a whole number of frames -- an exact
        # translation on the uniform grid (no sub-frame regridding). A 1-frame
        # step difference at TR=2 snaps to exactly one frame; the soft-ramp
        # provenance keys are gone.
        n, tr = 30, 2.0
        t = _minutes(n, tr)
        d, prov = compute_aif_tissue_delay(
            _step(n, 6), _step(n, 7), t, tr, baseline_frames=3,
        )
        assert prov["applied"] is True
        assert prov["delay_frames"] == 1.0            # one whole frame
        assert abs(d - tr) < 1e-9                     # exact multiple of TR
        assert "attenuated" not in prov
        assert "floor_seconds" not in prov
        assert "applied_delay_frames" not in prov
        # A 3-frame separation snaps to exactly three frames.
        d3, prov3 = compute_aif_tissue_delay(
            _step(n, 6), _step(n, 9), t, tr, baseline_frames=3,
        )
        assert prov3["delay_frames"] == 3.0
        assert abs(d3 - prov3["delay_frames"] * tr) < 1e-9

    def test_methods_match_flag(self):
        n, tr = 30, 2.0
        t = _minutes(n, tr)
        _, prov_match = compute_aif_tissue_delay(
            _step(n, 6), _step(n, 9), t, tr, baseline_frames=3,
        )
        assert prov_match["methods_match"] is True
        # tissue flat -> default method, AIF threshold -> mixed methods. The flag
        # is recorded even though the low-confidence guard no-ops the shift.
        _, prov_mixed = compute_aif_tissue_delay(
            _step(n, 6), np.zeros(n), t, tr, baseline_frames=3,
        )
        assert prov_mixed["methods_match"] is False


class TestResolveBatTissueReference:
    """The enhancing-tumor tissue-reference selector used by ``ComputeDCEMaps``
    for AIF BAT pre-alignment. The reference is the enhancing-tumor mask
    intersected with the PK fitting mask; selection no-ops (returns ``None`` and
    a reason) when no tumor mask, a grid mismatch, or too few voxels."""

    @staticmethod
    def _write_mask(path, arr: np.ndarray) -> str:
        import nibabel as nib
        nib.save(nib.Nifti1Image(arr.astype(np.uint8), np.eye(4)), str(path))
        return str(path)

    def test_no_tumor_reference_when_empty_path(self):
        fm = np.ones((4, 4, 4), dtype=bool)
        conc = np.ones((4, 4, 4, 10))
        roi, info = _resolve_bat_tissue_reference("", fm, conc)
        assert roi is None
        assert info == {"applied": False, "reason": "no_tumor_reference"}

    def test_grid_mismatch_noop(self, tmp_path):
        fm = np.ones((4, 4, 4), dtype=bool)
        conc = np.ones((4, 4, 4, 10))
        mask = np.ones((5, 5, 5))            # wrong spatial shape vs conc[:3]
        f = self._write_mask(tmp_path / "m.nii.gz", mask)
        roi, info = _resolve_bat_tissue_reference(f, fm, conc)
        assert roi is None
        assert info["reason"] == "tumor_reference_grid_mismatch"

    def test_too_small_noop_reports_voxel_count(self, tmp_path):
        fm = np.ones((4, 4, 4), dtype=bool)
        conc = np.ones((4, 4, 4, 10))
        mask = np.zeros((4, 4, 4))
        mask[0, 0, :3] = 1                   # 3 enhancing voxels < floor (10)
        f = self._write_mask(tmp_path / "m.nii.gz", mask)
        roi, info = _resolve_bat_tissue_reference(f, fm, conc)
        assert roi is None
        assert info["reason"] == "tumor_reference_too_small"
        assert info["n_reference_voxels"] == 3
        assert BAT_TUMOR_REFERENCE_MIN_VOXELS == 10

    def test_returns_roi_intersected_with_fitting_mask(self, tmp_path):
        conc = np.ones((4, 4, 4, 10))
        mask = np.zeros((4, 4, 4))
        mask[:, :, 0] = 1                    # 16-voxel enhancing slab on z=0
        fm = np.ones((4, 4, 4), dtype=bool)
        fm[0, :, 0] = False                  # fitting mask drops 4 of the 16
        f = self._write_mask(tmp_path / "m.nii.gz", mask)
        roi, info = _resolve_bat_tissue_reference(f, fm, conc)
        assert roi is not None
        expected = mask.astype(bool) & fm    # the intersection, not the bare mask
        assert np.array_equal(roi, expected)
        assert info["n_reference_voxels"] == int(expected.sum()) == 12
