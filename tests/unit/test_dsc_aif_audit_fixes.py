"""Test DSC AIF dose scaling, peak width and emitted-pool provenance."""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pmriprep.interfaces.dsc_aif import AIFDetection
from pmriprep.utils.dsc_aif import compute_aif_fwhm_frames


def _make_aif_curve(
    n_frames: int = 120,
    baseline_frames: int = 20,
    tr: float = 1.5,
    peak_frame: int = 25,
    peak_height: float = 30.0,
    fwhm_frames: int = 5,
    recirculation_height: float = 5.0,
    recirc_frame: int = 45,
) -> npt.NDArray[np.float64]:
    """Build a synthetic AIF concentration curve (first pass + recirculation).

    Mirrors the helper in ``test_dsc_aif_cascade`` so the two suites share a
    single curve model.
    """
    curve = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        if i >= peak_frame:
            dt = i - peak_frame
            curve[i] = peak_height * np.exp(-0.5 * (dt / fwhm_frames) ** 2)
        if i >= recirc_frame:
            dt = i - recirc_frame
            curve[i] += recirculation_height * np.exp(-0.5 * (dt / 8) ** 2)
    return curve


def _valid_metrics(**overrides: float) -> dict[str, float]:
    """Return a full gate-passing candidate-metrics dict.

    Copied from ``test_dsc_aif_validation._make_metrics`` and extended with a
    ``peak_conc`` key so the D1 peak-ΔR2* bounds block executes. Every value
    clears the ``strict`` thresholds in ``_score_candidate_cost`` when scored
    with ``median_s0=200.0``.
    """
    metrics: dict[str, float] = {
        "pre_bolus_frames": 15,
        "dt": 1.5,
        "ttp": 30.0,  # 30 s → 9 s after baseline end
        "baseline_return_ratio": 15.0,
        "peak_to_baseline_ratio": 5.0,
        "secondary_peak_ratio": 40.0,
        "post_peak_monotonicity": 70.0,
        "fwhm": 6.0,
        "ph": 50.0,
        "psr": 80.0,
        "s0": 50.0,
        "max_allowed_ttp": 100.0,
        "peak_conc": 40.0,
    }
    metrics.update(overrides)
    return metrics


# =============================================================================
# D1 — peak-ΔR2* bounds are B0- and dose-scaled
# =============================================================================


class TestPeakConcBoundsDoseScaled:
    """Peak-concentration floor/ceiling ride B0·dose, not a fixed 5.0."""

    def test_reduced_dose_lowers_floor_admits_low_peak(self) -> None:
        """At 3T, dose=0.75 → floor 3.75, so peak_conc=4.0 scores finite."""
        det = AIFDetection()
        metrics = _valid_metrics(peak_conc=4.0)
        cost = det._score_candidate_cost(
            metrics, criteria="strict", median_s0=200.0, dose=0.75,
        )
        assert np.isfinite(cost)

    def test_peak_below_scaled_floor_rejected(self) -> None:
        """At 3T, dose=0.75 → floor 3.75, so peak_conc=3.0 is rejected."""
        det = AIFDetection()
        metrics = _valid_metrics(peak_conc=3.0)
        cost = det._score_candidate_cost(
            metrics, criteria="strict", median_s0=200.0, dose=0.75,
        )
        assert cost == float("inf")

    def test_reference_regime_unchanged(self) -> None:
        """3T full-dose uses a 5.0 floor: 4.0 rejected, 40.0 finite."""
        det = AIFDetection()
        cost_low = det._score_candidate_cost(
            _valid_metrics(peak_conc=4.0), criteria="strict", median_s0=200.0,
        )
        cost_ok = det._score_candidate_cost(
            _valid_metrics(peak_conc=40.0), criteria="strict", median_s0=200.0,
        )
        assert cost_low == float("inf")
        assert np.isfinite(cost_ok)

    def test_field_strength_scales_ceiling(self) -> None:
        """1.5T full dose halves the ceiling to 60: 65 rejected, 55 finite.

        Pins the ``field_strength / 3.0`` factor via the ceiling (unconfounded
        by the floor clamp): peak_conc=65 clears the fixed 3T ceiling (120) but
        exceeds the B0-scaled 60, so only the scaled law rejects it.
        """
        det = AIFDetection()
        det.inputs.field_strength = 1.5
        cost_over = det._score_candidate_cost(
            _valid_metrics(peak_conc=65.0), criteria="strict", median_s0=200.0,
        )
        cost_under = det._score_candidate_cost(
            _valid_metrics(peak_conc=55.0), criteria="strict", median_s0=200.0,
        )
        assert cost_over == float("inf")
        assert np.isfinite(cost_under)

    def test_min_peak_floor_clamped(self) -> None:
        """Very low dose pins the floor at SYNTH_AIF_MIN_PEAK (3.0), not lower.

        At 3T with dose=0.2 the unclamped floor would be 5.0*0.2=1.0; the clamp
        holds it at 3.0, so peak_conc=2.0 is rejected while 3.5 passes.
        """
        det = AIFDetection()
        cost_below = det._score_candidate_cost(
            _valid_metrics(peak_conc=2.0), criteria="strict", median_s0=200.0,
            dose=0.2,
        )
        cost_above = det._score_candidate_cost(
            _valid_metrics(peak_conc=3.5), criteria="strict", median_s0=200.0,
            dose=0.2,
        )
        assert cost_below == float("inf")
        assert np.isfinite(cost_above)

    def test_superdose_raises_ceiling(self) -> None:
        """Super-dose lifts the ceiling above the 3T default via max(eff,1.0).

        At 3T, peak_conc=200 exceeds the default ceiling (120) and is rejected;
        with dose=2.0 the ceiling rises to 240 and the same peak passes.
        """
        det = AIFDetection()
        cost_default = det._score_candidate_cost(
            _valid_metrics(peak_conc=200.0), criteria="strict", median_s0=200.0,
        )
        cost_superdose = det._score_candidate_cost(
            _valid_metrics(peak_conc=200.0), criteria="strict", median_s0=200.0,
            dose=2.0,
        )
        assert cost_default == float("inf")
        assert np.isfinite(cost_superdose)


# =============================================================================
# D2 — tissue-TTP gate compares in seconds (TR-invariant)
# =============================================================================


class TestTissueTTPGateSeconds:
    """D2: causality gate uses seconds, so it is invariant to TR."""

    @staticmethod
    def _build_single_arterial_voxel(
        tr: float,
        baseline_frames: int = 20,
    ) -> tuple[
        AIFDetection,
        npt.NDArray[np.intp],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        float,
    ]:
        """Return interface + inputs for one gate-passing arterial voxel.

        The voxel at (2, 2, 2) carries a clean first-pass bolus; ``peak_conc``
        is fixed at 30 (inside the 3T floor/ceiling). Returns the candidate's
        own TTP in frames so the caller can place the tissue reference relative
        to it.
        """
        n_frames = 120
        curve = _make_aif_curve(
            n_frames=n_frames,
            baseline_frames=baseline_frames,
            tr=tr,
            peak_frame=25,
            peak_height=40.0,
            fwhm_frames=3,
            recirculation_height=3.0,
            recirc_frame=60,
        )
        conc_data = np.zeros((5, 5, 5, n_frames), dtype=np.float64)
        conc_data[2, 2, 2, :] = curve
        brain_mask = np.ones((5, 5, 5), dtype=np.float64)
        peak_conc_map = np.zeros((5, 5, 5), dtype=np.float64)
        peak_conc_map[2, 2, 2] = 30.0
        candidate_indices = np.array([[2, 2, 2]], dtype=np.intp)

        det = AIFDetection()
        m = AIFDetection._calculate_candidate_metrics(
            curve, None, tr, baseline_frames,
        )
        assert m is not None
        candidate_ttp_frame = m["ttp"] / tr
        return (
            det,
            candidate_indices,
            conc_data,
            brain_mask,
            peak_conc_map,
            candidate_ttp_frame,
        )

    @pytest.mark.parametrize("tr", [1.0, 2.0])
    def test_candidate_after_tissue_rejected_tr_invariant(self, tr: float) -> None:
        """Candidate peaking 3 s after tissue is rejected at any TR."""
        baseline_frames = 20
        (
            det,
            candidate_indices,
            conc_data,
            brain_mask,
            peak_conc_map,
            candidate_ttp_frame,
        ) = self._build_single_arterial_voxel(tr, baseline_frames)
        # Tissue peaks 3 s BEFORE the candidate → candidate is venous.
        tissue_ttp_frame = candidate_ttp_frame + (-3.0) / tr
        scored, _ = det._score_and_select_candidates(
            candidate_indices, conc_data, None, peak_conc_map,
            brain_mask, tr, baseline_frames, "permissive", tissue_ttp_frame,
        )
        assert scored == []

    def test_candidate_leading_tissue_retained(self) -> None:
        """Candidate peaking 3 s before tissue survives the causality gate."""
        tr = 1.0
        baseline_frames = 20
        (
            det,
            candidate_indices,
            conc_data,
            brain_mask,
            peak_conc_map,
            candidate_ttp_frame,
        ) = self._build_single_arterial_voxel(tr, baseline_frames)
        # Tissue peaks 3 s AFTER the candidate → arterial causality holds.
        tissue_ttp_frame = candidate_ttp_frame + 3.0 / tr
        scored, _ = det._score_and_select_candidates(
            candidate_indices, conc_data, None, peak_conc_map,
            brain_mask, tr, baseline_frames, "permissive", tissue_ttp_frame,
        )
        assert len(scored) == 1
        assert scored[0][1] == (2, 2, 2)
        assert np.isfinite(scored[0][0])


# =============================================================================
# D3 — FWHM measures the contiguous first pass
# =============================================================================


class TestContiguousFirstPassFWHM:
    """FWHM measures the contiguous first-pass lobe, excluding separate recirculation."""

    def test_recirculation_above_half_max_not_counted(self) -> None:
        """Narrow first pass + tall recirc → FWHM stays narrow, below the full above-half-maximum span."""
        tr = 1.0
        baseline_frames = 20
        curve = _make_aif_curve(
            n_frames=120,
            baseline_frames=baseline_frames,
            tr=tr,
            peak_frame=25,
            peak_height=30.0,
            fwhm_frames=3,
            recirculation_height=20.0,  # above half-max (15)
            recirc_frame=70,
        )
        m = AIFDetection._calculate_candidate_metrics(
            curve, None, tr, baseline_frames,
        )
        assert m is not None
        new_fwhm = m["fwhm"]

        # Comparison: first-to-last frame above half-max across the whole
        # curve — the recirculation lobe would stretch this far past the peak.
        half_max = m["peak_conc"] / 2.0
        above = np.where(curve > half_max)[0]
        old_span = float(above[-1] - above[0]) * tr

        assert new_fwhm < 8.0
        assert new_fwhm < old_span

    def test_genuinely_broad_first_pass_still_wide(self) -> None:
        """A real broad contiguous bolus keeps a large FWHM."""
        tr = 1.0
        baseline_frames = 20
        curve = _make_aif_curve(
            n_frames=120,
            baseline_frames=baseline_frames,
            tr=tr,
            peak_frame=25,
            peak_height=30.0,
            fwhm_frames=10,
            recirculation_height=0.0,
        )
        m = AIFDetection._calculate_candidate_metrics(
            curve, None, tr, baseline_frames,
        )
        assert m is not None
        assert m["fwhm"] > 8.0


# =============================================================================
# D4 — compute_aif_fwhm_frames guards a baseline-window maximum
# =============================================================================


class TestFWHMFramesBaselineGuard:
    """D4: a global max inside the baseline returns 0, not a spurious width."""

    def test_spike_in_baseline_returns_zero(self) -> None:
        """Max at a baseline frame → 0 (no measurable first pass)."""
        n = 40
        baseline_frames = 5
        aif = np.zeros(n, dtype=np.float64)
        aif[3] = 100.0  # spike inside the baseline window
        aif[20] = 50.0  # smaller genuine post-baseline bump
        assert compute_aif_fwhm_frames(aif, baseline_frames=baseline_frames) == 0

    def test_post_baseline_bolus_positive(self) -> None:
        """A normal post-baseline Gaussian bolus yields a positive FWHM."""
        n = 40
        baseline_frames = 5
        t = np.arange(n)
        aif = np.zeros(n, dtype=np.float64)
        aif[baseline_frames:] = 10.0 * np.exp(
            -0.5 * ((t[baseline_frames:] - 15.0) / 3.0) ** 2
        )
        assert compute_aif_fwhm_frames(aif, baseline_frames=baseline_frames) > 0


# =============================================================================
# D5 — single-best fallback provenance sums to one voxel
# =============================================================================


class TestSingleBestProvenance:
    """D5: single-best fallback provenance describes exactly one voxel."""

    def test_single_best_tier_provenance_sums_to_one(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Multi-voxel union → single-best fallback collapses provenance to 1.

        Two sharp anterior arterial voxels (both beyond the physical edge and
        posterior-midline exclusions) pass per-voxel scoring and enter the cascade's
        union pool, so ``union_tier_provenance`` accumulates 2 before any
        fallback (asserted via ``n_aif_voxels_available == 2``). The background
        is a low, late, broad tissue bolus whose peak concentration (~4) sits
        below the arterial peak floor (5), so those voxels are rejected as AIF
        candidates while still supplying a sane tissue-TTP reference. With only
        two survivors the per-tier and union averaged-pool gates (which need
        ``>= MIN_AIF_CANDIDATES_ACCEPT`` = 3 voxels) cannot fire, so the cascade
        falls through to the single-best branch, which emits ONE voxel.

        The available pool contains two voxels, but emitted provenance must
        count only the single selected voxel.
        """
        import nibabel as nib  # type: ignore[import-untyped]

        shape = (13, 13, 13)
        n_frames = 80
        tr = 1.5
        baseline_frames = 10

        t = np.arange(n_frames)
        tissue = 4.0 * np.exp(-0.5 * ((t - 22.0) / 6.0) ** 2)
        conc = np.broadcast_to(tissue, (*shape, n_frames)).astype(np.float64).copy()
        curve = _make_aif_curve(
            n_frames=n_frames,
            baseline_frames=baseline_frames,
            tr=tr,
            peak_frame=16,
            peak_height=40.0,
            fwhm_frames=3,
            recirculation_height=6.0,
            recirc_frame=40,
        )
        # Both voxels are physically interior and anterior to the venous band.
        for coord in ((5, 9, 6), (8, 9, 6)):
            conc[coord[0], coord[1], coord[2], :] = curve

        brain = np.ones(shape, dtype=np.float64)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        conc_img = nib.Nifti1Image(conc, affine)
        conc_img.header.set_zooms((2.0, 2.0, 2.0, tr))
        brain_img = nib.Nifti1Image(brain, affine)

        conc_file = tmp_path / "conc.nii.gz"
        brain_file = tmp_path / "brain_mask.nii.gz"
        nib.save(conc_img, str(conc_file))
        nib.save(brain_img, str(brain_file))

        monkeypatch.chdir(tmp_path)
        det = AIFDetection()
        det.inputs.concentration_file = str(conc_file)
        det.inputs.brain_mask_file = str(brain_file)
        det.inputs.tr = tr
        det.inputs.baseline_frames = baseline_frames

        result = det.run()
        qc = result.outputs.qc_metrics
        detection_level = result.outputs.detection_level

        # Single-best branch taken: a real detection tier (not synthetic) with a
        # genuinely multi-voxel pre-fallback union.
        assert detection_level in ("brain_relaxed", "brain_permissive", "atlas_strict", "atlas_relaxed", "atlas_permissive")
        assert qc["n_aif_voxels_available"] == 2
        # Provenance describes the single emitted voxel.
        assert sum(qc["aif_tier_provenance"].values()) == 1
