"""Ground-truth-anchored DCE AIF fidelity gate (regression sentinel).

This is the can-fail test that ends the "audit-20x" cycle for the DCE AIF
quality scorer. It pins two properties that a prior version of the code
violated (the monotonic per-candidate washout term ``min(1, washout/0.7)``
*rewarded* a collapsed tail, so a partial-volume-diluted AIF out-scored a
faithful one):

1. The per-candidate scorer prefers a physiologic-washout arterial curve over
   an over-washed (tail-collapsed) one — the exact inversion that let the
   diluting voxels into the pooled AIF (OSIPI sub-0091: AIF 3->10 voxels,
   Ktrans overestimate +66% -> +111%).
2. The averaged-AIF fidelity scalar rank-orders the true AIF above both
   contamination modes: tail-collapse (V1) and genuine tissue partial-volume
   (V2), which move the tail in *opposite* directions.

The fidelity scalar is a QC / regression-sentinel number, NOT a runtime
accept/reject gate (real detected AIFs cluster low because they systematically
under-recover the real-MCA late tail, a second-order effect for Ktrans), so the
assertions are rank-orderings, not absolute thresholds.
"""

from __future__ import annotations

import numpy as np
import pytest

from pmriprep.interfaces.dce_aif import (
    AIFDetection,
    parker_population_aif,
    score_averaged_aif_fidelity,
)

TR = 4.8  # OSIPI DCE synthetic temporal resolution (Shalom 2024)
BASELINE_FRAMES = 5
N_FRAMES = 65

def _clean_arterial() -> np.ndarray:
    """Build a physiologic arterial AIF: Parker first pass + a recirculation tail.

    Pure ``parker_population_aif`` is thin-tailed (washout ~0.92); a real MCA AIF
    (the OSIPI DRO ground truth) carries a heavier recirculation tail (washout
    ~0.78). Floor the post-peak curve at ~0.22x peak so the reference reflects a
    faithful artery, not the idealized thin Parker form.
    """
    curve = np.zeros(N_FRAMES)
    t = np.arange(N_FRAMES - BASELINE_FRAMES) * TR
    bolus = parker_population_aif(t, 1.0)
    peak = float(bolus.max())
    peak_i = int(np.argmax(bolus))
    bolus[peak_i:] = np.maximum(bolus[peak_i:], 0.22 * peak)
    curve[BASELINE_FRAMES:] = bolus
    return curve


def _tail_collapse(curve: np.ndarray) -> np.ndarray:
    """V1: same first-pass peak, tail attenuated toward baseline (over-washout).

    Reproduces the e745c05 3->10-voxel signature: peak preserved, washout up,
    normalized tail down.
    """
    v1 = curve.copy()
    peak_idx = int(np.argmax(curve))
    v1[peak_idx:] = curve[peak_idx:] * np.linspace(1.0, 0.35, N_FRAMES - peak_idx)
    return v1


def _partial_volume(curve: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """V2: genuine tissue partial-volume, alpha*arterial + (1-alpha)*tissue.

    A tissue mixture *lowers* the peak and *raises* the tail (tissue does not
    wash out) — the opposite direction to V1. The gate must catch both.
    """
    tissue = np.zeros(N_FRAMES)
    ramp = np.linspace(0.0, 0.9, N_FRAMES - BASELINE_FRAMES) ** 0.6
    tissue[BASELINE_FRAMES:] = ramp * 3.0
    return alpha * curve + (1.0 - alpha) * tissue


# ---------------------------------------------------------------------------
# 1. Per-candidate scorer: the ranking inversion is fixed (can-fail vs old code)
# ---------------------------------------------------------------------------

class TestPerCandidateWashoutRanking:
    """The two-sided washout term must prefer physiologic over over-washed.

    Under the old monotonic ``min(1, washout/0.7)`` the over-washed curve scored
    >= the physiologic one; this asserts the corrected ordering.
    """

    def test_physiologic_beats_over_washed(self) -> None:
        """Asymmetric washout: a physiologic candidate out-scores an over-washed one."""
        det = AIFDetection()
        clean = _clean_arterial()
        over_washed = _tail_collapse(clean)
        s_clean, m_clean = det._score_dce_candidate(clean, TR, BASELINE_FRAMES, "relaxed")
        s_ow, m_ow = det._score_dce_candidate(over_washed, TR, BASELINE_FRAMES, "relaxed")
        # The over-washed candidate has the HIGHER washout fraction...
        assert m_ow["washout"] > m_clean["washout"]
        # ...but must NOT be rewarded for it (the old bug did exactly that).
        assert s_clean > s_ow, (
            f"over-washed candidate out-scored physiologic: {s_ow:.3f} >= {s_clean:.3f}"
        )

    def test_over_wash_penalized_leniently(self) -> None:
        """Over-washing is penalized less than under-washing.

        The asymmetric band is lenient above centre (benign long-scan drift) and
        strict below (contamination).
        """
        from pmriprep.interfaces.dce_aif import _asymmetric_band_score
        from pmriprep.utils.dce_constants import (
            AIF_WASHOUT_BAND_HALFWIDTH,
            AIF_WASHOUT_BAND_HALFWIDTH_UPPER,
            AIF_WASHOUT_PHYSIOLOGIC_CENTER,
        )
        c = AIF_WASHOUT_PHYSIOLOGIC_CENTER
        delta = 0.10
        over = _asymmetric_band_score(
            c + delta, c, AIF_WASHOUT_BAND_HALFWIDTH, AIF_WASHOUT_BAND_HALFWIDTH_UPPER
        )
        under = _asymmetric_band_score(
            c - delta, c, AIF_WASHOUT_BAND_HALFWIDTH, AIF_WASHOUT_BAND_HALFWIDTH_UPPER
        )
        assert over > under  # over-washing (long scan) penalized less than under-washing


# ---------------------------------------------------------------------------
# 2. Averaged-AIF fidelity: truth ranks above both contamination modes
# ---------------------------------------------------------------------------

class TestAveragedFidelityRanking:
    """The averaged-AIF fidelity scalar rank-orders truth above contamination."""

    def test_truth_above_tail_collapse(self) -> None:
        """A faithful AIF out-scores an over-washed (tail-collapsed) one."""
        clean = _clean_arterial()
        v1 = _tail_collapse(clean)
        f_clean, _ = score_averaged_aif_fidelity(
            clean, TR, BASELINE_FRAMES, hematocrit=0.0
        )
        f_v1, _ = score_averaged_aif_fidelity(
            v1, TR, BASELINE_FRAMES, hematocrit=0.0
        )
        assert f_clean > f_v1

    def test_truth_above_partial_volume(self) -> None:
        """A faithful AIF out-scores a tissue partial-volume-diluted one."""
        clean = _clean_arterial()
        v2 = _partial_volume(clean)
        f_clean, _ = score_averaged_aif_fidelity(
            clean, TR, BASELINE_FRAMES, hematocrit=0.0
        )
        f_v2, _ = score_averaged_aif_fidelity(
            v2, TR, BASELINE_FRAMES, hematocrit=0.0
        )
        assert f_clean > f_v2

    def test_partial_volume_two_sided_tail(self) -> None:
        """V2's tissue-raised tail must NOT earn full marks (two-sided tail term)."""
        v2 = _partial_volume(_clean_arterial())
        _, m = score_averaged_aif_fidelity(
            v2, TR, BASELINE_FRAMES, hematocrit=0.0
        )
        assert m["tail_ratio"] > 0.35  # tissue-diluted tail sits high
        assert m["tail_score"] < 0.5   # ...and is penalized, not rewarded


class TestPlasmaReferenceConsistency:
    """Reference scaling must be invariant to a consistent Hct conversion."""

    @pytest.mark.parametrize("hematocrit", [0.20, 0.42, 0.55])
    def test_consistent_blood_to_plasma_conversion_is_score_invariant(
        self, hematocrit: float,
    ) -> None:
        """The same blood curve receives the same score at any stated Hct."""
        blood_curve = _clean_arterial()
        reference_score, reference_metrics = score_averaged_aif_fidelity(
            blood_curve, TR, BASELINE_FRAMES, hematocrit=0.0
        )
        plasma_curve = blood_curve / (1.0 - hematocrit)
        plasma_score, plasma_metrics = score_averaged_aif_fidelity(
            plasma_curve, TR, BASELINE_FRAMES, hematocrit=hematocrit
        )

        assert plasma_score == pytest.approx(reference_score, abs=1e-12)
        assert plasma_metrics["scale_score"] == pytest.approx(
            reference_metrics["scale_score"], abs=1e-12
        )
        assert plasma_metrics["auc_ratio"] == pytest.approx(
            reference_metrics["auc_ratio"], abs=1e-12
        )

    @pytest.mark.parametrize("hematocrit", [-0.01, 1.0, 1.1])
    def test_invalid_hematocrit_is_rejected(self, hematocrit: float) -> None:
        """Invalid hematocrit cannot define a plasma reference scale."""
        with pytest.raises(ValueError, match="hematocrit must be in"):
            score_averaged_aif_fidelity(
                _clean_arterial(), TR, BASELINE_FRAMES, hematocrit=hematocrit
            )
