"""Test DCE AIF score ordering on synthetic curve perturbations.

The reference is a sampled Parker curve with an imposed tail plateau.
Compare it with a reduced tail and a mixture with a synthetic tissue curve.
These fixtures check score ordering; they are not measured arterial truth
or independent validation of the scoring thresholds. The pooled fidelity
score is diagnostic, while candidate scores participate in selection.
"""

from __future__ import annotations

import numpy as np
import pytest

from pmriprep.interfaces.dce_aif import (
    AIFDetection,
    parker_population_aif,
    score_averaged_aif_fidelity,
)

TR = 4.8  # Synthetic volume interval in seconds
BASELINE_FRAMES = 5
N_FRAMES = 65

def _clean_arterial() -> np.ndarray:
    """Build a Parker-shaped reference with a post-peak floor of 22% of its peak."""
    curve = np.zeros(N_FRAMES)
    t = np.arange(N_FRAMES - BASELINE_FRAMES) * TR
    bolus = parker_population_aif(t, 1.0)
    peak = float(bolus.max())
    peak_i = int(np.argmax(bolus))
    bolus[peak_i:] = np.maximum(bolus[peak_i:], 0.22 * peak)
    curve[BASELINE_FRAMES:] = bolus
    return curve


def _tail_collapse(curve: np.ndarray) -> np.ndarray:
    """Preserve the peak while progressively attenuating the post-peak tail."""
    v1 = curve.copy()
    peak_idx = int(np.argmax(curve))
    v1[peak_idx:] = curve[peak_idx:] * np.linspace(1.0, 0.35, N_FRAMES - peak_idx)
    return v1


def _partial_volume(curve: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Mix the reference AIF with a slowly rising synthetic tissue curve."""
    tissue = np.zeros(N_FRAMES)
    ramp = np.linspace(0.0, 0.9, N_FRAMES - BASELINE_FRAMES) ** 0.6
    tissue[BASELINE_FRAMES:] = ramp * 3.0
    return alpha * curve + (1.0 - alpha) * tissue


# ---------------------------------------------------------------------------
# Per-candidate washout score ordering
# ---------------------------------------------------------------------------

class TestPerCandidateWashoutRanking:
    """Compare candidate scores for the reference and attenuated-tail fixtures."""

    def test_physiologic_beats_over_washed(self) -> None:
        """Asymmetric washout: a physiologic candidate out-scores an over-washed one."""
        det = AIFDetection()
        clean = _clean_arterial()
        over_washed = _tail_collapse(clean)
        s_clean, m_clean = det._score_dce_candidate(clean, TR, BASELINE_FRAMES, "relaxed")
        s_ow, m_ow = det._score_dce_candidate(over_washed, TR, BASELINE_FRAMES, "relaxed")
        # The over-washed candidate has the HIGHER washout fraction...
        assert m_ow["washout"] > m_clean["washout"]
        # A larger washout fraction does not imply a larger score.
        assert s_clean > s_ow, (
            f"over-washed candidate out-scored physiologic: {s_ow:.3f} >= {s_clean:.3f}"
        )

    def test_over_wash_penalized_leniently(self) -> None:
        """Over-washing is penalized less than under-washing.

        The configured upper half-width exceeds the lower half-width.
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
