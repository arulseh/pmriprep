"""OSIPI ground-truth validation of DCE AIF fidelity scoring."""

from __future__ import annotations

import numpy as np

from pmriprep.interfaces.dce_aif import (
    compute_dce_aif_washout,
    score_averaged_aif_fidelity,
)
from pmriprep.utils.dce_constants import (
    AIF_WASHOUT_BAND_HALFWIDTH,
    AIF_WASHOUT_PHYSIOLOGIC_CENTER,
)
from tests.dro.test_dce_aif_fidelity import (
    BASELINE_FRAMES,
    TR,
    _clean_arterial,
    _partial_volume,
    _tail_collapse,
)

from .conftest import load_osipi_aif, skip_no_osipi_aif


@skip_no_osipi_aif
class TestOsipiGroundTruthAnchor:
    """Compare DCE AIF fidelity scores with the OSIPI MCA reference AIF."""

    @staticmethod
    def _ground_truth() -> np.ndarray:
        """Load the non-negative OSIPI plasma AIF."""
        aif = load_osipi_aif()
        assert aif is not None
        return np.clip(aif, 0.0, None)

    def test_gt_scores_high_and_above_contamination(self) -> None:
        """The reference MCA AIF scores above both contamination modes."""
        ground_truth = self._ground_truth()
        ground_truth_score, _ = score_averaged_aif_fidelity(
            ground_truth, TR, BASELINE_FRAMES, hematocrit=0.0
        )
        clean = _clean_arterial()
        tail_collapse_score, _ = score_averaged_aif_fidelity(
            _tail_collapse(clean), TR, BASELINE_FRAMES, hematocrit=0.0
        )
        partial_volume_score, _ = score_averaged_aif_fidelity(
            _partial_volume(clean), TR, BASELINE_FRAMES, hematocrit=0.0
        )

        assert ground_truth_score > 0.6
        assert ground_truth_score > tail_collapse_score
        assert ground_truth_score > partial_volume_score

    def test_gt_washout_is_physiologic(self) -> None:
        """The reference AIF washout lies within the physiologic band."""
        washout = compute_dce_aif_washout(self._ground_truth(), BASELINE_FRAMES)
        assert (
            abs(washout - AIF_WASHOUT_PHYSIOLOGIC_CENTER)
            < AIF_WASHOUT_BAND_HALFWIDTH
        )
