"""Unit tests for DSC AIF metric functions."""

import numpy as np

from pmriprep.utils.dsc_aif import (
    compute_aif_fwhm_frames,
)


class TestComputeAIFFWHMFrames:
    """Tests for compute_aif_fwhm_frames."""

    def test_compute_aif_fwhm_frames(self) -> None:
        """Gaussian-like AIF has FWHM matching expected width."""
        # Create a Gaussian-like peak on top of zero baseline
        n = 40
        baseline_frames = 5
        aif = np.zeros(n)
        # Gaussian centered at frame 15, sigma=3 → FWHM ≈ 2.355*3 ≈ 7
        t = np.arange(n)
        aif[baseline_frames:] = 10.0 * np.exp(-0.5 * ((t[baseline_frames:] - 15.0) / 3.0) ** 2)

        fwhm = compute_aif_fwhm_frames(aif, baseline_frames=baseline_frames)
        # Analytic FWHM for Gaussian with sigma=3 is ~7.07, discrete ≈ 7
        assert 6 <= fwhm <= 9
