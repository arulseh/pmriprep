"""Unit tests for DSC synthetic AIF generation functions."""

import numpy as np
import pytest

from pmriprep.utils.dsc_aif import (
    SyntheticAIFResult,
    _estimate_dose_from_global_signal,
    generate_synthetic_aif,
)


class TestGenerateSyntheticAIF:
    """Tests for generate_synthetic_aif."""

    def test_full_dose_3T(self) -> None:
        """Full dose at 3T produces correct shape, synthetic flags, and zero baseline."""
        result = generate_synthetic_aif(
            n_frames=60, baseline_frames=5, dose_level=1.0, field_strength=3.0
        )

        assert result.is_synthetic is True
        assert result.absolute_quantification_valid is False
        assert result.aif_curve.shape == (60,)
        # Baseline frames are exactly zero
        assert np.all(result.aif_curve[:5] == 0.0)
        # Peak is positive
        assert float(np.max(result.aif_curve)) > 0.0

    def test_half_dose_lower_peak(self) -> None:
        """Half dose produces a lower peak than full dose."""
        full = generate_synthetic_aif(
            n_frames=60, baseline_frames=5, dose_level=1.0, field_strength=3.0
        )
        half = generate_synthetic_aif(
            n_frames=60, baseline_frames=5, dose_level=0.5, field_strength=3.0
        )

        assert float(np.max(half.aif_curve)) < float(np.max(full.aif_curve))

    def test_double_dose_higher_peak(self) -> None:
        """Double dose produces a higher peak than full dose."""
        full = generate_synthetic_aif(
            n_frames=60, baseline_frames=5, dose_level=1.0, field_strength=3.0
        )
        double = generate_synthetic_aif(
            n_frames=60, baseline_frames=5, dose_level=2.0, field_strength=3.0
        )

        assert float(np.max(double.aif_curve)) > float(np.max(full.aif_curve))

    def test_field_strength_scaling(self) -> None:
        """Lower field strength (1.5T) produces lower peak than 3T."""
        result_3t = generate_synthetic_aif(
            n_frames=60, baseline_frames=5, dose_level=1.0, field_strength=3.0
        )
        result_15t = generate_synthetic_aif(
            n_frames=60, baseline_frames=5, dose_level=1.0, field_strength=1.5
        )

        assert float(np.max(result_15t.aif_curve)) < float(np.max(result_3t.aif_curve))

    def test_baseline_frames_zero(self) -> None:
        """First baseline_frames values of aif_curve are exactly zero."""
        baseline_frames = 8
        result = generate_synthetic_aif(
            n_frames=60, baseline_frames=baseline_frames, dose_level=1.0
        )

        assert np.all(result.aif_curve[:baseline_frames] == 0.0)

    def test_no_negative_values(self) -> None:
        """All values in the generated AIF curve are non-negative."""
        result = generate_synthetic_aif(
            n_frames=60, baseline_frames=5, dose_level=1.0, field_strength=3.0
        )

        assert np.all(result.aif_curve >= 0.0)

    def test_custom_scaling(self) -> None:
        """Custom global_scaling produces a different amplitude than default."""
        default = generate_synthetic_aif(
            n_frames=60, baseline_frames=5, dose_level=1.0
        )
        scaled = generate_synthetic_aif(
            n_frames=60, baseline_frames=5, dose_level=1.0, global_scaling=2.0
        )

        default_peak = float(np.max(default.aif_curve))
        scaled_peak = float(np.max(scaled.aif_curve))
        # Scaled peak should be ~2x default
        assert scaled_peak == pytest.approx(2.0 * default_peak, rel=0.01)


class TestEstimateDoseFromGlobalSignal:
    """Tests for _estimate_dose_from_global_signal."""

    def test_low_signal_half_dose(self) -> None:
        """Low peak deltaR2 (<10) is classified as half dose."""
        rng = np.random.default_rng(42)
        # Create 4D data: 10x10x5 brain, 30 frames
        shape = (10, 10, 5, 30)
        brain_mask = np.ones(shape[:3], dtype=np.uint8)

        # Low concentration: peak ~5 s^-1
        conc = np.zeros(shape, dtype=np.float64)
        for t_idx in range(5, 30):
            conc[:, :, :, t_idx] = 5.0 * np.exp(-0.3 * (t_idx - 10) ** 2 / 10.0)
        conc += rng.normal(0, 0.1, size=shape)
        conc = np.maximum(conc, 0.0)

        dose_level, peak_deltaR2, scaling_factor = _estimate_dose_from_global_signal(
            conc, brain_mask, baseline_frames=5
        )

        assert dose_level < 0.5  # Low signal → low dose multiplier

    def test_medium_signal_full_dose(self) -> None:
        """Medium peak deltaR2 (~45) yields dose multiplier near 1.0."""
        # Create 4D data: 10x10x5 brain, 30 frames
        shape = (10, 10, 5, 30)
        brain_mask = np.ones(shape[:3], dtype=np.uint8)

        # Medium concentration: peak ~45 s^-1
        conc = np.zeros(shape, dtype=np.float64)
        for t_idx in range(5, 30):
            conc[:, :, :, t_idx] = 45.0 * np.exp(-0.3 * (t_idx - 10) ** 2 / 10.0)

        dose_level, peak_deltaR2, scaling_factor = _estimate_dose_from_global_signal(
            conc, brain_mask, baseline_frames=5
        )

        assert 0.8 < dose_level < 1.5  # ~45/40 ≈ 1.125


class TestSyntheticAIFResultDataclass:
    """Tests for SyntheticAIFResult dataclass."""

    def test_fields(self) -> None:
        """Warning message is non-empty and qc_metrics is a dict."""
        result = generate_synthetic_aif(
            n_frames=60, baseline_frames=5, dose_level=1.0
        )

        assert isinstance(result.warning_message, str)
        assert len(result.warning_message) > 0
        assert isinstance(result.qc_metrics, dict)
        assert "source" in result.qc_metrics
        assert result.qc_metrics["source"] == "synthetic"
