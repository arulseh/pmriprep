"""Tests for synthetic AIF peak calibration.

Verifies the population-median, B0-scaled, dose-aware peak formula:
    peak = max(SYNTH_AIF_BASE_PEAK_3T * (B0 / 3.0) * effective_dose,
               SYNTH_AIF_MIN_PEAK)
"""


import numpy as np
import pytest

from pmriprep.utils.dsc_constants import (
    SYNTH_AIF_MIN_PEAK,
    SYNTH_AIF_RECIRC_ALPHA,
    SYNTH_AIF_RECIRC_DELAY_S,
    SYNTH_AIF_RECIRC_FRACTION,
)


def _make_aif_detection(**trait_overrides):
    """Create a mock AIFDetection with specified input traits."""
    from pmriprep.interfaces.dsc_aif import AIFDetection

    det = AIFDetection()
    for key, val in trait_overrides.items():
        setattr(det.inputs, key, val)
    return det


# -----------------------------------------------------------------------
# _effective_dose helper
# -----------------------------------------------------------------------

class TestEffectiveDose:

    def test_standard_dose(self):
        det = _make_aif_detection(dose=1.0, agent_standard_dose=0.1, gd_multiplicity=1)
        assert det._effective_dose() == pytest.approx(1.0)

    def test_double_dose(self):
        det = _make_aif_detection(dose=2.0, agent_standard_dose=0.1, gd_multiplicity=1)
        assert det._effective_dose() == pytest.approx(2.0)

    def test_gadoquatrane(self):
        det = _make_aif_detection(dose=1.0, agent_standard_dose=0.04, gd_multiplicity=4)
        assert det._effective_dose() == pytest.approx(0.4)

    def test_raw_value_not_clamped(self):
        det = _make_aif_detection(dose=5.0, agent_standard_dose=0.2, gd_multiplicity=4)
        raw = det._effective_dose()
        assert raw == pytest.approx(10.0)

    def test_explicit_overrides(self):
        det = _make_aif_detection(dose=1.0, agent_standard_dose=0.1, gd_multiplicity=1)
        val = det._effective_dose(dose=0.5, agent_standard_dose=0.1)
        assert val == pytest.approx(0.5)

    def test_all_defaults(self):
        det = _make_aif_detection()
        assert det._effective_dose() == pytest.approx(1.0)


# -----------------------------------------------------------------------
# _generate_synthetic_aif peak formula
# -----------------------------------------------------------------------

class TestSyntheticAIFPeak:

    @pytest.mark.parametrize(
        "field_strength, effective_dose, expected_peak",
        [
            (3.0, 1.0, 35.0),            # standard single dose 3T
            (3.0, 0.75, 26.25),           # three-quarter dose
            (3.0, 2.0, 70.0),             # double dose
            (1.5, 1.0, 17.5),             # 1.5T scaling
            (7.0, 1.0, 35.0 * 7 / 3),    # 7T scaling
            (3.0, 0.4, 14.0),             # gadoquatrane effective dose
            (3.0, 2.5, 87.5),             # dose cap boundary
        ],
        ids=[
            "standard_3T",
            "penn_sub02_075",
            "double_dose",
            "1.5T",
            "7T",
            "gadoquatrane",
            "dose_cap",
        ],
    )
    def test_peak_formula(self, field_strength, effective_dose, expected_peak):
        det = _make_aif_detection()
        aif, metrics = det._generate_synthetic_aif(
            n_timepoints=60,
            tr=1.5,
            baseline_frames=5,
            field_strength=field_strength,
            effective_dose=effective_dose,
        )
        assert metrics["synthetic_aif_target_peak"] == pytest.approx(expected_peak)
        assert aif.max() > 0

    def test_min_peak_guard(self):
        det = _make_aif_detection()
        aif, metrics = det._generate_synthetic_aif(
            n_timepoints=60,
            tr=1.5,
            baseline_frames=5,
            field_strength=3.0,
            effective_dose=0.01,
        )
        assert metrics["synthetic_aif_target_peak"] == pytest.approx(SYNTH_AIF_MIN_PEAK)

    def test_tissue_max_does_not_affect_peak(self):
        det = _make_aif_detection()
        _, metrics_no_tissue = det._generate_synthetic_aif(
            n_timepoints=60, field_strength=3.0, effective_dose=1.0,
        )
        _, metrics_high_tissue = det._generate_synthetic_aif(
            n_timepoints=60, field_strength=3.0, effective_dose=1.0,
            tissue_max_delta_r2star=500.0,
        )
        assert (
            metrics_no_tissue["synthetic_aif_target_peak"]
            == metrics_high_tissue["synthetic_aif_target_peak"]
        )

    def test_tissue_max_none_uses_formula(self):
        """An absent tissue maximum leaves the formula-based target peak at 35."""
        det = _make_aif_detection()
        _, metrics = det._generate_synthetic_aif(
            n_timepoints=60, field_strength=3.0, effective_dose=1.0,
            tissue_max_delta_r2star=None,
        )
        assert metrics["synthetic_aif_target_peak"] == pytest.approx(35.0)


# -----------------------------------------------------------------------
# QC metrics
# -----------------------------------------------------------------------

class TestSyntheticAIFQCMetrics:

    def test_new_qc_keys_present(self):
        det = _make_aif_detection()
        _, metrics = det._generate_synthetic_aif(
            n_timepoints=60, field_strength=3.0, effective_dose=1.0,
            tissue_max_delta_r2star=7.0,
        )
        assert metrics["synthetic_aif_peak_formula"] == "population_median_b0_scaled"
        assert metrics["synthetic_aif_effective_dose"] == pytest.approx(1.0)
        assert metrics["synthetic_aif_field_strength"] == pytest.approx(3.0)
        assert metrics["synthetic_aif_amplitude_source"] == "population_calibrated"

    def test_peak_to_tissue_ratio(self):
        det = _make_aif_detection()
        _, metrics = det._generate_synthetic_aif(
            n_timepoints=60, field_strength=3.0, effective_dose=1.0,
            tissue_max_delta_r2star=7.0,
        )
        assert metrics["synthetic_aif_peak_to_tissue_ratio"] == pytest.approx(5.0)

    def test_tissue_p95_qc_key(self):
        det = _make_aif_detection()
        _, metrics = det._generate_synthetic_aif(
            n_timepoints=60, field_strength=3.0, effective_dose=1.0,
            tissue_max_delta_r2star=34.0,
        )
        assert metrics["synthetic_aif_tissue_p95_delta_r2star"] == pytest.approx(34.0)

    def test_no_tissue_max_no_ratio_key(self):
        det = _make_aif_detection()
        _, metrics = det._generate_synthetic_aif(
            n_timepoints=60, field_strength=3.0, effective_dose=1.0,
        )
        assert "synthetic_aif_peak_to_tissue_ratio" not in metrics
        assert "synthetic_aif_tissue_p95_delta_r2star" not in metrics

    def test_tissue_max_zero_no_division_error(self):
        det = _make_aif_detection()
        _, metrics = det._generate_synthetic_aif(
            n_timepoints=60, field_strength=3.0, effective_dose=1.0,
            tissue_max_delta_r2star=0.0,
        )
        assert "synthetic_aif_peak_to_tissue_ratio" not in metrics
        assert metrics["synthetic_aif_tissue_p95_delta_r2star"] == pytest.approx(0.0)

    def test_tissue_max_negative_no_ratio(self):
        det = _make_aif_detection()
        _, metrics = det._generate_synthetic_aif(
            n_timepoints=60, field_strength=3.0, effective_dose=1.0,
            tissue_max_delta_r2star=-1.0,
        )
        assert "synthetic_aif_peak_to_tissue_ratio" not in metrics


# -----------------------------------------------------------------------
# Shape preservation
# -----------------------------------------------------------------------

class TestShapePreserved:

    def test_shape_parameters_unchanged(self):
        det = _make_aif_detection()
        _, metrics = det._generate_synthetic_aif(
            n_timepoints=60, field_strength=3.0, effective_dose=1.0,
        )
        assert metrics["synthetic_aif_alpha"] == pytest.approx(4.0)
        assert metrics["synthetic_aif_time_to_peak_s"] == pytest.approx(4.5)

    def test_recirculation_scales_with_peak(self):
        det = _make_aif_detection()
        _, metrics_std = det._generate_synthetic_aif(
            n_timepoints=60, field_strength=3.0, effective_dose=1.0,
        )
        _, metrics_dbl = det._generate_synthetic_aif(
            n_timepoints=60, field_strength=3.0, effective_dose=2.0,
        )
        # Dose-invariant fraction, pinned to the physics-governed constant.
        assert (metrics_std["synthetic_aif_recirc_fraction"]
                == pytest.approx(SYNTH_AIF_RECIRC_FRACTION))
        assert (metrics_dbl["synthetic_aif_recirc_fraction"]
                == pytest.approx(SYNTH_AIF_RECIRC_FRACTION))

    def test_recirculation_lobe_is_physiological(self):
        """The recirculation lobe is small and dispersed vs the first pass:
        second-peak amplitude ~0.12x first pass and recirc area ~0.25x first-pass
        area for this configured synthetic AIF."""
        det = _make_aif_detection()
        tr, baseline_frames = 1.5, 5
        aif, metrics = det._generate_synthetic_aif(
            n_timepoints=60, tr=tr, baseline_frames=baseline_frames,
        )
        assert (metrics["synthetic_aif_recirc_alpha"]
                == pytest.approx(SYNTH_AIF_RECIRC_ALPHA))

        # Split first-pass vs recirculation at the recirc onset (derived from
        # constants; the first pass has decayed to ~1e-5 of peak by then).
        arrival = baseline_frames + 2
        recirc_t0 = (arrival * tr
                     + metrics["synthetic_aif_time_to_peak_s"]
                     + SYNTH_AIF_RECIRC_DELAY_S)
        idx = round(recirc_t0 / tr)
        first_pass, recirc = aif[:idx], aif[idx:]

        amp_ratio = recirc.max() / first_pass.max()
        assert amp_ratio == pytest.approx(SYNTH_AIF_RECIRC_FRACTION, abs=0.01)

        area_ratio = np.trapezoid(recirc) / np.trapezoid(first_pass)
        assert 0.20 <= area_ratio <= 0.27, area_ratio

    def test_baseline_is_zero(self):
        det = _make_aif_detection()
        aif, _ = det._generate_synthetic_aif(
            n_timepoints=60, tr=1.5, baseline_frames=5,
            field_strength=3.0, effective_dose=1.0,
        )
        np.testing.assert_array_equal(aif[:5], 0.0)

    def test_no_negative_values(self):
        det = _make_aif_detection()
        aif, _ = det._generate_synthetic_aif(
            n_timepoints=60, field_strength=3.0, effective_dose=1.0,
        )
        assert np.all(aif >= 0)
