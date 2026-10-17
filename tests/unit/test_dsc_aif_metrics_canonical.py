"""Pin the canonical AIF metrics shape produced by `aif_detect`.

The DSC AIF detection node emits its detection-time metrics in the nested
``aif_metrics`` block on ``qc_metrics``. Tests check the per-detection-mode
key set and its use in prior qualification.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pmriprep.interfaces.dsc_aif import AIFDetection, _CascadeDetectionResult


def _candidate_metrics() -> dict[str, Any]:
    return {
        "ttp": 22.5,
        "fwhm": 6.0,
        "ph": 41.2,
        "psr": 78.0,
        "baseline_return_ratio": 92.0,
    }


def _measured_pool(
    ph: float, other_tail_fraction: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, tuple[int, int, int], dict]]]:
    """Three arterial signals with unequal costs and known weighted signal drop."""
    pulse = np.maximum(0.0, 1.0 - ((np.arange(80) - 25) / 7.0) ** 2)
    drops = np.array([ph + 2, ph - 2, ph - 2]) / 100.0
    signal = 1000.0 * (1.0 - drops[:, None] * pulse)
    peaks = -np.log1p(-drops) / 0.03
    tails = peaks * [0.1, other_tail_fraction, other_tail_fraction]
    signal[:, 45:65] = 1000.0 * np.exp(-0.03 * tails[:, None])
    concentration = -np.log(signal / 1000.0) / 0.03
    candidates = [
        (cost, (i, 0, 0), {
            **_candidate_metrics(), "ph": float(100 * drops[i]),
            "baseline_return_ratio": 10.0 if i == 0 else 100 * other_tail_fraction,
        })
        for i, cost in enumerate((0.1, 0.2, 0.2))
    ]
    return concentration[:, None, None, :], signal[:, None, None, :], candidates


class TestAutoDetectionFullShape:
    """Measured-pool PH and BRR reach the prior gate with matching support."""

    def test_canonical_aif_metrics_use_accepted_roi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Single-candidate ROI QC uses its masked neighbors and finalized baseline."""
        pulse = np.maximum(0.0, 1.0 - ((np.arange(80) - 30) / 7.0) ** 2)
        signal = np.broadcast_to(1000.0 * (1.0 - 0.8 * pulse), (3, 3, 3, 80)).copy()
        mask = np.zeros((3, 3, 3), dtype=bool)
        for coord, drop in (((1, 1, 1), 0.6), ((1, 1, 0), 0.3), ((1, 0, 1), 0.0)):
            signal[coord] = 1000.0 * (1.0 - drop * pulse)
            signal[coord][:10] = 800.0
            mask[coord] = True
        concentration = -np.log(signal / 1000.0) / 0.03
        candidates = [(0.5, (1, 1, 1), _candidate_metrics())]
        curve, accepted_mask = AIFDetection._compute_roi_averaged_aif(
            candidates, concentration, mask, 1.5, roi_size=3,
        )
        cascade = _CascadeDetectionResult(
            (curve, accepted_mask, 3), candidates, 1, "atlas_strict", {},
            "atlas_strict", ["atlas_strict"], {"atlas_strict": 1}, {}, {},
        )
        context = SimpleNamespace(
            concentration=concentration, signal=signal, brain_mask=mask,
            affine=np.eye(4), tr=1.5, n_candidates=6, baseline_frames=5,
            lesion_mask=None, enhancing_mask=None,
        )
        detector = AIFDetection()
        monkeypatch.setattr(
            detector, "_refine_selected_aif_baseline", lambda *args, **kwargs: (20, {}),
        )
        result = detector._finalize_detected_aif(cascade, context, 5)
        am = result.qc_metrics["aif_metrics"]
        np.testing.assert_array_equal(result.curve, curve)
        assert result.baseline_frames == 20
        assert am["peak_conc"] == pytest.approx(float(np.max(curve)))
        assert am["ttp"] == 22.5
        assert am["fwhm"] == 6.0
        assert am["psr"] == 78.0
        # S0 is 900 over the finalized 20-frame baseline; pooled minimum is 700.
        assert am["ph"] == pytest.approx(100 * (900 - 700) / 900)
        assert am["brr"] == pytest.approx(0.0)

    @pytest.mark.parametrize("ph, other_tail_fraction, expected_confidence", [
        (14.0, 0.3, "low"), (20.0, 0.3, "moderate"), (30.0, 0.3, "high"),
        (30.0, 0.9, "low"),
    ])
    def test_pooled_ph_and_brr_qualify_prior(
        self, ph: float, other_tail_fraction: float, expected_confidence: str,
    ) -> None:
        """The measured pool controls eligibility even when its best candidate passes."""
        from pmriprep.utils.dsc_cbv_constraint import (
            ConstraintMode, assess_dsc_cbv_quality, determine_constraint_mode,
        )

        concentration, signal, candidates = _measured_pool(ph, other_tail_fraction)
        mask = np.ones(concentration.shape[:3], dtype=bool)
        curve, _ = AIFDetection._compute_roi_averaged_aif(
            candidates, concentration, mask, 1.5, roi_size=1,
        )
        _, qc = AIFDetection._finalize_aif_metrics(
            curve, candidates, signal, mask, np.eye(4), 1.5, 6, 3, baseline_frames=10,
        )
        qc["cnr_whole_brain"] = {"fraction_cnr_gt_4": 0.99}
        quality = assess_dsc_cbv_quality(qc)
        config = determine_constraint_mode(
            quality, k2_metrics={"k2_reference_available": True},
        )
        # Costs give weights 1/2, 1/4, 1/4. Quadratic smoothing preserves the signal dip.
        assert quality.aif_peak_height == pytest.approx(ph / 100)
        assert quality.aif_baseline_return == pytest.approx(
            np.mean(curve[54:64]) / np.max(curve[12:36])
        )
        assert quality.aif_baseline_return != pytest.approx(0.1)
        assert quality.confidence == expected_confidence
        assert config.mode == (
            ConstraintMode.SKIP if expected_confidence == "low" else ConstraintMode.SOFT
        )

    @pytest.mark.parametrize("signal_state", ["missing", "nonfinite"])
    def test_unavailable_pooled_signal_withholds_prior(self, signal_state: str) -> None:
        """Unavailable pool signal leaves PH unavailable despite candidate metadata."""
        from pmriprep.utils.dsc_cbv_constraint import assess_dsc_cbv_quality

        concentration, signal, candidates = _measured_pool(30.0)
        mask = np.ones(concentration.shape[:3], dtype=bool)
        curve, _ = AIFDetection._compute_roi_averaged_aif(
            candidates, concentration, mask, 1.5, roi_size=1,
        )
        signal[1, 0, 0, 25] = np.nan
        _, qc = AIFDetection._finalize_aif_metrics(
            curve, candidates, None if signal_state == "missing" else signal,
            mask, np.eye(4), 1.5, 6, 3, baseline_frames=10,
        )
        quality = assess_dsc_cbv_quality(qc)
        assert "ph" not in qc["aif_metrics"]
        assert quality.aif_peak_height is None
        assert quality.aif_baseline_return is not None
        assert quality.use_constraint is False
        assert "AIF signal-drop fraction (PH) unavailable" in quality.reasons


class TestExternalAifSubsetShape:
    """External AIF metrics, provenance, and prior/concordance qualification."""

    def test_external_aif_subset_shape(self, tmp_path, monkeypatch):
        import nibabel as nib

        monkeypatch.chdir(tmp_path)

        # Synthetic concentration NIfTI: 60 timepoints
        n_t = 60
        conc = np.zeros((4, 4, 4, n_t), dtype=np.float32)
        # Inject a synthetic ΔR2* curve into one voxel — overall
        # peak only matters for the curve, not for the field set.
        conc[2, 2, 2, 8:14] = np.array([5.0, 12.0, 28.0, 22.0, 14.0, 8.0])
        conc_path = tmp_path / "dsc.nii.gz"
        nib.save(nib.Nifti1Image(conc, np.eye(4)), str(conc_path))

        # PMRIPrep-style external AIF TSV (s⁻¹), matching n_t timepoints.
        from pmriprep.interfaces.qc_writers import _write_aif_timeseries_tsv

        aif = np.zeros(n_t, dtype=np.float64)
        aif[8:14] = np.array([5.0, 12.0, 28.0, 22.0, 14.0, 8.0])
        aif_path = tmp_path / "ext_aif.tsv"
        _write_aif_timeseries_tsv(aif_path, aif, "DSC")

        interface = AIFDetection()
        interface.inputs.concentration_file = str(conc_path)
        interface.inputs.external_aif_file = str(aif_path)
        interface.inputs.baseline_frames = 5
        interface.inputs.tr = 1.5
        interface.inputs.field_strength = 3.0

        # Drive only the external-path body, not the full _run.
        # _handle_external_aif sets self._results which we read after.
        interface._results = {}
        interface._handle_external_aif(runtime=None)

        qc = interface._results["qc_metrics"]
        assert "aif_metrics" in qc
        am = qc["aif_metrics"]
        # Subset that the external path can derive
        assert "peak_conc" in am
        assert "fwhm" in am
        assert qc["washout_fraction"] == pytest.approx(1.0)
        assert qc["detection_method"] == "external"
        assert qc["cbv_auc_denominator_is_synthetic"] is False

        from pmriprep.utils.aif_provenance import build_aif_provenance_block
        from pmriprep.utils.dsc_cbv_constraint import prepare_vp_constraint_from_data
        from pmriprep.workflows.dce._helpers import _dsc_dce_validation_node

        provenance = build_aif_provenance_block(qc, modality="DSC")
        assert provenance["Source"] == "external"
        assert provenance["UsedAsCBVDenominator"] is True
        prior = prepare_vp_constraint_from_data(
            ncbv_file=None, aif_qc_data=qc, brain_mask_file=str(conc_path),
        )
        assert prior.config.mode.value == "skip"
        assert prior.vp_map is None
        assert prior.qc_report["skip_reason"] == "external_aif_qc_unavailable"
        concordance = _dsc_dce_validation_node(
            vp_file=str(conc_path),
            pk_qc={"dsc_cbv_constraint": prior.qc_report},
        )
        assert concordance["skipped"] is True
        assert concordance["skip_reason"] == "external_aif_qc_unavailable"
        assert concordance["reason"] == prior.config.reason
