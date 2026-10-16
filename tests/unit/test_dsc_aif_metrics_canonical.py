"""Pin the canonical AIF metrics shape produced by `aif_detect`.

The DSC AIF detection node emits its detection-time metrics in the nested
``aif_metrics`` block on ``qc_metrics``. Tests check the per-detection-mode
key set and the absence of flat ``best_*`` duplicates.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pmriprep.interfaces.dsc_aif import AIFDetection


_FLAT_FORBIDDEN = ("best_psr", "best_brr", "best_ttp", "best_fwhm", "best_ph")


def _candidate_metrics() -> dict[str, Any]:
    return {
        "ttp": 22.5,
        "fwhm": 6.0,
        "ph": 41.2,
        "psr": 78.0,
        "baseline_return_ratio": 92.0,
    }


class TestAutoDetectionFullShape:
    """``_finalize_aif_metrics`` (auto/atlas path) emits the full key set."""

    def test_canonical_aif_metrics_keys_present(self):
        aif_curve = np.zeros(60, dtype=np.float64)
        # Synthetic bolus: small baseline, sharp peak, return-to-baseline.
        aif_curve[10:14] = np.array([5.0, 25.0, 18.0, 9.0])
        aif_curve[14:] = 4.0

        top_candidates = [
            (0.5, (10, 10, 5), _candidate_metrics()),
        ]
        conc_data = np.zeros((20, 20, 10, 60), dtype=np.float64)
        conc_data[:, :, :, :] = aif_curve[None, None, None, :]
        brain_mask = np.ones((20, 20, 10), dtype=np.uint8)
        affine = np.eye(4)

        out_curve, qc = AIFDetection._finalize_aif_metrics(
            aif_curve=aif_curve,
            top_candidates=top_candidates,
            conc_data=conc_data,
            brain_mask=brain_mask,
            affine=affine,
            tr=1.5,
            n_candidates_requested=6,
            n_available=1,
            baseline_frames=10,
        )

        assert "aif_metrics" in qc
        am = qc["aif_metrics"]
        for key in ("peak_conc", "ttp", "fwhm", "ph", "psr", "brr", "arrival"):
            assert key in am, f"aif_metrics missing canonical key: {key}"

        assert am["peak_conc"] == pytest.approx(float(np.max(aif_curve)))
        assert am["ttp"] == pytest.approx(22.5)
        assert am["fwhm"] == pytest.approx(6.0)
        assert am["ph"] == pytest.approx(41.2)
        assert am["psr"] == pytest.approx(78.0)
        assert am["brr"] == pytest.approx(92.0)

    def test_no_flat_best_keys_at_top_level(self):
        aif_curve = np.zeros(60, dtype=np.float64)
        aif_curve[10:14] = np.array([5.0, 25.0, 18.0, 9.0])
        aif_curve[14:] = 4.0

        top_candidates = [
            (0.5, (10, 10, 5), _candidate_metrics()),
        ]
        conc_data = np.zeros((20, 20, 10, 60), dtype=np.float64)
        conc_data[:, :, :, :] = aif_curve[None, None, None, :]
        brain_mask = np.ones((20, 20, 10), dtype=np.uint8)
        affine = np.eye(4)

        _, qc = AIFDetection._finalize_aif_metrics(
            aif_curve=aif_curve,
            top_candidates=top_candidates,
            conc_data=conc_data,
            brain_mask=brain_mask,
            affine=affine,
            tr=1.5,
            n_candidates_requested=6,
            n_available=1,
            baseline_frames=10,
        )

        for k in _FLAT_FORBIDDEN:
            assert k not in qc, f"flat key {k!r} reintroduced into qc top level"

    @pytest.mark.parametrize("ph, expected_confidence", [
        (13.0, "low"), (15.0, "moderate"), (25.0, "high"),
    ])
    def test_candidate_ph_qualifies_prior(self, ph, expected_confidence):
        """Candidate PH controls prior eligibility independently of recovery."""
        from pmriprep.utils.dsc_cbv_constraint import (
            ConstraintMode, assess_dsc_cbv_quality, determine_constraint_mode,
        )

        aif_curve = np.zeros(60, dtype=np.float64)
        aif_curve[10:14] = np.array([5.0, 25.0, 18.0, 9.0])
        aif_curve[14:] = 4.0
        metrics = {**_candidate_metrics(), "ph": ph, "baseline_return_ratio": 20.0}
        top_candidates = [(0.5, (10 + i, 10, 5), metrics) for i in range(3)]
        conc_data = np.zeros((20, 20, 10, 60), dtype=np.float64)
        conc_data[:, :, :, :] = aif_curve[None, None, None, :]
        brain_mask = np.ones((20, 20, 10), dtype=np.uint8)
        _, qc = AIFDetection._finalize_aif_metrics(
            aif_curve=aif_curve,
            top_candidates=top_candidates,
            conc_data=conc_data,
            brain_mask=brain_mask,
            affine=np.eye(4),
            tr=1.5,
            n_candidates_requested=6,
            n_available=3,
            baseline_frames=10,
        )
        qc["cnr_whole_brain"] = {"fraction_cnr_gt_4": 0.99}
        quality = assess_dsc_cbv_quality(qc)
        config = determine_constraint_mode(
            quality, k2_metrics={"k2_reference_available": True},
        )
        assert quality.aif_peak_height == pytest.approx(ph / 100)
        assert quality.confidence == expected_confidence
        assert config.mode == (
            ConstraintMode.SKIP if ph < 15 else ConstraintMode.SOFT
        )


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
