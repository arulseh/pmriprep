"""AIF curve sidecar uses flat ``AIF*``-prefixed top-level keys.

Asserts that ``build_aif_timeseries_sidecar`` (and the qc_writers AIF
sidecar emitter) produce a JSON sidecar with flat top-level keys at the
sidecar surface, with detection-time metrics sourced from the
producer-shape ``aif_qc.aif_metrics`` nested block.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib  # type: ignore[import-untyped]
import numpy as np
import pytest

from pmriprep.utils.sidecar_builders import (
    aif_timeseries_columns,
    build_aif_timeseries_sidecar,
)


@pytest.fixture
def aif_qc() -> dict[str, Any]:
    """Producer-shape AIF QC dict (subset matching dsc_aif.py output)."""
    return {
        "detection_method": "atlas_strict",
        "n_aif_voxels_selected": 6,
        "quality_score": 0.81,
        "cascade_final_level": "atlas_strict",
        "hemisphere_coverage": "bilateral",
        "gamma_fit_applied": True,
        "gamma_fit_r_squared": 0.94,
        "gamma_fit_auc_ratio": 0.97,
        "synthetic": False,
        "cbv_auc_denominator_is_synthetic": False,
        "hematocrit": 0.42,
        "aif_metrics": {
            "peak_conc": 12.5,
            "fwhm": 6.2,
            "ttp": 18.0,
            "arrival": 9.5,
            "psr": 78.0,    # → 0.78 after /100 in cosmetic writer
            "brr": 92.0,    # → 0.92 after /100 in cosmetic writer
        },
    }


@pytest.fixture
def bids_root(tmp_path) -> Path:
    bids = tmp_path / "bids"
    perf = bids / "sub-X" / "ses-Y" / "perf"
    perf.mkdir(parents=True)
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )
    src = perf / "sub-X_ses-Y_dsc.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), src)
    src.with_name("sub-X_ses-Y_dsc.json").write_text(
        json.dumps({"RepetitionTime": 1.5})
    )
    return bids


class TestFlatKeyShape:
    def test_dsc_flat_keys_present(self, aif_qc, bids_root):
        sidecar = build_aif_timeseries_sidecar(
            modality="DSC",
            source_paths=[bids_root / "sub-X" / "ses-Y" / "perf" / "sub-X_ses-Y_dsc.nii.gz"],
            bids_dir=bids_root,
            subject="X",
            session="Y",
            aif_qc=aif_qc,
        )
        # Flat top-level keys, NOT nested
        assert sidecar["AIFMethod"] == "atlas_strict"
        assert sidecar["AIFNumberOfVoxelsSelected"] == 6
        assert sidecar["AIFPeakAmplitude"] == pytest.approx(12.5)
        assert sidecar["AIFFWHM"] == pytest.approx(6.2)
        assert sidecar["AIFTimeToPeak"] == pytest.approx(18.0)
        assert "AIFArrival" not in sidecar
        assert "AIFBaselineReturnRatio" not in sidecar
        assert "AIFPeakSignalRatio" not in sidecar
        assert sidecar["AIFQualityScore"] == pytest.approx(0.81)
        assert sidecar["AIFArterialCascadeTier"] == "atlas_strict"
        assert sidecar["AIFHemisphereCoverage"] == "bilateral"
        assert sidecar["AIFGammaFitApplied"] is True
        assert sidecar["AIFGammaFitRSquared"] == pytest.approx(0.94)
        assert sidecar["AIFGammaFitAUCRatio"] == pytest.approx(0.97)
        assert "AIFFallback" not in sidecar
        assert sidecar["AIFUsedAsCBVDenominator"] is False
        assert sidecar["AIFHematocrit"] == pytest.approx(0.42)

    def test_no_legacy_nested_block(self, aif_qc, bids_root):
        """Regression: must not emit a nested AIFDetection block."""
        sidecar = build_aif_timeseries_sidecar(
            modality="DSC",
            source_paths=[],
            bids_dir=bids_root,
            subject="X",
            session=None,
            aif_qc=aif_qc,
        )
        assert "AIFDetection" not in sidecar
        # Old PascalCase-with-units keys must not appear
        assert "PeakConcentration_mM" not in sidecar
        assert "FWHM_seconds" not in sidecar

    def test_dce_modality_units(self, aif_qc, bids_root):
        sidecar = build_aif_timeseries_sidecar(
            modality="DCE",
            source_paths=[],
            bids_dir=bids_root,
            subject="X",
            session=None,
            aif_qc=aif_qc,
        )
        assert sidecar["Units"] == "mM"
        # Same flat AIF* keys regardless of modality
        assert sidecar["AIFPeakAmplitude"] == pytest.approx(12.5)


class TestOptionalKeysOmitted:
    def test_missing_aif_qc(self, bids_root):
        sidecar = build_aif_timeseries_sidecar(
            modality="DSC",
            source_paths=[],
            bids_dir=bids_root,
            subject="X",
            session=None,
            aif_qc=None,
        )
        # No AIF* keys at all
        assert all(not k.startswith("AIF") for k in sidecar)
        # But base sidecar still has Description / Columns / Units
        assert sidecar["Description"].startswith("Arterial input function")
        assert sidecar["Units"] == "1/s"

    def test_partial_aif_qc(self, bids_root):
        sidecar = build_aif_timeseries_sidecar(
            modality="DSC",
            source_paths=[],
            bids_dir=bids_root,
            subject="X",
            session=None,
            aif_qc={"detection_method": "synthetic", "synthetic": True},
        )
        assert sidecar["AIFMethod"] == "synthetic"
        assert sidecar["AIFFallback"] is True
        # Keys not in source dict are omitted, not null
        assert "AIFFWHM" not in sidecar
        assert "AIFGammaFitApplied" not in sidecar


class TestRejectsBadModality:
    def test_invalid_modality_raises(self, bids_root):
        with pytest.raises(ValueError, match="modality must be"):
            build_aif_timeseries_sidecar(
                modality="ASL",
                source_paths=[],
                bids_dir=bids_root,
                subject="X",
                session=None,
                aif_qc=None,
            )


def test_writer_emits_base_sidecar_without_qc(bids_root, tmp_path) -> None:
    """An emitted AIF TSV always has columns, units, sources, and provenance."""
    from pmriprep.interfaces.qc_writers import DSCFiguresWriter

    source = bids_root / "sub-X" / "ses-Y" / "perf" / "sub-X_ses-Y_dsc.nii.gz"
    curve = tmp_path / "aif.txt"
    np.savetxt(curve, np.array([1.0, 2.0, 3.0]))
    writer = DSCFiguresWriter(
        output_dir=str(tmp_path / "out"),
        subject_id="X",
        session_id="Y",
        bids_dir=str(bids_root),
        source_dsc=str(source),
        aif_curve=str(curve),
    )

    result = writer.run()
    sidecar_path = Path(result.outputs.aif_curve_txt).with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())

    assert sidecar["Columns"] == ["delta_r2_star"]
    assert sidecar["Units"] == "1/s"
    assert sidecar["Sources"] == [
        "bids:raw:sub-X/ses-Y/perf/sub-X_ses-Y_dsc.nii.gz"
    ]
    assert sidecar["SamplingFrequency"] == pytest.approx(1.0 / 1.5)
    assert sidecar["StartTime"] == 0.0
    assert all(not key.startswith("AIF") for key in sidecar)


def test_aif_sidecar_sampling_metadata() -> None:
    """Sampling metadata describes rows without adding a redundant time column."""
    sidecar = build_aif_timeseries_sidecar(
        modality="DCE",
        source_paths=[],
        bids_dir=".",
        subject="X",
        session=None,
        volume_tr_s=4.0,
    )
    assert sidecar["SamplingFrequency"] == pytest.approx(0.25)
    assert sidecar["StartTime"] == 0.0


class TestColumnHeaderSingleSource:
    """The TSV column header and the sidecar ``Columns`` share one source.

    Regression guard for the bug where the AIF ``_timeseries.tsv`` header was
    hardcoded ``concentration_mM`` (wrong for DSC, which is ΔR2* in 1/s) and
    drifted from the sidecar's declared ``Columns``.
    """

    @pytest.mark.parametrize(
        "modality, name, units",
        [
            ("DSC", "delta_r2_star", "1/s"),
            ("DCE", "plasma_concentration_mM", "mM"),
        ],
    )
    def test_aif_timeseries_columns_values(self, modality, name, units):
        col_name, col_units, col_desc = aif_timeseries_columns(modality)
        assert (col_name, col_units) == (name, units)
        assert col_desc  # non-empty description
        # DSC AIF is ΔR2* (1/s) — explicitly NOT a mM concentration label
        if modality == "DSC":
            assert "mM" not in col_name and col_units == "1/s"

    def test_aif_timeseries_columns_bad_modality(self):
        with pytest.raises(ValueError, match="modality must be"):
            aif_timeseries_columns("ASL")

    @pytest.mark.parametrize("modality", ["DSC", "DCE"])
    def test_written_tsv_header_matches_sidecar(self, modality, bids_root, tmp_path):
        """The on-disk TSV header equals the sidecar ``Columns[0]``.

        Exercises the real write path (``_write_aif_timeseries_tsv``), so a
        future re-hardcode in qc_writers is caught — a helper-vs-sidecar
        comparison alone would not.
        """
        from pmriprep.interfaces.qc_writers import _write_aif_timeseries_tsv

        out = tmp_path / f"sub-X_ses-Y_desc-{modality.lower()}Aif_timeseries.tsv"
        _write_aif_timeseries_tsv(out, np.array([1.0, 2.0, 3.0]), modality)

        written_header = out.read_text().splitlines()[0]
        sidecar = build_aif_timeseries_sidecar(
            modality=modality,
            source_paths=[],
            bids_dir=bids_root,
            subject="X",
            session=None,
            aif_qc=None,
        )
        assert written_header == sidecar["Columns"][0]
