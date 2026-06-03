"""_anat_nested_from_sidecars — round-trip of brain-extraction QC fields.

Brain volume, qc_status, and warnings for the ``brain_extraction`` block
land on the brain-mask sidecar as flat ``BrainExtraction*`` keys.
This test verifies the writer + loader round-trip by writing a synthetic
mask sidecar and asserting the loader reconstructs the nested
shape that ``section_anat.html`` consumes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pmriprep.utils.report_utils import _anat_nested_from_sidecars
from pmriprep.utils.sidecar_builders import enrich_brain_mask_sidecar


def _write_brain_mask_sidecar(
    anat_dir: Path,
    prefix: str,
    qc_metrics: dict,
) -> None:
    sidecar = {
        "Description": "Brain mask",
    }
    enrich_brain_mask_sidecar(sidecar, qc_metrics)
    out = anat_dir / f"{prefix}_space-T1w_desc-brain_mask.json"
    out.write_text(json.dumps(sidecar, indent=2))


@pytest.fixture
def ses_dir(tmp_path) -> Path:
    anat = tmp_path / "anat"
    anat.mkdir()
    return tmp_path


class TestAnatNestedFromSidecars:
    def test_brain_extraction_full_round_trip(self, ses_dir):
        prefix = "sub-X_ses-Y"
        qc = {
            "brain_extraction": {
                "tool": "HD-BET",
                "version": "1.2.3",
                "voxel_count": 1742445,
                "volume_ml": 1742.445,
                "qc_status": "PASS",
                "warnings": [],
            },
        }
        _write_brain_mask_sidecar(ses_dir / "anat", prefix, qc)

        out = _anat_nested_from_sidecars(ses_dir, prefix, anchor={})
        be = out["brain_extraction"]
        # Tool name normalized to lowercase token (HD-BET → hdbet)
        assert be["tool"] == "hdbet"
        assert be["version"] == "1.2.3"
        assert be["voxel_count"] == 1742445
        assert be["brain_volume_ml"] == pytest.approx(1742.445)
        assert be["brain_volume_mm3"] == pytest.approx(1742445.0)
        assert be["qc_status"] == "PASS"
        # Empty warnings list intentionally omitted by the writer (per
        # the truthiness guard in enrich_brain_mask_sidecar).
        assert be.get("warnings", []) == []

    def test_brain_volume_keys_match_producer_shape(self, ses_dir):
        """The HD-BET producer emits ``brain_volume_ml`` and
        ``brain_volume_mm3`` (with ``brain_volume_`` prefix). The writer
        + loader must round-trip both, preferring the explicit mm3 key
        over reconstruction-via-multiply.
        """
        prefix = "sub-X_ses-Y"
        qc = {
            "brain_extraction": {
                "tool": "HD-BET",
                "brain_volume_ml": 1742.445,
                "brain_volume_mm3": 1742445.0,
            },
        }
        _write_brain_mask_sidecar(ses_dir / "anat", prefix, qc)

        out = _anat_nested_from_sidecars(ses_dir, prefix, anchor={})
        be = out["brain_extraction"]
        assert be["brain_volume_ml"] == pytest.approx(1742.445)
        # Preferring the explicit mm3 key keeps producer-side precision.
        assert be["brain_volume_mm3"] == pytest.approx(1742445.0)

    def test_brain_extraction_with_warnings(self, ses_dir):
        prefix = "sub-X_ses-Y"
        qc = {
            "brain_extraction": {
                "tool": "HD-BET",
                "voxel_count": 2300000,
                "volume_ml": 2300.0,
                "qc_status": "WARN",
                "warnings": [
                    "Brain volume 2300 mL above expected maximum 2000 mL",
                ],
            },
        }
        _write_brain_mask_sidecar(ses_dir / "anat", prefix, qc)

        out = _anat_nested_from_sidecars(ses_dir, prefix, anchor={})
        be = out["brain_extraction"]
        assert be["qc_status"] == "WARN"
        assert len(be["warnings"]) == 1
        assert "above expected maximum" in be["warnings"][0]

    def test_missing_qc_status_does_not_break(self, ses_dir):
        """If the producer didn't run QC validation (older outputs),
        the loader still returns a populated tool/version block.
        """
        prefix = "sub-X_ses-Y"
        qc = {
            "brain_extraction": {
                "tool": "HD-BET",
                "version": "1.2.3",
            },
        }
        _write_brain_mask_sidecar(ses_dir / "anat", prefix, qc)

        out = _anat_nested_from_sidecars(ses_dir, prefix, anchor={})
        be = out["brain_extraction"]
        assert be["tool"] == "hdbet"
        assert "qc_status" not in be
        assert "brain_volume_ml" not in be
