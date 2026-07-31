"""_anat_nested_from_sidecars — round-trip of brain-extraction measurements.

Brain volume for the ``brain_extraction`` block lands on the brain-mask
sidecar as flat ``BrainExtraction*`` keys.
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
                "brain_volume_ml": 1742.445,
                "brain_volume_mm3": 1742445.0,
            },
        }
        _write_brain_mask_sidecar(ses_dir / "anat", prefix, qc)

        out = _anat_nested_from_sidecars(ses_dir, prefix, anchor={})
        be = out["brain_extraction"]
        # Tool name normalized to lowercase token (HD-BET → hdbet)
        assert be["tool"] == "hdbet"
        # Tool version retired from per-file sidecars → dataset-level GeneratedBy.
        assert "version" not in be
        assert be["brain_volume_ml"] == pytest.approx(1742.445)
        assert be["brain_volume_mm3"] == pytest.approx(1742445.0)

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

    def test_missing_volume_does_not_break(self, ses_dir):
        """The loader still returns the extraction tool without a volume."""
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
        assert "brain_volume_ml" not in be
