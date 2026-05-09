"""TumorSegResumeWriter round-trip + loader integration.

Verifies the writer emits ``anat/<prefix>_desc-tumorResume_pmriprep.json``
carrying the producer-shape ``qc_metrics`` envelope, and that
``_load_pipeline_sidecars`` reads the envelope back into the nested
shape the HTML aggregator consumes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pmriprep.interfaces.qc_writers import TumorSegResumeWriter
from pmriprep.utils.report_utils import _load_pipeline_sidecars
from pmriprep.utils.sidecar_builders import build_tumor_resume_sidecar


@pytest.fixture
def representative_tumor_qc() -> dict[str, Any]:
    return {
        "preprocessing": {
            "t1w_shape": [256, 256, 192],
            "flair_shape": [256, 256, 192],
        },
        "segmentation": {
            "validation": {
                "enhancing_voxels": 4012,
                "non_enhancing_voxels": 15401,
                "total_voxels": 19413,
                "enhancing_volume_ml": 4.012,
                "non_enhancing_volume_ml": 15.401,
                "total_volume_ml": 19.413,
            },
        },
        "nawm": {
            "selected_hemisphere": "right",
            "n_voxels": 67301,
            "volume_ml": 67.301,
            "valid": True,
            "atlas_used": True,
            "qc_steps": [
                {"step": "atlas_select", "n_voxels": 70010},
                {"step": "erode", "n_voxels": 67301},
            ],
        },
    }


class TestBuildTumorResumeSidecar:
    def test_envelope_round_trip(self, representative_tumor_qc):
        sidecar = build_tumor_resume_sidecar(qc_metrics=representative_tumor_qc)
        assert sidecar["qc_metrics"] == representative_tumor_qc
        assert sidecar["PMRIPrepInternal"] is True
        assert "Description" in sidecar

    def test_omits_when_empty(self):
        sidecar = build_tumor_resume_sidecar(qc_metrics=None)
        assert "qc_metrics" not in sidecar
        assert sidecar["PMRIPrepInternal"] is True


class TestTumorSegResumeWriter:
    def test_writer_lands_at_anat_path(self, tmp_path, representative_tumor_qc):
        writer = TumorSegResumeWriter()
        writer.inputs.output_dir = str(tmp_path)
        writer.inputs.subject_id = "X"
        writer.inputs.session_id = "Y"
        writer.inputs.qc_metrics = representative_tumor_qc

        result = writer.run()
        out_path = Path(result.outputs.out_file)
        expected = (
            tmp_path / "pmriprep" / "sub-X" / "ses-Y" / "anat"
            / "sub-X_ses-Y_desc-tumorResume_pmriprep.json"
        )
        assert out_path == expected
        assert out_path.exists()

        sidecar = json.loads(out_path.read_text())
        assert sidecar["qc_metrics"]["nawm"]["n_voxels"] == 67301
        assert sidecar["qc_metrics"]["nawm"]["volume_ml"] == pytest.approx(67.301)
        assert sidecar["qc_metrics"]["segmentation"]["validation"][
            "enhancing_voxels"
        ] == 4012

    def test_writer_omits_session_segment_when_unset(
        self, tmp_path, representative_tumor_qc,
    ):
        writer = TumorSegResumeWriter()
        writer.inputs.output_dir = str(tmp_path)
        writer.inputs.subject_id = "X"
        writer.inputs.qc_metrics = representative_tumor_qc

        result = writer.run()
        out_path = Path(result.outputs.out_file)
        # session=""  → no ses- segment in path or filename
        expected = (
            tmp_path / "pmriprep" / "sub-X" / "anat"
            / "sub-X_desc-tumorResume_pmriprep.json"
        )
        assert out_path == expected


class TestTumorSegResumeLoaderIntegration:
    def test_loader_reads_envelope(self, tmp_path, representative_tumor_qc):
        writer = TumorSegResumeWriter()
        writer.inputs.output_dir = str(tmp_path)
        writer.inputs.subject_id = "X"
        writer.inputs.session_id = "Y"
        writer.inputs.qc_metrics = representative_tumor_qc
        writer.run()

        ses_dir = tmp_path / "pmriprep" / "sub-X" / "ses-Y"
        # Stub the presence-probe anchor so _load_pipeline_sidecars picks
        # up the resume sidecar (anat/dseg.json sidecar must exist).
        anat_dir = ses_dir / "anat"
        (anat_dir / "sub-X_ses-Y_space-T1w_desc-tumorseg_dseg.json").write_text(
            json.dumps({}),
        )

        qc = _load_pipeline_sidecars(ses_dir, "sub-X_ses-Y", "tumor_seg")
        assert qc["nawm"]["n_voxels"] == 67301
        assert qc["nawm"]["volume_ml"] == pytest.approx(67.301)
        assert qc["nawm"]["valid"] is True
        assert qc["segmentation"]["validation"]["enhancing_voxels"] == 4012
