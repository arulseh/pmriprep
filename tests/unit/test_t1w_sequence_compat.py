"""Tests for T1w sequence compatibility check in discovery module."""

import json
from pathlib import Path
from typing import Optional

from pmriprep.workflows.discovery import _get_t1w_sequence_mismatches


class _MockLayout:
    """Minimal stand-in: reads per-file JSON sidecar and parses BIDS entities."""

    def get_metadata(self, path: str) -> Optional[dict]:
        sidecar = Path(path).with_name(
            Path(path).name.replace(".nii.gz", ".json").replace(".nii", ".json")
        )
        if not sidecar.is_file():
            return None
        try:
            return json.loads(sidecar.read_text())
        except json.JSONDecodeError:
            return None

    def parse_file_entities(self, path: str) -> dict:
        import re

        entities: dict = {}
        fname = Path(path).name
        m = re.search(r"_acq-([a-zA-Z0-9]+)", fname)
        if m:
            entities["acquisition"] = m.group(1)
        return entities


class _RaisingLayout(_MockLayout):
    """Layout that raises on metadata access."""

    def get_metadata(self, path: str) -> Optional[dict]:
        raise RuntimeError("simulated metadata failure")

    def parse_file_entities(self, path: str) -> dict:
        raise RuntimeError("simulated parse failure")


class TestT1wSequenceCompatibility:
    """Tests for _get_t1w_sequence_mismatches."""

    def test_matching_acq_entities_matches(
        self, tmp_path: Path
    ) -> None:
        pre = tmp_path / "sub-01_acq-mprage_T1w.nii.gz"
        post = tmp_path / "sub-01_acq-mprage_ce-GAD_T1w.nii.gz"
        pre.touch()
        post.touch()
        (tmp_path / "sub-01_acq-mprage_T1w.json").write_text(
            json.dumps({"ScanningSequence": ["GR", "IR"], "MRAcquisitionType": "3D"})
        )
        (tmp_path / "sub-01_acq-mprage_ce-GAD_T1w.json").write_text(
            json.dumps({"ScanningSequence": ["GR", "IR"], "MRAcquisitionType": "3D"})
        )

        mismatches = _get_t1w_sequence_mismatches(pre, post, _MockLayout())

        assert mismatches == ()

    def test_mismatched_acq_entities_records_mismatch(
        self, tmp_path: Path
    ) -> None:
        pre = tmp_path / "sub-01_acq-mprage_T1w.nii.gz"
        post = tmp_path / "sub-01_acq-spgr_ce-GAD_T1w.nii.gz"
        pre.touch()
        post.touch()

        mismatches = _get_t1w_sequence_mismatches(pre, post, _MockLayout())

        assert mismatches
        assert len(mismatches) >= 1
        assert "acquisition label" in mismatches[0]
        assert "mprage" in mismatches[0]
        assert "spgr" in mismatches[0]

    def test_no_acq_matching_scanning_sequence_matches(
        self, tmp_path: Path
    ) -> None:
        pre = tmp_path / "sub-01_T1w.nii.gz"
        post = tmp_path / "sub-01_ce-GAD_T1w.nii.gz"
        pre.touch()
        post.touch()
        (tmp_path / "sub-01_T1w.json").write_text(
            json.dumps({"ScanningSequence": "GR", "MRAcquisitionType": "3D"})
        )
        (tmp_path / "sub-01_ce-GAD_T1w.json").write_text(
            json.dumps({"ScanningSequence": "GR", "MRAcquisitionType": "3D"})
        )

        mismatches = _get_t1w_sequence_mismatches(pre, post, _MockLayout())

        assert mismatches == ()

    def test_mismatched_scanning_sequence_records_mismatch(
        self, tmp_path: Path
    ) -> None:
        pre = tmp_path / "sub-01_T1w.nii.gz"
        post = tmp_path / "sub-01_ce-GAD_T1w.nii.gz"
        pre.touch()
        post.touch()
        (tmp_path / "sub-01_T1w.json").write_text(
            json.dumps({"ScanningSequence": ["GR", "IR"]})
        )
        (tmp_path / "sub-01_ce-GAD_T1w.json").write_text(
            json.dumps({"ScanningSequence": "GR"})
        )

        mismatches = _get_t1w_sequence_mismatches(pre, post, _MockLayout())

        assert len(mismatches) >= 1
        assert "ScanningSequence" in mismatches[0]

    def test_inversion_time_present_absent_records_mismatch(
        self, tmp_path: Path
    ) -> None:
        pre = tmp_path / "sub-01_T1w.nii.gz"
        post = tmp_path / "sub-01_ce-GAD_T1w.nii.gz"
        pre.touch()
        post.touch()
        (tmp_path / "sub-01_T1w.json").write_text(
            json.dumps({"ScanningSequence": "GR", "InversionTime": 0.9})
        )
        (tmp_path / "sub-01_ce-GAD_T1w.json").write_text(
            json.dumps({"ScanningSequence": "GR"})
        )

        mismatches = _get_t1w_sequence_mismatches(pre, post, _MockLayout())

        assert len(mismatches) >= 1
        assert "InversionTime metadata" in mismatches[0]

    def test_ti_difference_over_50ms_records_mismatch(
        self, tmp_path: Path
    ) -> None:
        pre = tmp_path / "sub-01_T1w.nii.gz"
        post = tmp_path / "sub-01_ce-GAD_T1w.nii.gz"
        pre.touch()
        post.touch()
        (tmp_path / "sub-01_T1w.json").write_text(
            json.dumps({"ScanningSequence": "GR", "InversionTime": 0.9})
        )
        (tmp_path / "sub-01_ce-GAD_T1w.json").write_text(
            json.dumps({"ScanningSequence": "GR", "InversionTime": 1.1})
        )

        mismatches = _get_t1w_sequence_mismatches(pre, post, _MockLayout())

        assert len(mismatches) >= 1
        assert "InversionTime" in mismatches[0]

    def test_tr_ratio_over_1_5x_records_mismatch(
        self, tmp_path: Path
    ) -> None:
        pre = tmp_path / "sub-01_T1w.nii.gz"
        post = tmp_path / "sub-01_ce-GAD_T1w.nii.gz"
        pre.touch()
        post.touch()
        (tmp_path / "sub-01_T1w.json").write_text(
            json.dumps({"ScanningSequence": "GR", "RepetitionTime": 2.3})
        )
        (tmp_path / "sub-01_ce-GAD_T1w.json").write_text(
            json.dumps({"ScanningSequence": "GR", "RepetitionTime": 0.008})
        )

        mismatches = _get_t1w_sequence_mismatches(pre, post, _MockLayout())

        assert len(mismatches) >= 1
        assert "RepetitionTime" in mismatches[0]

    def test_flip_angle_difference_over_3_deg_records_mismatch(
        self, tmp_path: Path
    ) -> None:
        pre = tmp_path / "sub-01_T1w.nii.gz"
        post = tmp_path / "sub-01_ce-GAD_T1w.nii.gz"
        pre.touch()
        post.touch()
        (tmp_path / "sub-01_T1w.json").write_text(
            json.dumps({"ScanningSequence": "GR", "FlipAngle": 9})
        )
        (tmp_path / "sub-01_ce-GAD_T1w.json").write_text(
            json.dumps({"ScanningSequence": "GR", "FlipAngle": 15})
        )

        mismatches = _get_t1w_sequence_mismatches(pre, post, _MockLayout())

        assert len(mismatches) >= 1
        assert "FlipAngle" in mismatches[0]

    def test_mracquisitiontype_mismatch_records_mismatch(
        self, tmp_path: Path
    ) -> None:
        pre = tmp_path / "sub-01_T1w.nii.gz"
        post = tmp_path / "sub-01_ce-GAD_T1w.nii.gz"
        pre.touch()
        post.touch()
        (tmp_path / "sub-01_T1w.json").write_text(
            json.dumps({"ScanningSequence": "GR", "MRAcquisitionType": "3D"})
        )
        (tmp_path / "sub-01_ce-GAD_T1w.json").write_text(
            json.dumps({"ScanningSequence": "GR", "MRAcquisitionType": "2D"})
        )

        mismatches = _get_t1w_sequence_mismatches(pre, post, _MockLayout())

        assert len(mismatches) >= 1
        assert "MRAcquisitionType" in mismatches[0]

    def test_missing_sidecars_no_crash(
        self, tmp_path: Path
    ) -> None:
        pre = tmp_path / "sub-01_T1w.nii.gz"
        post = tmp_path / "sub-01_ce-GAD_T1w.nii.gz"
        pre.touch()
        post.touch()

        mismatches = _get_t1w_sequence_mismatches(pre, post, _MockLayout())

        assert mismatches == ()

    def test_list_typed_metadata_philips(
        self, tmp_path: Path
    ) -> None:
        pre = tmp_path / "sub-01_T1w.nii.gz"
        post = tmp_path / "sub-01_ce-GAD_T1w.nii.gz"
        pre.touch()
        post.touch()
        (tmp_path / "sub-01_T1w.json").write_text(
            json.dumps({"ScanningSequence": "GR", "InversionTime": [0.9]})
        )
        (tmp_path / "sub-01_ce-GAD_T1w.json").write_text(
            json.dumps({"ScanningSequence": "GR", "InversionTime": [0.9]})
        )

        mismatches = _get_t1w_sequence_mismatches(pre, post, _MockLayout())

        assert mismatches == ()

    def test_malformed_metadata_no_crash(
        self, tmp_path: Path
    ) -> None:
        """Non-numeric and unexpected-type metadata must not raise."""
        pre = tmp_path / "sub-01_T1w.nii.gz"
        post = tmp_path / "sub-01_ce-GAD_T1w.nii.gz"
        pre.touch()
        post.touch()
        (tmp_path / "sub-01_T1w.json").write_text(
            json.dumps({
                "ScanningSequence": [1, 2],
                "InversionTime": "N/A",
                "RepetitionTime": "unknown",
                "FlipAngle": [],
                "MRAcquisitionType": 3,
            })
        )
        (tmp_path / "sub-01_ce-GAD_T1w.json").write_text(
            json.dumps({
                "ScanningSequence": True,
                "InversionTime": {"nested": "dict"},
                "RepetitionTime": None,
                "FlipAngle": "not_a_number",
            })
        )

        mismatches = _get_t1w_sequence_mismatches(pre, post, _MockLayout())
        assert mismatches == ()

    def test_tier2_suppressed_when_tier1_fires(
        self, tmp_path: Path
    ) -> None:
        """Sequence-type mismatches take precedence over parameter differences."""
        pre = tmp_path / "sub-01_acq-mprage_T1w.nii.gz"
        post = tmp_path / "sub-01_acq-spgr_ce-GAD_T1w.nii.gz"
        pre.touch()
        post.touch()
        (tmp_path / "sub-01_acq-mprage_T1w.json").write_text(
            json.dumps({"RepetitionTime": 2.3, "FlipAngle": 9})
        )
        (tmp_path / "sub-01_acq-spgr_ce-GAD_T1w.json").write_text(
            json.dumps({"RepetitionTime": 0.008, "FlipAngle": 20})
        )

        mismatches = _get_t1w_sequence_mismatches(pre, post, _MockLayout())

        assert len(mismatches) == 1
        assert "acquisition label" in mismatches[0]
        assert "RepetitionTime" not in mismatches[0]

    def test_layout_raises_no_crash(
        self, tmp_path: Path
    ) -> None:
        pre = tmp_path / "sub-01_acq-mprage_T1w.nii.gz"
        post = tmp_path / "sub-01_acq-mprage_ce-GAD_T1w.nii.gz"
        pre.touch()
        post.touch()

        mismatches = _get_t1w_sequence_mismatches(pre, post, _RaisingLayout())

        assert mismatches == ()
