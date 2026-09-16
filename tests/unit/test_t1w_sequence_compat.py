"""Tests for T1w sequence compatibility check in discovery module."""

import json
import logging
from pathlib import Path
from typing import Optional

import pytest

from pmriprep.workflows.discovery import _check_t1w_sequence_compatibility


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
    """Tests for _check_t1w_sequence_compatibility."""

    def test_matching_acq_entities_no_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
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

        with caplog.at_level(logging.DEBUG, logger="pmriprep.workflows.discovery"):
            compatible = _check_t1w_sequence_compatibility(pre, post, _MockLayout())

        assert compatible is True
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)
        assert any("No T1w sequence mismatch" in r.message for r in caplog.records)

    def test_mismatched_acq_entities_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pre = tmp_path / "sub-01_acq-mprage_T1w.nii.gz"
        post = tmp_path / "sub-01_acq-spgr_ce-GAD_T1w.nii.gz"
        pre.touch()
        post.touch()

        with caplog.at_level(logging.DEBUG, logger="pmriprep.workflows.discovery"):
            compatible = _check_t1w_sequence_compatibility(pre, post, _MockLayout())

        assert compatible is False
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) >= 1
        assert "different pulse sequences" in warnings[0].message
        assert "mprage" in warnings[0].message
        assert "spgr" in warnings[0].message

    def test_no_acq_matching_scanning_sequence_no_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
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

        with caplog.at_level(logging.DEBUG, logger="pmriprep.workflows.discovery"):
            _check_t1w_sequence_compatibility(pre, post, _MockLayout())

        assert not any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_mismatched_scanning_sequence_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
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

        with caplog.at_level(logging.DEBUG, logger="pmriprep.workflows.discovery"):
            _check_t1w_sequence_compatibility(pre, post, _MockLayout())

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) >= 1
        assert "ScanningSequence" in warnings[0].message

    def test_inversion_time_present_absent_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
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

        with caplog.at_level(logging.DEBUG, logger="pmriprep.workflows.discovery"):
            _check_t1w_sequence_compatibility(pre, post, _MockLayout())

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) >= 1
        assert "inversion recovery" in warnings[0].message.lower()

    def test_ti_difference_over_50ms_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
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

        with caplog.at_level(logging.DEBUG, logger="pmriprep.workflows.discovery"):
            _check_t1w_sequence_compatibility(pre, post, _MockLayout())

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) >= 1
        assert "InversionTime" in warnings[0].message

    def test_tr_ratio_over_1_5x_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
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

        with caplog.at_level(logging.DEBUG, logger="pmriprep.workflows.discovery"):
            _check_t1w_sequence_compatibility(pre, post, _MockLayout())

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) >= 1
        assert "RepetitionTime" in warnings[0].message

    def test_flip_angle_difference_over_3_deg_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
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

        with caplog.at_level(logging.DEBUG, logger="pmriprep.workflows.discovery"):
            _check_t1w_sequence_compatibility(pre, post, _MockLayout())

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) >= 1
        assert "FlipAngle" in warnings[0].message

    def test_mracquisitiontype_mismatch_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
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

        with caplog.at_level(logging.DEBUG, logger="pmriprep.workflows.discovery"):
            _check_t1w_sequence_compatibility(pre, post, _MockLayout())

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) >= 1
        assert "Slice profile" in warnings[0].message

    def test_missing_sidecars_no_crash(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pre = tmp_path / "sub-01_T1w.nii.gz"
        post = tmp_path / "sub-01_ce-GAD_T1w.nii.gz"
        pre.touch()
        post.touch()

        with caplog.at_level(logging.DEBUG, logger="pmriprep.workflows.discovery"):
            _check_t1w_sequence_compatibility(pre, post, _MockLayout())

        assert not any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_list_typed_metadata_philips(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
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

        with caplog.at_level(logging.DEBUG, logger="pmriprep.workflows.discovery"):
            _check_t1w_sequence_compatibility(pre, post, _MockLayout())

        assert not any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_malformed_metadata_no_crash(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
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

        with caplog.at_level(logging.DEBUG, logger="pmriprep.workflows.discovery"):
            _check_t1w_sequence_compatibility(pre, post, _MockLayout())

    def test_tier2_suppressed_when_tier1_fires(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Tier 2 parameter warnings should not fire when Tier 1 already warned."""
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

        with caplog.at_level(logging.DEBUG, logger="pmriprep.workflows.discovery"):
            _check_t1w_sequence_compatibility(pre, post, _MockLayout())

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "different pulse sequences" in warnings[0].message
        assert "RepetitionTime" not in warnings[0].message

    def test_layout_raises_no_crash(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pre = tmp_path / "sub-01_acq-mprage_T1w.nii.gz"
        post = tmp_path / "sub-01_acq-mprage_ce-GAD_T1w.nii.gz"
        pre.touch()
        post.touch()

        with caplog.at_level(logging.DEBUG, logger="pmriprep.workflows.discovery"):
            _check_t1w_sequence_compatibility(pre, post, _RaisingLayout())

        assert not any(r.levelno >= logging.ERROR for r in caplog.records)
