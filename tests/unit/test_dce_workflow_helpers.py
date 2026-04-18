"""Unit tests for private workflow helpers in ``pmriprep/workflows/dce/base.py``
and the skip-DSC glob helper in ``pmriprep/workflows/base.py``.

These tests target path-dispatch invariants that are not covered by the
validation/constraint unit tests: specifically, that the Function-node
wrappers correctly route nCBV vs absolute CBV to the right downstream
consumer, and that the skip-DSC glob handles pre-v0.25.0 output layouts
without silently feeding biased data to the new pathway.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest import mock

import numpy as np
import pytest


def _write_nifti(path: Any, data: np.ndarray) -> str:
    import nibabel as nib  # type: ignore[import-untyped]

    img = nib.Nifti1Image(data.astype(np.float32), affine=np.eye(4))
    nib.save(img, str(path))
    return str(path)


# =============================================================================
# _dsc_dce_validation_node dispatch
# =============================================================================


class TestDscDceValidationNodeDispatch:
    """Path-dispatch invariants for ``_dsc_dce_validation_node``."""

    @staticmethod
    def _capture_validator():
        """Return a (mock-side-effect, captured-dict) pair for validate_vp_cbv_from_files."""
        captured: dict[str, Any] = {}

        def fake_validate(**kwargs: Any) -> Any:
            captured.update(kwargs)

            class _Result:
                def to_dict(self) -> dict[str, Any]:
                    return {"status": "PASS"}

            return _Result()

        return fake_validate, captured

    def test_missing_cbv_file_does_not_reach_validator(
        self, tmp_path: Any,
    ) -> None:
        """When only nCBV exists on disk, the node passes ``cbv_path=None``
        (never the phantom) and validation uses the nCBV path.

        This is the dispatch contract guard: the `os.path.isfile` check at
        call-time prevents a non-existent `dsc_cbv_file` string from ever
        reaching `validate_vp_cbv_from_files`, which would otherwise raise
        FileNotFoundError when resolving the abs-CBV path.
        """
        from pmriprep.workflows.dce.base import _dsc_dce_validation_node

        ncbv_file = _write_nifti(
            tmp_path / "ncbv.nii.gz", np.ones((4, 4, 4))
        )
        mask_file = _write_nifti(
            tmp_path / "mask.nii.gz",
            np.ones((4, 4, 4), dtype=np.float32),
        )
        vp_file = _write_nifti(
            tmp_path / "vp.nii.gz",
            0.02 * np.ones((4, 4, 4), dtype=np.float32),
        )
        nonexistent_cbv = str(tmp_path / "does_not_exist.nii.gz")

        fake_validate, captured = self._capture_validator()

        with mock.patch(
            "pmriprep.utils.dce_dsc_validation.validate_vp_cbv_from_files",
            side_effect=fake_validate,
        ):
            out = _dsc_dce_validation_node(
                vp_file=vp_file,
                dsc_cbv_file=nonexistent_cbv,
                dsc_ncbv_file=ncbv_file,
                brain_mask=mask_file,
            )

        assert out == {"status": "PASS"}
        assert captured["ncbv_path"] == ncbv_file
        # The key dispatch invariant: no phantom path leaks through.
        assert captured["cbv_path"] is None

    def test_both_files_present_prefers_ncbv(self, tmp_path: Any) -> None:
        """When both exist, the node passes both paths to the validator and
        the nCBV path is what drives `use_ncbv_scale` inside the validator.

        This pins the workflow-side preference in place alongside the
        validator-side semantics, so a future refactor that flips the
        order fails loudly here rather than silently on a production run.
        """
        from pmriprep.workflows.dce.base import _dsc_dce_validation_node

        ncbv_file = _write_nifti(
            tmp_path / "ncbv.nii.gz", np.ones((4, 4, 4))
        )
        cbv_file = _write_nifti(
            tmp_path / "cbv.nii.gz",
            3.0 * np.ones((4, 4, 4), dtype=np.float32),
        )
        mask_file = _write_nifti(
            tmp_path / "mask.nii.gz",
            np.ones((4, 4, 4), dtype=np.float32),
        )
        vp_file = _write_nifti(
            tmp_path / "vp.nii.gz",
            0.02 * np.ones((4, 4, 4), dtype=np.float32),
        )

        fake_validate, captured = self._capture_validator()

        with mock.patch(
            "pmriprep.utils.dce_dsc_validation.validate_vp_cbv_from_files",
            side_effect=fake_validate,
        ):
            _dsc_dce_validation_node(
                vp_file=vp_file,
                dsc_cbv_file=cbv_file,
                dsc_ncbv_file=ncbv_file,
                brain_mask=mask_file,
            )

        # Both paths are forwarded; validator dispatches on ncbv_path.
        assert captured["cbv_path"] == cbv_file
        assert captured["ncbv_path"] == ncbv_file

    def test_only_cbv_present_uses_legacy_path(self, tmp_path: Any) -> None:
        """When only absolute CBV exists, the node forwards it and leaves
        ``ncbv_path=None`` so the validator takes the legacy path."""
        from pmriprep.workflows.dce.base import _dsc_dce_validation_node

        cbv_file = _write_nifti(
            tmp_path / "cbv.nii.gz",
            3.0 * np.ones((4, 4, 4), dtype=np.float32),
        )
        mask_file = _write_nifti(
            tmp_path / "mask.nii.gz",
            np.ones((4, 4, 4), dtype=np.float32),
        )
        vp_file = _write_nifti(
            tmp_path / "vp.nii.gz",
            0.02 * np.ones((4, 4, 4), dtype=np.float32),
        )

        fake_validate, captured = self._capture_validator()

        with mock.patch(
            "pmriprep.utils.dce_dsc_validation.validate_vp_cbv_from_files",
            side_effect=fake_validate,
        ):
            _dsc_dce_validation_node(
                vp_file=vp_file,
                dsc_cbv_file=cbv_file,
                dsc_ncbv_file=None,
                brain_mask=mask_file,
            )

        assert captured["cbv_path"] == cbv_file
        assert captured["ncbv_path"] is None

    def test_no_cbv_no_ncbv_returns_skipped(self, tmp_path: Any) -> None:
        """No CBV source at all → validation is cleanly skipped."""
        from pmriprep.workflows.dce.base import _dsc_dce_validation_node

        vp_file = _write_nifti(tmp_path / "vp.nii.gz", np.zeros((3, 3, 3)))
        mask_file = _write_nifti(
            tmp_path / "mask.nii.gz",
            np.ones((3, 3, 3), dtype=np.float32),
        )

        out = _dsc_dce_validation_node(
            vp_file=vp_file,
            dsc_cbv_file=None,
            dsc_ncbv_file=None,
            brain_mask=mask_file,
        )
        assert out["skipped"] is True
        assert out["status"] == "skipped"

    def test_exception_is_logged_with_traceback(
        self, tmp_path: Any, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Validation exceptions are logged with exc_info, not silently swallowed."""
        from pmriprep.workflows.dce.base import _dsc_dce_validation_node

        vp_file = _write_nifti(tmp_path / "vp.nii.gz", np.zeros((3, 3, 3)))
        mask_file = _write_nifti(
            tmp_path / "mask.nii.gz",
            np.ones((3, 3, 3), dtype=np.float32),
        )
        ncbv_file = _write_nifti(
            tmp_path / "ncbv.nii.gz", np.ones((3, 3, 3))
        )

        def boom(**_: Any) -> Any:
            raise RuntimeError("simulated validation failure")

        with mock.patch(
            "pmriprep.utils.dce_dsc_validation.validate_vp_cbv_from_files",
            side_effect=boom,
        ):
            with caplog.at_level(logging.WARNING, logger="pmriprep.workflows.dce"):
                out = _dsc_dce_validation_node(
                    vp_file=vp_file,
                    dsc_cbv_file=None,
                    dsc_ncbv_file=ncbv_file,
                    brain_mask=mask_file,
                )

        assert out["status"] == "error"
        assert "simulated validation failure" in out["reason"]
        assert any(
            "simulated validation failure" in r.message for r in caplog.records
        )


# =============================================================================
# _find_previous_dsc_outputs — skip-DSC glob for nCBV
# =============================================================================


class TestFindPreviousDscOutputsNcbv:
    """Skip-DSC path must discover the nCBV file, and warn if missing."""

    @staticmethod
    def _make_layout(tmp_path: Any, subject_id: str, session_id: str):
        perf_dir = (
            tmp_path
            / "pmriprep"
            / f"sub-{subject_id}"
            / f"ses-{session_id}"
            / "perf"
        )
        perf_dir.mkdir(parents=True)
        return perf_dir

    def test_ncbv_glob_discovers_file(self, tmp_path: Any) -> None:
        """Presence of ``*desc-dscNawmNorm_cbv.nii.gz`` populates ``ncbv_map``."""
        from pmriprep.workflows.base import _find_previous_dsc_outputs

        perf_dir = self._make_layout(tmp_path, "01", "01")
        ncbv_path = perf_dir / "sub-01_ses-01_desc-dscNawmNorm_cbv.nii.gz"
        _write_nifti(ncbv_path, np.ones((2, 2, 2)))
        _write_nifti(
            perf_dir / "sub-01_ses-01_desc-dscRelAUC_cbv.nii.gz",
            np.ones((2, 2, 2)),
        )

        outputs = _find_previous_dsc_outputs(
            output_dir=tmp_path,
            subject_id="01",
            session_id="01",
        )
        assert outputs.get("ncbv_map") == str(ncbv_path)

    def test_ncbv_missing_warns_and_sets_none(
        self, tmp_path: Any, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Pre-v0.25.0 layout (no dscNawmNorm_cbv) → ``ncbv_map is None`` and a warning fires."""
        from pmriprep.workflows.base import _find_previous_dsc_outputs

        perf_dir = self._make_layout(tmp_path, "02", "01")
        _write_nifti(
            perf_dir / "sub-02_ses-01_desc-dscRelAUC_cbv.nii.gz",
            np.ones((2, 2, 2)),
        )

        with caplog.at_level(logging.WARNING, logger="pmriprep.workflows"):
            outputs = _find_previous_dsc_outputs(
                output_dir=tmp_path,
                subject_id="02",
                session_id="01",
            )

        assert outputs.get("ncbv_map") is None
        joined = " ".join(r.message for r in caplog.records)
        assert ("NAWM" in joined) or ("nCBV" in joined) or ("ncbv" in joined)
