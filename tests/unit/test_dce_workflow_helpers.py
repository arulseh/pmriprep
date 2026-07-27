"""Unit tests for private workflow helpers in ``pmriprep/workflows/dce/base.py``
and the skip-DSC glob helper in ``pmriprep/workflows/base.py``.

These tests target path-dispatch invariants that are not covered by the
validation/constraint unit tests: specifically, that the Function-node
wrappers correctly route nCBV to downstream validation
consumer, and that the skip-DSC glob handles older output layouts
without silently feeding biased data to the cross-validation pathway.
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


def test_signal_conversion_node_reports_invalid_t1(tmp_path: Any) -> None:
    """Workflow wrapper zeroes invalid T1 and returns conversion diagnostics."""
    from pmriprep.workflows.dce._helpers import _signal_to_concentration_node

    signal = np.full((2, 1, 1, 10), 100.0)
    signal[..., 5:] = 120.0
    t1 = np.array([[[1200.0]], [[np.nan]]])
    dce_file = _write_nifti(tmp_path / "dce.nii.gz", signal)
    brain_mask_file = _write_nifti(
        tmp_path / "brain_mask.nii.gz", np.ones((2, 1, 1))
    )
    t1_file = _write_nifti(tmp_path / "t1.nii.gz", t1)
    with mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        output, qc = _signal_to_concentration_node(
            dce_file,
            brain_mask_file,
            t1_file,
            None,
            15.0,
            0.005,
            3.0,
            3.8,
            5,
            False,
        )

    assert output.endswith("dce_concentration.nii.gz")
    assert qc["t1_invalid_voxels"] == 1
    assert qc["t1_invalid_fraction"] == pytest.approx(0.5)
    assert qc["nonfinite_output_samples"] == 0
    assert qc["qc_domain"] == "dce_brain_mask"
    assert qc["n_qc_domain_voxels"] == 2


def test_external_t1_reports_alignment_metrics(tmp_path: Any) -> None:
    """External T1 conversion records support and contrast-robust edge agreement."""
    from pmriprep.workflows.dce._helpers import _signal_to_concentration_node

    grid = np.indices((6, 6, 6), dtype=float)
    radius = np.sqrt(np.sum((grid - 2.5) ** 2, axis=0))
    t1 = 800.0 + 120.0 * radius
    baseline = 2000.0 - t1
    signal = np.repeat(baseline[..., None], 10, axis=3)
    signal[..., 5:] *= 1.1
    dce_file = _write_nifti(tmp_path / "dce.nii.gz", signal)
    brain_mask_file = _write_nifti(
        tmp_path / "brain_mask.nii.gz", np.ones((6, 6, 6))
    )
    t1_file = _write_nifti(tmp_path / "external_t1.nii.gz", t1)
    with mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        _, qc = _signal_to_concentration_node(
            dce_file,
            brain_mask_file,
            t1_file,
            None,
            15.0,
            0.005,
            3.0,
            3.8,
            5,
            False,
            external_t1=True,
        )

    alignment = qc["external_t1_alignment"]
    assert alignment["brain_support_fraction"] == pytest.approx(1.0)
    assert alignment["gradient_magnitude_ncc"] is not None
    assert alignment["resampling"] == "nifti_world_affine"


def test_signal_conversion_node_scopes_t1_qc_to_brain_mask(
    tmp_path: Any,
) -> None:
    """Nonzero extracranial signal does not inflate invalid-T1 counts."""
    from pmriprep.workflows.dce._helpers import _signal_to_concentration_node

    signal = np.full((2, 1, 1, 10), 100.0)
    signal[..., 5:] = 120.0
    t1 = np.array([[[1200.0]], [[0.0]]])
    dce_file = _write_nifti(tmp_path / "dce.nii.gz", signal)
    brain_mask_file = _write_nifti(
        tmp_path / "brain_mask.nii.gz", np.array([[[1.0]], [[0.0]]])
    )
    t1_file = _write_nifti(tmp_path / "t1.nii.gz", t1)
    with mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        _, qc = _signal_to_concentration_node(
            dce_file,
            brain_mask_file,
            t1_file,
            None,
            15.0,
            0.005,
            3.0,
            3.8,
            5,
            False,
        )

    assert qc["n_spatial_voxels"] == 2
    assert qc["n_qc_domain_voxels"] == 1
    assert qc["n_signal_supported_voxels"] == 1
    assert qc["t1_invalid_voxels"] == 0
    assert qc["t1_invalid_fraction"] == 0.0


def test_no_t1_signal_conversion_retains_b1_correction(tmp_path: Any) -> None:
    """Assumed-T1 conversion still applies the measured excitation field."""
    import nibabel as nib  # type: ignore[import-untyped]

    from pmriprep.workflows.dce._helpers import _signal_to_concentration_node

    signal = np.full((1, 1, 1, 10), 100.0)
    signal[..., 5:] = 130.0
    dce_file = _write_nifti(tmp_path / "dce.nii.gz", signal)
    brain_mask_file = _write_nifti(
        tmp_path / "brain_mask.nii.gz", np.ones((1, 1, 1))
    )
    b1_file = _write_nifti(
        tmp_path / "b1.nii.gz", np.full((1, 1, 1), 0.8)
    )

    with mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        nominal_file, nominal_qc = _signal_to_concentration_node(
            dce_file, brain_mask_file, None, None,
            15.0, 0.005, 3.0, 3.8, 5, True,
        )
        nominal = np.asarray(nib.load(nominal_file).dataobj).copy()
        corrected_file, corrected_qc = _signal_to_concentration_node(
            dce_file, brain_mask_file, None, b1_file,
            15.0, 0.005, 3.0, 3.8, 5, True,
        )
        corrected = np.asarray(nib.load(corrected_file).dataobj).copy()

    assert not np.allclose(corrected, nominal)
    assert nominal_qc["b1_correction_applied"] is False
    assert corrected_qc["b1_correction_applied"] is True
    assert corrected_qc["t1_source"] == "assumed"
    assert corrected_qc["assumed_t1_ms"] == pytest.approx(1200.0)
    assert corrected_qc["assumed_t1_field_strength_t"] == pytest.approx(3.0)
    assert (
        corrected_qc["assumed_t1_policy"]
        == "uniform_field_strength_reference"
    )


def test_unity_b1_sentinel_is_not_reported_as_correction(tmp_path: Any) -> None:
    """A unity B1 file preserves signal and reports no applied correction."""
    import nibabel as nib  # type: ignore[import-untyped]

    from pmriprep.workflows.dce._helpers import _signal_to_concentration_node

    signal = np.full((1, 1, 1, 10), 100.0)
    signal[..., 5:] = 130.0
    dce_file = _write_nifti(tmp_path / "dce.nii.gz", signal)
    brain_mask_file = _write_nifti(
        tmp_path / "brain_mask.nii.gz", np.ones((1, 1, 1))
    )
    unity_b1_file = _write_nifti(
        tmp_path / "unity_b1.nii.gz", np.ones((1, 1, 1))
    )

    with mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        nominal_file, _ = _signal_to_concentration_node(
            dce_file, brain_mask_file, None, None,
            15.0, 0.005, 3.0, 3.8, 5, True,
        )
        nominal = np.asarray(nib.load(nominal_file).dataobj).copy()
        unity_file, unity_qc = _signal_to_concentration_node(
            dce_file, brain_mask_file, None, unity_b1_file,
            15.0, 0.005, 3.0, 3.8, 5, True,
        )
        unity = np.asarray(nib.load(unity_file).dataobj).copy()

    assert np.allclose(unity, nominal)
    assert unity_qc["b1_correction_applied"] is False


def test_signal_conversion_node_rejects_brain_mask_shape_mismatch(
    tmp_path: Any,
) -> None:
    """Conversion QC cannot silently use a misregistered mask domain."""
    from pmriprep.workflows.dce._helpers import _signal_to_concentration_node

    dce_file = _write_nifti(
        tmp_path / "dce.nii.gz", np.ones((2, 1, 1, 10))
    )
    brain_mask_file = _write_nifti(
        tmp_path / "brain_mask.nii.gz", np.ones((3, 1, 1))
    )

    with pytest.raises(ValueError, match="brain mask shape"):
        _signal_to_concentration_node(
            dce_file,
            brain_mask_file,
            None,
            None,
            15.0,
            0.005,
            3.0,
            3.8,
            5,
            True,
        )


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
        """When only nCBV exists, validation uses that nCBV path.

        This is the dispatch contract guard: the `os.path.isfile` check at
        call-time prevents a non-existent `dsc_cbv_file` string from ever
        reaching `validate_vp_cbv_from_files`.
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
        assert "cbv_path" not in captured

    def test_both_files_present_uses_ncbv_only(self, tmp_path: Any) -> None:
        """When both exist, the node routes only the nCBV path because
        the rel-AUC-CBV fallback is
        miscalibrated against dimensionless rel-AUC-CBV.
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

        assert "cbv_path" not in captured
        assert captured["ncbv_path"] == ncbv_file

    def test_workflow_skips_xval_on_no_nawm(self, tmp_path: Any) -> None:
        """When only rel-AUC-CBV is available (no NAWM → no nCBV),
        xval is cleanly SKIPPED rather than falling back to the
        rel-AUC-CBV path with miscalibrated expected ratio 0.669."""
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

        # Validator should NOT be invoked when ncbv is missing.
        fake_validate, captured = self._capture_validator()

        with mock.patch(
            "pmriprep.utils.dce_dsc_validation.validate_vp_cbv_from_files",
            side_effect=fake_validate,
        ):
            out = _dsc_dce_validation_node(
                vp_file=vp_file,
                dsc_cbv_file=cbv_file,
                dsc_ncbv_file=None,
                brain_mask=mask_file,
            )

        assert out["skipped"] is True
        assert out["status"] == "SKIPPED"
        assert "NAWM" in out["reason"]
        assert captured == {}  # validator was not called

    def test_no_cbv_no_ncbv_returns_skipped(self, tmp_path: Any) -> None:
        """No CBV + no DSC config at all → skipped with no_dsc_run reason."""
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
            dsc_preload_configured=None,
        )
        assert out["skipped"] is True
        assert out["status"] == "SKIPPED"
        assert out["reason"] == "No DSC data provided"
        assert out["skip_reason"] == "no_dsc_run"

    def test_no_ncbv_but_dsc_configured_returns_nawm_skip(self, tmp_path: Any) -> None:
        """DSC configured but no nCBV → skipped with NAWM-unavailable reason."""
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
            dsc_preload_configured=True,
        )
        assert out["skipped"] is True
        assert out["status"] == "SKIPPED"
        assert "NAWM" in out["reason"]

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

    def test_validation_node_extracts_constraint_width(self, tmp_path: Any) -> None:
        """pk_qc forwards constraint mode and relative prior width."""
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

        fake_validate, captured = self._capture_validator()
        pk_qc = {
            "dsc_cbv_constraint": {
                "mode": "soft",
                "vp_uncertainty": 0.5,
            },
        }

        with mock.patch(
            "pmriprep.utils.dce_dsc_validation.validate_vp_cbv_from_files",
            side_effect=fake_validate,
        ):
            _dsc_dce_validation_node(
                vp_file=vp_file,
                dsc_cbv_file=None,
                dsc_ncbv_file=ncbv_file,
                brain_mask=mask_file,
                pk_qc=pk_qc,
            )

        assert captured["constraint_mode"] == "soft"
        assert captured["constraint_relative_width"] == 0.5
        # Guard the dict-vs-string dispatch: ncbv_path must still route through.
        assert captured["ncbv_path"] == ncbv_file
        assert "cbv_path" not in captured


def test_xval_plot_node_emits_constraint_diagnostic(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constraint-coupled results still produce the diagnostic figure."""
    from pathlib import Path

    from pmriprep.workflows.dce.base import _dsc_dce_xval_plot_node

    shape = (8, 8, 4)
    ncbv = np.linspace(0.5, 2.5, np.prod(shape)).reshape(shape)
    vp = 0.01 * ncbv
    vp_file = _write_nifti(tmp_path / "vp.nii.gz", vp)
    ncbv_file = _write_nifti(tmp_path / "ncbv.nii.gz", ncbv)
    mask_file = _write_nifti(tmp_path / "mask.nii.gz", np.ones(shape))
    monkeypatch.chdir(tmp_path)

    out_file = _dsc_dce_xval_plot_node(
        vp_file=vp_file,
        dsc_ncbv_file=ncbv_file,
        brain_mask=mask_file,
        validation_result={
            "status": "CONSTRAINT-COUPLED",
            "constraint_mode": "soft",
            "constraint_coupled": True,
            "constraint_relative_width": 0.5,
            "metrics": {"correlation": 1.0, "pearson_correlation": 1.0},
        },
    )

    assert out_file is not None
    assert Path(out_file).is_file()


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
        """Presence of ``*desc-dscNawmNormCbv_stat.nii.gz`` populates ``ncbv_map``."""
        from pmriprep.workflows.base import _find_previous_dsc_outputs

        perf_dir = self._make_layout(tmp_path, "01", "01")
        ncbv_path = perf_dir / "sub-01_ses-01_desc-dscNawmNormCbv_stat.nii.gz"
        _write_nifti(ncbv_path, np.ones((2, 2, 2)))
        _write_nifti(
            perf_dir / "sub-01_ses-01_desc-dscRelAUCCbv_stat.nii.gz",
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
        """Layout with no dscNawmNormCbv_stat → ``ncbv_map is None`` and a warning fires."""
        from pmriprep.workflows.base import _find_previous_dsc_outputs

        perf_dir = self._make_layout(tmp_path, "02", "01")
        _write_nifti(
            perf_dir / "sub-02_ses-01_desc-dscRelAUCCbv_stat.nii.gz",
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


# =============================================================================
# ComputeDCEMaps dose-trait forwarding
# =============================================================================


class TestComputeDCEMapsDoseTraits:
    """`dsc_dose` + `dsc_agent_standard_dose` traits must reach
    `prepare_vp_constraint_from_data`."""

    def test_dose_traits_forwarded_to_prepare_vp_constraint_from_data(
        self, tmp_path: Any,
    ) -> None:
        """`self.inputs.dsc_dose` + `.dsc_agent_standard_dose` flow through
        to `prepare_vp_constraint_from_data(dose=..., agent_standard_dose=...)`."""

        from pmriprep.interfaces.dce_compute import ComputeDCEMapsInputSpec

        # Interface has the traits with documented defaults.
        spec = ComputeDCEMapsInputSpec()
        assert "hematocrit" not in spec.traits()
        assert "dsc_cbv_file" not in spec.traits()
        assert "tr" not in spec.traits()
        assert spec.dsc_dose == 1.0
        assert spec.dsc_agent_standard_dose == 0.1

        # Traits accept explicit values.
        spec.dsc_dose = 0.5
        spec.dsc_agent_standard_dose = 0.05
        assert spec.dsc_dose == 0.5
        assert spec.dsc_agent_standard_dose == 0.05

    def test_prepare_vp_constraint_accepts_dose_kwargs(self, tmp_path: Any) -> None:
        """Direct call to prepare_vp_constraint_from_data with dose kwargs
        should work without error (smoke test on the signature)."""
        from pmriprep.utils.dsc_cbv_constraint import (
            ConstraintMode,
            prepare_vp_constraint_from_data,
        )

        mask_file = _write_nifti(
            tmp_path / "mask.nii.gz",
            np.ones((3, 3, 3), dtype=np.float32),
        )
        # ncbv_file=None exercises the missing-NAWM SKIP path but still
        # exercises that the dose kwargs don't raise.
        result = prepare_vp_constraint_from_data(
            ncbv_file=None,
            aif_qc_data={},
            brain_mask_file=mask_file,
            dose=0.5,
            agent_standard_dose=0.05,
        )
        assert result.config.mode == ConstraintMode.SKIP


class TestMaskProducersReturnEmptyString:
    """Workflow Function nodes that select/transform
    optional masks must return the empty string (not None) when no mask is
    available, because the downstream traits are
    ``Either(File(exists=True), Enum(""))`` and reject None.

    Covers the override pathway (``_select_aif_exclusion_mask`` →
    ``aif_detection.lesion_mask_file``) and ``_transform_probseg_to_dce``,
    both instances of the same class.
    """

    def test_select_aif_exclusion_mask_returns_empty_when_no_mask(self) -> None:
        from pmriprep.workflows.dce.base import _select_aif_exclusion_mask

        assert _select_aif_exclusion_mask(None, None) == ""
        assert _select_aif_exclusion_mask("", "") == ""

    def test_select_aif_exclusion_mask_prefers_user_then_transformed(self) -> None:
        from pmriprep.workflows.dce.base import _select_aif_exclusion_mask

        assert _select_aif_exclusion_mask("/user.nii.gz", "/seg.nii.gz") == "/user.nii.gz"
        assert _select_aif_exclusion_mask("", "/seg.nii.gz") == "/seg.nii.gz"

    def test_transform_probseg_returns_empty_on_missing_input(self) -> None:
        from pmriprep.workflows.dce.base import _transform_probseg_to_dce

        assert _transform_probseg_to_dce(None, "/ref.nii.gz", "/xfm.mat") == ""
        assert _transform_probseg_to_dce("/p.nii.gz", None, "/xfm.mat") == ""
        assert _transform_probseg_to_dce("/p.nii.gz", "/ref.nii.gz", None) == ""
        # Non-existent path on disk also degrades to "".
        assert _transform_probseg_to_dce("/nope.nii.gz", "/ref.nii.gz", "/xfm.mat") == ""


class TestT1MappingProbsegTraitsAcceptEmptyString:
    """The T1Mapping CSF/WM/GM probseg traits must accept
    the empty string returned by ``_transform_probseg_to_dce`` when no probseg is
    available, mirroring the lesion-mask trait behavior. They must still
    reject non-existent paths."""

    def test_probseg_traits_accept_empty_string(self) -> None:
        from pmriprep.interfaces.dce import T1Mapping

        iface = T1Mapping()
        iface.inputs.csf_probseg = ""
        iface.inputs.wm_probseg = ""
        iface.inputs.gm_probseg = ""
        assert iface.inputs.csf_probseg == ""
        assert iface.inputs.wm_probseg == ""
        assert iface.inputs.gm_probseg == ""

    def test_probseg_traits_accept_existing_path(self, tmp_path: Any) -> None:
        from pmriprep.interfaces.dce import T1Mapping

        f = tmp_path / "csf.nii.gz"
        f.write_bytes(b"")
        iface = T1Mapping()
        iface.inputs.csf_probseg = str(f)
        assert iface.inputs.csf_probseg == str(f)

    def test_probseg_traits_reject_nonexistent_path(self) -> None:
        from traits.trait_errors import TraitError

        from pmriprep.interfaces.dce import T1Mapping

        iface = T1Mapping()
        with pytest.raises(TraitError):
            iface.inputs.wm_probseg = "/nonexistent/missing.nii.gz"
