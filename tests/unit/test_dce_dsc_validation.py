"""Tests for descriptive DSC-DCE vascular concordance."""

import numpy as np
import pytest

from pmriprep.utils.dce_dsc_validation import (
    MIN_VALID_VOXELS,
    build_vp_cbv_validation_mask,
    validate_vp_cbv,
)


def _make_data(n: int = 5000, seed: int = 42):
    """Create correlated maps with separate NAWM and tumor supports."""
    rng = np.random.default_rng(seed)
    vascular = np.exp(0.35 * rng.standard_normal(n))
    cbv = vascular * np.exp(0.12 * rng.standard_normal(n))
    vp = 0.012 * vascular * np.exp(0.12 * rng.standard_normal(n))
    shape = (n, 1, 1)
    brain = np.ones(shape, dtype=bool)
    nawm = np.zeros(shape, dtype=bool)
    nawm[:1000] = True
    tumor = np.zeros(shape, dtype=bool)
    tumor[2000:3000] = True
    return vp.reshape(shape), cbv.reshape(shape), brain, nawm, tumor


def test_validation_mask_is_jointly_finite_positive() -> None:
    vp = np.array([0.01, 0.0, -0.1, np.nan]).reshape(4, 1, 1)
    cbv = np.array([1.0, 1.0, 1.0, 1.0]).reshape(4, 1, 1)
    mask = np.ones((4, 1, 1), dtype=bool)
    assert build_vp_cbv_validation_mask(vp, cbv, mask)[:, 0, 0].tolist() == [
        True, False, False, False,
    ]


def test_validation_mask_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        build_vp_cbv_validation_mask(
            np.ones((2, 2, 2)), np.ones((2, 2, 2)),
            np.ones((2, 2, 1), dtype=bool),
        )


def test_independent_concordance_uses_measured_nawm_normalization() -> None:
    vp, cbv, brain, nawm, tumor = _make_data()
    result = validate_vp_cbv(
        vp, cbv, brain, nawm_mask=nawm, region_mask=tumor,
    )
    assert result.status == "INDEPENDENT"
    assert result.metrics.correlation > 0.8
    assert result.metrics.pearson_correlation > 0.8
    assert result.normalization["method"] == "within_subject_nawm_median"
    assert result.normalization["dce_vp_nawm_median"] == pytest.approx(
        np.median(vp[nawm])
    )
    assert result.region_metrics is not None
    assert result.region_metrics.n_valid_voxels == 1000


def test_prior_use_changes_label_not_measurement() -> None:
    vp, cbv, brain, nawm, _ = _make_data()
    independent = validate_vp_cbv(vp, cbv, brain, nawm_mask=nawm)
    informed = validate_vp_cbv(
        vp, cbv, brain, nawm_mask=nawm, prior_informed=True,
        prior_relative_sd=0.6, prior_informed_fraction=0.75,
    )
    assert informed.status == "PRIOR_INFORMED"
    assert informed.metrics.correlation == pytest.approx(
        independent.metrics.correlation
    )
    assert "not independent" in informed.interpretation


def test_small_or_absent_tumor_is_omitted_without_downgrade() -> None:
    vp, cbv, brain, nawm, tumor = _make_data()
    tumor[:] = False
    tumor[: MIN_VALID_VOXELS - 1] = True
    result = validate_vp_cbv(
        vp, cbv, brain, nawm_mask=nawm, region_mask=tumor,
    )
    assert result.status == "INDEPENDENT"
    assert result.region_metrics is None


def test_missing_or_small_nawm_fails_closed() -> None:
    vp, cbv, brain, nawm, _ = _make_data()
    with pytest.raises(ValueError, match="NAWM mask is required"):
        validate_vp_cbv(vp, cbv, brain)
    nawm[:] = False
    nawm[: MIN_VALID_VOXELS - 1] = True
    result = validate_vp_cbv(vp, cbv, brain, nawm_mask=nawm)
    assert result.status == "UNAVAILABLE"
    assert "at least" in result.interpretation


def test_insufficient_whole_brain_support_is_unavailable() -> None:
    vp, cbv, brain, nawm, _ = _make_data()
    brain[:] = False
    brain[: MIN_VALID_VOXELS - 1] = True
    result = validate_vp_cbv(vp, cbv, brain, nawm_mask=nawm)
    assert result.status == "UNAVAILABLE"


class TestDscDceValidationNodeSkipReasons:
    """The workflow wrapper preserves explicit skip reasons."""

    @staticmethod
    def _call(**kwargs):
        from pmriprep.workflows.dce.base import _dsc_dce_validation_node
        return _dsc_dce_validation_node(**kwargs)

    def test_reports_no_dsc_run(self, tmp_path):
        vp_file = tmp_path / "vp.nii.gz"
        vp_file.write_bytes(b"dummy")
        result = self._call(vp_file=str(vp_file), pk_qc=None)
        assert result["status"] == "SKIPPED"
        assert result["skip_reason"] == "no_dsc_run"

    def test_reports_missing_nawm(self, tmp_path):
        vp_file = tmp_path / "vp.nii.gz"
        vp_file.write_bytes(b"dummy")
        result = self._call(
            vp_file=str(vp_file), dsc_preload_configured=True, pk_qc=None,
        )
        assert result["status"] == "SKIPPED"
        assert "NAWM" in result["reason"]

    def test_reports_tofts_has_no_vp(self):
        result = self._call(dce_model="tofts", dsc_preload_configured=True)
        assert result["status"] == "SKIPPED"
        assert result["skip_reason"] == "vp_not_defined_for_model"


def test_prior_departure_is_bounded_with_near_zero_reference() -> None:
    """Near-zero prior values must remain visible without flattening the plot."""
    from pmriprep.utils.dsc_dce_xval_plots import (
        _finite_symmetric_relative_differences,
    )

    departure = _finite_symmetric_relative_differences(
        np.array([1.0, 1.0, 2.0]),
        np.array([1e-16, 1.0, 1.0]),
    )

    np.testing.assert_allclose(departure, [2.0, 0.0, 2.0 / 3.0])
    assert np.all(np.abs(departure) <= 2.0)


def test_prior_sd_bounds_are_mapped_to_symmetric_difference() -> None:
    """Multiplicative prior limits must use the plotted difference scale."""
    from pmriprep.utils.dsc_dce_xval_plots import _symmetric_prior_sd_bounds

    lower, upper = _symmetric_prior_sd_bounds(0.6)

    assert lower == pytest.approx(-6.0 / 7.0)
    assert upper == pytest.approx(6.0 / 13.0)

    with pytest.raises(ValueError, match="0 <= relative_sd < 1"):
        _symmetric_prior_sd_bounds(2.0)
