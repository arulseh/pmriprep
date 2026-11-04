"""Unit tests for dce_t1_validation module.

Tests uniform T1 fallback, valid-support classification,
auto-correction tiers, T1CorrectionResult serialization,
interior NaN gap-fill, and NLLS T1 fitting.
"""

from pathlib import Path

import numpy as np
import pytest

from pmriprep.utils.dce_t1_validation import (
    MIN_VALID_T1_MS,
    T1CorrectionResult,
    T1GapFillResult,
    T1_TISSUE_REFERENCE,
    apply_t1_correction,
    fill_t1_gaps,
    get_field_strength_key,
    resolve_t1_default_ms,
    valid_t1_support,
)


class TestValidT1Support:
    """Tests for the shared downstream T1-validity rule."""

    def test_rejects_nonfinite_zero_and_subthreshold_values(self) -> None:
        """QC support matches signal conversion rather than finiteness alone."""
        values = np.array([np.nan, np.inf, -1.0, 0.0, 99.9, MIN_VALID_T1_MS, 900.0])
        assert valid_t1_support(values).tolist() == [
            False, False, False, False, False, True, True,
        ]

    def test_field_strength_envelope_retains_pathology_and_rejects_extremes(self) -> None:
        """The 3T safety envelope retains prolonged tissue T1 through 6000 ms."""
        values = np.array([99.0, 100.0, 2500.0, 5000.0, 6000.0, 6000.1])
        assert valid_t1_support(values, 3.0).tolist() == [
            False, True, True, True, True, False,
        ]

    @pytest.mark.parametrize("field_strength,upper_bound", [(1.5, 5000.), (3., 6000.), (7., 7000.)])
    def test_uniform_fallback_obeys_measured_t1_limits(
        self, field_strength: float, upper_bound: float,
    ) -> None:
        """Catch seconds-as-ms and field-limit violations without excluding endpoints."""
        for value in (1.4, 99.9, upper_bound + 0.1):
            with pytest.raises(ValueError, match="milliseconds; did you enter seconds"):
                resolve_t1_default_ms(field_strength, value)
        for value in (100., 1400., upper_bound):
            assert resolve_t1_default_ms(field_strength, value) == value


class TestT1CorrectionResult:
    """Tests for T1CorrectionResult dataclass."""


    def test_to_dict_tissue_t1_true(self) -> None:
        result = T1CorrectionResult(tissue_t1_used=True, vfa_fit_nrmse=0.065)
        d = result.to_dict()
        assert d["tissue_t1_used"] is True
        assert d["vfa_fit_nrmse"] == pytest.approx(0.065)

    def test_to_dict_nan_fraction_none_excluded(self) -> None:
        """nan_fraction=None should not appear in dict."""
        result = T1CorrectionResult()
        d = result.to_dict()
        assert "nan_fraction" not in d


class TestApplyT1CorrectionAuto:
    """Tests for auto-mode T1 correction with tissue fallback."""

    @pytest.fixture
    def brain_mask(self) -> np.ndarray:
        """10x10x10 brain mask (inner 6x6x6)."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[2:8, 2:8, 2:8] = True
        return mask

    @pytest.mark.parametrize("t1_ms,with_probsegs,b1_qc_pass", [
        (400.0, True, True), (1090.0, False, True), (1600.0, False, False),
    ])
    def test_auto_preserves_measured_and_missing_support(
        self, brain_mask: np.ndarray, t1_ms: float,
        with_probsegs: bool, b1_qc_pass: bool,
    ) -> None:
        """Acquisition diagnostics retain the measured map and its missing voxels."""
        t1 = np.full(brain_mask.shape, np.nan)
        t1[brain_mask] = t1_ms
        t1[:5] = np.nan
        wm = np.where(brain_mask, 0.6, 0.0) if with_probsegs else None
        gm = np.where(brain_mask, 0.4, 0.0) if with_probsegs else None
        corrected, result = apply_t1_correction(
            t1, field_strength=3.0, correction_mode="auto", brain_mask=brain_mask,
            nan_fraction=0.5, e1_median=0.998, b1_corrected=True,
            b1_qc_pass=b1_qc_pass, wm_probseg=wm, gm_probseg=gm,
            t1_default_ms=850.,
        )
        np.testing.assert_array_equal(corrected, t1)
        assert result.correction_applied == "none"
        assert result.tissue_t1_used is False
        assert result.fallback_value_ms is None
        assert result.to_dict()["nan_fraction"] == 0.5
        if not b1_qc_pass:
            assert "B1 QC failed" in result.reason


    def test_unknown_correction_mode_raises(self, brain_mask: np.ndarray) -> None:
        """An unknown correction mode raises a clear error."""
        t1_map = np.full((10, 10, 10), 1200.0)
        t1_map[~brain_mask] = np.nan

        with pytest.raises(ValueError, match="Unknown correction mode: invalid"):
            apply_t1_correction(
                t1_map, field_strength=3.0, correction_mode="invalid",
                brain_mask=brain_mask,
            )


    def test_population_comparison_unavailable_without_csf_restriction(
        self,
        brain_mask: np.ndarray,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A whole-brain median is not compared with parenchymal norms."""
        t1_map = np.full((10, 10, 10), 2000.0)
        t1_map[~brain_mask] = np.nan

        with caplog.at_level("INFO"):
            _, result = apply_t1_correction(
                t1_map,
                field_strength=3.0,
                correction_mode="auto",
                brain_mask=brain_mask,
                population_reference_applicable=False,
            )

        assert result.population_reference_applicable is False
        assert result.deviation_type is None
        assert "comparison unavailable" in result.reason
        assert "no CSF-restricted QC mask was available" in caplog.text
        assert "numerical safety envelope" not in caplog.text

    def test_population_warning_explains_pathology_retention(
        self,
        brain_mask: np.ndarray,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A CSF-restricted deviation remains descriptive in pathology."""
        t1_map = np.full((10, 10, 10), 2000.0)
        t1_map[~brain_mask] = np.nan

        with caplog.at_level("WARNING"):
            _, result = apply_t1_correction(
                t1_map,
                field_strength=3.0,
                correction_mode="auto",
                brain_mask=brain_mask,
                population_reference_applicable=True,
            )

        assert result.population_reference_applicable is True
        assert result.deviation_type == "high"
        assert "may still include tumor or edema" in caplog.text
        assert "no population-T1 fallback was applied" in caplog.text


class TestSkipT1Paths:
    """Tests for skip_t1-equivalent behavior in apply_t1_correction."""

    def test_fallback_mode_returns_uniform(self) -> None:
        """Explicit fallback mode → uniform literature value."""
        shape = (5, 5, 5)
        mask = np.ones(shape, dtype=bool)
        t1_map = np.full(shape, 1050.0)
        t1_map[:2] = np.nan

        t1_corr, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="fallback",
            brain_mask=mask, t1_default_ms=850.,
        )
        assert result.correction_applied == "fallback"
        assert result.fallback_value_ms == 850.
        assert result.to_dict()["fallback_source"] == "configured"
        np.testing.assert_allclose(t1_corr[mask], result.fallback_value_ms)

    def test_auto_falls_back_only_when_no_measured_support_exists(self) -> None:
        """An entirely unavailable measured map uses the documented fallback."""
        shape = (5, 5, 5)
        mask = np.ones(shape, dtype=bool)
        corrected, result = apply_t1_correction(
            np.full(shape, np.nan),
            field_strength=3.0,
            correction_mode="auto",
            brain_mask=mask,
        )
        assert result.correction_applied == "fallback"
        assert result.reason == "No valid measured T1 values"
        np.testing.assert_allclose(
            corrected[mask], T1_TISSUE_REFERENCE[3.0]["default"],
        )

    def test_none_mode_returns_unchanged(self) -> None:
        """None mode → no changes."""
        shape = (5, 5, 5)
        mask = np.ones(shape, dtype=bool)
        t1_map = np.full(shape, 1050.0)

        t1_corr, result = apply_t1_correction(
            t1_map, field_strength=3.0, correction_mode="none",
            brain_mask=mask,
        )
        assert result.correction_applied == "none"
        np.testing.assert_array_equal(t1_corr, t1_map)


class TestT1GapFill:
    """Tests for interior NaN gap-fill in T1 maps."""

    @pytest.fixture
    def brain_mask_20(self) -> np.ndarray:
        """20x20x20 brain mask (inner 4:16 cube)."""
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[4:16, 4:16, 4:16] = True
        return mask

    @pytest.fixture
    def t1_map_20(self, brain_mask_20: np.ndarray) -> np.ndarray:
        """20x20x20 T1 map with valid values in brain, NaN outside."""
        t1 = np.full((20, 20, 20), np.nan, dtype=np.float32)
        t1[brain_mask_20] = 1050.0  # typical 3T brain T1
        return t1

    def test_no_gaps_returns_unchanged(
        self, t1_map_20: np.ndarray, brain_mask_20: np.ndarray,
    ) -> None:
        """No NaN gaps → returns copy, method='none'."""
        t1_out, result = fill_t1_gaps(t1_map_20, brain_mask_20, field_strength=3.0)
        assert result.method == "none"
        assert result.n_nan_before == 0
        np.testing.assert_array_equal(t1_out[brain_mask_20], t1_map_20[brain_mask_20])

    def test_scattered_nans_filled_by_interp(
        self, t1_map_20: np.ndarray, brain_mask_20: np.ndarray,
    ) -> None:
        """Scattered interior NaN voxels filled by IDW interpolation."""
        rng = np.random.default_rng(42)
        # Add spatial variation so interpolation is meaningful
        t1_map_20[brain_mask_20] = rng.normal(1050.0, 50.0, int(np.sum(brain_mask_20))).astype(np.float32)
        # Punch scattered NaN holes in interior (far from boundary)
        interior = np.zeros_like(brain_mask_20)
        interior[7:13, 7:13, 7:13] = True
        gap_coords = np.argwhere(interior)
        chosen = rng.choice(len(gap_coords), size=20, replace=False)
        for idx in chosen:
            c = tuple(gap_coords[idx])
            t1_map_20[c] = np.nan

        t1_out, result = fill_t1_gaps(
            t1_map_20, brain_mask_20, field_strength=3.0,
            voxel_sizes_mm=np.array([1.5, 1.5, 1.5]),
        )
        assert result.n_nan_before == 20
        assert result.n_filled_interp > 0
        assert result.n_remaining == 0
        assert "interp" in result.method

    def test_large_gap_stays_nan(
        self, t1_map_20: np.ndarray, brain_mask_20: np.ndarray,
    ) -> None:
        """Large NaN hole with < min_neighbors → stays NaN (no tissue fallback)."""
        # Create a big hole that can't be filled by IDW
        t1_map_20[6:14, 6:14, 6:14] = np.nan

        t1_out, result = fill_t1_gaps(
            t1_map_20, brain_mask_20, field_strength=3.0,
            voxel_sizes_mm=np.array([1.5, 1.5, 1.5]),
        )
        assert result.n_filled_tissue == 0
        assert result.n_remaining > 0
        # Interior of the large hole should still be NaN
        assert np.any(np.isnan(t1_out[brain_mask_20]))


    def test_outside_brain_mask_stays_nan(
        self, t1_map_20: np.ndarray, brain_mask_20: np.ndarray,
    ) -> None:
        """Voxels outside brain mask must remain NaN."""
        # Punch some gaps in brain
        t1_map_20[8:12, 8:12, 8:12] = np.nan

        t1_out, result = fill_t1_gaps(
            t1_map_20, brain_mask_20, field_strength=3.0,
        )
        assert np.all(np.isnan(t1_out[~brain_mask_20]))

    def test_valid_values_never_changed(
        self, t1_map_20: np.ndarray, brain_mask_20: np.ndarray,
    ) -> None:
        """Original valid T1 values must not be modified (bitwise equality)."""
        # Punch a few holes
        rng = np.random.default_rng(123)
        t1_map_20[brain_mask_20] = rng.normal(1050.0, 50.0, int(np.sum(brain_mask_20))).astype(np.float32)
        t1_map_20.copy()
        interior = np.zeros_like(brain_mask_20)
        interior[8:12, 8:12, 8:12] = True
        gap_coords = np.argwhere(interior)
        chosen = rng.choice(len(gap_coords), size=10, replace=False)
        for idx in chosen:
            c = tuple(gap_coords[idx])
            t1_map_20[c] = np.nan

        valid_before = np.isfinite(t1_map_20) & brain_mask_20
        t1_out, _ = fill_t1_gaps(
            t1_map_20, brain_mask_20, field_strength=3.0,
            voxel_sizes_mm=np.array([1.5, 1.5, 1.5]),
        )
        # Boundary smoothing only applies to filled voxels, not original valid voxels
        np.testing.assert_array_equal(
            t1_out[valid_before], t1_map_20[valid_before],
        )

    def test_metrics_sum_correctly(
        self, t1_map_20: np.ndarray, brain_mask_20: np.ndarray,
    ) -> None:
        """n_nan_before == n_filled_interp + n_filled_tissue + n_remaining."""
        t1_map_20[7:13, 7:13, 7:13] = np.nan

        _, result = fill_t1_gaps(
            t1_map_20, brain_mask_20, field_strength=3.0,
            voxel_sizes_mm=np.array([1.5, 1.5, 1.5]),
        )
        assert result.n_filled_tissue == 0
        assert result.n_nan_before == (
            result.n_filled_interp + result.n_filled_tissue + result.n_remaining
        )

    def test_output_dtype_float32(
        self, t1_map_20: np.ndarray, brain_mask_20: np.ndarray,
    ) -> None:
        """Output must be float32."""
        t1_map_20[8:12, 8:12, 8:12] = np.nan
        t1_out, _ = fill_t1_gaps(t1_map_20, brain_mask_20, field_strength=3.0)
        assert t1_out.dtype == np.float32

    @pytest.mark.parametrize("field_strength", [1.5, 3.0, 7.0])
    def test_field_strengths(
        self, brain_mask_20: np.ndarray, field_strength: float,
    ) -> None:
        """Gap-fill works at all supported field strengths (scattered gaps)."""
        t1 = np.full((20, 20, 20), np.nan, dtype=np.float32)
        expected = T1_TISSUE_REFERENCE[get_field_strength_key(field_strength)]["expected_median"]
        t1[brain_mask_20] = expected
        # Punch scattered single-voxel holes (well-surrounded by valid neighbors)
        rng = np.random.default_rng(42)
        interior = np.zeros_like(brain_mask_20)
        interior[7:13, 7:13, 7:13] = True
        gap_coords = np.argwhere(interior)
        chosen = rng.choice(len(gap_coords), size=10, replace=False)
        for idx in chosen:
            c = tuple(gap_coords[idx])
            t1[c] = np.nan

        t1_out, result = fill_t1_gaps(
            t1, brain_mask_20, field_strength=field_strength,
            voxel_sizes_mm=np.array([1.5, 1.5, 1.5]),
        )
        assert result.n_nan_before > 0
        assert result.n_remaining == 0

    def test_t1_gap_fill_result_to_dict(self) -> None:
        """T1GapFillResult.to_dict() returns expected keys."""
        result = T1GapFillResult(
            n_nan_before=100, n_filled_interp=80,
            n_filled_tissue=0, n_remaining=20, method="interp_only",
        )
        d = result.to_dict()
        assert d["n_nan_before"] == 100
        assert d["n_filled_interp"] == 80
        assert d["n_filled_tissue"] == 0
        assert d["n_remaining"] == 20
        assert d["method"] == "interp_only"

    def test_anisotropic_voxels(
        self, brain_mask_20: np.ndarray,
    ) -> None:
        """Thick-slice DCE (anisotropic voxels) uses ellipsoidal kernel."""
        t1 = np.full((20, 20, 20), np.nan, dtype=np.float32)
        t1[brain_mask_20] = 1050.0
        # Punch a single interior gap
        t1[10, 10, 10] = np.nan

        t1_out, result = fill_t1_gaps(
            t1, brain_mask_20, field_strength=3.0,
            voxel_sizes_mm=np.array([0.5, 0.5, 5.0]),
        )
        assert result.n_nan_before == 1
        assert result.n_filled_interp == 1
        assert np.isfinite(t1_out[10, 10, 10])

    def test_single_voxel_gap(
        self, t1_map_20: np.ndarray, brain_mask_20: np.ndarray,
    ) -> None:
        """One NaN surrounded by valid → filled near neighbor mean."""
        t1_map_20[10, 10, 10] = np.nan

        t1_out, result = fill_t1_gaps(
            t1_map_20, brain_mask_20, field_strength=3.0,
            voxel_sizes_mm=np.array([1.5, 1.5, 1.5]),
        )
        assert result.n_nan_before == 1
        assert result.n_filled_interp == 1
        assert result.n_remaining == 0
        # Filled value should be close to surrounding value (1050.0)
        assert t1_out[10, 10, 10] == pytest.approx(1050.0, rel=0.01)

    def test_all_nan_brain(self, brain_mask_20: np.ndarray) -> None:
        """Entire brain NaN → graceful return, n_remaining == n_nan_before."""
        t1 = np.full((20, 20, 20), np.nan, dtype=np.float32)

        t1_out, result = fill_t1_gaps(
            t1, brain_mask_20, field_strength=3.0,
            voxel_sizes_mm=np.array([1.5, 1.5, 1.5]),
        )
        assert result.n_nan_before == int(np.sum(brain_mask_20))
        assert result.n_remaining == result.n_nan_before
        assert result.n_filled_interp == 0
        assert result.n_filled_tissue == 0
        # All brain voxels still NaN
        assert np.all(np.isnan(t1_out[brain_mask_20]))

    def test_gap_support_depends_on_physical_resolution(self) -> None:
        """Six isolated neighbors suffice at 3 mm but not at 1 mm spacing."""
        t1 = np.full((11, 11, 11), np.nan, dtype=np.float32)
        for axis in range(3):
            for offset in (-1, 1):
                index = [5, 5, 5]
                index[axis] += offset
                t1[tuple(index)] = 1050.0
        mask = np.ones(t1.shape, dtype=bool)
        coarse, _ = fill_t1_gaps(t1, mask, 3.0, voxel_sizes_mm=np.full(3, 3.0))
        fine, _ = fill_t1_gaps(t1, mask, 3.0, voxel_sizes_mm=np.ones(3))
        assert coarse[5, 5, 5] == pytest.approx(1050.0)
        assert np.isnan(fine[5, 5, 5])


def _make_ernst_signal(
    s0: float, t1_s: float, tr_s: float, fa_deg: np.ndarray, b1: float = 1.0,
) -> np.ndarray:
    """Generate noiseless Ernst equation signal for VFA."""
    fa_rad = np.radians(fa_deg) * b1
    e1 = np.exp(-tr_s / t1_s)
    return s0 * np.sin(fa_rad) * (1 - e1) / (1 - np.cos(fa_rad) * e1)


def test_t1_mapping_preserves_compact_measured_structure(tmp_path: Path) -> None:
    """Quantitative T1 output must not spatially vote away a compact lesion."""
    import nibabel as nib

    from pmriprep.interfaces.dce import T1Mapping

    shape = (9, 9, 9)
    affine = np.diag([1.0, 1.0, 5.0, 1.0])
    truth_ms = np.full(shape, 1000.0)
    center = np.array([4, 4, 4])
    lesion = np.zeros(shape, dtype=bool)
    lesion[tuple(center)] = True
    for axis in range(3):
        for direction in (-1, 1):
            neighbor = center.copy()
            neighbor[axis] += direction
            lesion[tuple(neighbor)] = True
    truth_ms[lesion] = 1600.0

    flip_angles = np.array([5.0, 15.0, 25.0])
    tr_ms = 7.0
    vfa_files = []
    for index, signal in enumerate(
        _make_ernst_signal(
            5000.0,
            truth_ms / 1000.0,
            tr_ms / 1000.0,
            flip_angles[:, None, None, None],
        ),
    ):
        path = tmp_path / f"vfa_{index}.nii.gz"
        nib.save(nib.Nifti1Image(signal.astype(np.float32), affine), path)
        vfa_files.append(str(path))

    mask_file = tmp_path / "brain_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.uint8), affine), mask_file)
    interface = T1Mapping(
        vfa_files=vfa_files,
        flip_angles=flip_angles.tolist(),
        tr=tr_ms,
        brain_mask=str(mask_file),
        t1_method="nlls",
        t1_correction_mode="none",
        t1_default_ms=850.,
        out_file=str(tmp_path / "t1_map.nii.gz"),
    )
    result = interface.run(cwd=str(tmp_path))

    measured = nib.load(result.outputs.out_file).get_fdata()
    support = nib.load(result.outputs.support_file).get_fdata()
    assert np.all(measured[lesion] > 1500.0)
    assert np.all(measured[~lesion & np.isfinite(measured)] < 1100.0)
    assert set(np.unique(support)).issubset({1.0, 2.0, 3.0})
    support_qc = result.outputs.qc_metrics["t1_support"]
    assert sum(
        support_qc[key]
        for key in (
            "directly_measured_voxels", "locally_interpolated_voxels",
            "unavailable_voxels",
            "uniform_assumption_voxels",
        )
    ) == support_qc["analysis_domain_voxels"]
    assert result.outputs.qc_metrics["median_filtered"] is False
    assert (
        result.outputs.qc_metrics["t1_correction"][
            "population_reference_applicable"
        ]
        is False
    )


@pytest.mark.parametrize("skip", [True, False])
def test_t1_interface_uniform_fallback_reaches_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, skip: bool,
) -> None:
    """Skipped/failed mapping uses the override and reports its actual source."""
    import nibabel as nib
    from pmriprep.interfaces.dce import T1Mapping
    from pmriprep.workflows.dce._helpers import _signal_to_concentration_node
    from pmriprep.utils.dce_signal import signal_to_concentration

    monkeypatch.chdir(tmp_path)
    shape = (7, 7, 3)
    vfa_files = []
    for index in range(2):
        path = tmp_path / f"vfa_{index}.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), np.eye(4)), path)
        vfa_files.append(str(path))
    mask_file = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape, np.uint8), np.eye(4)), mask_file)
    mapped = T1Mapping(vfa_files=vfa_files, flip_angles=[5., 15.], tr=7.,
                       brain_mask=str(mask_file), skip_t1=skip, t1_default_ms=850.).run().outputs
    values = nib.load(mapped.out_file).get_fdata()
    np.testing.assert_allclose(values[np.isfinite(values)], 850.)
    qc = mapped.qc_metrics
    if skip:
        assert qc["assumed_t1_ms"] == 850.
        assert qc["assumed_t1_policy"] == "uniform_configured"
    else:
        assert qc["t1_correction"]["fallback_value_ms"] == 850.
        assert qc["t1_correction"]["fallback_source"] == "configured"
        assert qc["t1_correction"]["median_t1_corrected"] == 850.
        signal = np.full(shape + (10,), 100., dtype=np.float32)
        signal[..., 5:] = 130.
        signal_file = tmp_path / "dce.nii.gz"
        nib.save(nib.Nifti1Image(signal, np.eye(4)), signal_file)
        output, conversion = _signal_to_concentration_node(
            str(signal_file), str(mask_file), mapped.out_file, None,
            15., .005, 3., 3.8, 5, False, t1_default_ms=850., t1_mapping_qc=qc,
        )
        expected = signal_to_concentration(signal, 15., .005, t1_map=values,
                                           r1_relaxivity=3.8, baseline_frames=5,
                                           convert_to_plasma=False)
        np.testing.assert_allclose(nib.load(output).get_fdata(), expected.concentration,
                                   rtol=1e-6, atol=1e-7)
        assert conversion["t1_source"] == "assumed"
        assert conversion["assumed_t1_policy"] == "uniform_configured"
        assert conversion["assumed_t1_ms"] == 850.


def test_t1_mapping_csf_probseg_restricts_qc_not_support(tmp_path: Path) -> None:
    """Anatomical CSF classification must not remove quantitative T1 support."""
    import nibabel as nib

    from pmriprep.interfaces.dce import T1Mapping

    shape = (5, 5, 5)
    affine = np.eye(4)
    flip_angles = np.array([5.0, 15.0, 25.0])
    tr_ms = 7.0
    truth_ms = np.full(shape, 1200.0)
    truth_ms[2, 2, 2] = 4000.0
    truth_ms[1, 1, 1] = 4000.0
    vfa_files = []
    for index, signal in enumerate(
        _make_ernst_signal(
            5000.0,
            truth_ms / 1000.0,
            tr_ms / 1000.0,
            flip_angles[:, None, None, None],
        ),
    ):
        path = tmp_path / f"vfa_{index}.nii.gz"
        nib.save(nib.Nifti1Image(signal.astype(np.float32), affine), path)
        vfa_files.append(str(path))

    brain = np.ones(shape, dtype=np.uint8)
    csf = np.zeros(shape, dtype=np.float32)
    csf[2, 2, 2] = 1.0
    csf[1, 1, 1] = 1.0
    paths = {}
    for name, data in (("brain", brain), ("csf", csf)):
        path = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), path)
        paths[name] = str(path)

    result = T1Mapping(
        vfa_files=vfa_files,
        flip_angles=flip_angles.tolist(),
        tr=tr_ms,
        brain_mask=paths["brain"],
        csf_probseg=paths["csf"],
        t1_method="nlls",
        t1_correction_mode="none",
        out_file=str(tmp_path / "t1_map.nii.gz"),
    ).run(cwd=str(tmp_path))

    support = nib.load(result.outputs.support_file).get_fdata()
    assert support[2, 2, 2] == 1
    assert support[1, 1, 1] == 1
    support_qc = result.outputs.qc_metrics["t1_support"]
    assert support_qc["csf_qc_restricted_voxels"] == 2
    assert result.outputs.qc_metrics["fraction_valid"] == 1.0
    assert result.outputs.qc_metrics["median_t1_ms"] == pytest.approx(1200.0, abs=1.0)
    assert (
        result.outputs.qc_metrics["t1_correction"][
            "population_reference_applicable"
        ]
        is True
    )


class TestNLLSFitting:
    """Tests for nonlinear least squares T1 fitting."""

    def _make_t1mapping_interface(self):
        """Create a minimal T1Mapping interface for calling fitting methods."""
        from unittest.mock import MagicMock

        from pmriprep.interfaces.dce import T1Mapping

        t1m = T1Mapping.__new__(T1Mapping)
        t1m.inputs = MagicMock()
        return t1m

    def test_nlls_recovers_long_t1_noiseless(self) -> None:
        """NLLS recovers T1=5.0s at TR=4.4ms, noiseless (the OSIPI failure case)."""
        t1m = self._make_t1mapping_interface()
        tr_s = 0.0044
        fa_deg = np.array([2.0, 12.0])
        t1_true = 5.0  # seconds
        s0_true = 1000.0

        shape = (5, 5, 5)
        n_fa = len(fa_deg)
        signal_data = np.zeros((n_fa,) + shape)
        for i, fa in enumerate(fa_deg):
            signal_data[i] = _make_ernst_signal(s0_true, t1_true, tr_s, np.array([fa]))

        fa_rad = np.radians(fa_deg)
        sin_fa = np.sin(fa_rad[:, None, None, None] * np.ones(shape))
        cos_fa = np.cos(fa_rad[:, None, None, None] * np.ones(shape))

        t1_map = t1m._fit_t1_nlls(signal_data, sin_fa, cos_fa, tr_s, shape)

        # Should recover T1 within 1% of ground truth (noiseless)
        median_t1 = float(np.median(t1_map[t1_map > 0]))
        assert abs(median_t1 - t1_true) / t1_true < 0.01, (
            f"NLLS median T1={median_t1:.3f}s, expected {t1_true}s"
        )

    def test_wlls_inaccurate_long_t1_with_noise(self) -> None:
        """Document: WLLS is inaccurate for T1=5.0s at TR=4.4ms with noise.

        When E1 ~ 0.999, the linearized DESPOT1 variables X=S/tan(a) and
        Y=S/sin(a) share correlated noise from the same signal S. This makes
        WLLS unstable — small noise perturbations cause large T1 swings.
        The error can go in either direction depending on noise realization.
        """
        t1m = self._make_t1mapping_interface()
        tr_s = 0.0044
        fa_deg = np.array([2.0, 12.0])
        t1_true = 5.0
        s0_true = 1000.0

        shape = (20, 20, 20)
        rng = np.random.default_rng(42)
        signal_data = np.zeros((2,) + shape)
        for i, fa in enumerate(fa_deg):
            clean = _make_ernst_signal(s0_true, t1_true, tr_s, np.array([fa]))
            signal_data[i] = clean + rng.normal(0, s0_true / 30, shape)  # SNR ~30
        signal_data = np.maximum(signal_data, 1.0)

        fa_rad = np.radians(fa_deg)
        sin_fa = np.sin(fa_rad[:, None, None, None] * np.ones(shape))
        cos_fa = np.cos(fa_rad[:, None, None, None] * np.ones(shape))
        tan_fa = np.tan(fa_rad[:, None, None, None] * np.ones(shape))

        y_data = signal_data / sin_fa
        x_data = signal_data / tan_fa

        t1_wlls = t1m._fit_t1_wlls(x_data, y_data, sin_fa, cos_fa, tr_s, shape)

        # WLLS with noise at long T1 should be significantly inaccurate (>50% error)
        median_wlls = float(np.median(t1_wlls[t1_wlls > 0]))
        relative_error = abs(median_wlls - t1_true) / t1_true
        assert relative_error > 0.5, (
            f"Expected WLLS to be inaccurate (>50% error) at T1=5s, TR=4.4ms, "
            f"SNR=30, got {median_wlls:.3f}s (error={relative_error:.1%})"
        )

    def test_wlls_initialized_nlls_improves_long_t1(self) -> None:
        """Default NLLS initialization improves a noisy long-T1 estimate."""
        tr_s = 0.0044
        t1_true = 5.0
        t1m = self._make_t1mapping_interface()
        fa_deg = np.array([2.0, 12.0])
        s0_true = 1000.0
        shape = (15, 15, 15)

        rng = np.random.default_rng(42)
        signal_data = np.zeros((2,) + shape)
        for i, fa in enumerate(fa_deg):
            clean = _make_ernst_signal(s0_true, t1_true, tr_s, np.array([fa]))
            signal_data[i] = clean + rng.normal(0, s0_true / 30, shape)
        signal_data = np.maximum(signal_data, 1.0)

        fa_rad = np.radians(fa_deg)
        sin_fa = np.sin(fa_rad[:, None, None, None] * np.ones(shape))
        cos_fa = np.cos(fa_rad[:, None, None, None] * np.ones(shape))
        tan_fa = np.tan(fa_rad[:, None, None, None] * np.ones(shape))

        # WLLS result
        y_data = signal_data / sin_fa
        x_data = signal_data / tan_fa
        t1_wlls = t1m._fit_t1_wlls(x_data, y_data, sin_fa, cos_fa, tr_s, shape)

        # No explicit seed: production NLLS derives its WLLS warm start.
        refined = t1m._fit_t1_nlls(
            signal_data, sin_fa, cos_fa, tr_s, shape,
        )
        e1_wlls = np.exp(-tr_s / np.where(t1_wlls > 0, t1_wlls, 1.0))
        explicit_warm_start = t1m._fit_t1_nlls(
            signal_data, sin_fa, cos_fa, tr_s, shape,
            e1_init=e1_wlls.ravel(),
        )
        np.testing.assert_allclose(refined, explicit_warm_start)
        median_refined = float(np.median(refined[refined > 0]))
        median_wlls = float(np.median(t1_wlls[t1_wlls > 0]))

        # NLLS should be closer to ground truth than WLLS
        wlls_error = abs(median_wlls - t1_true)
        nlls_error = abs(median_refined - t1_true)
        assert nlls_error < wlls_error, (
            f"NLLS error ({nlls_error:.3f}) should be < WLLS error ({wlls_error:.3f})"
        )

    def test_nlls_2_flip_angles(self) -> None:
        """Exactly-determined fit with 2 FAs — verify convergence."""
        t1m = self._make_t1mapping_interface()
        tr_s = 0.005
        fa_deg = np.array([5.0, 15.0])
        t1_true = 1.5
        s0_true = 500.0

        shape = (2, 2, 2)
        signal_data = np.zeros((2,) + shape)
        for i, fa in enumerate(fa_deg):
            signal_data[i] = _make_ernst_signal(s0_true, t1_true, tr_s, np.array([fa]))

        fa_rad = np.radians(fa_deg)
        sin_fa = np.sin(fa_rad[:, None, None, None] * np.ones(shape))
        cos_fa = np.cos(fa_rad[:, None, None, None] * np.ones(shape))

        t1_map = t1m._fit_t1_nlls(signal_data, sin_fa, cos_fa, tr_s, shape)

        # Noiseless 2-FA: should converge to exact solution
        for val in t1_map.ravel():
            assert abs(val - t1_true) / t1_true < 0.01, (
                f"Expected T1≈{t1_true}s, got {val:.4f}s"
            )

    def test_nlls_noisy_graceful_degradation(self) -> None:
        """SNR=1: NLLS no worse than WLLS (both noisy, neither should crash)."""
        t1m = self._make_t1mapping_interface()
        tr_s = 0.005
        fa_deg = np.array([5.0, 15.0])
        t1_true = 1.0
        s0_true = 100.0

        shape = (10, 10, 10)
        rng = np.random.default_rng(123)
        signal_data = np.zeros((2,) + shape)
        for i, fa in enumerate(fa_deg):
            clean = _make_ernst_signal(s0_true, t1_true, tr_s, np.array([fa]))
            signal_data[i] = clean + rng.normal(0, s0_true, shape)  # SNR ≈ 1
        signal_data = np.maximum(signal_data, 1.0)

        fa_rad = np.radians(fa_deg)
        sin_fa = np.sin(fa_rad[:, None, None, None] * np.ones(shape))
        cos_fa = np.cos(fa_rad[:, None, None, None] * np.ones(shape))

        t1_map = t1m._fit_t1_nlls(signal_data, sin_fa, cos_fa, tr_s, shape)

        # Should not crash, all values should be finite and within bounds
        valid = t1_map[t1_map > 0]
        assert len(valid) > 0, "Should have some valid T1 values"
        assert np.all(np.isfinite(valid)), "All fitted T1 values should be finite"
        assert np.all(valid >= 0.01), "T1 >= 10ms"
        assert np.all(valid <= 20.0), "T1 <= 20s"

    def test_nlls_r1_bounds(self) -> None:
        """T1 stays within [10ms, 20s] with pathological inputs."""
        t1m = self._make_t1mapping_interface()
        tr_s = 0.005
        fa_deg = np.array([5.0, 15.0])
        shape = (3, 3, 3)

        # Pathological: constant signal across flip angles (degenerate)
        signal_data = np.ones((2,) + shape) * 100.0
        fa_rad = np.radians(fa_deg)
        sin_fa = np.sin(fa_rad[:, None, None, None] * np.ones(shape))
        cos_fa = np.cos(fa_rad[:, None, None, None] * np.ones(shape))

        t1_map = t1m._fit_t1_nlls(signal_data, sin_fa, cos_fa, tr_s, shape)

        # All values should be within bounds
        valid = t1_map[t1_map > 0]
        if len(valid) > 0:
            assert np.all(valid >= 0.01), f"Min T1={np.min(valid):.4f}s < 10ms"
            assert np.all(valid <= 20.0), f"Max T1={np.max(valid):.4f}s > 20s"


class TestExternalT1:
    """Tests for external T1 map support."""

    def test_external_t1_unit_detection_seconds(self, tmp_path) -> None:
        """Verify auto-detection of seconds vs milliseconds."""
        import nibabel as nib

        # Create a T1 map in seconds (median ~1.0)
        t1_data = np.full((10, 10, 10), 1.0, dtype=np.float32)
        t1_data[5:, :, :] = 1.5  # Some GM
        affine = np.eye(4)
        img = nib.Nifti1Image(t1_data, affine)
        t1_file = str(tmp_path / "t1_seconds.nii.gz")
        nib.save(img, t1_file)

        # Also create a reference image
        ref_data = np.ones((10, 10, 10), dtype=np.float32)
        ref_img = nib.Nifti1Image(ref_data, affine)
        ref_file = str(tmp_path / "ref.nii.gz")
        nib.save(ref_img, ref_file)

        # The function should detect seconds and convert
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            from pmriprep.workflows.dce.base import _transform_external_t1_to_dce
            result = _transform_external_t1_to_dce(t1_file, ref_file)
            # Load the result and check its millisecond units
            result_img = nib.load(result)
            result_data = result_img.get_fdata()
            median_val = float(np.nanmedian(result_data[result_data > 0]))
            assert median_val > 500, (
                f"Expected T1 in ms (>500), got median={median_val:.1f}"
            )
        finally:
            os.chdir(original_cwd)

    def test_external_t1_unit_detection_milliseconds(self, tmp_path) -> None:
        """T1 already in ms should not be converted."""
        import nibabel as nib

        t1_data = np.full((10, 10, 10), 1000.0, dtype=np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(t1_data, affine)
        t1_file = str(tmp_path / "t1_ms.nii.gz")
        nib.save(img, t1_file)

        ref_data = np.ones((10, 10, 10), dtype=np.float32)
        ref_img = nib.Nifti1Image(ref_data, affine)
        ref_file = str(tmp_path / "ref.nii.gz")
        nib.save(ref_img, ref_file)

        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            from pmriprep.workflows.dce.base import _transform_external_t1_to_dce
            result = _transform_external_t1_to_dce(t1_file, ref_file)
            result_img = nib.load(result)
            result_data = result_img.get_fdata()
            median_val = float(np.nanmedian(result_data[result_data > 0]))
            # Should still be ~1000ms (not 1e6)
            assert 500 < median_val < 2000, (
                f"Expected ~1000ms, got median={median_val:.1f}"
            )
        finally:
            os.chdir(original_cwd)


class TestConfigMutualExclusivity:
    """Tests for DCEConfig t1_map_file / no_t1 validation."""

    def test_no_t1_and_t1_map_file_raises(self) -> None:
        """Cannot set both no_t1=True and t1_map_file."""
        from pmriprep.config.dce import DCEConfig

        with pytest.raises(ValueError, match="Cannot set both"):
            DCEConfig(no_t1=True, t1_map_file=Path("/tmp/t1.nii"))

    def test_t1_map_file_alone_ok(self) -> None:
        """t1_map_file without no_t1 is valid."""
        from pmriprep.config.dce import DCEConfig

        config = DCEConfig(t1_map_file=Path("/tmp/t1.nii"))
        assert config.t1_map_file == Path("/tmp/t1.nii")
        assert not config.no_t1

    def test_t1_method_default_is_nlls(self) -> None:
        """Default T1 fitting uses WLLS-initialized NLLS."""
        from pmriprep.config.dce import DCEConfig

        config = DCEConfig()
        assert config.t1_method == "nlls"
