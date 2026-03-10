"""Unit tests for lesion/tumor ROI statistics extraction utilities."""

import numpy as np
import pytest

from pmriprep.utils.lesion_stats import (
    LesionStatisticsResult,
    MapStatistics,
    RegionStatistics,
    compute_dce_derived_metrics,
    compute_dsc_derived_metrics,
    compute_fraction_above_threshold,
    compute_hotspot_value,
    compute_map_statistics,
    compute_nawm_reference,
    compute_region_volume,
    detect_mask_type,
    extract_label_mask,
)


class TestComputeMapStatistics:
    """Tests for compute_map_statistics()."""

    def test_known_values(self) -> None:
        """Known 3D array + mask produces correct statistics."""
        data = np.zeros((2, 5, 1))
        mask = np.ones((2, 5, 1), dtype=bool)

        # Place known values: 1..10 inside the mask (10 voxels total)
        values = np.arange(1.0, 11.0)
        data.flat[:10] = values

        stats = compute_map_statistics(data, mask)
        assert stats is not None
        assert stats.mean == pytest.approx(np.mean(values), abs=1e-5)
        assert stats.std == pytest.approx(np.std(values), abs=1e-5)
        assert stats.median == pytest.approx(np.median(values), abs=1e-5)
        assert stats.min == pytest.approx(1.0, abs=1e-5)
        assert stats.max == pytest.approx(10.0, abs=1e-5)
        assert stats.p10 == pytest.approx(np.percentile(values, 10), abs=1e-5)
        assert stats.p25 == pytest.approx(np.percentile(values, 25), abs=1e-5)
        assert stats.p75 == pytest.approx(np.percentile(values, 75), abs=1e-5)
        assert stats.p90 == pytest.approx(np.percentile(values, 90), abs=1e-5)
        assert stats.p95 == pytest.approx(np.percentile(values, 95), abs=1e-5)
        assert stats.cv is not None
        expected_cv = round(float(np.std(values)) / abs(float(np.mean(values))), 4)
        assert stats.cv == pytest.approx(expected_cv, abs=1e-4)
        assert stats.iqr is not None
        expected_iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
        assert stats.iqr == pytest.approx(expected_iqr, abs=1e-5)

    def test_empty_mask_returns_none(self) -> None:
        """Empty mask returns None."""
        data = np.ones((5, 5, 5))
        mask = np.zeros((5, 5, 5), dtype=bool)
        assert compute_map_statistics(data, mask) is None

    def test_all_nan_values_returns_stats_with_zero_valid(self) -> None:
        """All NaN values within non-empty mask returns stats with n_valid=0."""
        data = np.full((5, 5, 5), np.nan)
        mask = np.ones((5, 5, 5), dtype=bool)
        stats = compute_map_statistics(data, mask)
        assert stats is not None
        assert stats.n_valid == 0
        assert stats.n_total == 125
        assert stats.mean is None

    def test_mean_zero_cv_is_none(self) -> None:
        """When mean is zero, coefficient of variation is None."""
        data = np.zeros((3, 3, 3))
        # Create symmetric values that sum to zero
        data[0, 0, 0] = -1.0
        data[0, 0, 1] = 1.0
        mask = np.zeros((3, 3, 3), dtype=bool)
        mask[0, 0, 0] = True
        mask[0, 0, 1] = True

        stats = compute_map_statistics(data, mask)
        assert stats is not None
        assert stats.cv is None


class TestComputeFractionAboveThreshold:
    """Tests for compute_fraction_above_threshold()."""

    def test_known_fraction(self) -> None:
        """10 voxels, 6 above threshold of 4.5 returns 0.6."""
        data = np.zeros((2, 5, 1))
        mask = np.ones((2, 5, 1), dtype=bool)
        # Values 1..10
        values = np.arange(1.0, 11.0)
        data.flat[:10] = values
        # 6 values above 4.5: 5, 6, 7, 8, 9, 10
        fraction = compute_fraction_above_threshold(data, mask, 4.5)
        assert fraction == pytest.approx(0.6, abs=1e-4)

    def test_empty_mask_returns_none(self) -> None:
        """Empty mask returns None."""
        data = np.ones((3, 3, 3))
        mask = np.zeros((3, 3, 3), dtype=bool)
        assert compute_fraction_above_threshold(data, mask, 0.5) is None


class TestComputeHotspotValue:
    """Tests for compute_hotspot_value()."""

    def test_default_90th_percentile(self) -> None:
        """Known data returns correct 90th percentile."""
        data = np.zeros((10, 10, 1))
        mask = np.ones((10, 10, 1), dtype=bool)
        values = np.arange(1.0, 101.0)
        data.flat[:100] = values

        result = compute_hotspot_value(data, mask)
        expected = float(np.percentile(values, 90))
        assert result == pytest.approx(expected, abs=1e-5)

    def test_custom_95th_percentile(self) -> None:
        """Custom 95th percentile returns correct value."""
        data = np.zeros((10, 10, 1))
        mask = np.ones((10, 10, 1), dtype=bool)
        values = np.arange(1.0, 101.0)
        data.flat[:100] = values

        result = compute_hotspot_value(data, mask, percentile=95.0)
        expected = float(np.percentile(values, 95))
        assert result == pytest.approx(expected, abs=1e-5)

    def test_empty_mask_returns_none(self) -> None:
        """Empty mask returns None."""
        data = np.ones((3, 3, 3))
        mask = np.zeros((3, 3, 3), dtype=bool)
        assert compute_hotspot_value(data, mask) is None


class TestDetectMaskType:
    """Tests for detect_mask_type()."""

    def test_multi_label(self) -> None:
        """Max label >= 2 detected as multi_label_tumor_seg."""
        mask_data = np.array([[[0, 1, 2, 3]]])
        mask_type, max_label = detect_mask_type(mask_data)
        assert mask_type == "multi_label_tumor_seg"
        assert max_label == 3

    def test_binary(self) -> None:
        """Max label = 1 detected as binary."""
        mask_data = np.array([[[0, 0, 1, 1]]])
        mask_type, max_label = detect_mask_type(mask_data)
        assert mask_type == "binary"
        assert max_label == 1


class TestExtractLabelMask:
    """Tests for extract_label_mask()."""

    def test_extract_label_2(self) -> None:
        """Extracting label 2 from multi-label data returns correct boolean mask."""
        mask_data = np.array([[[0, 1, 2, 2, 3, 1]]])
        result = extract_label_mask(mask_data, 2)
        expected = np.array([[[False, False, True, True, False, False]]])
        np.testing.assert_array_equal(result, expected)


class TestComputeRegionVolume:
    """Tests for compute_region_volume()."""

    def test_known_volume(self) -> None:
        """100 voxels with 1mm^3 voxels gives (100, 100.0, 0.1)."""
        mask = np.zeros((10, 10, 1), dtype=bool)
        mask[:, :, 0] = True  # 100 True voxels
        voxel_sizes = (1.0, 1.0, 1.0)

        count, vol_mm3, vol_ml = compute_region_volume(mask, voxel_sizes)
        assert count == 100
        assert vol_mm3 == pytest.approx(100.0, abs=1e-6)
        assert vol_ml == pytest.approx(0.1, abs=1e-6)


class TestComputeDSCDerivedMetrics:
    """Tests for compute_dsc_derived_metrics()."""

    def test_with_ncbv_data(self) -> None:
        """nCBV data + mask produces fractional_volumes and hotspot_summary keys."""
        mask = np.ones((10, 10, 1), dtype=bool)
        # nCBV values: 0.5..5.4 in steps of ~0.05
        ncbv = np.zeros((10, 10, 1))
        ncbv.flat[:100] = np.linspace(0.5, 5.4, 100)

        derived = compute_dsc_derived_metrics(
            ncbv_data=ncbv,
            ncbf_data=None,
            cbv_data=None,
            cbf_data=None,
            mask=mask,
        )
        assert "fractional_volumes" in derived
        assert "nCBV_above_1.75" in derived["fractional_volumes"]
        assert "nCBV_above_2.5" in derived["fractional_volumes"]
        assert "nCBV_above_4.0" in derived["fractional_volumes"]
        assert "hotspot_summary" in derived

    def test_all_none_maps_returns_empty(self) -> None:
        """All None maps returns empty dict."""
        mask = np.ones((3, 3, 3), dtype=bool)
        derived = compute_dsc_derived_metrics(
            ncbv_data=None,
            ncbf_data=None,
            cbv_data=None,
            cbf_data=None,
            mask=mask,
        )
        assert derived == {}


class TestComputeDCEDerivedMetrics:
    """Tests for compute_dce_derived_metrics()."""

    def test_with_ktrans_and_ve(self) -> None:
        """Ktrans + ve data produces correct fractional volumes."""
        mask = np.ones((10, 10, 1), dtype=bool)
        ktrans = np.zeros((10, 10, 1))
        ktrans.flat[:100] = np.linspace(0.0, 0.3, 100)
        ve = np.zeros((10, 10, 1))
        ve.flat[:100] = np.linspace(0.0, 0.5, 100)

        derived = compute_dce_derived_metrics(
            ktrans_data=ktrans,
            nktrans_data=None,
            ve_data=ve,
            vp_data=None,
            mask=mask,
        )
        assert "fractional_volumes" in derived
        assert "Ktrans_above_0.1" in derived["fractional_volumes"]
        assert "Ktrans_above_0.2" in derived["fractional_volumes"]
        assert "hotspot_summary" in derived
        assert "ktrans_p90" in derived["hotspot_summary"]
        assert "ve_p90" in derived["hotspot_summary"]


class TestComputeNAWMReference:
    """Tests for compute_nawm_reference()."""

    def test_nawm_stats(self) -> None:
        """Maps + NAWM mask produces per-map stats with median, std, qc_status=PASS."""
        nawm_mask = np.zeros((5, 5, 5), dtype=bool)
        nawm_mask[1:4, 1:4, 1:4] = True  # 27 voxels
        rng = np.random.default_rng(42)
        cbv_data = rng.uniform(2.0, 4.0, (5, 5, 5))

        maps = {"cbv": cbv_data}
        voxel_sizes = (1.0, 1.0, 1.0)
        ref = compute_nawm_reference(nawm_mask, maps, voxel_sizes)

        assert ref["voxel_count"] == 27
        assert "cbv" in ref
        assert "median" in ref["cbv"]
        assert "std" in ref["cbv"]
        assert ref["qc_status"] == "PASS"

    def test_extreme_cbv_warns(self) -> None:
        """Extreme CBV median produces qc_status=WARNING."""
        nawm_mask = np.zeros((5, 5, 5), dtype=bool)
        nawm_mask[0, 0, 0] = True
        # CBV far outside expected range (> 10)
        cbv_data = np.full((5, 5, 5), 15.0)

        maps = {"cbv": cbv_data}
        voxel_sizes = (1.0, 1.0, 1.0)
        ref = compute_nawm_reference(nawm_mask, maps, voxel_sizes)

        assert ref["qc_status"] == "WARNING"


class TestMapStatisticsToDict:
    """Tests for MapStatistics.to_dict()."""

    def test_excludes_none_values(self) -> None:
        """to_dict() excludes fields that are None."""
        stats = MapStatistics(
            mean=1.0,
            std=0.5,
            median=1.0,
            min=0.0,
            max=2.0,
            p10=0.2,
            p25=0.5,
            p75=1.5,
            p90=1.8,
            p95=1.9,
            cv=None,
            iqr=None,
        )
        d = stats.to_dict()
        assert "cv" not in d
        assert "iqr" not in d
        assert "mean" in d
        assert d["mean"] == 1.0


class TestRegionStatisticsToDict:
    """Tests for RegionStatistics.to_dict()."""

    def test_correct_nested_structure(self) -> None:
        """to_dict() returns correct nested structure."""
        map_stats = MapStatistics(
            mean=5.0, std=1.0, median=5.0, min=3.0, max=7.0,
            p10=3.5, p25=4.0, p75=6.0, p90=6.5, p95=6.8,
            cv=0.2, iqr=2.0,
        )
        region = RegionStatistics(
            region_name="enhancing_tumor",
            voxel_count=200,
            volume_mm3=1600.0,
            volume_ml=1.6,
            perfusion_maps={"ncbv": map_stats},
        )
        d = region.to_dict()
        assert d["voxel_count"] == 200
        assert d["volume_mm3"] == 1600.0
        assert d["volume_ml"] == 1.6
        assert "ncbv" in d["perfusion_maps"]
        assert d["perfusion_maps"]["ncbv"]["mean"] == 5.0


class TestLesionStatisticsResultToDict:
    """Tests for LesionStatisticsResult.to_dict()."""

    def test_bids_compliant_keys(self) -> None:
        """to_dict() contains BIDS-compliant keys."""
        result = LesionStatisticsResult(
            subject_id="0001",
            session_id="0002",
            modality="DSC",
            mask_type="binary",
            mask_source="/path/to/mask.nii.gz",
            total_voxels=500,
        )
        d = result.to_dict()
        assert "Description" in d
        assert "GeneratedBy" in d
        assert d["GeneratedBy"][0]["Name"] == "pmriprep"
        assert "GeneratedAt" in d
        assert "Subject" in d
        assert d["Subject"] == "sub-0001"
        assert "Session" in d
        assert d["Session"] == "ses-0002"
        assert "LesionMask" in d
        assert "LesionMaskType" in d
        assert "TotalLesionVoxels" in d
        assert "Regions" in d
