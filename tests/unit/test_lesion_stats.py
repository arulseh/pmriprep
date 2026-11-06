"""Unit tests for lesion/tumor ROI statistics extraction utilities."""

from pathlib import Path

import numpy as np
import pytest

from pmriprep.utils.lesion_stats import (
    LesionStatisticsResult,
    MapStatistics,
    RegionStatistics,
    combine_tumor_masks,
    compute_dce_derived_metrics,
    compute_dsc_derived_metrics,
    compute_fraction_above_threshold,
    compute_map_statistics,
    compute_nawm_reference,
    compute_region_volume,
    detect_mask_type,
    extract_label_mask,
    is_integer_valued,
    validate_mask_ndim,
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
        count, vol_mm3, vol_ml = compute_region_volume(mask, 1.0)
        assert count == 100
        assert vol_mm3 == pytest.approx(100.0, abs=1e-6)
        assert vol_ml == pytest.approx(0.1, abs=1e-6)


class TestComputeDSCDerivedMetrics:
    """Tests for compute_dsc_derived_metrics()."""

    def test_with_ncbv_data(self) -> None:
        """nCBV data + mask produces per-map fractional_volumes (no hotspot_summary)."""
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
        assert "ncbv" in derived["fractional_volumes"]
        ncbv_fv = derived["fractional_volumes"]["ncbv"]
        for t in (1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0):
            assert f"above_{t}" in ncbv_fv

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
        """Ktrans data produces per-map fractional volumes (no hotspot_summary)."""
        mask = np.ones((10, 10, 1), dtype=bool)
        ktrans = np.zeros((10, 10, 1))
        ktrans.flat[:100] = np.linspace(0.0, 0.3, 100)
        ve = np.zeros((10, 10, 1))
        ve.flat[:100] = np.linspace(0.0, 0.5, 100)

        derived = compute_dce_derived_metrics(
            ktrans_data=ktrans,
            ve_data=ve,
            vp_data=None,
            mask=mask,
        )
        assert "fractional_volumes" in derived
        assert "ktrans" in derived["fractional_volumes"]
        ktrans_fv = derived["fractional_volumes"]["ktrans"]
        for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
            assert f"above_{t}" in ktrans_fv

    def test_vp_fractional_volumes(self) -> None:
        """vp data produces per-map fractional volumes."""
        mask = np.ones((10, 10, 1), dtype=bool)
        vp = np.zeros((10, 10, 1))
        vp.flat[:100] = np.linspace(0.0, 0.12, 100)

        derived = compute_dce_derived_metrics(
            ktrans_data=None,
            ve_data=None, vp_data=vp, mask=mask,
        )
        assert "vp" in derived["fractional_volumes"]
        for t in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
            assert f"above_{t}" in derived["fractional_volumes"]["vp"]


class TestComputeNAWMReference:
    """Tests for compute_nawm_reference()."""

    def test_nawm_reference_statistics_and_coverage(self) -> None:
        """NAWM statistics retain their map values and finite coverage."""
        nawm_mask = np.zeros((10, 10, 10), dtype=bool)
        nawm_mask[1:9, 1:9, 1:9] = True  # 512 voxels
        rng = np.random.default_rng(42)
        cbv_data = rng.uniform(2.0, 4.0, (10, 10, 10))

        maps = {"cbv": cbv_data}
        ref = compute_nawm_reference(nawm_mask, maps, 1.0)

        assert ref["voxel_count"] == 512
        assert "cbv" in ref
        assert "mean" in ref["cbv"]
        assert "median" in ref["cbv"]
        assert "std" in ref["cbv"]
        assert ref["cbv"]["n_valid"] == 512
        assert ref["cbv"]["n_total"] == 512

    def test_quantitative_iauc60_reference_is_reported(self) -> None:
        """Raw IAUC60 retains NAWM context without constructing a ratio map."""
        nawm_mask = np.zeros((10, 10, 10), dtype=bool)
        nawm_mask[1:9, 1:9, 1:9] = True
        iauc60 = np.full((10, 10, 10), 0.03)

        ref = compute_nawm_reference(
            nawm_mask, {"iauc60": iauc60}, 1.0,
        )

        assert ref["iauc60"]["median"] == pytest.approx(0.03)
        assert ref["iauc60"]["std"] == pytest.approx(0.0)

    def test_small_nawm_mask_emits_size_without_editorial(self) -> None:
        """Small NAWM masks retain size and descriptive statistics.

        NAWM may come from upstream auto-NAWM (config.tumor_seg.nawm_min_voxels=500),
        bilateral fallback, OR a user-supplied mask via the force_binary ROI path.
        Each path has its own validity contract, so this function reports size as
        informational only and does not editorialize.
        """
        nawm_mask = np.zeros((5, 5, 5), dtype=bool)
        nawm_mask[0, 0, 0] = True  # 1 voxel
        cbv_data = np.full((5, 5, 5), 2.5)

        ref = compute_nawm_reference(
            nawm_mask, {"cbv": cbv_data}, 1.0,
        )
        assert ref["voxel_count"] == 1


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
            iqr=None,
        )
        d = stats.to_dict()
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
            iqr=2.0,
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

    def test_to_dict_pascal_case_keys(self) -> None:
        """to_dict() emits PascalCase for NAWMReference and DerivedMetrics."""
        result = LesionStatisticsResult(
            subject_id="0001",
            session_id="0002",
            modality="DSC",
            mask_type="binary",
            mask_source="/path/to/mask.nii.gz",
            total_voxels=500,
            nawm_reference={"cbv": {"median": 1.0}},
            derived_metrics={"fractional_volumes": {}},
        )
        d = result.to_dict()
        assert "NAWMReference" in d
        assert "DerivedMetrics" in d
        assert "nawm_reference" not in d
        assert "derived_metrics" not in d


class TestLesionStatisticsAifProvenance:
    """PerfusionProcessing.AIF sidecar block."""

    def test_measured_aif_provenance(self) -> None:
        aif_block = {
            "Source": "measured",
            "DetectionLevel": "atlas_strict",
            "QualityFlag": "pass",
        }
        result = LesionStatisticsResult(
            subject_id="0001", session_id="0002", modality="DSC",
            mask_type="binary", mask_source="m.nii.gz", total_voxels=500,
            aif_provenance=aif_block,
        )
        d = result.to_dict()
        assert d["PerfusionProcessing"]["AIF"]["Source"] == "measured"

    def test_unavailable_aif_provenance(self) -> None:
        aif_block = {
            "Source": "unavailable",
            "DetectionLevel": "unavailable",
            "QualityFlag": "poor",
        }
        result = LesionStatisticsResult(
            subject_id="0001", session_id="0002", modality="DSC",
            mask_type="binary", mask_source="m.nii.gz", total_voxels=500,
            aif_provenance=aif_block,
        )
        d = result.to_dict()
        assert d["PerfusionProcessing"]["AIF"]["Source"] == "unavailable"

    def test_fields_absent_by_default(self) -> None:
        """DCE lesion-stats (no AIF provenance) omits the block cleanly."""
        result = LesionStatisticsResult(
            subject_id="0001", session_id="0002", modality="DCE",
            mask_type="binary", mask_source="m.nii.gz", total_voxels=500,
        )
        d = result.to_dict()
        assert "PerfusionProcessing" not in d


class TestCombineTumorMasksReturnsEmptyString:
    """Contract: when no masks present, return ("", "") not (None, "").

    Downstream consumers (DCE/DSC lesion-stats interfaces) declare
    ``lesion_mask`` as ``traits.Str``; a Function-node returning ``None``
    triggers a TraitError when nipype propagates the value to the next
    node's input. Empty string is the proper "not set" sentinel —
    ``has_input()`` treats it as missing and skips processing.
    """

    def test_no_masks_returns_empty_strings(self) -> None:
        result = combine_tumor_masks(
            tumor_segmentation_file=None,
            whole_tumor_mask_file=None,
            reference_file="ref.nii.gz",
        )
        assert result == ("", "")

    def test_undefined_inputs_return_empty_strings(self) -> None:
        from nipype.interfaces.base import Undefined

        result = combine_tumor_masks(
            tumor_segmentation_file=Undefined,
            whole_tumor_mask_file=Undefined,
            reference_file="ref.nii.gz",
        )
        assert result == ("", "")


class TestTransformHelpersReturnEmptyString:
    """Contract: workflow Function nodes returning paths must use "" not None.

    See TestCombineTumorMasksReturnsEmptyString for the trait rationale.
    """

    def test_transform_lesion_mask_to_dce_no_input(self) -> None:
        from pmriprep.workflows.dce.base import _transform_lesion_mask_to_dce

        assert _transform_lesion_mask_to_dce(
            lesion_mask_file=None,
            reference_file="ref.nii.gz",
            transform_matrix="x.mat",
        ) == ""

    def test_transform_lesion_mask_to_dsc_no_input(self) -> None:
        from pmriprep.workflows.dsc.base import _transform_lesion_mask_to_dsc

        assert _transform_lesion_mask_to_dsc(
            lesion_mask_file=None,
            reference_file="ref.nii.gz",
            transform_matrix="x.mat",
        ) == ""

    def test_transform_nawm_to_dsc_no_input(self) -> None:
        from pmriprep.workflows.dsc.base import _transform_nawm_to_dsc

        assert _transform_nawm_to_dsc(
            nawm_mask_file=None,
            reference_file="ref.nii.gz",
            transform_matrix="x.mat",
        ) == ""


class TestNormalizeHelpersReturnEmptyString:
    """Contract: NAWM normalization Function nodes return ``""`` (not ``None``).

    These helpers feed ``traits.Str`` consumers in DSC lesion-stats
    (``ncbv_map``, ``ncbf_map``). Returning
    ``None`` triggers a TraitError that under the MultiProc plugin manifests
    as a silent dispatch hang (no work-dir created, no log line) — the same
    failure mode :class:`TestCombineTumorMasksReturnsEmptyString` documents.
    """

    def test_normalize_cbv_no_nawm(self) -> None:
        from pmriprep.workflows.dsc.base import _normalize_cbv

        assert _normalize_cbv(
            cbv_file="/nonexistent/cbv.nii.gz",
            derivatives_dir="/tmp",
            subject_id="test01",
            session_id="visit1",
            nawm_mask_file=None,
        ) == ("", None)

    def test_normalize_cbf_no_nawm(self) -> None:
        from pmriprep.workflows.dsc.base import _normalize_cbf

        assert _normalize_cbf(
            cbf_file="/nonexistent/cbf.nii.gz",
            derivatives_dir="/tmp",
            subject_id="test01",
            session_id="visit1",
            nawm_mask_file=None,
        ) == ("", None)


class TestLesionStatsAcceptsEmptyMapTrait:
    """Normalized-map traits accept the empty-string sentinel emitted when
    no normalization reference is available.
    """

    def test_dsc_lesion_stats_accepts_empty_string_for_normalized_maps(self) -> None:
        from pmriprep.interfaces.dsc_lesion_stats import DSCLesionStats

        iface = DSCLesionStats()
        iface.inputs.subject_id = "test01"
        iface.inputs.session_id = "visit1"
        # Should not raise TraitError — these are the values produced by
        # _normalize_cbv / _normalize_cbf on the no-NAWM path.
        iface.inputs.ncbv_map = ""
        iface.inputs.ncbf_map = ""
        assert iface.inputs.ncbv_map == ""
        assert iface.inputs.ncbf_map == ""


class TestMapStatisticsDistributionMetrics:
    """Tests for upper-tail and distribution-shape summaries."""

    def test_advanced_metrics_present_on_large_roi(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.gamma(2.0, 0.05, size=(20, 20, 20))
        mask = np.ones_like(data, dtype=bool)
        s = compute_map_statistics(data, mask, compute_advanced=True)
        assert s is not None
        assert s.p98 is not None
        assert s.p99 is not None
        assert s.skewness is not None
        assert s.kurtosis is not None
        assert s.p98 < s.p99 <= s.max

    def test_advanced_metrics_omitted_when_disabled(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.gamma(2.0, 0.05, size=100)
        mask = np.ones_like(data, dtype=bool)
        s = compute_map_statistics(data, mask, compute_advanced=False)
        assert s.p98 is None
        assert s.p99 is None
        assert s.skewness is None
        assert s.kurtosis is None

    def test_advanced_metrics_omitted_when_n_valid_below_10(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.ones_like(data, dtype=bool)
        s = compute_map_statistics(data, mask, compute_advanced=True)
        assert s.p98 is None
        assert s.p99 is None

    def test_constant_data_shape_metrics_are_none(self) -> None:
        data = np.ones(200)
        mask = np.ones_like(data, dtype=bool)
        s = compute_map_statistics(data, mask, compute_advanced=True)
        assert s.skewness is None
        assert s.kurtosis is None


class TestCohensDAugmentation:
    """pooled-SD Cohen's d vs NAWM."""

    def test_pooled_d_attached_when_nawm_matches(self) -> None:
        result = LesionStatisticsResult(
            subject_id="X", session_id="Y", modality="DSC",
            mask_type="binary", mask_source="m.nii.gz", total_voxels=10,
        )
        region = RegionStatistics("tumor", 100, 100.0, 0.1)
        region.perfusion_maps["cbv"] = MapStatistics(
            mean=2.5, median=2.4, std=0.8, n_valid=100, n_total=100,
        )
        result.regions["tumor"] = region
        result.nawm_reference = {
            "cbv": {"mean": 1.5, "median": 1.4, "std": 0.4,
                    "n_valid": 50, "n_total": 50, "p25": 1.2, "p75": 1.8},
        }
        result.augment_with_cohens_d()
        d = result.regions["tumor"].perfusion_maps["cbv"].cohens_d_vs_nawm
        # Population SDs imply within-group sums of squares n*std².
        pooled_sd = np.sqrt((100 * 0.8**2 + 50 * 0.4**2) / 148)
        assert d == pytest.approx((2.5 - 1.5) / pooled_sd, abs=5e-5)

    def test_no_nawm_yields_no_cohen_d(self) -> None:
        result = LesionStatisticsResult(
            subject_id="X", session_id="Y", modality="DSC",
            mask_type="binary", mask_source="m.nii.gz", total_voxels=10,
        )
        region = RegionStatistics("tumor", 100, 100.0, 0.1)
        region.perfusion_maps["cbv"] = MapStatistics(
            median=2.5, std=0.8, n_valid=100, n_total=100,
        )
        result.regions["tumor"] = region
        # nawm_reference is None
        result.augment_with_cohens_d()
        assert result.regions["tumor"].perfusion_maps["cbv"].cohens_d_vs_nawm is None

    def test_map_without_nawm_entry_skipped(self) -> None:
        result = LesionStatisticsResult(
            subject_id="X", session_id="Y", modality="DSC",
            mask_type="binary", mask_source="m.nii.gz", total_voxels=10,
        )
        region = RegionStatistics("tumor", 100, 100.0, 0.1)
        region.perfusion_maps["mtt"] = MapStatistics(
            median=12.0, std=3.0, n_valid=100, n_total=100,
        )
        result.regions["tumor"] = region
        result.nawm_reference = {  # only cbv, no mtt
            "cbv": {"median": 1.5, "std": 0.4, "p25": 1.2, "p75": 1.8},
        }
        result.augment_with_cohens_d()
        assert result.regions["tumor"].perfusion_maps["mtt"].cohens_d_vs_nawm is None


class TestExtractPerMapLesionStatsEnrichment:
    """extract_per_map_lesion_stats forwards modality context."""

    @pytest.fixture
    def aggregate(self):
        return {
            "LesionMaskType": "multi_label_tumor_seg",
            "TotalLesionVoxels": 1000,
            "Regions": {
                "contrast_enhancing_tumor": {
                    "voxel_count": 100, "volume_mm3": 100.0, "volume_ml": 0.1,
                    "perfusion_maps": {
                        "ncbv": {"mean": 1.8, "std": 0.5, "median": 1.7,
                                 "p98": 3.2, "n_valid": 100, "n_total": 100},
                        "cbv": {"mean": 0.6, "std": 0.2, "median": 0.55,
                                "n_valid": 100, "n_total": 100},
                        "ktrans": {"mean": 0.12, "std": 0.05, "median": 0.10,
                                   "n_valid": 100, "n_total": 100},
                        "ttp": {"median": 5.0, "std": 1.0,
                                         "n_valid": 100, "n_total": 100},
                    },
                },
            },
            "NAWMReference": {
                "voxel_count": 1000, "volume_ml": 50.0,
                "ncbv": {"median": 1.0, "std": 0.1, "p25": 0.95, "p75": 1.05},
                "cbv": {"median": 0.34, "std": 0.05, "p25": 0.30, "p75": 0.38},
                "mtt": {"median": 5.5, "std": 1.0, "p25": 4.8, "p75": 6.2},
            },
            "DerivedMetrics": {
                "fractional_volumes": {
                    "ncbv": {"above_1.75": 0.45, "above_2.5": 0.18,
                             "above_3.0": 0.09, "above_4.0": 0.02},
                    "ktrans": {"above_0.05": 0.20, "above_0.1": 0.10},
                },
            },
            "QCFilteredRegions": {
                "contrast_enhancing_tumor": {
                    "perfusion_maps": {
                        "ktrans": {"median": 0.14, "n_valid": 60, "n_total": 100},
                    },
                },
            },
            "QCFilteredNAWMReference": {
                "voxel_count": 1000,
                "ktrans": {"median": 0.002, "n_valid": 200, "n_total": 1000},
            },
            "QCFilteredDerivedMetrics": {
                "fractional_volumes": {
                    "ktrans": {"above_0.05": 0.25, "above_0.1": 0.15},
                },
            },
        }

    def test_normalized_map_gets_fv(self, aggregate):
        from pmriprep.utils.lesion_stats import extract_per_map_lesion_stats

        out = extract_per_map_lesion_stats(aggregate, "ncbv")
        assert out["LesionMaskType"] == "multi_label_tumor_seg"
        assert "Regions" in out
        assert "FractionalVolumes" in out
        assert "above_3.0" in out["FractionalVolumes"]
        # NAWM in normalized units, slice for ncbv only
        assert out["NAWMReference"]["median"] == 1.0
        assert out["NAWMReference"]["voxel_count"] == 1000

    def test_unnormalized_companion_no_fv(self, aggregate):
        from pmriprep.utils.lesion_stats import extract_per_map_lesion_stats

        out = extract_per_map_lesion_stats(aggregate, "cbv")
        assert "FractionalVolumes" not in out
        # NAWM in deconv units
        assert out["NAWMReference"]["median"] == 0.34

    def test_ktrans_gets_fractional_volumes(self, aggregate):
        from pmriprep.utils.lesion_stats import extract_per_map_lesion_stats

        out = extract_per_map_lesion_stats(aggregate, "ktrans")
        assert "Regions" in out
        assert "FractionalVolumes" in out
        assert "above_0.05" in out["FractionalVolumes"]
        assert out["QCFilteredRegions"]["contrast_enhancing_tumor"]["stats"][
            "n_valid"
        ] == 60
        assert "voxel_count" not in out["QCFilteredRegions"][
            "contrast_enhancing_tumor"
        ]
        assert "volume_ml" not in out["QCFilteredRegions"][
            "contrast_enhancing_tumor"
        ]
        assert out["QCFilteredNAWMReference"]["median"] == pytest.approx(0.002)
        assert out["QCFilteredNAWMReference"]["n_valid"] == 200
        assert out["QCFilteredNAWMReference"]["n_total"] == 1000
        assert "voxel_count" not in out["QCFilteredNAWMReference"]
        assert "volume_ml" not in out["QCFilteredNAWMReference"]
        assert out["QCFilteredFractionalVolumes"]["above_0.05"] == pytest.approx(0.25)

    def test_map_with_no_fv_just_stats(self, aggregate):
        from pmriprep.utils.lesion_stats import extract_per_map_lesion_stats

        out = extract_per_map_lesion_stats(aggregate, "ttp")
        assert "Regions" in out
        assert "FractionalVolumes" not in out
        assert "NAWMReference" not in out

    def test_map_absent_from_regions_but_nawm_present_returns_nawm_only(self, aggregate):
        """A map with no per-region stats but NAWM ref still emits NAWMReference."""
        from pmriprep.utils.lesion_stats import extract_per_map_lesion_stats

        out = extract_per_map_lesion_stats(aggregate, "mtt")
        assert "Regions" not in out
        assert "NAWMReference" in out
        assert out["NAWMReference"]["median"] == 5.5

    def test_empty_aggregate_returns_none(self):
        from pmriprep.utils.lesion_stats import extract_per_map_lesion_stats

        assert extract_per_map_lesion_stats(None, "ncbv") is None
        assert extract_per_map_lesion_stats({}, "ncbv") is None


def test_dsc_load_maps_uses_configured_timing_name(tmp_path) -> None:
    """TTP and parametric arrival delay retain distinct names and values."""
    import nibabel as nib

    from pmriprep.interfaces.dsc_lesion_stats import DSCLesionStats

    timing_file = tmp_path / "timing.nii.gz"
    nib.save(
        nib.Nifti1Image(np.full((2, 2, 2), -0.25, dtype=np.float32), np.eye(4)),
        timing_file,
    )
    ttp_file = tmp_path / "ttp.nii.gz"
    nib.save(
        nib.Nifti1Image(np.full((2, 2, 2), 8.0, dtype=np.float32), np.eye(4)),
        ttp_file,
    )
    iface = DSCLesionStats()
    iface.inputs.tmax_map = str(timing_file)
    iface.inputs.ttp_map = str(ttp_file)
    iface.inputs.timing_map_name = "arrival_delay"

    maps = iface._load_maps()

    assert "arrival_delay" in maps
    assert "tmax" not in maps
    np.testing.assert_array_equal(maps["arrival_delay"], -0.25)
    np.testing.assert_array_equal(maps["ttp"], 8.0)


def test_dsc_regional_statistics_disclose_actual_osvd_fallback(tmp_path) -> None:
    """CPI replacement fractions accompany only affected regional map slices."""
    import json
    from pathlib import Path

    import nibabel as nib

    from pmriprep.interfaces.dsc_lesion_stats import DSCLesionStats
    from pmriprep.utils.lesion_stats import extract_per_map_lesion_stats

    def save(name: str, data: np.ndarray) -> str:
        path = tmp_path / f"{name}.nii.gz"
        nib.save(
            nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4)),
            path,
        )
        return str(path)

    shape = (2, 2, 1)
    iface = DSCLesionStats()
    iface.inputs.subject_id = "01"
    iface.inputs.session_id = "01"
    iface.inputs.lesion_mask = save("lesion", np.ones(shape))
    iface.inputs.nawm_mask = save(
        "nawm", np.array([[[1], [1]], [[0], [0]]])
    )
    iface.inputs.fallback_mask = save(
        "fallback", np.array([[[1], [0]], [[1], [0]]])
    )
    iface.inputs.cbf_map = save("cbf", np.arange(1, 5).reshape(shape))
    iface.inputs.cbv_auc_map = save("cbv_auc", np.arange(5, 9).reshape(shape))

    result = iface.run(cwd=tmp_path)
    aggregate = json.loads(Path(result.outputs.stats_json).read_text())

    expected_region = {
        "n_osvd": 2,
        "fraction": 0.5,
        "denominator": "region_voxels",
    }
    expected_nawm = {
        "n_osvd": 1,
        "fraction": 0.5,
        "denominator": "region_voxels",
    }
    assert aggregate["Regions"]["tumor"]["cpi_fallback"] == expected_region
    assert aggregate["NAWMReference"]["cpi_fallback"] == expected_nawm

    cbf_fragment = extract_per_map_lesion_stats(aggregate, "cbf")
    assert cbf_fragment is not None
    assert cbf_fragment["Regions"]["tumor"]["cpi_fallback"] == expected_region
    assert cbf_fragment["NAWMReference"]["cpi_fallback"] == expected_nawm

    auc_fragment = extract_per_map_lesion_stats(aggregate, "cbv_auc")
    assert auc_fragment is not None
    assert "cpi_fallback" not in auc_fragment["Regions"]["tumor"]
    assert "cpi_fallback" not in auc_fragment["NAWMReference"]

    no_fallback_dir = tmp_path / "without_fallback"
    no_fallback_dir.mkdir()
    no_fallback = DSCLesionStats()
    no_fallback.inputs.subject_id = "01"
    no_fallback.inputs.session_id = "01"
    no_fallback.inputs.lesion_mask = iface.inputs.lesion_mask
    no_fallback.inputs.cbf_map = iface.inputs.cbf_map
    no_fallback_result = no_fallback.run(cwd=no_fallback_dir)
    no_fallback_aggregate = json.loads(
        Path(no_fallback_result.outputs.stats_json).read_text()
    )
    assert "cpi_fallback" not in no_fallback_aggregate["Regions"]["tumor"]


class TestDCELesionStatsMapKeyCoverage:
    """Model-free DCE maps get LesionStats."""

    def test_mf_dispatcher_covers_intended_maps(self):
        from pmriprep.workflows.dce.datasinks import (
            _MF_LESIONSTATS_MAP_KEY,
        )

        expected = {
            "iauc60_map": "iauc60",
            "iauc90_map": "iauc90",
            "iauc120_map": "iauc120",
            "peak_enhancement_map": "peak_enhancement",
            "time_to_peak_map": "ttp",
            "ser_map": "ser",
        }
        assert _MF_LESIONSTATS_MAP_KEY == expected


class TestDCEStatisticsMapSupport:
    """DCE statistics use the fit-quality support without changing maps."""

    @staticmethod
    def _save(tmp_path, name, data):
        import nibabel as nib

        path = tmp_path / name
        nib.save(
            nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4)),
            path,
        )
        return str(path)

    def test_pk_and_model_free_maps_retain_native_support(self, tmp_path) -> None:
        from pmriprep.interfaces.dce_lesion_stats import DCELesionStats

        values = np.arange(1, 5, dtype=np.float32).reshape((2, 2, 1))
        iface = DCELesionStats()
        iface.inputs.ktrans_map = self._save(
            tmp_path, "ktrans.nii.gz", values
        )
        iface.inputs.iauc60_map = self._save(
            tmp_path, "iauc60.nii.gz", values + 10
        )
        iface.inputs.fp_map = self._save(tmp_path, "fp.nii.gz", values + 20)
        iface.inputs.ps_map = self._save(tmp_path, "ps.nii.gz", values + 30)
        iface.inputs.e_map = self._save(tmp_path, "e.nii.gz", values / 10)

        maps = iface._load_maps()

        np.testing.assert_array_equal(maps["ktrans"], values)
        np.testing.assert_array_equal(maps["iauc60"], values + 10)
        np.testing.assert_array_equal(maps["fp"], values + 20)
        np.testing.assert_array_equal(maps["ps"], values + 30)
        np.testing.assert_allclose(maps["e"], values / 10)
        pk_stats = compute_map_statistics(
            maps["ktrans"], np.ones_like(values, dtype=bool)
        )
        model_free_stats = compute_map_statistics(
            maps["iauc60"], np.ones_like(values, dtype=bool)
        )
        assert pk_stats is not None
        assert pk_stats.n_valid == pk_stats.n_total == 4
        assert pk_stats.median == pytest.approx(2.5)
        assert model_free_stats is not None
        assert model_free_stats.n_valid == model_free_stats.n_total == 4
        assert model_free_stats.median == pytest.approx(12.5)

    def test_qc_mask_filters_lesion_and_nawm_values(self, tmp_path) -> None:
        """Lesion and NAWM summaries use the same QC-passing population."""
        import json

        from pmriprep.interfaces.dce_lesion_stats import DCELesionStats

        shape = (2, 2, 1)
        values = np.arange(1, 5, dtype=np.float32).reshape(shape)
        iface = DCELesionStats()
        iface.inputs.subject_id = "01"
        iface.inputs.session_id = "01"
        iface.inputs.lesion_mask = self._save(
            tmp_path, "lesion.nii.gz", np.ones(shape)
        )
        iface.inputs.nawm_mask = self._save(
            tmp_path, "nawm.nii.gz", np.ones(shape)
        )
        iface.inputs.qc_mask = self._save(
            tmp_path, "qc.nii.gz", np.array([[[1], [0]], [[1], [0]]])
        )
        iface.inputs.ktrans_map = self._save(
            tmp_path, "ktrans.nii.gz", values
        )

        iface.run(cwd=tmp_path)
        result = json.loads(
            (tmp_path / "sub-01_ses-01_lesion-stats.json").read_text()
        )

        lesion = result["Regions"]["tumor"]["perfusion_maps"]["ktrans"]
        reference = result["NAWMReference"]["ktrans"]
        qc_lesion = result["QCFilteredRegions"]["tumor"]["perfusion_maps"][
            "ktrans"
        ]
        qc_reference = result["QCFilteredNAWMReference"]["ktrans"]
        assert lesion["n_valid"] == reference["n_valid"] == 4
        assert lesion["median"] == reference["median"] == pytest.approx(2.5)
        assert qc_lesion["n_total"] == qc_reference["n_total"] == 4
        assert qc_lesion["n_valid"] == qc_reference["n_valid"] == 2
        assert qc_lesion["median"] == qc_reference["median"] == pytest.approx(2.0)
        assert set(result["QCFilteredRegions"]["tumor"]) == {"perfusion_maps"}
        assert "voxel_count" not in result["QCFilteredNAWMReference"]
        assert "volume_ml" not in result["QCFilteredNAWMReference"]

    def test_qc_mask_does_not_filter_model_free_maps(self, tmp_path) -> None:
        """PK fit quality must not condition model-free DCE statistics."""
        import json

        from pmriprep.interfaces.dce_lesion_stats import DCELesionStats

        shape = (2, 2, 1)
        values = np.arange(1, 5, dtype=np.float32).reshape(shape)
        iface = DCELesionStats()
        iface.inputs.subject_id = "01"
        iface.inputs.session_id = "01"
        iface.inputs.lesion_mask = self._save(
            tmp_path, "lesion.nii.gz", np.ones(shape)
        )
        iface.inputs.nawm_mask = self._save(
            tmp_path, "nawm.nii.gz", np.ones(shape)
        )
        iface.inputs.qc_mask = self._save(
            tmp_path, "qc.nii.gz", np.array([[[1], [0]], [[1], [0]]])
        )
        iface.inputs.ktrans_map = self._save(
            tmp_path, "ktrans.nii.gz", values
        )
        iface.inputs.iauc60_map = self._save(
            tmp_path, "iauc60.nii.gz", values
        )

        iface.run(cwd=tmp_path)
        result = json.loads(
            (tmp_path / "sub-01_ses-01_lesion-stats.json").read_text()
        )

        native_maps = result["Regions"]["tumor"]["perfusion_maps"]
        qc_maps = result["QCFilteredRegions"]["tumor"]["perfusion_maps"]
        assert native_maps["iauc60"]["n_valid"] == 4
        assert qc_maps["ktrans"]["n_valid"] == 2
        assert "iauc60" not in qc_maps
        assert "iauc60" not in result["QCFilteredNAWMReference"]

    def test_dce_map_registry_classifies_every_loaded_map(self) -> None:
        """The single loader registry explicitly classifies PK-QC use."""
        from pmriprep.interfaces.dce_lesion_stats import DCELesionStats

        names = [name for _, name, _ in DCELesionStats._MAP_ATTRS]
        assert len(names) == len(set(names))
        classification = {
            name: apply_pk_qc
            for _, name, apply_pk_qc in DCELesionStats._MAP_ATTRS
        }
        assert classification == {
            "ktrans": True,
            "ve": True,
            "vp": True,
            "kep": True,
            "fp": True,
            "ps": True,
            "e": True,
            "iauc60": False,
            "iauc90": False,
            "iauc120": False,
            "peak_enhancement": False,
            "ser": False,
            "ttp": False,
        }

    def test_qc_mask_geometry_must_match_maps(self, tmp_path) -> None:
        """QC-filtered companions reject a mismatched mask grid."""
        from pmriprep.interfaces.dce_lesion_stats import DCELesionStats

        iface = DCELesionStats()
        iface.inputs.ktrans_map = self._save(
            tmp_path, "ktrans.nii.gz", np.ones((2, 2, 1))
        )
        iface.inputs.qc_mask = self._save(
            tmp_path, "qc.nii.gz", np.ones((3, 2, 1))
        )

        maps = iface._load_maps()
        with pytest.raises(ValueError, match="QC mask geometry.*ktrans"):
            iface._qc_filtered_maps(maps)

    def test_all_iauc_windows_are_nawm_references(self, tmp_path) -> None:
        """Every published IAUC window receives the same NAWM context."""
        import json

        from pmriprep.interfaces.dce_lesion_stats import DCELesionStats

        shape = (2, 2, 1)
        iface = DCELesionStats()
        iface.inputs.subject_id = "01"
        iface.inputs.session_id = "01"
        iface.inputs.lesion_mask = self._save(
            tmp_path, "lesion.nii.gz", np.ones(shape)
        )
        iface.inputs.nawm_mask = self._save(
            tmp_path, "nawm.nii.gz", np.ones(shape)
        )
        for seconds, value in ((60, 0.03), (90, 0.04), (120, 0.05)):
            setattr(
                iface.inputs,
                f"iauc{seconds}_map",
                self._save(tmp_path, f"iauc{seconds}.nii.gz", np.full(shape, value)),
            )

        result = iface.run(cwd=tmp_path)
        stats = json.loads((tmp_path / "sub-01_ses-01_lesion-stats.json").read_text())

        assert result.outputs.stats_json.endswith("_lesion-stats.json")
        for seconds, value in ((60, 0.03), (90, 0.04), (120, 0.05)):
            reference = stats["NAWMReference"][f"iauc{seconds}"]
            assert reference["median"] == pytest.approx(value)
            assert reference["std"] == pytest.approx(0.0)

class TestValidateMaskNdim:
    """Tests for validate_mask_ndim()."""

    def test_3d_passes(self, tmp_path) -> None:
        """3D mask raises no error."""
        import nibabel as nib
        img = nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.float32), np.eye(4))
        path = tmp_path / "mask_3d.nii.gz"
        nib.save(img, path)
        validate_mask_ndim(path, "test_mask")

    def test_4d_raises(self, tmp_path) -> None:
        """4D mask raises ValueError."""
        import nibabel as nib
        img = nib.Nifti1Image(np.ones((5, 5, 5, 3), dtype=np.float32), np.eye(4))
        path = tmp_path / "mask_4d.nii.gz"
        nib.save(img, path)
        with pytest.raises(ValueError, match="expected a 3D mask.*got 4D"):
            validate_mask_ndim(path, "test_mask")

    def test_5d_raises(self, tmp_path) -> None:
        """5D mask raises ValueError."""
        import nibabel as nib
        img = nib.Nifti1Image(np.ones((5, 5, 5, 3, 2), dtype=np.float32), np.eye(4))
        path = tmp_path / "mask_5d.nii.gz"
        nib.save(img, path)
        with pytest.raises(ValueError, match="expected a 3D mask.*got 5D"):
            validate_mask_ndim(path, "test_mask")

    def test_error_includes_source_name(self, tmp_path) -> None:
        """Error message includes source_name."""
        import nibabel as nib
        img = nib.Nifti1Image(np.ones((5, 5, 5, 2), dtype=np.float32), np.eye(4))
        path = tmp_path / "bad.nii.gz"
        nib.save(img, path)
        with pytest.raises(ValueError, match="my_lesion_mask"):
            validate_mask_ndim(path, "my_lesion_mask")

    def test_2d_raises(self, tmp_path) -> None:
        """2D mask raises ValueError."""
        import nibabel as nib
        img = nib.Nifti1Image(np.ones((5, 5), dtype=np.float32), np.eye(4))
        path = tmp_path / "mask_2d.nii.gz"
        nib.save(img, path)
        with pytest.raises(ValueError, match="expected a 3D mask.*got 2D"):
            validate_mask_ndim(path, "test_mask")

    def test_accepts_str_path(self, tmp_path) -> None:
        """Accepts str path without crashing."""
        import nibabel as nib
        img = nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.float32), np.eye(4))
        path = tmp_path / "mask_3d.nii.gz"
        nib.save(img, path)
        validate_mask_ndim(str(path), "test_mask")


class TestIsIntegerValued:
    """Tests for is_integer_valued()."""

    def test_integer_dtype(self) -> None:
        assert is_integer_valued(np.array([0, 1, 2, 3], dtype=np.int16))

    def test_float_with_integer_values(self) -> None:
        assert is_integer_valued(np.array([0.0, 1.0, 2.0, 3.0]))

    def test_float_with_non_integer_values(self) -> None:
        assert not is_integer_valued(np.array([0.0, 0.5, 1.0, 2.5]))

    def test_all_zeros(self) -> None:
        assert is_integer_valued(np.zeros(10))

    def test_nan_ignored_finite_values_checked(self) -> None:
        assert is_integer_valued(np.array([0.0, 1.0, np.nan]))

    def test_inf_ignored_finite_values_checked(self) -> None:
        assert is_integer_valued(np.array([0.0, 1.0, np.inf]))

    def test_empty_array(self) -> None:
        assert is_integer_valued(np.array([]))


class TestMultiLabelRoiStats:
    """Tests for multi-label ROI stats via force_binary path."""

    @pytest.fixture(autouse=True)
    def _run_in_tmp(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

    def _make_mask(self, tmp_path, data: np.ndarray) -> str:
        import nibabel as nib
        img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
        path = tmp_path / "roi_mask.nii.gz"
        nib.save(img, path)
        return str(path)

    def _make_map(self, tmp_path, shape: tuple, value: float, name: str) -> str:
        import nibabel as nib
        data = np.full(shape, value, dtype=np.float32)
        path = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), path)
        return str(path)

    def test_multi_label_produces_per_label_regions(self, tmp_path) -> None:
        """3-label mask with force_binary=True produces label_1, label_2, label_3."""
        from pmriprep.interfaces.dsc_lesion_stats import DSCLesionStats

        shape = (10, 10, 10)
        mask_data = np.zeros(shape)
        mask_data[1:3, 1:3, 1:3] = 1
        mask_data[4:6, 4:6, 4:6] = 2
        mask_data[7:9, 7:9, 7:9] = 3

        iface = DSCLesionStats()
        iface.inputs.lesion_mask = self._make_mask(tmp_path, mask_data)
        iface.inputs.subject_id = "01"
        iface.inputs.session_id = "01"
        iface.inputs.force_binary = True
        iface.inputs.cbv_map = self._make_map(tmp_path, shape, 2.0, "cbv")

        result = iface.run()
        import json
        stats = json.loads(open(result.outputs.stats_json).read())
        regions = stats["Regions"]
        assert "label_1" in regions
        assert "label_2" in regions
        assert "label_3" in regions
        assert "roi" not in regions
        assert stats["LesionMaskType"] == "multi_label_roi"

    def test_non_contiguous_labels(self, tmp_path) -> None:
        """Non-contiguous labels (1, 3, 7) produce correct keys."""
        from pmriprep.interfaces.dsc_lesion_stats import DSCLesionStats

        shape = (10, 10, 10)
        mask_data = np.zeros(shape)
        mask_data[1:3, 1:3, 1:3] = 1
        mask_data[4:6, 4:6, 4:6] = 3
        mask_data[7:9, 7:9, 7:9] = 7

        iface = DSCLesionStats()
        iface.inputs.lesion_mask = self._make_mask(tmp_path, mask_data)
        iface.inputs.subject_id = "01"
        iface.inputs.session_id = "01"
        iface.inputs.force_binary = True
        iface.inputs.cbv_map = self._make_map(tmp_path, shape, 2.0, "cbv")

        result = iface.run()
        import json
        stats = json.loads(open(result.outputs.stats_json).read())
        region_keys = set(stats["Regions"].keys())
        assert region_keys == {"label_1", "label_3", "label_7"}

    def test_binary_mask_backward_compat(self, tmp_path) -> None:
        """Single-label binary mask still produces single 'roi' region."""
        from pmriprep.interfaces.dsc_lesion_stats import DSCLesionStats

        shape = (10, 10, 10)
        mask_data = np.zeros(shape)
        mask_data[2:5, 2:5, 2:5] = 1

        iface = DSCLesionStats()
        iface.inputs.lesion_mask = self._make_mask(tmp_path, mask_data)
        iface.inputs.subject_id = "01"
        iface.inputs.session_id = "01"
        iface.inputs.force_binary = True
        iface.inputs.cbv_map = self._make_map(tmp_path, shape, 2.0, "cbv")

        result = iface.run()
        import json
        stats = json.loads(open(result.outputs.stats_json).read())
        assert set(stats["Regions"].keys()) == {"roi"}
        assert stats["LesionMaskType"] == "binary"

    def test_float_mask_falls_back_to_binary(self, tmp_path) -> None:
        """Float-valued mask with max >= 2 falls back to binary."""
        from pmriprep.interfaces.dsc_lesion_stats import DSCLesionStats

        shape = (10, 10, 10)
        mask_data = np.zeros(shape)
        mask_data[1:3, 1:3, 1:3] = 0.5
        mask_data[4:6, 4:6, 4:6] = 2.7

        iface = DSCLesionStats()
        iface.inputs.lesion_mask = self._make_mask(tmp_path, mask_data)
        iface.inputs.subject_id = "01"
        iface.inputs.session_id = "01"
        iface.inputs.force_binary = True
        iface.inputs.cbv_map = self._make_map(tmp_path, shape, 2.0, "cbv")

        result = iface.run()
        import json
        stats = json.loads(open(result.outputs.stats_json).read())
        assert set(stats["Regions"].keys()) == {"roi"}
        assert stats["LesionMaskType"] == "binary"


@pytest.mark.parametrize('modality', ['dsc', 'dce'])
def test_regional_statistics_exclude_background_but_keep_measured_zero(tmp_path, modality) -> None:
    """The image support, not a positive-value test or fit QC, defines coverage."""
    import json
    import nibabel as nib
    from pmriprep.interfaces.dsc_lesion_stats import DSCLesionStats
    from pmriprep.interfaces.dce_lesion_stats import DCELesionStats
    from pmriprep.utils.lesion_stats import extract_per_map_lesion_stats

    def save(name, values):
        path = tmp_path / (name + '.nii.gz')
        nib.save(nib.Nifti1Image(np.array(values, dtype=np.float32).reshape(2, 2, 1), np.eye(4)), path)
        return str(path)

    inputs = dict(
        subject_id='test', session_id='01', force_binary=True,
        lesion_mask=save('roi', [1, 1, 1, 1]),
        nawm_mask=save('nawm', [1, 1, 1, 1]),
        brain_mask=save('brain', [1, 1, 1, 0]),
    )
    data = save('parameter', [0, 2, np.nan, 0])
    interface = DSCLesionStats if modality == 'dsc' else DCELesionStats
    key = 'cbv' if modality == 'dsc' else 'ktrans'
    inputs[key + '_map'] = data
    output = interface(**inputs).run(cwd=str(tmp_path))
    result = json.loads(Path(output.outputs.stats_json).read_text())
    fragment = extract_per_map_lesion_stats(result, key)
    region = fragment['Regions']['roi']
    assert region['voxel_count'] == 4
    assert region['outside_brain_mask_voxels'] == 1
    assert region['stats']['n_valid'] == 2
    assert region['stats']['n_total'] == 4
    assert region['stats']['mean'] == 1.0
    assert fragment['NAWMReference']['outside_brain_mask_voxels'] == 1
    assert fragment['NAWMReference']['n_valid'] == 2
    assert nib.load(data).get_fdata().flat[3] == 0
