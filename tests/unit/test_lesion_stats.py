"""Unit tests for lesion/tumor ROI statistics extraction utilities."""

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
        # New per-map dict shape (Phase 1b refactor):
        assert "ncbv" in derived["fractional_volumes"]
        ncbv_fv = derived["fractional_volumes"]["ncbv"]
        # Phase 2d expanded thresholds: nCBV {1.75, 2.5, 3.0, 4.0}
        for t in (1.75, 2.5, 3.0, 4.0):
            assert f"above_{t}" in ncbv_fv
        # hotspot_summary deliberately removed — per-map p98 is the canonical
        # hotspot now and lives on each MapStatistics block.
        assert "hotspot_summary" not in derived

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
            nktrans_data=None,
            ve_data=ve,
            vp_data=None,
            mask=mask,
        )
        assert "fractional_volumes" in derived
        assert "Ktrans" in derived["fractional_volumes"]
        ktrans_fv = derived["fractional_volumes"]["Ktrans"]
        # Phase 2d expanded thresholds: Ktrans {0.05, 0.1, 0.2}
        for t in (0.05, 0.1, 0.2):
            assert f"above_{t}" in ktrans_fv
        # hotspot_summary removed — per-map p98 is canonical now.
        assert "hotspot_summary" not in derived


class TestComputeNAWMReference:
    """Tests for compute_nawm_reference()."""

    def test_nawm_stats(self) -> None:
        """Maps + NAWM mask produces per-map stats with value-level QC by field strength."""
        nawm_mask = np.zeros((10, 10, 10), dtype=bool)
        nawm_mask[1:9, 1:9, 1:9] = True  # 512 voxels
        rng = np.random.default_rng(42)
        cbv_data = rng.uniform(2.0, 4.0, (10, 10, 10))

        maps = {"cbv": cbv_data}
        voxel_sizes = (1.0, 1.0, 1.0)
        ref = compute_nawm_reference(
            nawm_mask, maps, voxel_sizes, field_strength_t=3.0,
        )

        assert ref["voxel_count"] == 512
        assert "cbv" in ref
        assert "median" in ref["cbv"]
        assert "std" in ref["cbv"]
        # CBV ~3 lies in the 3T range (1.5, 4) → PASS at value level.
        assert ref["value_qc_status"] == "PASS"
        # Per-map verdict also surfaces:
        assert ref["cbv"]["value_qc_status"] == "PASS"
        # No mask-size editorial — voxel_count is informational only.
        assert "mask_qc_status" not in ref

    def test_extreme_cbv_warns(self) -> None:
        """Extreme CBV median produces value_qc_status=WARNING (3T table)."""
        nawm_mask = np.zeros((10, 10, 10), dtype=bool)
        nawm_mask[1:9, 1:9, 1:9] = True  # 512 voxels
        # CBV far outside 3T range (1.5, 4)
        cbv_data = np.full((10, 10, 10), 15.0)

        maps = {"cbv": cbv_data}
        voxel_sizes = (1.0, 1.0, 1.0)
        ref = compute_nawm_reference(
            nawm_mask, maps, voxel_sizes, field_strength_t=3.0,
        )

        assert ref["value_qc_status"] == "WARNING"
        assert "cbv" in ref["value_qc_notes"]
        # Hematocrit annotation surfaces in the QC notes.
        assert "hematocrit" in ref["value_qc_notes"].lower()

    def test_unknown_field_strength_skips_value_qc(self) -> None:
        """field_strength_t=None reports value_qc_status=UNKNOWN."""
        nawm_mask = np.zeros((10, 10, 10), dtype=bool)
        nawm_mask[1:9, 1:9, 1:9] = True
        cbv_data = np.full((10, 10, 10), 15.0)  # would WARN at 3T

        maps = {"cbv": cbv_data}
        ref = compute_nawm_reference(
            nawm_mask, maps, (1.0, 1.0, 1.0), field_strength_t=None,
        )

        assert ref["value_qc_status"] == "UNKNOWN"
        assert "Field strength unknown" in ref["value_qc_notes"]

    def test_field_strength_outside_validated_range_skips(self) -> None:
        """7T (outside [1.5, 3.0]) reports value_qc_status=UNKNOWN, no auto-gate."""
        nawm_mask = np.zeros((10, 10, 10), dtype=bool)
        nawm_mask[1:9, 1:9, 1:9] = True
        cbv_data = np.full((10, 10, 10), 5.0)

        maps = {"cbv": cbv_data}
        ref = compute_nawm_reference(
            nawm_mask, maps, (1.0, 1.0, 1.0), field_strength_t=7.0,
        )

        assert ref["value_qc_status"] == "UNKNOWN"
        assert "7" in ref["value_qc_notes"]
        assert "outside validated range" in ref["value_qc_notes"]

    def test_small_nawm_mask_emits_size_without_editorial(self) -> None:
        """Small NAWM mask: voxel_count + volume_ml emitted; no mask_qc_status fabricated.

        NAWM may come from upstream auto-NAWM (config.tumor_seg.nawm_min_voxels=500),
        bilateral fallback, OR a user-supplied mask via the force_binary ROI path.
        Each path has its own validity contract, so this function reports size as
        informational only and does not editorialize.
        """
        nawm_mask = np.zeros((5, 5, 5), dtype=bool)
        nawm_mask[0, 0, 0] = True  # 1 voxel
        cbv_data = np.full((5, 5, 5), 2.5)

        ref = compute_nawm_reference(
            nawm_mask, {"cbv": cbv_data}, (1.0, 1.0, 1.0),
            field_strength_t=3.0,
        )
        assert ref["voxel_count"] == 1
        assert "mask_qc_status" not in ref
        assert "mask_qc_notes" not in ref

    def test_no_maps_in_range_table_reports_unknown_not_pass(self) -> None:
        """Field strength provided but emitted maps have no range entries → UNKNOWN.

        Reproduces the M1 bug from the v0.50.0 code review: previously
        compute_nawm_reference returned ``value_qc_status="PASS"`` with a
        misleading "All NAWM medians within ranges" note when zero maps
        had registered ranges (e.g., DCE-only modality with kep / iauc60).
        """
        nawm_mask = np.zeros((10, 10, 10), dtype=bool)
        nawm_mask[1:9, 1:9, 1:9] = True
        # kep and peak_enhancement aren't in NAWM_RANGES_3T
        maps = {
            "kep": np.full((10, 10, 10), 0.5),
            "peak_enhancement": np.full((10, 10, 10), 50.0),
        }
        ref = compute_nawm_reference(
            nawm_mask, maps, (1.0, 1.0, 1.0), field_strength_t=3.0,
        )
        assert ref["value_qc_status"] == "UNKNOWN"
        assert "No emitted maps" in ref["value_qc_notes"]


class TestSliceNawmForMap:
    """Phase 1a: per-map NAWM slicing avoids cross-map qc_notes contamination."""

    def test_pass_map_does_not_carry_sibling_warning_notes(self) -> None:
        """A PASSing map must not ship modality-level notes about sibling violations."""
        from pmriprep.utils.lesion_stats import _slice_nawm_for_map

        nawm = {
            "voxel_count": 1000, "volume_ml": 50.0,
            "ncbv": {"median": 1.0, "std": 0.1, "p25": 0.95, "p75": 1.05,
                     "value_qc_status": "PASS"},
            "cbv": {"median": 0.34, "std": 0.05, "p25": 0.30, "p75": 0.38,
                    "value_qc_status": "WARNING"},
            "value_qc_status": "WARNING",
            "value_qc_notes": "Out-of-range NAWM medians: cbv. Hematocrit=0.45.",
        }
        # ncbv individually PASSed → must not carry the sibling violation note
        out = _slice_nawm_for_map(nawm, "ncbv")
        assert out["value_qc_status"] == "PASS"
        assert "value_qc_notes" not in out

    def test_warning_map_does_carry_modality_notes(self) -> None:
        """A WARNING map should surface the modality-level violation context."""
        from pmriprep.utils.lesion_stats import _slice_nawm_for_map

        nawm = {
            "voxel_count": 1000, "volume_ml": 50.0,
            "cbv": {"median": 0.34, "std": 0.05, "p25": 0.30, "p75": 0.38,
                    "value_qc_status": "WARNING"},
            "value_qc_status": "WARNING",
            "value_qc_notes": "Out-of-range NAWM medians: cbv. Hematocrit=0.45.",
        }
        out = _slice_nawm_for_map(nawm, "cbv")
        assert out["value_qc_status"] == "WARNING"
        assert "Out-of-range" in out["value_qc_notes"]

    def test_map_without_per_map_verdict_inherits_modality_status(self) -> None:
        """When per-map QC wasn't applied (no range entry), surface modality status."""
        from pmriprep.utils.lesion_stats import _slice_nawm_for_map

        nawm = {
            "voxel_count": 1000, "volume_ml": 50.0,
            "kep": {"median": 0.5, "std": 0.1, "p25": 0.4, "p75": 0.6},
            "value_qc_status": "UNKNOWN",
            "value_qc_notes": "Field strength unknown; NAWM ranges not applied.",
        }
        out = _slice_nawm_for_map(nawm, "kep")
        # Inherits modality-level UNKNOWN (no per-map verdict on kep)
        assert out["value_qc_status"] == "UNKNOWN"
        assert "Field strength unknown" in out["value_qc_notes"]


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
    """PerfusionProcessing.AIF sidecar block (v0.29.0)."""

    def test_measured_aif_provenance(self) -> None:
        aif_block = {
            "Source": "measured",
            "DetectionLevel": "atlas_strict",
            "QualityFlag": "pass",
            "GammaFitApplied": False,
            "CrossSubjectComparable": True,
        }
        result = LesionStatisticsResult(
            subject_id="0001", session_id="0002", modality="DSC",
            mask_type="binary", mask_source="m.nii.gz", total_voxels=500,
            aif_provenance=aif_block,
        )
        d = result.to_dict()
        assert d["PerfusionProcessing"]["AIF"]["Source"] == "measured"
        assert d["PerfusionProcessing"]["AIF"]["CrossSubjectComparable"] is True

    def test_template_aif_provenance(self) -> None:
        aif_block = {
            "Source": "template",
            "DetectionLevel": "synthetic",
            "QualityFlag": "poor",
            "GammaFitApplied": False,
            "CrossSubjectComparable": False,
            "SyntheticFallbackReason": "no_candidates",
        }
        result = LesionStatisticsResult(
            subject_id="0001", session_id="0002", modality="DSC",
            mask_type="binary", mask_source="m.nii.gz", total_voxels=500,
            aif_provenance=aif_block,
        )
        d = result.to_dict()
        assert d["PerfusionProcessing"]["AIF"]["Source"] == "template"
        assert d["PerfusionProcessing"]["AIF"]["SyntheticFallbackReason"] == "no_candidates"

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

    These helpers feed ``traits.Str`` consumers in DCE/DSC lesion-stats
    (``nktrans_map``, ``niauc60_map``, ``ncbv_map``, ``ncbf_map``). Returning
    ``None`` triggers a TraitError that under the MultiProc plugin manifests
    as a silent dispatch hang (no work-dir created, no log line) — the same
    failure mode :class:`TestCombineTumorMasksReturnsEmptyString` documents.
    """

    def test_normalize_by_nawm_no_inputs(self) -> None:
        from pmriprep.workflows.dce.base import _normalize_by_nawm

        assert _normalize_by_nawm(param_map=None, nawm_mask=None) == ""

    def test_normalize_by_nawm_nonexistent_nawm(self) -> None:
        from pmriprep.workflows.dce.base import _normalize_by_nawm

        assert _normalize_by_nawm(
            param_map="/nonexistent/ktrans.nii.gz",
            nawm_mask="/nonexistent/nawm.nii.gz",
        ) == ""

    def test_normalize_cbv_no_nawm(self) -> None:
        from pmriprep.workflows.dsc.base import _normalize_cbv

        assert _normalize_cbv(
            cbv_file="/nonexistent/cbv.nii.gz",
            brain_mask_file=None,
            derivatives_dir="/tmp",
            subject_id="0081",
            session_id="0001",
            nawm_mask_file=None,
        ) == ("", None)

    def test_normalize_cbf_no_nawm(self) -> None:
        from pmriprep.workflows.dsc.base import _normalize_cbf

        assert _normalize_cbf(
            cbf_file="/nonexistent/cbf.nii.gz",
            brain_mask_file=None,
            derivatives_dir="/tmp",
            subject_id="0081",
            session_id="0001",
            nawm_mask_file=None,
        ) == ("", None)


class TestLesionStatsAcceptsEmptyMapTrait:
    """Dispatch-time invariant: setting ``""`` on a normalized-map trait must not
    raise TraitError.

    The plain pure-function tests above lock the producer contract; this test
    locks the consumer contract. If a future refactor re-widens
    :func:`_normalize_by_nawm` (or its DSC siblings) to return ``None`` while
    the consumer trait is still strict ``traits.Str``, the silent MultiProc
    hang documented in 0.50.4's commit message resurfaces. This guards
    against that regression.
    """

    def test_dce_lesion_stats_accepts_empty_string_for_normalized_maps(self) -> None:
        from pmriprep.interfaces.dce_lesion_stats import DCELesionStats

        iface = DCELesionStats()
        iface.inputs.subject_id = "0081"
        iface.inputs.session_id = "0001"
        # Should not raise TraitError — these are the values produced by
        # _normalize_by_nawm on the no-NAWM path.
        iface.inputs.nktrans_map = ""
        iface.inputs.niauc60_map = ""
        assert iface.inputs.nktrans_map == ""
        assert iface.inputs.niauc60_map == ""

    def test_dsc_lesion_stats_accepts_empty_string_for_normalized_maps(self) -> None:
        from pmriprep.interfaces.dsc_lesion_stats import DSCLesionStats

        iface = DSCLesionStats()
        iface.inputs.subject_id = "0081"
        iface.inputs.session_id = "0001"
        # Should not raise TraitError — these are the values produced by
        # _normalize_cbv / _normalize_cbf on the no-NAWM path.
        iface.inputs.ncbv_map = ""
        iface.inputs.ncbf_map = ""
        assert iface.inputs.ncbv_map == ""
        assert iface.inputs.ncbf_map == ""


class TestMapStatisticsHistogramMetrics:
    """Phase 2a: p98/p99/skewness/kurtosis/entropy + compute_advanced gating."""

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
        assert s.entropy is not None
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
        assert s.entropy is None

    def test_advanced_metrics_omitted_when_n_valid_below_10(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.ones_like(data, dtype=bool)
        s = compute_map_statistics(data, mask, compute_advanced=True)
        assert s.p98 is None
        assert s.skewness is None

    def test_constant_data_skew_kurt_entropy_none(self) -> None:
        data = np.ones(200)
        mask = np.ones_like(data, dtype=bool)
        s = compute_map_statistics(data, mask, compute_advanced=True)
        assert s.skewness is None
        assert s.kurtosis is None
        assert s.entropy is None


class TestPPP:
    """Phase 2c: percent positive pixels for permeability biomarkers."""

    def test_ktrans_ppp_above_0p01(self) -> None:
        # 5 of 12 voxels strictly > 0.01: {0.02, 0.05, 0.1, 0.15, 0.2}
        data = np.array([0.0, 0.005, 0.02, 0.05, 0.1, 0.15, 0.2,
                         0.0, 0.0, 0.0, 0.0, 0.0])
        mask = np.ones_like(data, dtype=bool)
        s = compute_map_statistics(data, mask, map_name="ktrans")
        assert abs(s.ppp - 5 / 12) < 1e-3

    def test_vp_ppp_above_0p001(self) -> None:
        # 3 of 6 voxels strictly > 0.001: {0.002, 0.005, 0.01}
        data = np.array([0.0, 0.0005, 0.002, 0.005, 0.01, 0.0])
        mask = np.ones_like(data, dtype=bool)
        s = compute_map_statistics(data, mask, map_name="vp")
        assert s.ppp == 0.5

    def test_k2_ppp_above_0p001(self) -> None:
        # K2 noise floor at 0.001 s⁻¹ (Bjornerud 2011)
        data = np.array([0.0, 0.0005, 0.001, 0.002, 0.005])
        mask = np.ones_like(data, dtype=bool)
        s = compute_map_statistics(data, mask, map_name="k2")
        # 2 of 5 strictly > 0.001
        assert abs(s.ppp - 0.4) < 1e-3

    def test_non_permeability_map_no_ppp(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.ones_like(data, dtype=bool)
        s = compute_map_statistics(data, mask, map_name="cbv")
        assert s.ppp is None


class TestCohensDAugmentation:
    """Phase 2b: pooled-SD Cohen's d vs NAWM."""

    def test_pooled_d_attached_when_nawm_matches(self) -> None:
        result = LesionStatisticsResult(
            subject_id="X", session_id="Y", modality="DSC",
            mask_type="binary", mask_source="m.nii.gz", total_voxels=10,
        )
        region = RegionStatistics("tumor", 100, 100.0, 0.1)
        region.perfusion_maps["cbv"] = MapStatistics(
            median=2.5, std=0.8, n_valid=100, n_total=100,
        )
        result.regions["tumor"] = region
        result.nawm_reference = {
            "cbv": {"median": 1.5, "std": 0.4, "p25": 1.2, "p75": 1.8,
                    "value_qc_status": "PASS"},
        }
        result.augment_with_cohens_d()
        d = result.regions["tumor"].perfusion_maps["cbv"].cohens_d_vs_nawm
        # pooled SD = sqrt((0.64 + 0.16)/2) = sqrt(0.4) ≈ 0.6325
        # d = (2.5 - 1.5) / 0.6325 ≈ 1.5811
        assert abs(d - 1.5811) < 1e-3

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
    """Phase 1a: extract_per_map_lesion_stats forwards modality context."""

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
                        "k2": {"median": 0.0008, "std": 0.0005, "ppp": 0.12,
                               "n_valid": 100, "n_total": 100},
                        "ttp": {"median": 65.0, "std": 5.0,
                                "n_valid": 100, "n_total": 100},
                        "ttpCorrected": {"median": 5.0, "std": 1.0,
                                         "n_valid": 100, "n_total": 100},
                    },
                },
            },
            "NAWMReference": {
                "voxel_count": 1000, "volume_ml": 50.0,
                "ncbv": {"median": 1.0, "std": 0.1, "p25": 0.95, "p75": 1.05,
                         "value_qc_status": "PASS"},
                "cbv": {"median": 0.34, "std": 0.05, "p25": 0.30, "p75": 0.38,
                        "value_qc_status": "WARNING"},
                "mtt": {"median": 5.5, "std": 1.0, "p25": 4.8, "p75": 6.2,
                        "value_qc_status": "PASS"},
                "value_qc_status": "WARNING",
                "value_qc_notes": "Out-of-range NAWM medians: cbv. Hematocrit=0.45.",
            },
            "DerivedMetrics": {
                "fractional_volumes": {
                    "ncbv": {"above_1.75": 0.45, "above_2.5": 0.18,
                             "above_3.0": 0.09, "above_4.0": 0.02},
                    "Ktrans": {"above_0.05": 0.20, "above_0.1": 0.10},
                },
            },
        }

    def test_normalized_map_gets_direct_threshold_and_fv(self, aggregate):
        from pmriprep.utils.lesion_stats import extract_per_map_lesion_stats

        out = extract_per_map_lesion_stats(aggregate, "ncbv")
        assert out["LesionMaskType"] == "multi_label_tumor_seg"
        assert "Regions" in out
        assert out["ClinicalThreshold"] == {"high_grade": 1.75}
        assert "FractionalVolumes" in out
        assert "above_3.0" in out["FractionalVolumes"]
        # NAWM in normalized units, slice for ncbv only
        assert out["NAWMReference"]["median"] == 1.0
        assert out["NAWMReference"]["voxel_count"] == 1000

    def test_unnormalized_companion_gets_informational_breadcrumb(self, aggregate):
        from pmriprep.utils.lesion_stats import extract_per_map_lesion_stats

        out = extract_per_map_lesion_stats(aggregate, "cbv")
        assert out["ClinicalThreshold"] == {"high_grade": 1.75, "applies_to": "ncbv"}
        # FractionalVolumes does NOT live on the un-normalized companion
        assert "FractionalVolumes" not in out
        # NAWM in deconv units
        assert out["NAWMReference"]["median"] == 0.34

    def test_aif_corrected_flag_on_ttp_variants(self, aggregate):
        from pmriprep.utils.lesion_stats import extract_per_map_lesion_stats

        raw = extract_per_map_lesion_stats(aggregate, "ttp")
        corrected = extract_per_map_lesion_stats(aggregate, "ttpCorrected")
        assert raw["AIFCorrected"] is False
        assert corrected["AIFCorrected"] is True

    def test_map_with_no_threshold_no_fv_just_stats(self, aggregate):
        from pmriprep.utils.lesion_stats import extract_per_map_lesion_stats

        out = extract_per_map_lesion_stats(aggregate, "k2")
        assert "Regions" in out
        assert "ClinicalThreshold" not in out
        assert "FractionalVolumes" not in out
        # No NAWM entry for k2 → no NAWMReference
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


class TestDSCStatsKeyTtpCorrected:
    """Phase 1c: DSC interface stores camelCase 'ttpCorrected', not snake_case."""

    def test_dsc_load_maps_uses_camelcase_ttp(self):
        from pmriprep.interfaces.dsc_lesion_stats import DSCLesionStats

        # Inspect the source for the map_attrs list (no need to instantiate)
        import inspect
        src = inspect.getsource(DSCLesionStats._load_maps)
        assert '"ttpCorrected"' in src
        assert '"ttp_corrected"' not in src


class TestDCELesionStatsMapKeyCoverage:
    """Phase 1d: model-free + NAWM-norm DCE maps now get LesionStats."""

    def test_mf_dispatcher_covers_intended_maps(self):
        from pmriprep.workflows.dce.datasinks import (
            _MF_LESIONSTATS_MAP_KEY,
        )

        # Maps that SHOULD carry per-region LesionStats per Phase 1d.
        expected = {
            "iauc60_map": "iauc60",
            "iauc90_map": "iauc90",
            "iauc120_map": "iauc120",
            "peak_enhancement_map": "peak_enhancement",
            "time_to_peak_map": "ttp",
        }
        assert _MF_LESIONSTATS_MAP_KEY == expected

    def test_nawmnorm_dispatcher_covers_intended_maps(self):
        from pmriprep.workflows.dce.datasinks import (
            _NAWMNORM_LESIONSTATS_MAP_KEY,
        )

        assert _NAWMNORM_LESIONSTATS_MAP_KEY == {
            "nktrans_map": "nktrans",
            "niauc60_map": "niauc60",
        }
