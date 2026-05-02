"""DCE lesion/tumor ROI statistics extraction Nipype interface.

This module implements a Nipype interface for extracting comprehensive
statistics from DCE perfusion maps within lesion/tumor regions of interest.

Supports:
- Multi-label masks (HD-GLIO: 1=non-enhancing, 2=enhancing)
- Binary tumor masks
- NAWM reference values for normalization context
- Clinical interpretation thresholds
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import nibabel as nib
import numpy as np
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    traits,
)

from pmriprep.utils.lesion_stats import (
    LesionStatisticsResult,
    RegionStatistics,
    compute_dce_derived_metrics,
    compute_map_statistics,
    compute_nawm_reference,
    compute_region_volume,
    detect_mask_type,
    extract_label_mask,
    get_dce_clinical_context,
)


# Region label definitions (enhancing first for clinical priority)
TUMOR_SEG_LABELS = {
    2: "contrast_enhancing_tumor",
    1: "non_enhancing_tumor",
}


class DCELesionStatsInputSpec(BaseInterfaceInputSpec):
    """Input specification for DCE lesion statistics extraction."""

    # Required inputs
    lesion_mask = traits.Either(
        traits.Str, None, desc="Lesion/tumor mask in DCE space (optional)"
    )
    subject_id = traits.Str(mandatory=True, desc="Subject ID (without sub- prefix)")
    session_id = traits.Str(mandatory=True, desc="Session ID (without ses- prefix)")

    # Optional NAWM mask for reference values
    nawm_mask = traits.Either(
        traits.Str, None, desc="NAWM mask in DCE space for reference values"
    )

    # Core pharmacokinetic maps (all optional - stats computed for available maps)
    ktrans_map = traits.Either(traits.Str, None, desc="Ktrans map path (min^-1)")
    ve_map = traits.Either(traits.Str, None, desc="ve (EES volume fraction) map path")
    vp_map = traits.Either(traits.Str, None, desc="vp (plasma volume fraction) map path")
    kep_map = traits.Either(traits.Str, None, desc="kep (rate constant) map path")

    # Normalized maps (optional - only available when NAWM mask present)
    nktrans_map = traits.Either(traits.Str, None, desc="Normalized Ktrans map path")

    # IAUC maps (all optional)
    iauc60_map = traits.Either(traits.Str, None, desc="IAUC at 60 seconds map path")
    iauc90_map = traits.Either(traits.Str, None, desc="IAUC at 90 seconds map path")
    iauc120_map = traits.Either(traits.Str, None, desc="IAUC at 120 seconds map path")
    niauc60_map = traits.Either(traits.Str, None, desc="Normalized IAUC60 map path")

    # Mask origin: "tumor_segmentation" or "whole_tumor" (from combine_tumor_masks)
    mask_origin = traits.Str(
        "", usedefault=True,
        desc="Source of the lesion mask: 'tumor_segmentation' or 'whole_tumor'",
    )

    # ROI mode: treat mask as binary, label region as "roi", skip clinical context
    force_binary = traits.Bool(
        False, usedefault=True,
        desc="Force binary mask interpretation (for generic ROI stats)",
    )

    # Additional derived maps (all optional)
    peak_enhancement_map = traits.Either(traits.Str, None, desc="Peak enhancement map path")
    wash_in_map = traits.Either(traits.Str, None, desc="Wash-in rate map path")
    wash_out_map = traits.Either(traits.Str, None, desc="Wash-out rate map path")
    ttp_map = traits.Either(traits.Str, None, desc="Time to peak map path")
    auc_map = traits.Either(traits.Str, None, desc="Total AUC map path")


class DCELesionStatsOutputSpec(TraitedSpec):
    """Output specification for DCE lesion statistics extraction."""

    stats_json = traits.Str(desc="Path to JSON statistics file")


class DCELesionStats(BaseInterface):
    """Extract lesion/tumor ROI statistics from DCE perfusion maps.

    This interface extracts comprehensive statistics from DCE perfusion
    maps within lesion/tumor regions, supporting:

    - Multi-label masks (HD-GLIO: label 1 = non-enhancing, label 2 = enhancing)
    - Binary tumor masks
    - NAWM reference values for normalization context
    - Clinical interpretation thresholds (Ktrans > 0.1 = high-grade)

    Output files:
        {prefix}_lesion-stats.json  - Comprehensive JSON statistics
    """

    input_spec = DCELesionStatsInputSpec
    output_spec = DCELesionStatsOutputSpec

    def _run_interface(self, runtime: Any) -> Any:
        """Execute lesion statistics extraction."""
        # Initialize results dictionary
        self._results: Dict[str, str] = {}

        # Check if lesion mask is available
        if not self._has_input("lesion_mask"):
            # No lesion mask - skip processing
            self._results["stats_json"] = ""
            return runtime

        # Load lesion mask
        lesion_img = nib.load(self.inputs.lesion_mask)
        lesion_data = np.asarray(lesion_img.dataobj)  # type: ignore[attr-defined]
        voxel_sizes: Tuple[float, float, float] = lesion_img.header.get_zooms()[:3]  # type: ignore[attr-defined]

        # Load all available perfusion maps
        maps = self._load_maps()

        if self.inputs.force_binary:
            # ROI mode: treat as binary, label as "roi", skip clinical context
            mask_type = "binary"
            result = LesionStatisticsResult(
                subject_id=self.inputs.subject_id,
                session_id=self.inputs.session_id,
                modality="DCE",
                mask_type="binary",
                mask_source=Path(self.inputs.lesion_mask).name,
                total_voxels=int(np.sum(lesion_data > 0)),
            )
            binary_mask = lesion_data > 0
            region_stats = self._compute_region_stats(
                "roi", binary_mask, maps, voxel_sizes
            )
            if region_stats is not None:
                result.regions["roi"] = region_stats
        else:
            # Standard lesion stats mode
            mask_type, max_label = detect_mask_type(lesion_data)

            result = LesionStatisticsResult(
                subject_id=self.inputs.subject_id,
                session_id=self.inputs.session_id,
                modality="DCE",
                mask_type=mask_type,
                mask_source=Path(self.inputs.lesion_mask).name,
                total_voxels=int(np.sum(lesion_data > 0)),
            )

            # Extract statistics for each region
            if mask_type == "multi_label_tumor_seg":
                for label, region_name in TUMOR_SEG_LABELS.items():
                    if label > max_label:
                        continue
                    label_mask = extract_label_mask(lesion_data, label)
                    if np.sum(label_mask) == 0:
                        continue

                    region_stats = self._compute_region_stats(
                        region_name, label_mask, maps, voxel_sizes
                    )
                    if region_stats is not None:
                        result.regions[region_name] = region_stats
            else:
                binary_mask = lesion_data > 0
                # When mask comes from tumor_segmentation with only label 1
                # (non-enhancing), use the specific label name instead of
                # generic "tumor"
                mask_origin = self.inputs.mask_origin
                if mask_origin == "tumor_segmentation" and max_label == 1:
                    region_name = TUMOR_SEG_LABELS[1]  # "non_enhancing_tumor"
                else:
                    region_name = "tumor"
                region_stats = self._compute_region_stats(
                    region_name, binary_mask, maps, voxel_sizes
                )
                if region_stats is not None:
                    result.regions[region_name] = region_stats

            # Compute NAWM reference if mask provided
            if self._has_input("nawm_mask"):
                nawm_img = nib.load(self.inputs.nawm_mask)
                nawm_data = np.asarray(nawm_img.dataobj) > 0  # type: ignore[attr-defined]
                nawm_maps = {
                    k: v for k, v in maps.items()
                    if k in ["ktrans", "nktrans", "ve", "vp", "kep"]
                }
                result.nawm_reference = compute_nawm_reference(
                    nawm_data, nawm_maps, voxel_sizes
                )

            # Compute derived metrics for primary region
            primary_mask = lesion_data > 0
            result.derived_metrics = compute_dce_derived_metrics(
                ktrans_data=maps.get("ktrans"),
                nktrans_data=maps.get("nktrans"),
                ve_data=maps.get("ve"),
                vp_data=maps.get("vp"),
                mask=primary_mask,
            )

            # Add clinical context
            result.clinical_context = get_dce_clinical_context()

        # Write outputs
        output_dir = Path(runtime.cwd)

        prefix = f"sub-{self.inputs.subject_id}"
        if self.inputs.session_id:
            prefix += f"_ses-{self.inputs.session_id}"

        # JSON output
        desc = "roi-stats" if self.inputs.force_binary else "lesion-stats"
        json_path = output_dir / f"{prefix}_{desc}.json"
        with open(json_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        # Set outputs
        self._results["stats_json"] = str(json_path)

        return runtime

    def _list_outputs(self) -> Dict[str, str]:
        """Return outputs."""
        return self._results

    def _load_maps(self) -> Dict[str, np.ndarray]:
        """Load all available perfusion maps.

        Returns:
            Dictionary mapping map name to 3D numpy array.
        """
        maps: Dict[str, np.ndarray] = {}

        map_attrs = [
            ("ktrans_map", "ktrans"),
            ("ve_map", "ve"),
            ("vp_map", "vp"),
            ("kep_map", "kep"),
            ("nktrans_map", "nktrans"),
            ("iauc60_map", "iauc60"),
            ("iauc90_map", "iauc90"),
            ("iauc120_map", "iauc120"),
            ("niauc60_map", "niauc60"),
            ("peak_enhancement_map", "peak_enhancement"),
            ("wash_in_map", "wash_in"),
            ("wash_out_map", "wash_out"),
            ("ttp_map", "ttp"),
            ("auc_map", "auc"),
        ]

        for attr, name in map_attrs:
            if self._has_input(attr):
                path = getattr(self.inputs, attr)
                if Path(path).exists():
                    img = nib.load(path)
                    maps[name] = np.asarray(img.dataobj)  # type: ignore[attr-defined]

        return maps

    def _has_input(self, attr: str) -> bool:
        """Check if an input is defined and points to an existing file."""
        from pmriprep.utils.outputs import has_input
        return has_input(self.inputs, attr)

    def _compute_region_stats(
        self,
        region_name: str,
        mask: np.ndarray,
        maps: Dict[str, np.ndarray],
        voxel_sizes: Tuple[float, float, float],
    ) -> Optional[RegionStatistics]:
        """Compute statistics for a single region.

        Args:
            region_name: Name of the region.
            mask: Boolean mask for the region.
            maps: Dictionary of perfusion maps.
            voxel_sizes: Voxel dimensions in mm.

        Returns:
            RegionStatistics object or None if mask is empty.
        """
        voxel_count, volume_mm3, volume_ml = compute_region_volume(mask, voxel_sizes)
        if voxel_count == 0:
            return None

        region = RegionStatistics(
            region_name=region_name,
            voxel_count=voxel_count,
            volume_mm3=volume_mm3,
            volume_ml=volume_ml,
        )

        # Compute statistics for each available map
        for map_name, map_data in maps.items():
            stats = compute_map_statistics(map_data, mask)
            if stats is not None:
                region.perfusion_maps[map_name] = stats

        return region

