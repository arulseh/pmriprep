"""DCE lesion/tumor ROI statistics extraction Nipype interface.

This module implements a Nipype interface for extracting comprehensive
statistics from DCE perfusion maps within lesion/tumor regions of interest.

Supports:
- Multi-label masks (HD-GLIO: 1=non-enhancing, 2=enhancing)
- Binary tumor masks
- NAWM reference values for normalization context
- Clinical interpretation thresholds
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import nibabel as nib
import numpy as np
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    traits,
)

from pmriprep.config.constants import HDGLIO_LABEL_TO_NAME
from pmriprep.utils.lesion_stats import (
    LesionStatisticsResult,
    RegionStatistics,
    compute_dce_derived_metrics,
    compute_map_statistics,
    compute_nawm_reference,
    compute_region_volume,
    detect_mask_type,
    extract_label_mask,
    is_integer_valued,
)


class DCELesionStatsInputSpec(BaseInterfaceInputSpec):
    """Input specification for DCE lesion statistics extraction."""

    # Required inputs
    lesion_mask = traits.Str(desc="Lesion/tumor mask in DCE space")
    subject_id = traits.Str(mandatory=True, desc="Subject ID (without sub- prefix)")
    session_id = traits.Str(mandatory=True, desc="Session ID (without ses- prefix)")

    # Optional NAWM mask for reference values
    nawm_mask = traits.Str(desc="NAWM mask in DCE space for reference values")
    qc_mask = traits.Str(
        desc="DCE quantitative QC mask applied only to PK-map statistics"
    )

    # Core pharmacokinetic maps (all optional - stats computed for available maps)
    ktrans_map = traits.Str(desc="Ktrans map path (min^-1)")
    ve_map = traits.Str(desc="ve (EES volume fraction) map path")
    vp_map = traits.Str(desc="vp (plasma volume fraction) map path")
    kep_map = traits.Str(desc="kep (rate constant) map path")

    # IAUC maps (all optional)
    iauc60_map = traits.Str(desc="IAUC at 60 seconds map path")
    iauc90_map = traits.Str(desc="IAUC at 90 seconds map path")
    iauc120_map = traits.Str(desc="IAUC at 120 seconds map path")

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
    peak_enhancement_map = traits.Str(desc="Peak enhancement map path")
    ser_map = traits.Str(desc="Signal enhancement ratio map path")
    ttp_map = traits.Str(desc="Time to peak map path")


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
            import logging
            _log = logging.getLogger(__name__)

            finite_vals = lesion_data[np.isfinite(lesion_data)]
            max_label = int(np.max(finite_vals)) if len(finite_vals) > 0 else 0
            use_multi_label = (
                max_label >= 2
                and is_integer_valued(lesion_data)
            )

            if use_multi_label:
                unique_labels = sorted(
                    int(lbl) for lbl in np.unique(finite_vals) if lbl > 0
                )
                if len(unique_labels) > 50:
                    _log.warning(
                        "ROI mask has %d labels; stats output will be large",
                        len(unique_labels),
                    )
                result = LesionStatisticsResult(
                    subject_id=self.inputs.subject_id,
                    session_id=self.inputs.session_id,
                    modality="DCE",
                    mask_type="multi_label_roi",
                    mask_source=Path(self.inputs.lesion_mask).name,
                    total_voxels=int(np.sum(lesion_data > 0)),
                )
                for label in unique_labels:
                    label_mask = extract_label_mask(lesion_data, label)
                    if np.sum(label_mask) == 0:
                        continue
                    region_stats = self._compute_region_stats(
                        f"label_{label}", label_mask, maps, voxel_sizes
                    )
                    if region_stats is not None:
                        result.regions[f"label_{label}"] = region_stats
            else:
                if max_label >= 2 and not is_integer_valued(lesion_data):
                    _log.warning(
                        "ROI mask has non-integer values; treating as binary"
                    )
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
                for label, region_name in HDGLIO_LABEL_TO_NAME.items():
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
                    region_name = HDGLIO_LABEL_TO_NAME[1]  # "non_enhancing_tumor"
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
                if k in ["ktrans", "ve", "vp", "kep", "iauc60"]
            }
            result.nawm_reference = compute_nawm_reference(
                nawm_data, nawm_maps, voxel_sizes,
            )

        # Compute derived metrics for primary region
        primary_mask = lesion_data > 0
        result.derived_metrics = compute_dce_derived_metrics(
            ktrans_data=maps.get("ktrans"),
            ve_data=maps.get("ve"),
            vp_data=maps.get("vp"),
            mask=primary_mask,
        )

        # Augment per-region per-map stats with Cohen's d vs NAWM
        # (no-op when nawm_reference is empty).
        result.augment_with_cohens_d()

        # Write outputs
        output_dir = Path(runtime.cwd)

        prefix = f"sub-{self.inputs.subject_id}"
        if self.inputs.session_id:
            prefix += f"_ses-{self.inputs.session_id}"

        # JSON output
        desc = "roi-stats" if self.inputs.force_binary else "lesion-stats"
        json_path = output_dir / f"{prefix}_{desc}.json"
        from pmriprep.utils.sidecar_builders import dump_sidecar_json
        with open(json_path, "w") as f:
            f.write(dump_sidecar_json(result.to_dict()))

        # Set outputs
        self._results["stats_json"] = str(json_path)

        return runtime

    def _list_outputs(self) -> Dict[str, str]:
        """Return outputs."""
        return self._results

    def _load_maps(self) -> Dict[str, np.ndarray]:
        """Load maps and mark QC-rejected PK estimates unavailable.

        Returns
        -------
        dict
            Dictionary mapping map name to 3D numpy array.
        """
        maps: Dict[str, np.ndarray] = {}

        map_attrs = [
            ("ktrans_map", "ktrans"),
            ("ve_map", "ve"),
            ("vp_map", "vp"),
            ("kep_map", "kep"),
            ("iauc60_map", "iauc60"),
            ("iauc90_map", "iauc90"),
            ("iauc120_map", "iauc120"),
            ("peak_enhancement_map", "peak_enhancement"),
            ("ser_map", "ser"),
            ("ttp_map", "ttp"),
        ]

        for attr, name in map_attrs:
            if self._has_input(attr):
                path = getattr(self.inputs, attr)
                if Path(path).exists():
                    img = nib.load(path)
                    maps[name] = np.asarray(img.dataobj)  # type: ignore[attr-defined]

        if self._has_input("qc_mask"):
            qc_data = np.asarray(
                nib.load(self.inputs.qc_mask).dataobj  # type: ignore[attr-defined]
            ) > 0
            for name in ("ktrans", "ve", "vp", "kep"):
                if name not in maps:
                    continue
                if maps[name].shape != qc_data.shape:
                    raise ValueError(
                        f"DCE QC mask shape {qc_data.shape} does not match "
                        f"{name} map shape {maps[name].shape}"
                    )
                maps[name] = np.where(qc_data, maps[name], np.nan)

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

        Parameters
        ----------
        region_name : str
            Name of the region.
        mask : np.ndarray
            Boolean mask for the region.
        maps : dict
            Dictionary of perfusion maps.
        voxel_sizes : tuple of float
            Voxel dimensions in mm.

        Returns
        -------
        RegionStatistics or None
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

        # Compute statistics for each available map. compute_advanced=True
        # turns on histogram metrics (p98/p99/skewness/kurtosis/entropy);
        # map_name enables PPP for permeability biomarkers (Ktrans/vp).
        for map_name, map_data in maps.items():
            stats = compute_map_statistics(
                map_data, mask, map_name=map_name, compute_advanced=True,
            )
            if stats is not None:
                region.perfusion_maps[map_name] = stats

        return region
