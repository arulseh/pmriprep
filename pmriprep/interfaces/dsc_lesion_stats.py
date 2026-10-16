"""DSC lesion/tumor ROI statistics extraction Nipype interface.

This module implements a Nipype interface for extracting comprehensive
statistics from DSC perfusion maps within lesion/tumor regions of interest.

Supports:
- Multi-label masks (HD-GLIO: 1=non-enhancing, 2=enhancing)
- Binary tumor masks
- NAWM reference values for normalization context
- Descriptive distribution summaries
"""

from pathlib import Path
from typing import Any, Dict, Optional

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
    compute_dsc_derived_metrics,
    compute_map_statistics,
    compute_nawm_reference,
    compute_region_volume,
    detect_mask_type,
    extract_label_mask,
    is_integer_valued,
)
from pmriprep.utils.nifti import physical_voxel_volume_mm3


class DSCLesionStatsInputSpec(BaseInterfaceInputSpec):
    """Input specification for DSC lesion statistics extraction."""

    # Required inputs
    lesion_mask = traits.Str(desc="Lesion/tumor mask in DSC space")
    subject_id = traits.Str(mandatory=True, desc="Subject ID (without sub- prefix)")
    session_id = traits.Str(mandatory=True, desc="Session ID (without ses- prefix)")

    # Optional NAWM mask for reference values
    nawm_mask = traits.Str(desc="NAWM mask in DSC space for reference values")

    # Optional CPI-to-oSVD replacement mask for regional provenance
    fallback_mask = traits.Str(
        desc="Binary mask identifying voxels replaced by oSVD after rejected CPI fits"
    )

    # Core perfusion maps (all optional - stats computed for available maps)
    cbf_map = traits.Str(desc="CBF map path")
    cbv_map = traits.Str(desc="Deconvolution residue-integral CBV map path")
    cbv_auc_map = traits.Str(desc="AUC-based CBV map path")
    mtt_map = traits.Str(desc="MTT map path")
    tmax_map = traits.Str(desc="Method-specific deconvolution timing map path")
    timing_map_name = traits.Enum(
        "tmax",
        "arrival_delay",
        usedefault=True,
        desc="Public name for the method-specific timing measurement",
    )

    # Normalized maps (optional - only available when NAWM mask present)
    ncbv_map = traits.Str(desc="Normalized CBV map path")
    ncbf_map = traits.Str(desc="Normalized CBF map path")

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

    # Additional maps (all optional)
    psr_map = traits.Str(desc="PSR map path")
    ttp_map = traits.Str(desc="AIF-corrected TTP map path")

    # AIF provenance — passed from the cascade QC so the lesion-stats
    # sidecar carries the PerfusionProcessing.AIF block (Source /
    # DetectionLevel / QualityFlag / etc).
    aif_qc = traits.Dict(
        desc="AIF QC dict from cascade, used for the "
             "PerfusionProcessing.AIF provenance block",
    )


class DSCLesionStatsOutputSpec(TraitedSpec):
    """Output specification for DSC lesion statistics extraction."""

    stats_json = traits.Str(desc="Path to JSON statistics file")


class DSCLesionStats(BaseInterface):
    """Extract lesion/tumor ROI statistics from DSC perfusion maps.

    This interface extracts comprehensive statistics from DSC perfusion
    maps within lesion/tumor regions, supporting:

    - Multi-label masks (HD-GLIO: label 1 = non-enhancing, label 2 = enhancing)
    - Binary tumor masks
    - NAWM reference values for normalization context
    - Descriptive fractional-volume summaries over a predefined value grid

    Output files:
        {prefix}_lesion-stats.json  - Comprehensive JSON statistics
    """

    input_spec = DSCLesionStatsInputSpec
    output_spec = DSCLesionStatsOutputSpec

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
        voxel_volume_mm3 = physical_voxel_volume_mm3(lesion_img.affine)  # type: ignore[attr-defined]

        # Load all available perfusion maps
        maps = self._load_maps()
        fallback_mask = self._load_fallback_mask(lesion_data.shape)

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
                    modality="DSC",
                    mask_type="multi_label_roi",
                    mask_source=Path(self.inputs.lesion_mask).name,
                    total_voxels=int(np.sum(lesion_data > 0)),
                )
                for label in unique_labels:
                    label_mask = extract_label_mask(lesion_data, label)
                    if np.sum(label_mask) == 0:
                        continue
                    region_stats = self._compute_region_stats(
                        f"label_{label}", label_mask, maps, voxel_volume_mm3,
                        fallback_mask,
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
                    modality="DSC",
                    mask_type="binary",
                    mask_source=Path(self.inputs.lesion_mask).name,
                    total_voxels=int(np.sum(lesion_data > 0)),
                )
                binary_mask = lesion_data > 0
                region_stats = self._compute_region_stats(
                    "roi", binary_mask, maps, voxel_volume_mm3, fallback_mask
                )
                if region_stats is not None:
                    result.regions["roi"] = region_stats
        else:
            # Standard lesion stats mode
            mask_type, max_label = detect_mask_type(lesion_data)

            result = LesionStatisticsResult(
                subject_id=self.inputs.subject_id,
                session_id=self.inputs.session_id,
                modality="DSC",
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
                        region_name, label_mask, maps, voxel_volume_mm3,
                        fallback_mask,
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
                    region_name, binary_mask, maps, voxel_volume_mm3, fallback_mask
                )
                if region_stats is not None:
                    result.regions[region_name] = region_stats

        # Compute NAWM reference if mask provided
        if self._has_input("nawm_mask"):
            nawm_img = nib.load(self.inputs.nawm_mask)
            nawm_data = np.asarray(nawm_img.dataobj) > 0  # type: ignore[attr-defined]
            nawm_maps = {
                k: v for k, v in maps.items()
                if k in ["cbv", "cbv_auc", "cbf", "ncbv", "ncbf", "mtt"]
            }
            result.nawm_reference = compute_nawm_reference(
                nawm_data, nawm_maps, voxel_volume_mm3,
            )
            if result.nawm_reference and fallback_mask is not None:
                result.nawm_reference["cpi_fallback"] = (
                    self._compute_fallback_summary(nawm_data, fallback_mask)
                )

        # Compute derived metrics for primary region
        primary_mask = lesion_data > 0
        result.derived_metrics = compute_dsc_derived_metrics(
            ncbv_data=maps.get("ncbv"),
            ncbf_data=maps.get("ncbf"),
            cbv_data=maps.get("cbv"),
            cbf_data=maps.get("cbf"),
            mask=primary_mask,
        )

        # Augment per-region per-map stats with Cohen's d vs NAWM
        # (no-op when nawm_reference is empty).
        result.augment_with_cohens_d()

        # Carry the AIF source, detection and quality metadata into the
        # regional statistics alongside the selected reference tissue.
        if self._has_input("aif_qc"):
            from pmriprep.utils.aif_provenance import (
                build_aif_provenance_block,
            )
            result.aif_provenance = build_aif_provenance_block(
                self.inputs.aif_qc or {}
            ) or None

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
        """Load all available perfusion maps.

        Returns
        -------
        dict
            Dictionary mapping map name to 3D numpy array.
        """
        maps: Dict[str, np.ndarray] = {}

        map_attrs = [
            ("cbf_map", "cbf"),
            ("cbv_map", "cbv"),
            ("cbv_auc_map", "cbv_auc"),
            ("mtt_map", "mtt"),
            ("tmax_map", str(self.inputs.timing_map_name)),
            ("ncbv_map", "ncbv"),
            ("ncbf_map", "ncbf"),
            ("psr_map", "psr"),
            ("ttp_map", "ttp"),
        ]

        for attr, name in map_attrs:
            if self._has_input(attr):
                path = getattr(self.inputs, attr)
                if Path(path).is_file():
                    img = nib.load(path)
                    maps[name] = np.asarray(img.dataobj)  # type: ignore[attr-defined]

        return maps

    def _has_input(self, attr: str) -> bool:
        """Check if an input is defined and points to an existing file."""
        from pmriprep.utils.outputs import has_input
        return has_input(self.inputs, attr)

    def _load_fallback_mask(
        self,
        expected_shape: tuple[int, ...],
    ) -> Optional[np.ndarray]:
        """Load the optional CPI-to-oSVD replacement mask on the lesion grid."""
        if not self._has_input("fallback_mask"):
            return None
        fallback_img: Any = nib.load(self.inputs.fallback_mask)
        fallback_data: np.ndarray = np.asarray(fallback_img.dataobj)
        if fallback_data.shape != expected_shape:
            raise ValueError(
                "CPI fallback mask shape "
                f"{fallback_data.shape} does not match lesion mask shape "
                f"{expected_shape}"
            )
        return np.asarray(fallback_data == 1, dtype=bool)

    @staticmethod
    def _compute_fallback_summary(
        region_mask: np.ndarray,
        fallback_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """Count successful oSVD replacements over all voxels in a region."""
        region = np.asarray(region_mask, dtype=bool)
        if region.shape != fallback_mask.shape:
            raise ValueError(
                "CPI fallback and regional masks must share the same shape"
            )
        voxel_count = int(np.count_nonzero(region))
        n_osvd = int(np.count_nonzero(region & fallback_mask))
        return {
            "n_osvd": n_osvd,
            "fraction": n_osvd / voxel_count if voxel_count else 0.0,
            "denominator": "region_voxels",
        }

    def _compute_region_stats(
        self,
        region_name: str,
        mask: np.ndarray,
        maps: Dict[str, np.ndarray],
        voxel_volume_mm3: float,
        fallback_mask: Optional[np.ndarray] = None,
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
        voxel_volume_mm3 : float
            Physical voxel-cell volume in cubic millimetres.
        fallback_mask : np.ndarray, optional
            Binary mask of successful CPI-to-oSVD replacements.

        Returns
        -------
        RegionStatistics or None
            RegionStatistics object or None if mask is empty.
        """
        voxel_count, volume_mm3, volume_ml = compute_region_volume(
            mask, voxel_volume_mm3,
        )
        if voxel_count == 0:
            return None

        region = RegionStatistics(
            region_name=region_name,
            voxel_count=voxel_count,
            volume_mm3=volume_mm3,
            volume_ml=volume_ml,
        )
        if fallback_mask is not None:
            region.cpi_fallback = self._compute_fallback_summary(
                mask, fallback_mask
            )

        # Compute statistics for each available map. compute_advanced=True
        # turns on upper-tail and distribution-shape summaries.
        for map_name, map_data in maps.items():
            stats = compute_map_statistics(
                map_data, mask, compute_advanced=True,
            )
            if stats is not None:
                region.perfusion_maps[map_name] = stats

        return region
