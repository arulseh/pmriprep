1.0.0rc2.dev12 (unreleased)
===========================

* FIX: Use an 8 mm physical in-plane brain-edge exclusion for every DSC AIF
  tier and replace scanner-axis whole-brain venous cuts with a symmetry-aligned
  posterior midline exclusion before candidate selection.
* FIX: Consolidate DCE whole-brain AIF spatial exclusions around physical
  brain/slab distances and a symmetry-aligned posterior midline venous band.
* ENH: Record whether the transformed DCE arterial-atlas search was selected,
  exhausted, or unavailable, together with its usable voxel support.
* FIX: Restrict the fractional superior DCE AIF exclusion to whole-brain
  fallback, preserving the full arterial-atlas territory above slab rolloff.
* FIX: Replace the universal CPI/vascular R² fallback threshold with a
  minimum fit-adequacy rule, preserve rejected-fit QC, and hide internal DSC
  inference engines.
* FIX: Preserve measured DCE AIF temporal samples without noncausal smoothing.
* FIX: Keep DSC gamma-variate initialization within solver bounds and retain an
  accepted measured AIF when the optional gamma fit fails.
* FIX: Prevent pre-bolus noise from initializing DSC gamma-variate bolus onset.
* FIX: Use a uniform positive-R² criterion in the DCE fit-quality mask.
* FIX: Retain HD-BET test-time augmentation for normal GPU execution, with a
  scaled timeout and recorded execution QC.
* MNT: Remove the obsolete post-cascade DCE AIF retry policy while preserving
  the active G1--G6 candidate-pool trimming and singleton fallback.
* MNT: Re-store bundled arterial and anatomical region masks in RAS orientation
  on their existing TemplateFlow-equivalent MNI152NLin6Asym grids, correcting
  the region-mask space codes to MNI152 without changing world-space support.

1.0.0rc1 (2026-08-06)
======================

Initial release candidate.

* ENH: Add DSC preprocessing, leakage correction, AIF detection, and five
  deconvolution models.
* ENH: Add DCE preprocessing, T1/B1 correction, AIF detection, model-free maps,
  and four pharmacokinetic models.
* ENH: Add shared anatomical processing and optional tumor segmentation.
* ENH: Write provenance-rich derivatives and per-session HTML QC reports.
* CHG: Make T1w subtraction opt-in unless requested as a DSC or DCE underlay.
* MNT: Provide Python 3.11--3.12 packaging and Docker/Apptainer distribution.
