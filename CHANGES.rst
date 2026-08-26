1.0.0rc2 (2026-08-26)
=====================

* ENH: Publish the signed, brain-masked DSC K2 coefficient map when BSW leakage
  correction is applied.
* FIX: Serialize GPU-dependent workflow nodes automatically when multiple GPUs
  are visible and report the resulting scheduling policy accurately.
* FIX: Use physical brain-edge and symmetry-aligned posterior-midline
  exclusions during DSC whole-brain AIF search.
* FIX: Use physical brain- and slab-edge distances with a symmetry-aligned
  posterior-midline exclusion during DCE whole-brain AIF search, while
  preserving arterial-atlas territory.
* ENH: Expand DSC and DCE AIF provenance with atlas-search outcomes, spatial
  support, and final-pool laterality information.
* FIX: Preserve measured DCE AIF samples without noncausal temporal smoothing.
* FIX: Make DSC gamma-variate initialization robust to pre-bolus noise and
  solver bounds without discarding an otherwise accepted measured AIF.
* FIX: Replace the CPI and vascular fallback threshold with a minimum
  fit-adequacy rule and preserve rejected-fit QC.
* FIX: Use a uniform positive-R² criterion for QC-filtered DCE regional
  statistics.
* ENH: Retain HD-BET test-time augmentation during normal GPU execution, with
  an appropriate timeout and execution provenance.
* MNT: Store bundled atlas masks in RAS orientation on their existing
  TemplateFlow-equivalent grids and correct the anatomical-region mask space
  codes without changing world-space support.

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
