1.0.0rc2.dev2 (unreleased)
==========================

* FIX: Preserve measured DCE AIF temporal samples without noncausal smoothing.
* FIX: Keep DSC gamma-variate initialization within solver bounds and retain an
  accepted measured AIF when the optional gamma fit fails.
* FIX: Prevent pre-bolus noise from initializing DSC gamma-variate bolus onset.

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
