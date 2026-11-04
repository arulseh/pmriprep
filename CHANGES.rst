1.0.0rc6.dev7 (2026-11-04)
==========================

* ENH: Package bounded Patlak and rate-profile Extended Tofts fitting with
  the shared delayed input. Preserve the joint ET volume constraint and
  recompute residual sums directly. Workflow integration remains pending.

1.0.0rc6.dev6 (2026-11-04)
==========================

* ENH: Package the shared signed-delay PCHIP input operator and direct
  Standard Tofts rate-profile fitter. Workflow integration remains pending.

1.0.0rc6.dev5 (2026-11-04)
==========================

* ENH: Add dce.t1_default_ms for uniform T1 fallback in milliseconds. Null
  retains field-strength defaults; usable measured/external maps take precedence.
  Record the applied fallback value and source in conversion QC and sidecars.
  Reject fallback values outside the existing field-dependent T1 safety limits.

1.0.0rc6.dev4 (2026-11-04)
==========================

* FIX: Report reduced confidence and its reason for single-voxel DCE AIFs,
  preserving the emitted curve, calibration and numerical uncertainty factors.

1.0.0rc6.dev3 (2026-11-04)
==========================

* CHG: Use unit temporal weights for DSC/CPI and DCE fitting and model
  comparison, including inference. Motion estimates remain available for QC.

1.0.0rc6.dev2 (2026-11-04)
==========================

* CHG: Use ANTs rigid motion correction for DSC and DCE with a fixed
  baseline-mean reference and linear resampling. Disabling correction retains
  the input frames while still estimating motion for QC.
* FIX: Use common physical-transform conversion and FD summaries across
  preprocessing and motion plots, including the configured QC threshold and
  exclusion of the initial FD placeholder from the mean.
* ENH: Bound ANTs motion threads by the pipeline CPU budget and per-process
  thread limit, with matching Nipype processor reservations.

1.0.0rc6.dev1 (2026-11-04)
==========================

* FIX: Save derived DSC/DCE temporal means and concentration images as
  float32, preventing integer input headers from quantizing derived values.

1.0.0rc5 (2026-10-20)
=====================

Numerical changes
-----------------

* CHG: Use L-curve upper-extremum selection on a 100-point regularization grid
  for DSC Tikhonov, with a modified first-difference penalty and full-record
  residue support.
* CHG: Preserve signed sSVD, oSVD and Tikhonov residues without post-fit
  clipping. Integrate the full fitted support and report unavailable
  residue-derived MTT when the signed residue area is nonpositive.
* FIX: Correct the oSVD convolution matrix's first-sample weighting to match
  its zero-padding convention.
* FIX: Preserve the configured population DCE AIF arrival time at coarse
  temporal resolution instead of resetting it when the sampled peak is low.
* CHG: Use sourced contrast-agent relaxivities and consistent mmol Gd/kg dose
  conventions. Require an explicit DCE r1 override for missing or unrecognized
  agents and field strengths above 5 T.
* FIX: Correct DSC-derived Bayesian vp-prior qualification and preserve its
  required QC inputs for fixed Patlak models and cached DSC derivatives.
* FIX: Use the finalized DSC baseline for CNR/tSNR summaries and the tissue-CNR
  assessment used by the DCE prior, matching the exported signal-quality maps.

Input and output contracts
--------------------------

* CHG: Replace synthetic DSC AIF fallback with partial outputs when no
  acceptable measured AIF is available. Retain tissue-based nCBV, PSR, TTP,
  CNR and tSNR, plus K2 when BSW correction is applied; withhold AIF-dependent
  perfusion maps.
* FIX: Constrain every DSC and DCE AIF search tier to a supplied search mask.
  Exhausted or empty masks follow the modality-specific fallback and record
  the reason without searching outside the supplied region.
* FIX: Reject conflicting DCE time-axis inputs and malformed DSC deconvolution
  settings. Isolate nonfinite tissue curves so valid neighbouring voxels
  continue to produce estimates.
* FIX: Support session-less DSC/DCE workflow construction and partial-output
  writers, and skip DSC atlas transforms when their required input is absent.

Quality control and maintenance
-------------------------------

* ENH: Report Tikhonov regularization-selection outcomes, signed-residue
  diagnostics and unmet oSVD oscillation targets, including CPI fallback.
  Otherwise valid oSVD results remain available when the target is unmet.
* FIX: Preserve DSC-informed DCE reporting when DSC derivatives are reused,
  and record applied uniform B1 corrections in T1 provenance.
* FIX: Align QC plots, sidecar metadata, generated methods and warnings with
  the baseline, input sources and processing actually used; correct T1 QC
  left/right labels.
* MNT: Consolidate regression coverage, enforce strict expected-failure
  handling and separate portable synthetic DSC tests from local external-data
  validation. Remove unused helpers and inactive configuration fields.

1.0.0rc4 (2026-10-09)
=====================

Numerical changes
-----------------

* FIX: Normalize DSC magnitude signals acquisition-wide before CPI inference
  so arbitrary scanner intensity scaling cannot alter fit availability or
  quantitative estimates.
* CHG: Keep Extended Tofts eligible during automatic DCE model selection at
  all supported sampling intervals and restrict automatic 2CXM eligibility to
  acquisitions sampled at 2 seconds or faster. Equivalent floating-point
  representations of the 2-second boundary receive the same candidate set.

1.0.0rc3 (2026-10-07)
=====================

Numerical changes
-----------------

* CHG: Implement DSC CPI with joint signal-domain inference, including signed
  BSW-guided leakage estimation, isolation of local-fit failures, and optional
  oSVD fallback for rejected fits.
* FIX: Enforce consistent DCE parameter domains and finite-rate bounds across
  scalar, batched, and Bayesian fits; stabilize correlated bounded
  updates and reject numerical contact with configured rate ceilings.
* FIX: Prevent premature 2CXM convergence and Patlak uncertainty-frame
  overflow while preserving model-specific zero-rate behavior.
* FIX: Preserve DSC response tails for negative fitted delay and align
  parametric residue-shape and MTT support across fitting paths.
* FIX: Preserve supratentorial NAWM selection and exclude invalid NAWM support
  from quantitative normalization.
* FIX: Improve DCE reference selection, registration assessment, and AIF peak
  plausibility checks, failing closed when required quantitative checks cannot
  be completed.

Input and output contracts
--------------------------

* CHG: Remove the experimental vascular DSC method and its CTH output.
* ENH: Report fitted DSC delay and CPI spacing boundary contact without treating
  contact alone as a quality failure.
* FIX: Separate regional statistics for residue-integral CBV and AUC-CBV while
  retaining AUC-CBV as the source of normalized CBV.
* ENH: Disclose regional mixing caused by CPI-to-oSVD voxel fallback and align
  the generated DSC and DCE methods descriptions with executed processing.
* FIX: Emit interoperable affine-transform derivatives and record independent
  registration-quality measurements.
* FIX: Reject ambiguous, multi-echo, or hybrid DSC and DCE inputs and ambiguous
  DCE VFA/B1 calibration inputs instead of selecting a candidate silently.

Performance and maintenance
---------------------------

* ENH: Add CPI fitting progress and avoid repeated CPI spatial readout work
  between spatial variational sweeps.
* MNT: Consolidate duplicate and editorially coupled tests while preserving
  scientific recovery, workflow, provenance, and output-contract coverage.

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
