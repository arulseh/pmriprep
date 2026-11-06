1.0.0rc6.dev35 (2026-11-06)
===========================

* FIX: Label DSC motion-reference frames separately from the final baseline;
  label DCE model-free input baseline counts without implying configured values.

1.0.0rc6.dev34 (2026-11-06)
===========================

* CHG: Retire NAWM median range PASS/WARNING labels. Preserve descriptive
  statistics and coverage; fitting bounds and binary QC are unchanged.

1.0.0rc6.dev33 (2026-11-06)
===========================

* FIX: Restrict DSC NAWM normalization to the perfusion brain mask, preserving
  measured zeros and omitting normalized maps when that reference is unusable.

1.0.0rc6.dev32 (2026-11-06)
===========================

* FIX: Exclude outside-brain image background from DSC/DCE regional statistics
  while preserving legitimate zero values, original region sizes and excluded
  coverage counts.

1.0.0rc6.dev31 (2026-11-06)
===========================

* FIX: Scope DCE map provenance to its estimator, omit representative-voxel
  R-squared from map summaries, and remove perfusion-processing claims from
  registered anatomical images and masks.

1.0.0rc6.dev30 (2026-11-06)
===========================

* FIX: Limit DSC map sidecars to processing used by each estimator. Preserve
  AUC-CBV integration windows separately from deconvolution, retain method-specific
  MTT provenance, and omit unrelated AIF and leakage metadata.

1.0.0rc6.dev29 (2026-11-06)
===========================

* FIX: Report the executed DCE hematocrit consistently in shared and AIF
  provenance, including zero, and omit it when no value was recorded. Describe
  external AIFs as supplied plasma curves without hematocrit conversion.

1.0.0rc6.dev28 (2026-11-06)
===========================

* MNT: Repeat the 3-D mask check inside native mask validation for workflows
  built outside the master pipeline.

1.0.0rc6.dev27 (2026-11-06)
===========================

* FIX: Let DCE fitting run without waiting for DSC when the vp prior is
  disabled or unsupported by the selected model/inference. Keep DSC inputs
  connected to downstream vascular concordance.

1.0.0rc6.dev26 (2026-11-06)
===========================

* CHG: Centralize DCE run diagnostics in preprocessed-series QC, keep the QC
  definition with its mask and delay limits with the delay map, and use compact
  AIF provenance on DSC/DCE maps.

1.0.0rc6.dev25 (2026-11-06)
===========================

* FIX: Correct the Patlak-relative BIC guard description and record the actual
  candidate pool in model-selection sidecars.

1.0.0rc6.dev24 (2026-11-06)
===========================

* FIX: Normalize recorded ANTs motion versions in generated methods using the
  same formatter as other tool versions.

1.0.0rc6.dev23 (2026-11-06)
===========================

* CHG: Show motion-correction status beside the HTML motion-QC metrics instead
  of annotating DSC and DCE motion figures.

1.0.0rc6.dev22 (2026-11-06)
===========================

* FIX: Omit SER and IAUC derivatives when their acquisition windows are
  incomplete, including smoothed copies and regional statistics. Retain timing
  coverage in QC rather than exporting undefined maps with zero-background
  statistics.

1.0.0rc6.dev21 (2026-11-05)
===========================

* FIX: Describe executed RC6 motion, DCE fitting, T1 fallback, AIF support and
  identification QC in generated methods. Preserve testing-mode disclosure in
  copied methods and clarify the quick-start configuration tables.

1.0.0rc6.dev20 (2026-11-05)
===========================

* CHG: Make sloppy mode a run-wide, CLI-only testing preset with one HTML
  summary warning. Record approximate DCE QC in its sidecar policy.

1.0.0rc6.dev19 (2026-11-05)
===========================

* ENH: Emit the native selected DCE delay map and timing summary provenance
  without additional sensitivity refits or changes to parameter maps and binary
  QC. Distinguish selected/reference boundary counts and explain limit-selected
  offsets. Flag coarse-sampled 2CXM Fp/PS as timing limited in their sidecars.

1.0.0rc6.dev18 (2026-11-04)
===========================

* CHG: Combine reported-fit quality with selected-NLLS primary-parameter
  identification in the DCE binary QC mask, including VI runs. Preserve fitted
  maps and retention; record the screening policy and parameter-level failure
  counts in derivative provenance.

1.0.0rc6.dev17 (2026-11-04)
===========================

* ENH: Package signed one-frame DCE timing refits with per-parameter
  availability, raw fit evidence and endpoint-extension counts.
  Diagnostic output integration remains pending.

1.0.0rc6.dev16 (2026-11-04)
===========================

* ENH: Support signed fixed-delay inputs in native DCE fitting using the shared
  PCHIP geometry, for timing diagnostics outside the delay-selection grid.
  Diagnostic output integration remains pending.

1.0.0rc6.dev15 (2026-11-04)
===========================

* ENH: Package primary-parameter DCE identification from conditional NLLS
  covariance and delay profiles, with per-parameter failure reasons.
  Binary-mask and report integration remain pending.

1.0.0rc6.dev14 (2026-11-04)
===========================

* PERF: Reuse cached PCHIP convolution in Tofts and Extended Tofts delay
  profiles, sharing the input evaluator used by native 2CXM fitting.

1.0.0rc6.dev13 (2026-11-04)
===========================

* ENH: Route DCE NLLS and VI through local delay fitting with bounded slice
  execution and conditional uncertainty. Replace AIF pre-alignment with the
  delay_correction option and retire MAP/MCMC pipeline routing. Identification
  QC and timing-derivative integration remain pending.

1.0.0rc6.dev12 (2026-11-04)
===========================

* ENH: Bind VI refinement to the selected DCE family, delay and shared PCHIP
  input, with guarded NLLS initialization and same-delay fallback. Share the
  existing parameter-retention rules. Pipeline routing remains pending.

1.0.0rc6.dev11 (2026-11-04)
===========================

* ENH: Assemble complete-slice DCE profiles and local selections through one
  reusable worker pool with bounded fitting blocks. Preserve full neighborhood
  support, original frame identity and retry provenance across blocks. Abort
  unexpected worker failures explicitly. Record classified numerical failures
  as unavailable voxel/family profiles without discarding their neighbors.
  Workflow routing remains pending.

1.0.0rc6.dev10 (2026-11-04)
===========================

* ENH: Package in-plane local delay references, nested delay selection and
  capped G1/G2 automatic 2CXM admission. Preserve the Patlak model-selection
  guard and explicit-model choices. Record report-only delay-boundary flags
  and upper-edge truncation evidence. Workflow integration remains pending.

1.0.0rc6.dev9 (2026-11-04)
==========================

* ENH: Build conditional DCE delay profiles with common original observation
  indices and recorded continuation. Finalize failed native 2CXM points with
  bounded full-start retries after the sweep, preserving later seeds and fit
  status. Local selection and workflow routing remain pending.

1.0.0rc6.dev8 (2026-11-04)
==========================

* ENH: Package the native bounded 2CXM fitter and compiled PCHIP evaluator,
  preserving full starts and continuation inputs. Build through the package
  extension machinery with strict floating-point settings. Workflow integration
  and signed diagnostic fitting remain pending.
* FIX: Pin SciPy to the verified 1.17.0 release and check its compiled interfaces
  at DCE startup and image build time. Include BSD-3-Clause in image licence labels.

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
