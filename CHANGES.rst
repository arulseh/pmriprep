1.0.0rc6 (2026-11-10)
=====================

* CHG: Use ANTs rigid motion correction for DSC and DCE with a baseline-mean
  reference and linear resampling. Motion is still estimated for QC when
  correction is disabled. Use unit temporal weights for DSC/CPI and DCE fitting.
* ENH: Add local DCE delay correction using in-plane neighbourhood evidence,
  bounded model fitting and a shared PCHIP arterial input. Emit the selected
  delay map and run-level support and boundary summaries.
* ENH: Integrate bounded Patlak, Tofts, Extended Tofts and native compiled 2CXM
  fitting with reusable convolution calculations and slice-based worker pools.
  Automatic selection retains the Patlak BIC guard and applies additional
  2CXM admission checks within its maximum 2-s frame interval. Explicit model
  selection remains available at coarser sampling.
* CHG: DCE inference options are ``bls`` (bounded least squares) and ``vi``.
  VI refines parameters at the selected model and delay, with fallback to the
  same least-squares fit. MAP/MCMC are no longer pipeline inference options.
* CHG: Replace ``dce.aif_bat_alignment`` with boolean ``dce.delay_correction``.
  Use lowercase ``ssvd`` and ``osvd`` DSC configuration identifiers. Make
  ``--sloppy`` a run-wide, CLI-only testing preset with an HTML warning.
* CHG: Combine fit quality with primary-parameter identification at the
  delay-conditional least-squares estimate in the binary DCE QC mask,
  including VI runs. Preserve fitted maps and disclose the screening policy,
  parameter-level failure counts and timing limitations of coarse-sampled 2CXM.
* ENH: Add ``dce.t1_default_ms`` for uniform T1 fallback, with field-dependent
  range validation. Usable measured or external T1 maps take precedence;
  record the fallback value and source when it is applied.
* CHG: Apply pooled AIF quality and amplitude checks to DCE single-voxel
  selections, and report accepted singletons with reduced confidence. Apply
  the shared curve-quality checks to DSC single-voxel selections and remove
  the 3x3x3 neighbourhood retry. Failed DSC detection retains AIF-independent
  maps; failed DCE detection uses the population AIF.
* CHG: Require ten candidates across two slices for the DCE excessive-scaling
  rescue. Remove its bilateral-support exception and use brain symmetry for
  descriptive AIF laterality. Distinguish AIF detection outcome, confidence
  and QC status in HTML reports.
* FIX: Save derived DSC/DCE temporal means and concentration images as float32.
  Accept sub-micrometre header rounding in ANTs motion outputs and recover
  rigid transforms using the displacement field's physical grid.
  Restrict regional statistics and DSC NAWM normalization to the perfusion brain
  mask while preserving valid zeros and reporting excluded coverage. Include
  signed K2 in DSC regional statistics.
* FIX: Keep DCE posterior uncertainty unavailable where VI falls back to least
  squares, and restrict prior-departure summaries to retained prior-informed
  fits. Allow DCE fitting to proceed independently of DSC when its prior is
  disabled or unsupported by the selected model or inference.
* FIX: Omit SER and IAUC maps and statistics when their acquisition windows
  are incomplete. Allow smoothed Patlak outputs when native Kep is unavailable.
* CHG: Scope map sidecars to the processing used by each estimator. Separate
  AUC-CBV integration from deconvolution metadata, consolidate run-level QC,
  preserve unavailable measurements as null and use consistent descriptive
  field names. Correct hematocrit, contrast-agent, dose, baseline, registration,
  uncertainty and parameter-boundary provenance.
* CHG: Retain descriptive NAWM statistics and coverage without median-range
  PASS/WARNING labels. Show motion-correction status beside HTML QC metrics.
* FIX: Preserve existing derivative files on reruns and warn in the log about
  known unsmoothed outputs left by different settings. Reports use the latest
  figure set for each pipeline, including reused anatomy, and omit stale
  optional panels.
* CHG: Default to an automatic CPU budget capped at eight and bounded by CPU
  affinity, Slurm allocation and container quotas. Respect explicit budgets,
  bound preprocessing threads and workers, and require explicit CPU budgets
  for remote scheduler plugins. Match GPU HD-BET/HD-GLIO reservations to
  Nipype's shared CPU/GPU-slot accounting.
* FIX: Pin SciPy to the verified 1.17.0 release and check the native DCE
  interfaces at startup and image build time. Include the adapted solver's
  BSD licence in package and image metadata.
* DOC: Simplify generated methods while describing recorded processing choices,
  update ANTs citations and version formatting, and clarify quick-start
  configuration, inference terminology, GPU examples and brain-MRI requirements.


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
