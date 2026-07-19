1.0.0rc1 (July 19, 2026)
========================

First release candidate of PMRIPrep, a BIDS App for preprocessing and
quantifying dynamic susceptibility contrast (DSC) and dynamic contrast-enhanced
(DCE) MRI.

Highlights
----------

* Run DSC and DCE independently or together from a BIDS dataset, with shared
  anatomical processing and per-session HTML quality-control reports.
* Estimate DSC perfusion with sSVD, oSVD, Tikhonov, CPI, or vascular residue
  models, including leakage correction and automatic or supplied arterial input
  functions.
* Estimate DCE pharmacokinetic parameters with Tofts, extended Tofts, Patlak, or
  2CXM models, plus model-free enhancement maps and automatic, population, or
  supplied arterial input functions.
* Use variable-flip-angle T1 mapping with optional B1+ correction and optional
  tumor segmentation to construct normal-appearing white-matter reference
  regions.
* Write brain-extracted quantitative perfusion maps as BIDS-Derivatives with JSON
  provenance, AIF time series, transforms, confounds, and quality-control data.
* Run from the GHCR container, Apptainer image, or a Python 3.10–3.12 source
  installation with FSL and ANTs available.

Release-candidate contract
--------------------------

* Quantitative DSC outputs are relative rather than absolute CBF or CBV.
  Normalized nCBF and nCBV require a valid normal-appearing white-matter
  reference region.
* DCE fitting uses a plasma AIF. Detected AIF amplitudes are peak-anchored to the
  dose-adjusted population curve; population and external AIF modes remain
  available.
* Quantitative perfusion derivatives are brain-extracted. Motion-corrected DSC
  and DCE four-dimensional series are not skull stripped.
* Acquisition parameters are read from BIDS sidecars. Required fields and the
  supported optional inputs are documented in the README and user guide.
* External scientific validation datasets are optional local inputs rather than
  runtime dependencies or routine release-test requirements.

Known limitations
-----------------

* Tumor segmentation requires the separately managed HD-GLIO environment and a
  supported GPU; the remaining workflows continue when it is unavailable.
* DCE input time points must be finite, strictly increasing, and uniformly
  sampled. Nonuniform sampling is rejected before fitting.
* Bayesian uncertainty outputs depend on the selected inference method. DSC CBV
  uncertainty is not emitted where the joint covariance needed for defensible
  propagation is unavailable.
* The release candidate supports the documented BIDS input contract and tested
  Python versions; broader scanner, protocol, and contrast-agent validation will
  continue with the publication datasets.
