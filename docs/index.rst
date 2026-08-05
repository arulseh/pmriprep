PMRIPrep
========

.. include:: links.rst

PMRIPrep is a BIDS-App_ for processing Dynamic Susceptibility Contrast (DSC) and Dynamic
Contrast Enhanced (DCE) perfusion MRI. It follows the NiPreps_ framework conventions and
produces BIDS-Derivatives-style outputs with HTML QC reports.

Features
--------

- **DSC processing** -- five deconvolution methods (sSVD, oSVD, Tikhonov, CPI, Vascular) and BSW leakage correction.
- **DCE processing** -- four pharmacokinetic models (Tofts, Extended Tofts, Patlak, 2CXM) plus model-free maps.
- **Inference** -- nonlinear least squares and variational inference, plus spatial variational Bayes for DSC vascular models.
- **Automatic AIF detection** -- multi-level arterial input function detection with population fallback, or a user-supplied external AIF.
- **T1 mapping** -- variable flip angle T1 mapping with optional B1+ correction.
- **Tumor segmentation** -- HD-GLIO integration for glioma segmentation.
- **QC reports** -- per-session HTML reports with AIF quality, motion, registration, and parameter-map visualizations.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   outputs
   api
   changes
