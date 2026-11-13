Outputs
=======

.. include:: links.rst

PMRIPrep outputs follow BIDS-Derivatives_ conventions under ``<output_dir>/pmriprep/``.
Applicable NIfTI derivatives have JSON sidecars describing units, conventions,
processing, and provenance. A per-session HTML QC report is written alongside
the derivatives. Output availability depends on the supplied modalities and configuration.

Layout
------

This example shows a session-based dataset. Session directories and ``ses-``
entities are omitted for datasets without sessions. Filenames retain applicable
source entities such as ``acq``, ``ce``, and ``run``, omitted here for readability.

.. code-block:: text

   output_dir/pmriprep/
     dataset_description.json
     sub-01/ses-01/
       anat/
         sub-01_ses-01_space-T1w_desc-preproc_T1w.nii.gz
         sub-01_ses-01_space-T1w_desc-brain_mask.nii.gz
         sub-01_ses-01_space-T1w_label-{GM,WM,CSF}_probseg.nii.gz
         sub-01_ses-01_space-dceref_T1map.nii.gz
         sub-01_ses-01_space-T1w_desc-tumorseg_dseg.nii.gz
         ...
       perf/
         sub-01_ses-01_desc-dscPreproc_dsc.nii.gz
         sub-01_ses-01_desc-dscRelCbf_stat.nii.gz
         sub-01_ses-01_desc-dscRelCbv_stat.nii.gz
         sub-01_ses-01_desc-dscRelAUCCbv_stat.nii.gz
         sub-01_ses-01_desc-dscMtt_stat.nii.gz
         sub-01_ses-01_desc-dceKtrans_stat.nii.gz
         sub-01_ses-01_desc-dceVe_stat.nii.gz
         sub-01_ses-01_desc-dscAif_timeseries.tsv
         sub-01_ses-01_desc-dceAif_timeseries.tsv
         ...
       figures/
         ...
       sub-01_ses-01.html

``dscRelCbv`` contains residue-integral CBV; ``dscRelAUCCbv`` contains AUC-based
CBV. NAWM-normalized CBV uses AUC-based CBV.

Accepted VFA T1 mapping emits a millisecond ``T1map`` in the native DCE reference
grid. Population-T1 fallbacks and external T1 maps are consumed without being
republished. B1+ maps are used for correction but are not emitted as derivatives.

DCE outputs include a binary QC mask for fit quality and parameter identification.
Quantitative maps are not filtered. Regional statistics summarize finite voxels
within each region and the brain mask and, when a QC mask is available,
separately summarize voxels that pass QC.
The selected model delay is also written in seconds at native resolution; this
AIF-relative fitted offset is not a measure of vascular transit time.

When ``smooth_fwhm`` is set (DSC or DCE), smoothed map variants are written with a
``Sm{fwhm}`` tag in the ``desc-`` entity (e.g. ``desc-dscRelSm3Cbf_stat`` for 3 mm FWHM).
Unsmoothed maps are used for quantitative regional analysis. See the
`README output summary <https://github.com/arulseh/pmriprep#outputs-and-qc>`_
for available map types and their interpretation.
