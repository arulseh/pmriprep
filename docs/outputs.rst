Outputs
=======

.. include:: links.rst

PMRIPrep outputs follow BIDS-Derivatives_ conventions under ``<output_dir>/pmriprep/``.
Every NIfTI output has a paired ``.json`` sidecar carrying a description, units, and
provenance, and a per-session HTML QC report is written alongside the derivatives.

Layout
------

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
         sub-01_ses-01_desc-dscMtt_stat.nii.gz
         sub-01_ses-01_desc-dceKtrans_stat.nii.gz
         sub-01_ses-01_desc-dceVe_stat.nii.gz
         sub-01_ses-01_desc-dscAif_timeseries.tsv
         sub-01_ses-01_desc-dceAif_timeseries.tsv
         ...
       sub-01_ses-01.html

When ``smooth_fwhm`` is set (DSC or DCE), smoothed map variants are written with a
``Sm{fwhm}`` tag in the ``desc-`` entity (e.g. ``desc-dscRelSm3Cbf_stat`` for 3 mm FWHM).
The full per-file output table is documented in the repository ``README``.
