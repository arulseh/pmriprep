# Bundled mask resources

The NIfTI masks in this directory are distributed under the
[CC0 1.0 Universal dedication](https://creativecommons.org/publicdomain/zero/1.0/).
The dedication applies to the bundled mask files and PMRIPrep-authored
transformations described below; it does not alter the licenses of the software
or source datasets used to create them.

## Arterial masks (`mni_aif`)

The arterial masks derive from `vesselProbabilities.nii.gz` by Nils Daniel
Forkert and Pauline Mouches, published as *Cerebrovascular Probability Atlas*
([Figshare DOI 10.6084/m9.figshare.7026179.v1](https://doi.org/10.6084/m9.figshare.7026179.v1))
under CC0.

The source vessel-probability image was thresholded at 20% occurrence. Apparent
non-arterial structures were removed manually, the result was morphologically
processed, and the 0.5 mm image was resampled to the 2 mm FSL MNI152 grid.

## Anatomical region masks (`mni_regions`)

These masks were created by the PMRIPrep author from lobar masks assembled
following the FreeSurfer
[CorticalParcellation instructions](https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation).
The lobar masks were combined and manually edited in FSLview; left and right
components were split with `fslroi`. The bundled outputs are supratentorial and
cerebellar region masks, not lobar or NAWM masks.

The exact FreeSurfer version, source template or subject, original label list,
and manual-edit record are unavailable. These details are stated as unavailable
rather than reconstructed.
