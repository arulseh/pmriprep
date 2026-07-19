Usage
=====

.. include:: links.rst

PMRIPrep follows the BIDS-App_ command-line convention::

   pmriprep <bids_dir> <output_dir> participant [options]

Quick start
-----------

.. code-block:: bash

   pmriprep /path/to/bids /path/to/output participant \
       --participant-label 01 02 03

Input requirements
------------------

A normal run requires a BIDS ``dataset_description.json`` and at least one T1w image.
Post-contrast-only T1w processing is supported, although pre-contrast T1w is preferred.
DSC and DCE pipelines run when a corresponding 4D ``*_dsc.nii.gz`` or
``*_dce.nii.gz`` image is discovered.

Every perfusion NIfTI requires a matching JSON sidecar. DSC requires ``EchoTime``, volume
``RepetitionTime``, and ``MagneticFieldStrength``. DCE requires ``FlipAngle``, volume
``RepetitionTime``, and ``MagneticFieldStrength``; 3D acquisitions with a long volume TR
also require pulse ``RepetitionTimeExcitation``. Missing required fields stop workflow
construction because they enter the signal conversion or time axis directly.

``ContrastBolusAgent`` is strongly recommended for agent-specific dose and relaxivity
handling. Multiple VFA images, B1+ maps, structural modalities for tumor segmentation,
and user masks or AIFs are optional; see the README configuration tables for details.

Command-line arguments
----------------------

.. argparse::
   :ref: pmriprep.cli.run._build_parser
   :prog: pmriprep
   :nodefault:
