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

The default CPU budget is capped at 8; consider setting ``--nprocs`` if more CPUs
are available. ``--omp-nthreads`` limits threads per process. Remote scheduler
plugins require an explicit ``--nprocs``.

Input requirements
------------------

PMRIPrep requires brain MRI data; other anatomical regions are currently not supported.
A normal run requires ``dataset_description.json`` and an anatomical T1w image.
Post-contrast-only T1w anatomy is supported, but downstream options requiring a
pre-contrast T1w reference are unavailable. DSC and DCE run when corresponding
4D ``*_dsc.nii.gz`` and ``*_dce.nii.gz`` images are present. HD-GLIO requires pre-
and post-contrast T1w, T2w, and FLAIR.

DSC requires a single-echo gradient-echo time series; DCE requires a single-echo
spoiled gradient-echo time series. Each enabled modality requires exactly one
candidate per subject/session in the input dataset.

Every perfusion NIfTI requires a matching JSON sidecar. DSC requires ``EchoTime``, volume
``RepetitionTime``, and ``MagneticFieldStrength``. DCE requires ``FlipAngle``,
``MagneticFieldStrength``, volume TR, and excitation TR. For 3D-excitation DCE,
``RepetitionTime`` is the volume TR and ``RepetitionTimeExcitation`` is the excitation
TR. For 2D-excitation DCE, supply ``RepetitionTimeExcitation`` as well, unless each
slice is excited once per volume (single-shot, e.g. EPI), where ``RepetitionTime``
serves both.

DCE resolves relaxivity from ``ContrastBolusAgent`` and ``MagneticFieldStrength``.
A missing or unrecognized agent, or a field above 5 T, requires an explicit positive
``dce.r1_relaxivity`` value (or ``--dce-r1``), in s⁻¹ mM⁻¹ per Gd.

VFA images, B1+ maps, and user masks or AIFs are optional. PMRIPrep automatically
creates lesion and normal-appearing white-matter (NAWM) masks when tumor segmentation runs with the
required inputs; supplying masks through the configuration is optional. See the
`README configuration tables <https://github.com/arulseh/pmriprep#configuration>`_
for details.

Configuration
-------------

Use a YAML or JSON file to set configuration options:

.. code-block:: bash

   pmriprep /data/bids /data/out participant --config-file my_config.yaml

See the `configuration tables <https://github.com/arulseh/pmriprep#configuration>`_
for available options and defaults. CLI options override their file equivalents.

Command-line arguments
----------------------

.. argparse::
   :ref: pmriprep.cli.run._build_parser
   :prog: pmriprep
   :nodefault:
   :noepilog:
