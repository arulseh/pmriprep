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

Command-line arguments
----------------------

.. argparse::
   :ref: pmriprep.cli.run._build_parser
   :prog: pmriprep
   :nodefault:
