Installation
============

.. include:: links.rst

PMRIPrep is distributed as a Docker_ / Apptainer_ container image (recommended) and can
be installed from source for development. It is not yet published on PyPI.

Docker (recommended)
--------------------

.. code-block:: bash

   docker pull ghcr.io/arulseh/pmriprep:latest

   docker run --rm \
       -v /path/to/bids:/data:ro \
       -v /path/to/output:/out \
       -v /path/to/work:/work \
       ghcr.io/arulseh/pmriprep:latest \
       /data /out participant --participant-label 01 -w /work

Add ``--gpus all`` for GPU support (required for HD-GLIO tumor segmentation). When no GPU
is detected, tumor segmentation is skipped automatically and the remaining pipelines run.

Apptainer
---------

.. code-block:: bash

   apptainer pull pmriprep.sif docker://ghcr.io/arulseh/pmriprep:latest

   apptainer run --cleanenv --nv \
       --bind /path/to/bids:/data:ro \
       --bind /path/to/output:/out \
       --bind /path/to/work:/work \
       pmriprep.sif \
       /data /out participant --participant-label 01 -w /work

TemplateFlow templates are pre-populated at ``/opt/templateflow`` inside the image, so no
extra bind mounts are needed.

Source installation (development)
---------------------------------

.. code-block:: bash

   git clone https://github.com/arulseh/pmriprep.git
   cd pmriprep
   python -m pip install .

Requires FSL_ and ANTs_ on ``$PATH``; see the ``Dockerfile`` for the full system
dependency list.

System requirements
-------------------

Resource use depends on image dimensions, time points, enabled pipelines, inference method,
and segmentation tools; no universal RAM, disk, or runtime minimum has been validated. Run
one representative subject first and size batch jobs from its measured peak memory and work
directory use. The work directory can be substantially larger than the final derivatives,
so local scratch storage is recommended. Keep it until processing and report generation have
completed successfully.

A GPU is optional and used only for brain extraction (HD-BET) and tumor segmentation
(HD-GLIO). Runtime and memory concurrency scale with ``--nprocs`` and
``--omp-nthreads``.
