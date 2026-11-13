Installation
============

.. include:: links.rst

PMRIPrep is distributed as a Docker_ / Apptainer_ container image (recommended) and can
also be installed from source for development.

Docker (recommended)
--------------------

.. code-block:: bash

   docker pull ghcr.io/arulseh/pmriprep:latest

   docker run --rm --init --gpus 1 \
       -v /path/to/bids:/data:ro \
       -v /path/to/output:/out \
       -v /path/to/work:/work \
       ghcr.io/arulseh/pmriprep:latest \
       /data /out participant --participant-label 01 -w /work

Remove ``--gpus 1`` when running without an NVIDIA GPU. HD-BET falls back to CPU;
HD-GLIO tumor segmentation is skipped when GPU is unavailable.

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

Remove ``--nv`` when running without an NVIDIA GPU. HD-BET falls back to CPU;
HD-GLIO tumor segmentation is skipped when GPU is unavailable.

TemplateFlow templates are pre-populated at ``/opt/templateflow`` inside the image, so no
extra bind mounts are needed.

Source installation (development)
---------------------------------

.. code-block:: bash

   git clone https://github.com/arulseh/pmriprep.git
   cd pmriprep
   python -m pip install .

Building from source requires a C compiler and Python development headers.
Source installations require Python 3.11–3.12, FSL_, and ANTs_ on ``$PATH``.
HD-BET and HD-GLIO are optional: missing HD-BET falls back to ANTs for anatomical
images and to FSL BET for DSC/DCE images, while missing HD-GLIO skips tumor
segmentation. The container uses Python 3.11; see the ``Dockerfile`` for the
full system dependency list.

System requirements
-------------------

Runtime depends on image dimensions, time points, enabled pipelines, inference
method, and available hardware.
