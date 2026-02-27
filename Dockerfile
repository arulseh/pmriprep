# PMRIPrep: Perfusion MRI Preprocessing Pipeline
# BIDS-App container for DCE and DSC MRI processing
#
# Build:
#   docker build -t pmriprep:latest .
#
# Run:
#   docker run -it --rm \
#     -v /path/to/bids_dir:/data:ro \
#     -v /path/to/output:/out \
#     -v /path/to/work:/work \
#     pmriprep:latest \
#     /data /out participant \
#     --participant-label 01 \
#     -w /work
#
# Volume mounts required:
#   /data   - BIDS input directory (read-only recommended)
#   /out    - Output directory for derivatives
#   /work   - Working directory for intermediate files (optional, recommended)
#
# GPU support (NVIDIA):
#   docker run --gpus all -it --rm \
#     -v /path/to/bids_dir:/data:ro \
#     -v /path/to/output:/out \
#     pmriprep:latest \
#     /data /out participant --use-gpu

FROM condaforge/miniforge3:latest

LABEL maintainer="PMRIPrep Developers"
LABEL org.opencontainers.image.title="PMRIPrep"
LABEL org.opencontainers.image.description="Perfusion MRI Preprocessing Pipeline - A NiPreps-compatible BIDS-App"
LABEL org.opencontainers.image.version="0.3.0"
LABEL org.opencontainers.image.source="https://github.com/pmriprep/pmriprep"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Thread control — prevent MKL/OpenBLAS from spawning threads inside
# Nipype workers, which leads to oversubscription on HPC nodes.
ENV MKL_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

# Conda prefix
ENV CONDA_DIR=/opt/conda

# Matplotlib cache dir — prevents writes to ~/.config/matplotlib (read-only in Apptainer)
ENV MPLCONFIGDIR="/tmp/matplotlib"

# NiPreps container-detection convention
ENV IS_DOCKER_8395080871=1

# nnU-Net directory layout (used by HD-BET and HD-GLIO)
ENV nnUNet_raw_data_base=/opt/nnunet_data
ENV nnUNet_preprocessed=/opt/nnunet_preprocessed
ENV RESULTS_FOLDER=/opt/nnunet_results

# Install system dependencies (Ubuntu-based)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    bc \
    curl \
    dc \
    file \
    git \
    libfontconfig1 \
    libfreetype6 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgomp1 \
    libxext6 \
    libxrender1 \
    tar \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10, FSL, and ANTs via mamba (fast solver)
# FSL and ANTs are separate installs because the FSL channel does not
# carry ANTs packages (mixing channels in one solve can confuse the solver).
RUN mamba install -y -c conda-forge python=3.10 pip && mamba clean -afy

RUN mamba install -y \
    -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ \
    -c conda-forge \
    fsl-avwutils fsl-bet2 fsl-fast4 fsl-flirt fsl-mcflirt fsl-cluster \
    fsl-data_standard \
    && mamba clean -afy

RUN mamba install -y -c conda-forge ants && mamba clean -afy

# FSLDIR = conda prefix; FSL data lands at $CONDA_DIR/data/standard/
ENV FSLDIR=${CONDA_DIR}
ENV FSLOUTPUTTYPE=NIFTI_GZ

# Create directories for nnU-Net (used by HD-BET and HD-GLIO)
# nnunet.paths.py calls maybe_mkdir_p() at import time, creating subdirs.
# These must be world-writable for Apptainer (read-only container FS).
RUN mkdir -p ${nnUNet_raw_data_base}/nnUNet_raw_data \
             ${nnUNet_preprocessed} \
             ${RESULTS_FOLDER} && \
    chmod -R 777 ${nnUNet_raw_data_base} ${nnUNet_preprocessed} ${RESULTS_FOLDER}

# Install PyTorch with CUDA 12.1 support (GPU-accelerated tumor segmentation)
# The preflight check in decide_pipelines() auto-skips tumor seg at runtime
# if no GPU is detected, so shipping GPU PyTorch is always safe.
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# Install HD-BET for brain extraction and pre-download model weights
# (avoids runtime download failure on Apptainer's read-only filesystem)
RUN pip install --no-cache-dir hd-bet && \
    python -c "from HD_BET.checkpoint_download import maybe_download_parameters; maybe_download_parameters()"

# Install HD-GLIO for glioma segmentation and pre-download model weights
# (avoids runtime download failure on Apptainer's read-only filesystem)
# hd_glio/paths.py resolves ~ ($HOME) for weight storage; at build time
# $HOME=/root, so we download there and move to runtime user's home.
RUN pip install --no-cache-dir HD-GLIO || \
    pip install --no-cache-dir git+https://github.com/MIC-DKFZ/HD-GLIO.git
RUN python -c "from hd_glio.setup_hd_glio import maybe_download_weights; maybe_download_weights()" && \
    mkdir -p /home/pmriprep && \
    mv /root/hd_glio_params /home/pmriprep/hd_glio_params && \
    chmod -R a+rX /home/pmriprep/hd_glio_params

# Install CuPy (GPU-accelerated deconvolution) — prebuilt CUDA 12.x wheel,
# same pattern as PyTorch above (generic `cupy` has no prebuilt wheels).
RUN pip install --no-cache-dir cupy-cuda12x

# Copy source and install pmriprep
COPY --chown=1000:1000 . /src/pmriprep
WORKDIR /src/pmriprep

RUN pip install --no-cache-dir ".[test,dev]" && pip cache purge

# Download required TemplateFlow templates to a shared location
# (not under $HOME, which Apptainer auto-binds from the host)
ENV TEMPLATEFLOW_HOME="/opt/templateflow"
RUN python -c "import templateflow; \
    from templateflow import api as tflow; \
    tflow.get('MNI152NLin2009cAsym', resolution=1, desc='brain', suffix='mask'); \
    tflow.get('MNI152NLin2009cAsym', resolution=1, suffix='T1w'); \
    tflow.get('MNI152NLin6Asym', resolution=1, suffix='T1w')" && \
    chmod -R a+rX ${TEMPLATEFLOW_HOME}

# Build-time validation — fail the build early if something is broken
RUN python -c "\
import torch; import nibabel; import nipype; \
print(f'PyTorch {torch.__version__}'); \
print(f'CUDA available: {torch.cuda.is_available()}'); \
import HD_BET; print('HD-BET: OK'); \
import pmriprep; print('PMRIPrep: OK')"

# Set up user (non-root execution)
# chmod go=u lets Apptainer's host UID (mapped into the container) write to $HOME
RUN useradd -m -s /bin/bash -G users -u 1000 pmriprep || true
ENV HOME="/home/pmriprep"
RUN mkdir -p /home/pmriprep/.cache && \
    chmod -R go=u /home/pmriprep
WORKDIR /tmp
USER 1000

# Set up entry point
ENTRYPOINT ["pmriprep"]

# Default help message
CMD ["--help"]
