# PMRIPrep container image definition
#
# MIT License
#
# Copyright (c) 2024-2026 PMRIPrep Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# The PMRIPrep Python package installed by this definition is Apache-2.0.
#
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
#     /data /out participant --hdbet-device cuda --tumor-seg-device cuda

FROM condaforge/miniforge3:24.11.3-2

LABEL maintainer="PMRIPrep Developers"
LABEL org.opencontainers.image.title="PMRIPrep"
LABEL org.opencontainers.image.description="Perfusion MRI Preprocessing Pipeline - A NiPreps-compatible BIDS-App"
LABEL org.opencontainers.image.version="0.83.16"
LABEL org.opencontainers.image.source="https://github.com/arulseh/pmriprep"
LABEL org.opencontainers.image.licenses="Apache-2.0 AND CC0-1.0"

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

# NVIDIA container labels — enables proper GPU pass-through with
# `docker run --gpus all` and Apptainer's `--nv` flag.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

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
    graphviz \
    libfontconfig1 \
    libfreetype6 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgomp1 \
    libxext6 \
    libxrender1 \
    nodejs \
    npm \
    tar \
    tini \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install bids-validator (legacy JS CLI; the Deno v2 rewrite would require Deno 2)
RUN npm install -g bids-validator@1.14.10 && npm cache clean --force

# Install Python 3.10, FSL, and ANTs via mamba (fast solver)
# FSL and ANTs are separate installs because the FSL channel does not
# carry ANTs packages (mixing channels in one solve can confuse the solver).
RUN mamba install -y -c conda-forge python=3.10 pip=24.0 && mamba clean -afy

RUN mamba install -y \
    -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ \
    -c conda-forge \
    fsl-avwutils=2209.3 \
    fsl-bet2=2111.8 \
    fsl-fast4=2111.3 \
    fsl-flirt=2111.4 \
    fsl-mcflirt=2111.0 \
    && mamba clean -afy

RUN mamba install -y -c conda-forge ants=2.6.5 && mamba clean -afy

# FSLDIR = conda prefix; needed by FSL command-line tools (flirt, bet, etc.)
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
RUN pip install --no-cache-dir torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install HD-BET for brain extraction and pre-download model weights.
# Overwrite paths.py so weights live at /opt/ (not $HOME) — Apptainer
# auto-binds host $HOME over the container's, hiding baked-in weights.
RUN pip install --no-cache-dir hd-bet==2.0.1 && \
    HDBET_LOC=$(pip show hd-bet 2>/dev/null | grep -i "^Location:" | awk '{print $2}') && \
    [ -n "$HDBET_LOC" ] || { echo "ERROR: hd-bet not installed"; exit 1; } && \
    HDBET_PATHS="${HDBET_LOC}/HD_BET/paths.py" && \
    [ -f "$HDBET_PATHS" ] || { echo "ERROR: $HDBET_PATHS not found"; exit 1; } && \
    HDBET_URL=$(grep -oP "ZENODO_DOWNLOAD_URL\s*=\s*['\"]\\K[^'\"]*" "$HDBET_PATHS") && \
    printf "import os\nfolder_with_parameter_files = '/opt/hd-bet_params/release_2.0.0'\nZENODO_DOWNLOAD_URL = '%s'\n" "$HDBET_URL" > "$HDBET_PATHS" && \
    python -c "from HD_BET.paths import folder_with_parameter_files; assert folder_with_parameter_files.startswith('/opt/'), f'patch failed: {folder_with_parameter_files}'" && \
    python -c "from HD_BET.checkpoint_download import maybe_download_parameters; maybe_download_parameters()" && \
    chmod -R a+rX /opt/hd-bet_params

# Install HD-GLIO for glioma segmentation and pre-download model weights.
# Same /opt/ relocation as HD-BET above — Apptainer-safe.
RUN (pip install --no-cache-dir HD-GLIO==2.0 || \
     pip install --no-cache-dir git+https://github.com/CCI-Bonn/HD-GLIO.git@d1a37c79cfb0e5a253403b6e35cf66010950752b) && \
    HDGLIO_LOC=$(pip show hd-glio 2>/dev/null | grep -i "^Location:" | awk '{print $2}') && \
    [ -n "$HDGLIO_LOC" ] || { echo "ERROR: hd-glio not installed"; exit 1; } && \
    HDGLIO_PATHS="${HDGLIO_LOC}/hd_glio/paths.py" && \
    [ -f "$HDGLIO_PATHS" ] || { echo "ERROR: $HDGLIO_PATHS not found"; exit 1; } && \
    printf "import os\nfolder_with_parameter_files = '/opt/hd_glio_params'\n" > "$HDGLIO_PATHS" && \
    python -c "from hd_glio.paths import folder_with_parameter_files; assert folder_with_parameter_files.startswith('/opt/'), f'patch failed: {folder_with_parameter_files}'" && \
    python -c "from hd_glio.setup_hd_glio import maybe_download_weights; maybe_download_weights()" && \
    chmod -R a+rX /opt/hd_glio_params

# Copy source and install pmriprep
COPY --chown=1000:1000 . /src/pmriprep
WORKDIR /src/pmriprep

RUN pip install --no-cache-dir ".[test,dev]" && pip cache purge

# Download required TemplateFlow templates to a shared location
# (not under $HOME, which Apptainer auto-binds from the host)
ENV TEMPLATEFLOW_HOME="/opt/templateflow"
RUN python -c "from templateflow import api as tflow; \
    tflow.get('MNI152NLin6Asym', resolution=1, desc='brain', suffix='T1w')" && \
    chmod -R a+rX ${TEMPLATEFLOW_HOME}

# Build-time validation — fail the build early if something is broken
RUN python -c "\
import os, torch, nibabel, nipype; \
print(f'PyTorch {torch.__version__}'); \
print(f'CUDA available: {torch.cuda.is_available()}'); \
import HD_BET; print('HD-BET: OK'); \
assert os.path.isdir('/opt/hd-bet_params'), 'HD-BET weights missing'; \
print('HD-BET weights: OK'); \
assert os.path.isdir('/opt/hd_glio_params'), 'HD-GLIO weights missing'; \
print('HD-GLIO weights: OK'); \
import pmriprep; print('PMRIPrep: OK')"

RUN bids-validator --version

# Set up user (non-root execution)
# chmod go=u lets Apptainer's host UID (mapped into the container) write to $HOME
RUN useradd -m -s /bin/bash -G users -u 1000 pmriprep || true
ENV HOME="/home/pmriprep"
RUN mkdir -p /home/pmriprep/.cache && \
    chmod -R go=u /home/pmriprep
WORKDIR /tmp
USER 1000

# Set up entry point. tini reaps orphaned children from nipype multiproc
# + nnU-Net worker pools that would otherwise accumulate as zombies (a
# problem in JupyterHub-singleuser containers where the default init
# does not call wait() on SIGCHLD).
#
# The -s / --subreaper flag registers tini via prctl(PR_SET_CHILD_SUBREAPER)
# so it reaps orphaned descendants even when it is not PID 1 — required
# under Apptainer/Singularity, where the container runtime itself is PID 1.
ENTRYPOINT ["/usr/bin/tini", "-s", "--", "pmriprep"]

# Default help message
CMD ["--help"]
