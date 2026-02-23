# PMRIPrep: Perfusion MRI Preprocessing Pipeline
# BIDS-App container for DCE and DSC MRI processing
#
# Build (CPU, default):
#   docker build -t pmriprep:latest .
#
# Build (GPU / CUDA 12.1):
#   docker build -t pmriprep:latest --build-arg PYTORCH_VARIANT=cu121 .
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
LABEL org.opencontainers.image.version="0.2.0"
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
RUN mkdir -p ${nnUNet_raw_data_base} ${nnUNet_preprocessed} ${RESULTS_FOLDER}

# Install PyTorch (CPU by default; pass --build-arg PYTORCH_VARIANT=cu121 for GPU)
ARG PYTORCH_VARIANT=cpu
RUN if [ "$PYTORCH_VARIANT" != "cpu" ]; then \
        pip install --no-cache-dir torch torchvision \
            --index-url https://download.pytorch.org/whl/${PYTORCH_VARIANT}; \
    else \
        pip install --no-cache-dir torch torchvision \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install HD-BET for brain extraction
RUN pip install --no-cache-dir hd-bet

# Install HD-GLIO for glioma segmentation (requires CUDA at runtime)
RUN pip install --no-cache-dir HD-GLIO || \
    pip install --no-cache-dir git+https://github.com/MIC-DKFZ/HD-GLIO.git

# Copy source and install pmriprep
COPY --chown=1000:1000 . /src/pmriprep
WORKDIR /src/pmriprep

RUN if [ "$PYTORCH_VARIANT" != "cpu" ]; then \
        pip install --no-cache-dir -e ".[all]"; \
    else \
        pip install --no-cache-dir -e ".[dev]"; \
    fi && pip cache purge

# Download required TemplateFlow templates
RUN python -c "import templateflow; \
    from templateflow import api as tflow; \
    tflow.get('MNI152NLin2009cAsym', resolution=1, desc='brain', suffix='mask'); \
    tflow.get('MNI152NLin2009cAsym', resolution=1, suffix='T1w'); \
    tflow.get('MNI152NLin6Asym', resolution=1, suffix='T1w')"

# Build-time validation — fail the build early if something is broken
RUN python -c "\
import torch; import nibabel; import nipype; \
print(f'PyTorch {torch.__version__}'); \
print(f'CUDA available: {torch.cuda.is_available()}'); \
import HD_BET; print('HD-BET: OK'); \
import pmriprep; print('PMRIPrep: OK')"

# Set up user (non-root execution)
RUN useradd -m -u 1000 pmriprep || true
USER 1000
WORKDIR /home/pmriprep

# Create default working directories
RUN mkdir -p /home/pmriprep/.cache /home/pmriprep/work

# Set up entry point
ENTRYPOINT ["pmriprep"]

# Default help message
CMD ["--help"]
