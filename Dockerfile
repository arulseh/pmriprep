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
#     /data /out participant \
#     --dsc-use-gpu --dce-use-gpu

FROM nipreps/miniconda:latest

LABEL maintainer="PMRIPrep Developers"
LABEL org.opencontainers.image.title="PMRIPrep"
LABEL org.opencontainers.image.description="Perfusion MRI Preprocessing Pipeline - A NiPreps-compatible BIDS-App"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/pmriprep/pmriprep"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set up environment variables
ENV FSLDIR=/opt/fsl
ENV FSLOUTPUTTYPE=NIFTI_GZ
ENV PATH="${FSLDIR}/bin:${PATH}"
ENV ANTSPATH=/opt/ants/bin
ENV PATH="${ANTSPATH}:${PATH}"
ENV nnUNet_raw_data_base=/opt/nnunet_data
ENV nnUNet_preprocessed=/opt/nnunet_preprocessed
ENV RESULTS_FOLDER=/opt/nnunet_results

# Install system dependencies
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
    libquadmath0 \
    libtiff5 \
    libxext6 \
    libxrender1 \
    tar \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install FSL (minimal installation for FLIRT, FAST, MCFLIRT, etc.)
# Using NeuroDebian repository for FSL
RUN wget -q -O /tmp/neurodebian.gpg.key https://neuro.debian.net/lists/focal.de-fzj.full && \
    apt-key add /tmp/neurodebian.gpg.key && \
    echo "deb https://neuro.debian.net/debian focal main" > /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends fsl-core fsl-atlases && \
    rm -rf /var/lib/apt/lists/* && \
    rm /tmp/neurodebian.gpg.key

# Alternatively, download FSL from official source if NeuroDebian fails
# Uncomment below and comment out NeuroDebian section if needed:
# ARG FSL_VERSION=6.0.7.1
# RUN mkdir -p /opt/fsl && \
#     curl -fsSL https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py -o /tmp/fslinstaller.py && \
#     python /tmp/fslinstaller.py -d /opt/fsl -V ${FSL_VERSION} && \
#     rm /tmp/fslinstaller.py

# Install ANTs
ARG ANTS_VERSION=2.5.0
RUN mkdir -p /opt/ants && \
    curl -fsSL https://github.com/ANTsX/ANTs/releases/download/v${ANTS_VERSION}/ants-${ANTS_VERSION}-linux-ubuntu22-X64.zip \
    -o /tmp/ants.zip && \
    unzip -q /tmp/ants.zip -d /tmp && \
    mv /tmp/ants-${ANTS_VERSION}/bin/* /opt/ants/bin/ && \
    mv /tmp/ants-${ANTS_VERSION}/lib/* /opt/ants/lib/ 2>/dev/null || true && \
    rm -rf /tmp/ants.zip /tmp/ants-${ANTS_VERSION}

# Create directories for nnU-Net (used by HD-BET and HD-GLIO)
RUN mkdir -p ${nnUNet_raw_data_base} ${nnUNet_preprocessed} ${RESULTS_FOLDER}

# Set up conda environment for pmriprep
COPY --chown=1000:1000 . /src/pmriprep
WORKDIR /src/pmriprep

# Install Python dependencies and the package
RUN conda install -y python=3.10 pip && \
    conda clean -afy

# Install PyTorch for HD-BET and HD-GLIO (CPU version by default)
# For GPU support, users should use nvidia/cuda base image and install pytorch with CUDA
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install HD-BET for brain extraction
RUN pip install --no-cache-dir hd-bet

# Install HD-GLIO for glioma segmentation
# Note: HD-GLIO may require manual installation of model weights
RUN pip install --no-cache-dir HD-GLIO || \
    pip install --no-cache-dir git+https://github.com/MIC-DKFZ/HD-GLIO.git

# Install pmriprep package
RUN pip install --no-cache-dir -e ".[all]" && \
    pip cache purge

# Download required TemplateFlow templates
RUN python -c "import templateflow; \
    from templateflow import api as tflow; \
    tflow.get('MNI152NLin2009cAsym', resolution=1, desc='brain', suffix='mask'); \
    tflow.get('MNI152NLin2009cAsym', resolution=1, suffix='T1w'); \
    tflow.get('MNI152NLin6Asym', resolution=1, suffix='T1w')"

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
