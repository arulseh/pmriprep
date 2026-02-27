#!/usr/bin/env bash
#
# PMRIPrep Docker Build Script
#
# Build Docker and Apptainer containers for PMRIPrep.
#
# Usage:
#   ./docker/build.sh                    # Build local image
#   ./docker/build.sh --push             # Build and push to registry
#   ./docker/build.sh --multiarch        # Build for amd64 and arm64
#   ./docker/build.sh --multiarch --push # Build multiarch and push
#
# Environment variables:
#   DOCKER_REGISTRY   - Docker registry (default: docker.io)
#   DOCKER_REPO       - Repository name (default: pmriprep/pmriprep)
#   VERSION           - Override version (default: from pyproject.toml)
#
# Apptainer conversion (after building):
#   apptainer pull pmriprep.sif docker://pmriprep/pmriprep:latest
#   apptainer pull pmriprep.sif docker-daemon://pmriprep:latest
#
# See bottom of file for detailed Apptainer instructions.

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default settings
REGISTRY="${DOCKER_REGISTRY:-docker.io}"
REPO="${DOCKER_REPO:-pmriprep/pmriprep}"
PUSH=false
MULTIARCH=false
GPU=false
PYTORCH_VARIANT="cpu"
PLATFORMS="linux/amd64,linux/arm64"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --push)
                PUSH=true
                shift
                ;;
            --multiarch)
                MULTIARCH=true
                shift
                ;;
            --gpu)
                GPU=true
                PYTORCH_VARIANT="cu121"
                shift
                ;;
            --pytorch-variant)
                PYTORCH_VARIANT="$2"
                shift 2
                ;;
            --platforms)
                PLATFORMS="$2"
                shift 2
                ;;
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --repo)
                REPO="$2"
                shift 2
                ;;
            --version)
                VERSION="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Show help message
show_help() {
    cat << EOF
PMRIPrep Docker Build Script

Usage: $(basename "$0") [OPTIONS]

Options:
  --push              Push image to registry after building
  --multiarch         Build for multiple architectures (amd64, arm64)
  --gpu               Build with CUDA 12.1 PyTorch (for NVIDIA GPU support)
  --pytorch-variant V PyTorch variant (default: cpu, e.g. cu121, cu124)
  --platforms PLAT    Comma-separated list of platforms (default: linux/amd64,linux/arm64)
  --registry REG      Docker registry (default: docker.io, env: DOCKER_REGISTRY)
  --repo REPO         Repository name (default: pmriprep/pmriprep, env: DOCKER_REPO)
  --version VER       Override version (default: from pyproject.toml)
  -h, --help          Show this help message

Examples:
  # Build local CPU image
  ./docker/build.sh

  # Build with GPU support (CUDA 12.1)
  ./docker/build.sh --gpu

  # Build and push to Docker Hub
  ./docker/build.sh --push

  # Build for multiple architectures and push
  ./docker/build.sh --multiarch --push

  # Build for custom registry
  ./docker/build.sh --registry ghcr.io --repo myorg/pmriprep --push

Apptainer Conversion:
  After building, convert to Apptainer with:

  # From registry (after --push)
  apptainer pull pmriprep.sif docker://pmriprep/pmriprep:latest

  # From local Docker daemon
  apptainer pull pmriprep.sif docker-daemon://pmriprep:latest
EOF
}

# Extract version from pyproject.toml
get_version() {
    if [[ -n "${VERSION:-}" ]]; then
        echo "$VERSION"
        return
    fi

    local pyproject="${PROJECT_ROOT}/pyproject.toml"
    if [[ ! -f "$pyproject" ]]; then
        log_error "pyproject.toml not found at: $pyproject"
        exit 1
    fi

    # Extract version using grep and sed
    local version
    version=$(grep -E '^version\s*=' "$pyproject" | head -1 | sed 's/.*"\(.*\)".*/\1/')

    if [[ -z "$version" ]]; then
        log_error "Could not extract version from pyproject.toml"
        exit 1
    fi

    echo "$version"
}

# Check Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
}

# Check buildx is available for multiarch builds
check_buildx() {
    if ! docker buildx version &> /dev/null; then
        log_error "Docker buildx is not available. Install it for multi-arch builds."
        log_info "See: https://docs.docker.com/buildx/working-with-buildx/"
        exit 1
    fi
}

# Setup buildx builder for multiarch
setup_buildx() {
    local builder_name="pmriprep-builder"

    # Check if builder exists
    if ! docker buildx inspect "$builder_name" &> /dev/null; then
        log_info "Creating buildx builder: $builder_name"
        docker buildx create --name "$builder_name" --use --bootstrap
    else
        log_info "Using existing buildx builder: $builder_name"
        docker buildx use "$builder_name"
    fi
}

# Build single-arch image
build_single() {
    local version="$1"
    local full_image="${REPO}:${version}"
    local latest_image="${REPO}:latest"

    log_info "Building image: $full_image"

    docker build \
        --tag "$full_image" \
        --tag "$latest_image" \
        --build-arg "PYTORCH_VARIANT=${PYTORCH_VARIANT}" \
        --label "org.opencontainers.image.version=${version}" \
        --label "org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --file "${PROJECT_ROOT}/Dockerfile" \
        "$PROJECT_ROOT"

    log_success "Built: $full_image (PyTorch variant: ${PYTORCH_VARIANT})"
    log_success "Tagged as: $latest_image"

    if $PUSH; then
        log_info "Pushing to registry..."
        docker push "$full_image"
        docker push "$latest_image"
        log_success "Pushed: $full_image"
        log_success "Pushed: $latest_image"
    fi
}

# Build multi-arch image
build_multiarch() {
    local version="$1"
    local full_image="${REPO}:${version}"
    local latest_image="${REPO}:latest"

    log_info "Building multi-arch image: $full_image"
    log_info "Platforms: $PLATFORMS"

    local push_flag=""
    if $PUSH; then
        push_flag="--push"
    else
        push_flag="--load"
        # Note: --load only works for single platform
        log_warn "Multi-arch build without --push will only load the native platform image"
        # Get native platform
        local native_platform
        native_platform=$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')
        PLATFORMS="$native_platform"
        log_info "Using native platform: $PLATFORMS"
    fi

    docker buildx build \
        --platform "$PLATFORMS" \
        --tag "$full_image" \
        --tag "$latest_image" \
        --build-arg "PYTORCH_VARIANT=${PYTORCH_VARIANT}" \
        --label "org.opencontainers.image.version=${version}" \
        --label "org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --file "${PROJECT_ROOT}/Dockerfile" \
        $push_flag \
        "$PROJECT_ROOT"

    log_success "Built multi-arch: $full_image"
    log_success "Tagged as: $latest_image"

    if $PUSH; then
        log_success "Pushed to registry with platforms: $PLATFORMS"
    fi
}

# Print Apptainer conversion instructions
print_apptainer_instructions() {
    local version="$1"
    local full_image="${REPO}:${version}"

    echo ""
    echo "=============================================="
    echo "Apptainer Conversion Instructions"
    echo "=============================================="
    echo ""
    echo "Convert the Docker image to Apptainer:"
    echo ""

    if $PUSH; then
        echo "# From Docker Hub (recommended after push):"
        echo "apptainer pull pmriprep_${version}.sif docker://${full_image}"
        echo ""
        echo "# Or for latest:"
        echo "apptainer pull pmriprep_latest.sif docker://${REPO}:latest"
    else
        echo "# From local Docker daemon:"
        echo "apptainer pull pmriprep_${version}.sif docker-daemon://${REPO}:${version}"
        echo ""
        echo "# Note: This requires Docker images to be available locally."
        echo "# Run with --push to push to registry for easier Apptainer conversion."
    fi

    echo ""
    echo "TemplateFlow templates are pre-populated at /opt/templateflow inside the image."
    echo ""
    echo "Run Apptainer container:"
    echo "  apptainer run --cleanenv \\"
    echo "    --bind /path/to/bids_dir:/data:ro \\"
    echo "    --bind /path/to/output:/out \\"
    echo "    --bind /path/to/work:/work \\"
    echo "    pmriprep_${version}.sif \\"
    echo "    /data /out participant \\"
    echo "    --participant-label 01 \\"
    echo "    -w /work"
    echo ""
    echo "For GPU support (NVIDIA):"
    echo "  apptainer run --nv --cleanenv \\"
    echo "    --bind /path/to/bids_dir:/data:ro \\"
    echo "    --bind /path/to/output:/out \\"
    echo "    --bind /path/to/work:/work \\"
    echo "    pmriprep_${version}.sif \\"
    echo "    /data /out participant \\"
    echo "    --use-gpu"
    echo ""
    echo "=============================================="
}

# Main function
main() {
    parse_args "$@"

    log_info "PMRIPrep Docker Build Script"
    log_info "Project root: $PROJECT_ROOT"

    # Check prerequisites
    check_docker

    # Get version
    local version
    version=$(get_version)
    log_info "Version: $version"

    # Build
    if $MULTIARCH; then
        check_buildx
        setup_buildx
        build_multiarch "$version"
    else
        build_single "$version"
    fi

    # Print Apptainer instructions
    print_apptainer_instructions "$version"

    log_success "Build complete!"
}

# Run main
main "$@"
