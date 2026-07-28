#!/usr/bin/env bash
#
# PMRIPrep Docker Build Script
#
# Build the PMRIPrep Docker image and print Apptainer conversion instructions.
#
# Usage:
#   ./docker/build.sh        # Build the validated linux/amd64 CUDA image
#   ./docker/build.sh --push # Build and push to registry
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
PLATFORM="linux/amd64"

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
  --registry REG      Docker registry (default: docker.io, env: DOCKER_REGISTRY)
  --repo REPO         Repository name (default: pmriprep/pmriprep, env: DOCKER_REPO)
  --version VER       Override version (default: from pyproject.toml)
  -h, --help          Show this help message

Examples:
  # Build the validated linux/amd64 CUDA image locally
  ./docker/build.sh

  # Build and push to Docker Hub
  ./docker/build.sh --push

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

# Build the validated release architecture.
build_image() {
    local version="$1"
    local image_base="${REGISTRY}/${REPO}"
    local full_image="${image_base}:${version}"
    local latest_image="${image_base}:latest"

    log_info "Building image: $full_image"
    log_info "Platform: $PLATFORM (CUDA 12.1 PyTorch)"

    docker build \
        --platform "$PLATFORM" \
        --tag "$full_image" \
        --tag "$latest_image" \
        --label "org.opencontainers.image.version=${version}" \
        --label "org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --file "${PROJECT_ROOT}/Dockerfile" \
        "$PROJECT_ROOT"

    log_success "Built: $full_image"
    log_success "Tagged as: $latest_image"

    if $PUSH; then
        log_info "Pushing to registry..."
        docker push "$full_image"
        docker push "$latest_image"
        log_success "Pushed: $full_image"
        log_success "Pushed: $latest_image"
    fi
}

# Print Apptainer conversion instructions
print_apptainer_instructions() {
    local version="$1"
    local image_base="${REGISTRY}/${REPO}"
    local full_image="${image_base}:${version}"

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
        echo "apptainer pull pmriprep_latest.sif docker://${image_base}:latest"
    else
        echo "# From local Docker daemon:"
        echo "apptainer pull pmriprep_${version}.sif docker-daemon://${full_image}"
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

    build_image "$version"

    # Print Apptainer instructions
    print_apptainer_instructions "$version"

    log_success "Build complete!"
}

# Run main
main "$@"
