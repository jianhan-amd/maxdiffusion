#!/bin/bash
#
# Multi-Node Training: Wrapper Script
# 
# Description:
#   Unified wrapper script for multi-node training operations
#   Provides consistent interface for clean, build, and launch commands
#
# Usage:
#   bash wan_multinode_train.sh [nodes] [command]
#   bash wan_multinode_train.sh "node1,node2,node3" clean
#   bash wan_multinode_train.sh "node1,node2,node3" build
#   bash wan_multinode_train.sh "node1,node2,node3" launch
#   bash wan_multinode_train.sh "" clean  # Uses default node list
#
# Commands:
#   clean  - Cleanup containers and sync codebase
#   build  - Build Docker images on all nodes
#   launch - Launch training containers
#
# Environment Variables:
#   All environment variables supported by individual scripts
#   See scripts for details: wan_multinode_train_{clean,build,launch}.sh
#

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

# Get the directory where this script is located
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"

# Timestamp for this run
readonly TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Node list - first argument
if [ -z "${1:-}" ]; then
    # Default node list (edit this as needed)
    NODE_LIST="core42-5-a08u01,core42-1-a08u07,core42-3-a08u19,core42-4-a08u25"
else
    NODE_LIST="$1"
fi

# Command - second argument (required)
readonly CMD_RUN="${2:-}"

# Convert comma-separated list to array for validation
IFS=',' read -ra NODES <<< "$NODE_LIST"
readonly NNODES=${#NODES[@]}

# Default configuration values
readonly COORDINATOR_IP="${COORDINATOR_IP:-172.29.0.73}"
readonly IMAGE_TAG="${IMAGE_TAG:-maxdiffusion-multinode-train:v1}"
readonly MULTI_NODES_LOG_DIR="${MULTI_NODES_LOG_DIR:-/home/amd/jianhan/multi_node_log}"
readonly SHARE_DOCKERFILE_PATH="${SHARE_DOCKERFILE_PATH:-/home/amd/jianhan/github/maxdiffusion/multi_node/docker/jax_maxdiffusion_wan2.1_train_inference.ubuntu.amd.Dockerfile}"
readonly SHARED_CODE_BASE_PATH="${SHARED_CODE_BASE_PATH:-/home/amd/jianhan/github/maxdiffusion}"
readonly MAXDIFFUSION_DIR_IN_DOCKER="${MAXDIFFUSION_DIR_IN_DOCKER:-/app/maxdiffusion}"
readonly RUN_NAME="${RUN_NAME:-WAN_14B_FSDP8}"
readonly REMOVE_IMAGES="${REMOVE_IMAGES:-n}"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo "========================================"
    echo "$1"
    echo "========================================"
}

print_usage() {
    cat << EOF
Usage: $SCRIPT_NAME [nodes] [command]

Commands:
  clean  - Cleanup containers and sync codebase
  build  - Build Docker images on all nodes
  launch - Launch training containers

Arguments:
  nodes   - Comma-separated list of node hostnames
            Leave empty to use default node list

Environment Variables:
  COORDINATOR_IP            - JAX coordinator IP (default: 172.29.0.73)
  IMAGE_TAG                 - Docker image name (default: maxdiffusion-multinode-train:v1)
  MULTI_NODES_LOG_DIR       - Base log directory (default: /home/amd/jianhan/multi_node_log)
  SHARED_CODE_BASE_PATH     - Codebase path (default: /home/amd/jianhan/github/maxdiffusion)
  MAXDIFFUSION_DIR_IN_DOCKER - Docker mount path (default: /app/maxdiffusion)
  RUN_NAME                  - Experiment name prefix (default: WAN_14B_FSDP8)
  REMOVE_IMAGES             - Remove Docker images on clean? y/n (default: n)

Examples:
  # Clean and sync using specific nodes
  $SCRIPT_NAME "node1,node2,node3" clean

  # Build images using default nodes
  $SCRIPT_NAME "" build

  # Launch training with custom settings
  export RUN_NAME="WAN_1_3B_FSDP8"
  export REMOVE_IMAGES="y"
  $SCRIPT_NAME "node1,node2" launch

  # Full workflow
  $SCRIPT_NAME "node1,node2" clean
  $SCRIPT_NAME "node1,node2" build
  $SCRIPT_NAME "node1,node2" launch

EOF
}

print_config() {
    print_header "Multi-Node Training Wrapper"
    echo "Script:              $SCRIPT_NAME"
    echo "Command:             $CMD_RUN"
    echo "Timestamp:           $TIMESTAMP"
    echo "Nodes:               $NNODES (${NODES[*]})"
    echo ""
    echo "Configuration:"
    echo "  IMAGE_TAG:         $IMAGE_TAG"
    echo "  COORDINATOR_IP:    $COORDINATOR_IP"
    echo "  RUN_NAME:          $RUN_NAME"
    echo "  REMOVE_IMAGES:     $REMOVE_IMAGES"
    echo "  LOG_DIR:           $MULTI_NODES_LOG_DIR"
    echo "  CODE_PATH:         $SHARED_CODE_BASE_PATH"
    echo "========================================"
}

# ============================================================================
# OPERATION FUNCTIONS
# ============================================================================

run_clean() {
    print_header "Running: Clean and Sync"
    
    # Export environment variables for the clean script
    export REMOVE_IMAGES
    export IMAGE_TAG
    export MULTI_NODES_LOG_DIR
    export SHARED_CODE_BASE_PATH
    
    # Run clean script
    if bash "${SCRIPT_DIR}/wan_multinode_train_clean.sh" "$NODE_LIST"; then
        echo ""
        echo "✓ Clean and sync completed successfully"
        return 0
    else
        echo ""
        echo "✗ Clean and sync failed"
        return 1
    fi
}

run_build() {
    print_header "Running: Build Docker Images"
    
    # Export environment variables for the build script
    export IMAGE_TAG
    export MULTI_NODES_LOG_DIR
    export SHARE_DOCKERFILE_PATH
    
    # Run build script
    if bash "${SCRIPT_DIR}/wan_multinode_train_build_docker.sh" "$NODE_LIST"; then
        echo ""
        echo "✓ Docker build completed successfully"
        return 0
    else
        echo ""
        echo "✗ Docker build failed"
        return 1
    fi
}

run_launch() {
    print_header "Running: Launch Training"
    
    # Export environment variables for the launch script
    export COORDINATOR_IP
    export IMAGE_TAG
    export MULTI_NODES_LOG_DIR
    export SHARED_CODE_BASE_PATH
    export MAXDIFFUSION_DIR_IN_DOCKER
    export RUN_NAME
    
    # Run launch script
    if bash "${SCRIPT_DIR}/wan_multinode_train_launch.sh" "$NODE_LIST"; then
        echo ""
        echo "✓ Training launched successfully"
        return 0
    else
        echo ""
        echo "✗ Training launch failed"
        return 1
    fi
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Validate command
if [ -z "$CMD_RUN" ]; then
    echo "Error: No command specified"
    echo ""
    print_usage
    exit 1
fi

# Print configuration
print_config
echo ""

# Execute based on command
case "$CMD_RUN" in
    clean)
        run_clean
        exit_code=$?
        ;;
    build)
        run_build
        exit_code=$?
        ;;
    launch)
        run_launch
        exit_code=$?
        ;;
    *)
        echo "Error: Invalid command '$CMD_RUN'"
        echo ""
        print_usage
        exit 1
        ;;
esac

# Final status
echo ""
if [ $exit_code -eq 0 ]; then
    print_header "SUCCESS"
    echo "Command '$CMD_RUN' completed successfully"
else
    print_header "FAILED"
    echo "Command '$CMD_RUN' failed with exit code $exit_code"
fi

exit $exit_code
