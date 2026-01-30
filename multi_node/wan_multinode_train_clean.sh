#!/bin/bash
# Multi-node JAX training without SLURM
# Usage: bash wan_multinode_train_no_slurm.sh "node1,node2,node3"
# 
# Environment Variables:
#   MULTI_NODES_LOG_DIR - Base directory for logs and outputs (default: /home/amd/jianhan/github/maxdiffusion/multi_node)
#
# Example with custom log directory:
#   MULTI_NODES_LOG_DIR=/custom/path bash wan_multinode_train_no_slurm.sh "node1,node2"

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

# Node list - comma separated hostnames
# Can be passed as first argument or hardcoded here
if [ -z "$1" ]; then
    # Default node list (edit this)
    NODE_LIST="core42-2,core42-4"
else
    NODE_LIST="$1"
fi

# Convert comma-separated list to array
IFS=',' read -ra NODES <<< "$NODE_LIST"
NNODES=${#NODES[@]}

# Coordinator is the first node
COORDINATOR_IP="${NODES[0]}"

# Paths and configuration
timestamp=$(date +%Y%m%d-%H%M%S)
IMAGE_TAG="jianhan-wan-multinode-train:v1"

# Base directory for all multi-node logs and outputs (configurable)
MULTI_NODES_LOG_DIR="${MULTI_NODES_LOG_DIR:-/home/amd/jianhan/multi_node_log}"

echo "========================================"
echo "Multi-node Training Configuration"
echo "========================================"
echo "Total nodes: $NNODES"
echo "Node list: ${NODES[@]}"
echo "Coordinator: $COORDINATOR_IP"
echo "Base log directory: $MULTI_NODES_LOG_DIR"
echo "========================================"

SHARE_DOCKERFILE_PATH="/home/amd/jianhan/github/maxdiffusion/multi_node/docker/jax_maxdiffusion_wan2.1_train_inference.ubuntu.amd.Dockerfile"
SHARED_CODE_BASE_PATH="/home/amd/jianhan/github/maxdiffusion"
MAXDIFFUSION_DIR_IN_DOCKER="/app/maxdiffusion"

# Experiment name
export EXP_NAME="CLEAN_${NNODES}N_${timestamp}"

# Directories under MULTI_NODES_LOG_DIR
export LOG_DIR="${MULTI_NODES_LOG_DIR}/slurm_logs/${EXP_NAME}/${NNODES}-nodes"

# Create directories locally
mkdir -p ${LOG_DIR}

# Log files
export HOST_OUTPUT_LOG="${LOG_DIR}/host_output.out"
export HOST_ERROR_LOG="${LOG_DIR}/host_output.err"

# Redirect output
exec > >(tee -a "${HOST_OUTPUT_LOG}") 2> >(tee -a "${HOST_ERROR_LOG}" >&2)

echo "Log directory: $LOG_DIR"
echo ""

# ============================================================================
# STEP 1: CLEANUP EXISTING CONTAINERS
# ============================================================================

echo ""
echo "========================================"
echo "STEP 1: Cleaning up existing containers"
echo "========================================"

# docker image rm -f '"$IMAGE_TAG"' 2>/dev/null || true
for node in "${NODES[@]}"; do
    echo "Cleaning containers on $node..."
    ssh "$node" 'bash -c '\''
        echo "Cleaning up on $(hostname)..."
        docker stop $(docker ps -q) 2>/dev/null || true
        docker rm $(docker ps -aq) 2>/dev/null || true
        docker image rm -f '"$IMAGE_TAG"' 2>/dev/null || true
        echo "Cleanup completed on $(hostname)"
    '\''' &
done

echo "Container cleanup completed on all nodes"

# ============================================================================
# STEP 2: SYNC CODEBASE TO ALL NODES
# ============================================================================

echo ""
echo "========================================"
echo "STEP 2: Syncing codebase to all nodes"
echo "========================================"

for node in "${NODES[@]}"; do
    echo "Syncing $SHARED_CODE_BASE_PATH to $node..."
    ssh "$node" "mkdir -p $(dirname $SHARED_CODE_BASE_PATH)"
    rsync -az --delete -e "ssh" "$SHARED_CODE_BASE_PATH/" "$node:$SHARED_CODE_BASE_PATH/"
    echo "✓ Synced to $node"
done

echo "Codebase sync completed on all nodes"
echo ""

