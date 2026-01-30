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
export EXP_NAME="BUILD_DOCKER_${NNODES}N_${timestamp}"

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
# STEP 2: BUILD DOCKER IMAGE ON ALL NODES
# ============================================================================

echo ""
echo "========================================"
echo "STEP 2: Building Docker image on all nodes"
echo "========================================"

build_pids=()
for node in "${NODES[@]}"; do
    echo "Building image on $node..."
    ssh "$node" "bash -c '
        MAX_RETRIES=5
        INITIAL_DELAY=30
        MAX_DELAY=180
        RETRY_COUNT=0

        while true; do
            echo \"Node \$(hostname): Building image $IMAGE_TAG (Attempt \$((RETRY_COUNT + 1)))\"
            docker build --tag $IMAGE_TAG --file $SHARE_DOCKERFILE_PATH $(dirname $SHARE_DOCKERFILE_PATH)

            if [ \$? -eq 0 ]; then
                echo \"Image built successfully on \$(hostname)\"
                break
            else
                RETRY_COUNT=\$((RETRY_COUNT + 1))
                if [ \"\$RETRY_COUNT\" -ge \"\$MAX_RETRIES\" ]; then
                    echo \"Failed to build image after \$MAX_RETRIES attempts. Exiting.\"
                    exit 1
                fi

                CURRENT_DELAY=\$((INITIAL_DELAY * (2 ** (RETRY_COUNT - 1))))
                if [ \"\$CURRENT_DELAY\" -gt \"\$MAX_DELAY\" ]; then
                    CURRENT_DELAY=\"\$MAX_DELAY\"
                fi

                echo \"Build failed. Retrying in \$CURRENT_DELAY seconds...\"
                sleep \"\$CURRENT_DELAY\"
            fi
        done
    '" > "${LOG_DIR}/build_${node}.log" 2>&1 &
    build_pids+=($!)
done



