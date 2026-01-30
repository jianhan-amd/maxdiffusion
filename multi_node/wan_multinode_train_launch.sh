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
export EXP_NAME="WAN_1_3B_FSDP8_${NNODES}N_${timestamp}"

# Directories under MULTI_NODES_LOG_DIR
export LOG_DIR="${MULTI_NODES_LOG_DIR}/slurm_logs/${EXP_NAME}/${NNODES}-nodes"
OUTPUT_DIR="${MULTI_NODES_LOG_DIR}/output/${EXP_NAME}/${NNODES}-nodes"
OUTPUT_DIR_IN_DOCKER="/app/output-${EXP_NAME}/${NNODES}-nodes"

# Create directories locally
mkdir -p ${LOG_DIR}
mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/configs/models

# Log files
export HOST_OUTPUT_LOG="${LOG_DIR}/host_output.out"
export HOST_ERROR_LOG="${LOG_DIR}/host_output.err"

# Redirect output
exec > >(tee -a "${HOST_OUTPUT_LOG}") 2> >(tee -a "${HOST_ERROR_LOG}" >&2)

echo "Log directory: $LOG_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# ============================================================================
# STEP 3: LAUNCH TRAINING CONTAINERS
# ============================================================================

echo ""
echo "========================================"
echo "STEP 3: Launching training containers"
echo "========================================"
echo "Coordinator IP: $COORDINATOR_IP"
echo "JAX Coordinator Port: 12345"
echo ""

# --volume /dev/infiniband:/dev/infiniband \

launch_pids=()
for i in "${!NODES[@]}"; do
    NODE="${NODES[$i]}"
    NODE_RANK=$i
    
    echo "Launching training on $NODE (rank $NODE_RANK/$NNODES)..."
    
    ssh "$NODE" "bash -c '
        docker run --rm --privileged --network host \
        --cap-add=IPC_LOCK \
        --tmpfs /dev/shm:size=200G \
        --volume $OUTPUT_DIR:$OUTPUT_DIR_IN_DOCKER \
        --volume $SHARED_CODE_BASE_PATH:$MAXDIFFUSION_DIR_IN_DOCKER \
        -e JAX_COORDINATOR_IP=$COORDINATOR_IP \
        -e JAX_COORDINATOR_PORT=12345 \
        -e NNODES=$NNODES \
        -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        -e NODE_RANK=$NODE_RANK \
        -e JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT_SECONDS=1800 \
        -w $MAXDIFFUSION_DIR_IN_DOCKER \
        $IMAGE_TAG \
        /bin/bash -c \"
            set -ex
            set -o pipefail
            trap \\\"echo \\\\\\\"Error on line \\\$LINENO\\\\\\\"\\\" ERR

            echo \\\"Image: $IMAGE_TAG\\\"
            echo \\\"Starting node $NODE_RANK of $NNODES\\\"
            echo \\\"Coordinator IP: \\\$JAX_COORDINATOR_IP\\\"
            echo \\\"Node: \\\$(hostname)\\\"

            # Create output directory
            export BASE_OUTPUT_DIRECTORY=\\\"$OUTPUT_DIR_IN_DOCKER\\\"
            mkdir -p \\\${BASE_OUTPUT_DIRECTORY}

            chmod 777 $MAXDIFFUSION_DIR_IN_DOCKER -R
            cd $MAXDIFFUSION_DIR_IN_DOCKER
            bash launch.sh LOG_PATH=$OUTPUT_DIR_IN_DOCKER
        \"
    '" > "${LOG_DIR}/node_${NODE}_rank_${NODE_RANK}.log" 2>&1 &
    
    launch_pids+=($!)
    
    # Small delay between launches to avoid race conditions
    sleep 2
done

echo ""
echo "All containers launched. Waiting for completion..."
echo "Monitor logs in: $LOG_DIR"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "========================================"
echo "Training Launched!"
echo "========================================"
echo "Experiment: $EXP_NAME"
echo "Nodes: $NNODES"
echo "Logs: $LOG_DIR"
echo "Output: $OUTPUT_DIR"

