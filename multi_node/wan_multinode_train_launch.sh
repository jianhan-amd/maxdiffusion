#!/bin/bash
#
# Multi-Node Training: Launch Script
# 
# Description:
#   Launches distributed training containers on all nodes with JAX distributed setup
#
# Usage:
#   bash wan_multinode_train_launch.sh "node1,node2,node3"
#   bash wan_multinode_train_launch.sh  # Uses default node list
# 
# Environment Variables:
#   IMAGE_TAG                 - Docker image name (default: maxdiffusion-multinode-train:v1)
#   COORDINATOR_IP            - JAX coordinator IP (default: 172.29.0.73)
#   MULTI_NODES_LOG_DIR       - Base log directory (default: /home/amd/jianhan/multi_node_log)
#   SHARED_CODE_BASE_PATH     - Codebase path (default: /home/amd/jianhan/github/maxdiffusion)
#   MAXDIFFUSION_DIR_IN_DOCKER - Docker mount path (default: /app/maxdiffusion)
#   RUN_NAME                  - Experiment name prefix (default: WAN_14B_FSDP8)
#

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

readonly SCRIPT_NAME="$(basename "$0")"
readonly TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Node list - comma separated hostnames
if [ -z "${1:-}" ]; then
    # Default node list (edit this as needed)
    NODE_LIST="core42-5-a08u01,core42-1-a08u07,core42-3-a08u19,core42-4-a08u25"
else
    NODE_LIST="$1"
fi

# Convert comma-separated list to array
IFS=',' read -ra NODES <<< "$NODE_LIST"
readonly NNODES=${#NODES[@]}

# Docker and training configuration
readonly IMAGE_TAG="${IMAGE_TAG:-maxdiffusion-multinode-train:v1}"
readonly COORDINATOR_IP="${COORDINATOR_IP:-172.29.0.73}"
readonly JAX_COORDINATOR_PORT=12345
readonly COORDINATOR_TIMEOUT=1800

# Paths configuration
readonly MULTI_NODES_LOG_DIR="${MULTI_NODES_LOG_DIR:-/home/amd/jianhan/multi_node_log}"
readonly SHARED_CODE_BASE_PATH="${SHARED_CODE_BASE_PATH:-/home/amd/jianhan/github/maxdiffusion}"
readonly MAXDIFFUSION_DIR_IN_DOCKER="${MAXDIFFUSION_DIR_IN_DOCKER:-/app/maxdiffusion}"

# Experiment configuration
readonly RUN_NAME="${RUN_NAME:-WAN_14B_FSDP8}"
readonly EXP_NAME="${RUN_NAME}_${NNODES}N_${TIMESTAMP}"

# Log and output directories
readonly LOG_DIR="${MULTI_NODES_LOG_DIR}/slurm_logs/${EXP_NAME}"
readonly OUTPUT_DIR="${MULTI_NODES_LOG_DIR}/output/${EXP_NAME}"
readonly OUTPUT_DIR_IN_DOCKER="/app/${EXP_NAME}"
readonly HOST_OUTPUT_LOG="${LOG_DIR}/host_output.out"
readonly HOST_ERROR_LOG="${LOG_DIR}/host_output.err"

# GPU configuration
readonly NUM_GPUS_PER_NODE=8
readonly HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Docker configuration
readonly SHARED_MEMORY_SIZE="200G"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo "========================================"
    echo "$1"
    echo "========================================"
}

print_config() {
    print_header "Multi-Node Training Launch Configuration"
    echo "Script:            $SCRIPT_NAME"
    echo "Timestamp:         $TIMESTAMP"
    echo "Experiment:        $EXP_NAME"
    echo "Run name:          $RUN_NAME"
    echo ""
    echo "Cluster:"
    echo "  Total nodes:     $NNODES"
    echo "  Node list:       ${NODES[*]}"
    echo "  GPUs per node:   $NUM_GPUS_PER_NODE"
    echo "  Total GPUs:      $((NNODES * NUM_GPUS_PER_NODE))"
    echo ""
    echo "JAX Distributed:"
    echo "  Coordinator IP:  $COORDINATOR_IP"
    echo "  Coordinator port: $JAX_COORDINATOR_PORT"
    echo "  Init timeout:    ${COORDINATOR_TIMEOUT}s"
    echo ""
    echo "Docker:"
    echo "  Image:           $IMAGE_TAG"
    echo "  Shared memory:   $SHARED_MEMORY_SIZE"
    echo ""
    echo "Paths:"
    echo "  Code (host):     $SHARED_CODE_BASE_PATH"
    echo "  Code (docker):   $MAXDIFFUSION_DIR_IN_DOCKER"
    echo "  Output (host):   $OUTPUT_DIR"
    echo "  Output (docker): $OUTPUT_DIR_IN_DOCKER"
    echo "  Logs:            $LOG_DIR"
    echo "========================================"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/configs/models"

# Redirect output to log files
exec > >(tee -a "$HOST_OUTPUT_LOG") 2> >(tee -a "$HOST_ERROR_LOG" >&2)

# Print configuration
print_config

# ============================================================================
# LAUNCH TRAINING CONTAINERS
# ============================================================================

print_header "Launching training containers on all nodes"

launch_pids=()
for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    node_rank=$i
    
    echo "[${node}] Launching training container (rank ${node_rank}/${NNODES})..."
    
    ssh "$node" "bash -c '
        docker run --rm --privileged --network host \
            --cap-add=IPC_LOCK \
            --tmpfs /dev/shm:size=$SHARED_MEMORY_SIZE \
            --volume \"$OUTPUT_DIR:$OUTPUT_DIR_IN_DOCKER\" \
            --volume \"$SHARED_CODE_BASE_PATH:$MAXDIFFUSION_DIR_IN_DOCKER\" \
            -e JAX_COORDINATOR_IP=\"$COORDINATOR_IP\" \
            -e JAX_COORDINATOR_PORT=$JAX_COORDINATOR_PORT \
            -e NNODES=$NNODES \
            -e HIP_VISIBLE_DEVICES=\"$HIP_VISIBLE_DEVICES\" \
            -e NODE_RANK=$node_rank \
            -e JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT_SECONDS=$COORDINATOR_TIMEOUT \
            -w \"$MAXDIFFUSION_DIR_IN_DOCKER\" \
            \"$IMAGE_TAG\" \
            /bin/bash -c \"
                set -ex
                set -o pipefail
                trap \\\"echo \\\\\\\"✗ Error on line \\\\\$LINENO\\\\\\\"\\\" ERR
                
                echo \\\"========================================\\\"
                echo \\\"MaxDiffusion Training Container\\\"
                echo \\\"========================================\\\"
                echo \\\"Image:         $IMAGE_TAG\\\"
                echo \\\"Node:          \\\$(hostname)\\\"
                echo \\\"Node rank:     $node_rank of $NNODES\\\"
                echo \\\"Coordinator:   \\\$JAX_COORDINATOR_IP:\\\$JAX_COORDINATOR_PORT\\\"
                echo \\\"Output:        $OUTPUT_DIR_IN_DOCKER\\\"
                echo \\\"========================================\\\"
                echo \\\"\\\"
                
                # Create output directory
                export BASE_OUTPUT_DIRECTORY=\\\"$OUTPUT_DIR_IN_DOCKER\\\"
                mkdir -p \\\"\\\${BASE_OUTPUT_DIRECTORY}\\\"
                
                # Set permissions
                chmod 777 \\\"$MAXDIFFUSION_DIR_IN_DOCKER\\\" -R
                
                # Launch training
                cd \\\"$MAXDIFFUSION_DIR_IN_DOCKER\\\"
                echo \\\"Starting training...\\\"
                bash launch.sh LOG_PATH=\\\"$OUTPUT_DIR_IN_DOCKER\\\"
            \"
    '" > "${LOG_DIR}/node_${node}_rank_${node_rank}.log" 2>&1 &
    
    launch_pids+=($!)
    
    # Small delay to avoid race conditions in coordinator setup
    sleep 2
done

echo ""
echo "All training containers launched"
echo ""

# ============================================================================
# MONITORING
# ============================================================================

print_header "Training Monitoring"
echo "All containers are now running in the background."
echo ""
echo "Monitor training progress:"
echo "  # Watch coordinator (rank 0) log:"
echo "  tail -f ${LOG_DIR}/node_${NODES[0]}_rank_0.log"
echo ""
echo "  # Watch all nodes:"
echo "  tail -f ${LOG_DIR}/node_*.log"
echo ""
echo "  # Check for errors:"
echo "  grep -i error ${LOG_DIR}/node_*.log"
echo ""
echo "View outputs:"
echo "  ls -lh ${OUTPUT_DIR}/"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

print_header "Training Launch Summary"
echo "Experiment:       $EXP_NAME"
echo "Run name:         $RUN_NAME"
echo "Nodes:            $NNODES"
echo "Total GPUs:       $((NNODES * NUM_GPUS_PER_NODE))"
echo "Coordinator:      ${COORDINATOR_IP}:${JAX_COORDINATOR_PORT}"
echo "Logs:             $LOG_DIR"
echo "Output:           $OUTPUT_DIR"
echo "Status:           LAUNCHED"
echo "========================================"
echo ""
echo "Training is now running. Monitor logs for progress."
