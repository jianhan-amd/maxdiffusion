#!/bin/bash
#
# Multi-Node Training: Clean and Sync Script
# 
# Description:
#   Cleans up Docker containers and optionally images, then syncs codebase to all nodes
#
# Usage:
#   bash wan_multinode_train_clean.sh "node1,node2,node3"
#   bash wan_multinode_train_clean.sh  # Uses default node list
# 
# Environment Variables (required - should be set by wrapper script):
#   IMAGE_TAG              - Docker image name
#   REMOVE_IMAGES          - Remove Docker images? y/n
#   MULTI_NODES_LOG_DIR    - Base log directory
#   SHARED_CODE_BASE_PATH  - Codebase path to sync
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
    NODE_LIST="core42-4-a08u25"
else
    NODE_LIST="$1"
fi

# Convert comma-separated list to array
IFS=',' read -ra NODES <<< "$NODE_LIST"
readonly NNODES=${#NODES[@]}

# Docker configuration (should be set by wrapper script)
readonly IMAGE_TAG="${IMAGE_TAG}"
readonly REMOVE_IMAGES="${REMOVE_IMAGES}"

# Determine if images should be removed
if [[ "$REMOVE_IMAGES" =~ ^[Yy]$ ]]; then
    REMOVE_IMAGES_FLAG=true
else
    REMOVE_IMAGES_FLAG=false
fi

# Paths configuration (should be set by wrapper script)
readonly MULTI_NODES_LOG_DIR="${MULTI_NODES_LOG_DIR}"
readonly SHARED_CODE_BASE_PATH="${SHARED_CODE_BASE_PATH}"

# Experiment name and log directory
readonly EXP_NAME="CLEAN_${NNODES}N_${TIMESTAMP}"
readonly LOG_DIR="${MULTI_NODES_LOG_DIR}/slurm_logs/${EXP_NAME}"
readonly HOST_OUTPUT_LOG="${LOG_DIR}/host_output.out"
readonly HOST_ERROR_LOG="${LOG_DIR}/host_output.err"

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
    print_header "Multi-Node Cleanup Configuration"
    echo "Script:            $SCRIPT_NAME"
    echo "Timestamp:         $TIMESTAMP"
    echo "Total nodes:       $NNODES"
    echo "Node list:         ${NODES[*]}"
    echo "Docker image:      $IMAGE_TAG"
    echo "Remove images:     $REMOVE_IMAGES"
    echo "Log directory:     $LOG_DIR"
    echo "Codebase path:     $SHARED_CODE_BASE_PATH"
    echo "========================================"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Create log directory
mkdir -p "$LOG_DIR"

# Redirect output to log files
exec > >(tee -a "$HOST_OUTPUT_LOG") 2> >(tee -a "$HOST_ERROR_LOG" >&2)

# Print configuration
print_config

# ============================================================================
# STEP 1: CLEANUP EXISTING CONTAINERS
# ============================================================================

print_header "STEP 1: Cleaning up Docker containers"

cleanup_pids=()
for node in "${NODES[@]}"; do
    echo "[${node}] Starting cleanup..."

    if [ "$REMOVE_IMAGES_FLAG" = true ]; then
        IMAGE_RM_CMD="docker image rm -f '$IMAGE_TAG' 2>/dev/null || true"
        echo "[${node}] Will remove Docker image: $IMAGE_TAG"
    else
        IMAGE_RM_CMD="echo 'Skipping image removal'"
        echo "[${node}] Will keep Docker image"
    fi
    
    ssh "$node" "bash -c '
        set -e
        echo \"[$(hostname)] Stopping containers...\"
        docker stop \$(docker ps -q) 2>/dev/null || true
        
        echo \"[$(hostname)] Removing containers...\"
        docker rm \$(docker ps -aq) 2>/dev/null || true
        
        $IMAGE_RM_CMD
        
        echo \"[$(hostname)] ✓ Cleanup completed\"
    '" > "${LOG_DIR}/cleanup_${node}.log" 2>&1 &
    
    cleanup_pids+=($!)
done

# Wait for all cleanup jobs to complete
echo "Waiting for cleanup jobs to complete..."
for pid in "${cleanup_pids[@]}"; do
    if wait "$pid"; then
        echo "✓ Cleanup job $pid completed successfully"
    else
        echo "✗ Cleanup job $pid failed (exit code: $?)"
    fi
done

echo "✓ Container cleanup completed on all nodes"

# ============================================================================
# STEP 2: SYNC CODEBASE TO ALL NODES
# ============================================================================

print_header "STEP 2: Syncing codebase to all nodes"

echo "Source: $SHARED_CODE_BASE_PATH"
echo ""

sync_failed=()
for node in "${NODES[@]}"; do
    echo "[${node}] Syncing codebase..."
    
    if rsync -az --delete --info=progress2 -e "ssh" \
        "$SHARED_CODE_BASE_PATH/" "$node:$SHARED_CODE_BASE_PATH/" \
        > "${LOG_DIR}/sync_${node}.log" 2>&1; then
        echo "[${node}] ✓ Sync completed"
    else
        echo "[${node}] ✗ Sync failed"
        sync_failed+=("$node")
    fi
done

# Report sync status
echo ""
if [ ${#sync_failed[@]} -eq 0 ]; then
    echo "✓ Codebase sync completed successfully on all nodes"
else
    echo "✗ Codebase sync failed on ${#sync_failed[@]} node(s): ${sync_failed[*]}"
    exit 1
fi

# ============================================================================
# SUMMARY
# ============================================================================

print_header "Cleanup Summary"
echo "Experiment:    $EXP_NAME"
echo "Nodes cleaned: $NNODES"
echo "Logs:          $LOG_DIR"
echo "Status:        SUCCESS"
echo "========================================"
