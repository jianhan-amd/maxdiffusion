#!/bin/bash
#
# Multi-Node Training: Docker Build Script
# 
# Description:
#   Builds Docker images on all nodes in parallel with retry logic
#
# Usage:
#   bash wan_multinode_train_build_docker.sh "node1,node2,node3"
#   bash wan_multinode_train_build_docker.sh  # Uses default node list
# 
# Environment Variables (required - should be set by wrapper script):
#   IMAGE_TAG              - Docker image name
#   MULTI_NODES_LOG_DIR    - Base log directory
#   SHARE_DOCKERFILE_PATH  - Path to Dockerfile
#   REGISTRY_USERNAME      - Docker Hub username
#   REGISTRY_TOKEN         - Docker Hub token
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
readonly REGISTRY_USERNAME="${REGISTRY_USERNAME}"
readonly REGISTRY_TOKEN="${REGISTRY_TOKEN}"

# Paths configuration (should be set by wrapper script)
readonly MULTI_NODES_LOG_DIR="${MULTI_NODES_LOG_DIR}"
readonly SHARE_DOCKERFILE_PATH="${SHARE_DOCKERFILE_PATH}"
readonly DOCKERFILE_DIR="$(dirname "$SHARE_DOCKERFILE_PATH")"

# Retry configuration
readonly MAX_RETRIES=5
readonly INITIAL_DELAY=30
readonly MAX_DELAY=180

# Experiment name and log directory
readonly EXP_NAME="BUILD_DOCKER_${NNODES}N_${TIMESTAMP}"
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
    print_header "Multi-Node Docker Build Configuration"
    echo "Script:            $SCRIPT_NAME"
    echo "Timestamp:         $TIMESTAMP"
    echo "Total nodes:       $NNODES"
    echo "Node list:         ${NODES[*]}"
    echo "Docker image:      $IMAGE_TAG"
    echo "Dockerfile:        $SHARE_DOCKERFILE_PATH"
    echo "Max retries:       $MAX_RETRIES"
    echo "Log directory:     $LOG_DIR"
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
# BUILD DOCKER IMAGE ON ALL NODES
# ============================================================================

print_header "Building Docker images on all nodes"

build_pids=()
for node in "${NODES[@]}"; do
    echo "[${node}] Starting Docker build (background job)..."
    
    ssh "$node" "bash -c '
        set -e
        
        MAX_RETRIES=$MAX_RETRIES
        INITIAL_DELAY=$INITIAL_DELAY
        MAX_DELAY=$MAX_DELAY
        RETRY_COUNT=0
        
        while true; do
            echo \"========================================\"
            echo \"[$(hostname)] Build attempt \$((RETRY_COUNT + 1)) of \$MAX_RETRIES\"
            echo \"========================================\"
            
            # Login to Docker Hub
            echo \"[$(hostname)] Logging into Docker Hub as $REGISTRY_USERNAME...\"
            if echo \"$REGISTRY_TOKEN\" | docker login docker.io -u \"$REGISTRY_USERNAME\" --password-stdin; then
                echo \"[$(hostname)] ✓ Docker Hub login successful\"
            else
                echo \"[$(hostname)] ✗ Docker Hub login failed\"
            fi
            
            # Build Docker image
            echo \"[$(hostname)] Building image: $IMAGE_TAG\"
            echo \"[$(hostname)] Dockerfile: $SHARE_DOCKERFILE_PATH\"
            echo \"[$(hostname)] Context: $DOCKERFILE_DIR\"
            
            if docker build --tag \"$IMAGE_TAG\" \
                --file \"$SHARE_DOCKERFILE_PATH\" \
                \"$DOCKERFILE_DIR\"; then
                echo \"\"
                echo \"[$(hostname)] ========================================\"
                echo \"[$(hostname)] ✓ Image built successfully\"
                echo \"[$(hostname)] ========================================\"
                break
            else
                RETRY_COUNT=\$((RETRY_COUNT + 1))
                
                if [ \$RETRY_COUNT -ge \$MAX_RETRIES ]; then
                    echo \"[$(hostname)] ✗ Failed to build after \$MAX_RETRIES attempts\"
                    exit 1
                fi
                
                # Calculate exponential backoff delay
                CURRENT_DELAY=\$((INITIAL_DELAY * (2 ** (RETRY_COUNT - 1))))
                if [ \$CURRENT_DELAY -gt \$MAX_DELAY ]; then
                    CURRENT_DELAY=\$MAX_DELAY
                fi
                
                echo \"[$(hostname)] Build failed. Retrying in \$CURRENT_DELAY seconds...\"
                sleep \$CURRENT_DELAY
            fi
        done
    '" > "${LOG_DIR}/build_${node}.log" 2>&1 &
    
    build_pids+=($!)
done

echo ""
echo "All build jobs started. Monitor individual logs at:"
echo "  $LOG_DIR/build_*.log"
echo ""

# Wait for all builds to complete
echo "Waiting for all build jobs to complete..."
build_failed=()
for i in "${!build_pids[@]}"; do
    pid="${build_pids[$i]}"
    node="${NODES[$i]}"
    
    if wait "$pid"; then
        echo "[${node}] ✓ Build job completed successfully (PID: $pid)"
    else
        echo "[${node}] ✗ Build job failed (PID: $pid, exit code: $?)"
        build_failed+=("$node")
    fi
done

# ============================================================================
# SUMMARY
# ============================================================================

print_header "Docker Build Summary"
echo "Experiment:      $EXP_NAME"
echo "Image tag:       $IMAGE_TAG"
echo "Nodes built:     $NNODES"
echo "Logs:            $LOG_DIR"

if [ ${#build_failed[@]} -eq 0 ]; then
    echo "Status:          SUCCESS"
    echo "========================================"
else
    echo "Status:          FAILED"
    echo "Failed nodes:    ${build_failed[*]}"
    echo "========================================"
    exit 1
fi
