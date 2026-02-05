#!/bin/bash
#SBATCH --job-name=wan_multinode       # Job name
#SBATCH --nodes=2                      # Number of nodes (adjust as needed)
#SBATCH --ntasks-per-node=1            # One task per node
#SBATCH --gres=gpu:8                   # Assuming 8 GPUs per node, adjust if different
#SBATCH --mem=0                        # Use all available memory
#SBATCH --time=48:00:00                # Max runtime (48 hours)
#SBATCH --output=logs/wan_train_%j.out # Standard output (%j = job ID)
#SBATCH --error=logs/wan_train_%j.err  # Standard error
#SBATCH --exclusive                    # Exclusive node access
#SBATCH --account=amd-silo-tgr         # Your account/project name
#SBATCH --partition=amd-silo-tgr       # Partition/queue name
##SBATCH --nodelist=chi-mi300x-[002,005,013,019]


timestamp=$(date +%Y%m%d-%H%M%S)

MULTI_NODES_LOG_DIR="/home/jianhmei/code/github/multi_nodes"

# Docker image to use
IMAGE_TAG="jianhan-wan-multinode-train:v1"
SHARE_DOCKERFILE_PATH="${MULTI_NODES_LOG_DIR}/docker/jax_maxdiffusion_wan2.1_train_inference.ubuntu.amd.Dockerfile"

# shared code base path
SHARED_CODE_BASE_PATH="/home/jianhmei/code/github/upstream_maxdiff/maxdiffusion"
MAXDIFFUSION_DIR_IN_DOCKER="/app/maxdiffusion"

# scaling test exp NAME
export EXP_NAME=WAN_1_3B_FSDP8_2N_${timestamp}

export LOG_DIR="${MULTI_NODES_LOG_DIR}/slurm_logs-${EXP_NAME}/${SLURM_JOB_NUM_NODES}-${SLURM_JOB_ID}"
export SRUN_OUTPUT="${LOG_DIR}/node_%N_rank_%t.log"
# Create log directory
mkdir -p $LOG_DIR

export HOSTNAME=$(hostname)
OUTPUT_DIR="${MULTI_NODES_LOG_DIR}/output/output-${EXP_NAME}/${HOSTNAME}-${SLURM_JOB_NUM_NODES}-${SLURM_JOB_ID}"
OUTPUT_DIR_IN_DOCKER="/app/output-${EXP_NAME}/${HOSTNAME}-${SLURM_JOB_NUM_NODES}-${SLURM_JOB_ID}"
# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/configs/models

# Registry credentials
export REGISTRY_USERNAME=""
export REGISTRY_TOKEN=""

# Define log file paths with timestamp
export HOST_OUTPUT_LOG="${LOG_DIR}/host_output.out"
export HOST_ERROR_LOG="${LOG_DIR}/host_output.err"

# Redirect stdout and stderr to the new log files
exec > >(tee -a "${HOST_OUTPUT_LOG}") 2> >(tee -a "${HOST_ERROR_LOG}" >&2)


# Get the coordinator IP (first node in the allocation)
COORDINATOR_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "List of assigned hostnames:$(scontrol show hostnames $SLURM_JOB_NODELIST)"
NNODES=$SLURM_JOB_NUM_NODES

# Add error checking
if [ -z "$COORDINATOR_IP" ]; then
    echo "Error: Could not determine coordinator IP"
    exit 1
fi

# container cleanup:
srun bash -c '
    # Clean up any existing containers (with proper permission handling)
    if docker ps -q | grep -q .; then
        echo "Stopping existing containers on $(hostname)"
        docker stop $(docker ps -q)
    else
        echo "No running containers to stop on $(hostname)"
    fi
    
    # Remove stopped containers
    if docker ps -aq | grep -q .; then
        echo "Removing stopped containers on $(hostname)"
        docker rm $(docker ps -aq)
    else
        echo "No containers to remove on $(hostname)"
    fi

    docker image rm -f '"$IMAGE_TAG"'
'

# # Also update the Docker login section to use sudo:
# echo "Logging into Docker Hub on all nodes"
# srun --export=ALL,REGISTRY_USERNAME,REGISTRY_TOKEN bash -c '
#     echo "Node $(hostname): Logging into Docker Hub"
#     echo '"$REGISTRY_TOKEN"' | docker login docker.io -u '"$REGISTRY_USERNAME"' --password-stdin
    
#     if [ $? -ne 0 ]; then
#         echo "Failed to login to Docker Hub on $(hostname)!"
#         exit 1
#     else
#         echo "Successfully logged into Docker Hub on $(hostname)"
#     fi
# '


# Update the image pull section to use sudo:
echo "Building the container image on all nodes"
srun bash -c '
    MAX_RETRIES=5
    INITIAL_DELAY=30 # seconds
    MAX_DELAY=180 # seconds
    RETRY_COUNT=0

    while true; do
        echo "Node $(hostname): Pulling image '"$IMAGE_TAG"' (Attempt $((RETRY_COUNT + 1)))"
        echo "Attempting to pull image: $IMAGE_TO_PULL (Attempt $((RETRY_COUNT + 1)))"
        docker build --tag '"$IMAGE_TAG"' --file '"$SHARE_DOCKERFILE_PATH"' .

        if [ $? -eq 0 ]; then
            echo "Image built successfully."
            break
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then
                echo "Failed to build image after $MAX_RETRIES attempts. Exiting."
                exit 1
            fi

            CURRENT_DELAY=$((INITIAL_DELAY * (2 ** (RETRY_COUNT - 1))))
            if [ "$CURRENT_DELAY" -gt "$MAX_DELAY" ]; then
                CURRENT_DELAY="$MAX_DELAY"
            fi

            echo "Pull failed. Retrying in $CURRENT_DELAY seconds..."
            sleep "$CURRENT_DELAY"
        fi
    done

    echo "Done with image building"
'

srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES bash -c '
    echo "Cleaning containers on $(hostname)..."
    docker ps -aq | xargs -r docker rm -f
'

# Add a small delay to ensure cleanup is complete
sleep 5

# Modified srun command with proper variable expansion

# --mount type=bind,source='"${OUTPUT_DIR}"',target='"${OUTPUT_DIR_IN_DOCKER}"' \
# --mount type=bind,source='"${SHARED_CODE_BASE_PATH}"',target='"${MAXDIFFUSION_DIR_IN_DOCKER}"' \
echo "Starting the training on all nodes"
srun --output=$SRUN_OUTPUT --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES \
    --export=ALL \
    bash -c '
    NODE_RANK=$SLURM_PROCID
    NNODES=$SLURM_JOB_NUM_NODES
    docker run --rm --privileged --network host \
    --cap-add=IPC_LOCK \
    --volume /dev/infiniband:/dev/infiniband \
    --tmpfs /dev/shm:size=200G \
    --volume '"${OUTPUT_DIR}"':'"${OUTPUT_DIR_IN_DOCKER}"' \
    --volume '"${SHARED_CODE_BASE_PATH}"':'"${MAXDIFFUSION_DIR_IN_DOCKER}"' \
    -e JAX_COORDINATOR_IP='"${COORDINATOR_IP}"' \
    -e JAX_COORDINATOR_PORT=12345 \
    -e NNODES=$NNODES \
    -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e NODE_RANK=$NODE_RANK \
    -e JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT_SECONDS=1800 \
    -w '"${MAXDIFFUSION_DIR_IN_DOCKER}"' \
    '"${IMAGE_TAG}"' \
    /bin/bash -c "
        # Error handling
        set -ex
        set -o pipefail  # Make sure pipe failures are caught
        trap \"echo \\\"Error on line \$LINENO\\\"\" ERR

        echo '"${IMAGE_TAG}"'
        echo \"Starting node \$NODE_RANK of \$NNODES\"
        echo \"Coordinator IP: \$JAX_COORDINATOR_IP\"

        # Create output directory
        export BASE_OUTPUT_DIRECTORY=\"'"${OUTPUT_DIR_IN_DOCKER}"'\"
        mkdir -p \${BASE_OUTPUT_DIRECTORY}

        cd '"${MAXDIFFUSION_DIR_IN_DOCKER}"'
        bash launch.sh LOG_PATH='"${OUTPUT_DIR_IN_DOCKER}"'

    "
'

srun bash -c '
    docker image rm -f '"$IMAGE_TAG"'
'
