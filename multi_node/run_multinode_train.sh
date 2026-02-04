#!/bin/bash

# Required environment variables for wan_multinode_train.sh
export COORDINATOR_IP=172.29.0.73
export IMAGE_TAG=jianhan-wan-multinode-train:v1
export MULTI_NODES_LOG_DIR=/home/amd/jianhan/multi_node_log
export SHARE_DOCKERFILE_PATH=/home/amd/jianhan/github/maxdiffusion/multi_node/docker/jax_maxdiffusion_wan2.1_train_inference.ubuntu.amd.Dockerfile
export SHARED_CODE_BASE_PATH=/home/amd/jianhan/github/maxdiffusion
export MAXDIFFUSION_DIR_IN_DOCKER=/app/maxdiffusion
export RUN_NAME=WAN_1_3B_FSDP8
export REMOVE_IMAGES=n
export REGISTRY_USERNAME=""
export REGISTRY_TOKEN=""
export CHMOD_RUN=n

# Define node list
NODES="core42-5-a08u01,core42-1-a08u07,core42-3-a08u19,core42-4-a08u25"

# 1. Clean and sync codebase
# To remove Docker images during cleanup, uncomment the line below:
bash wan_multinode_train.sh "$NODES" clean

# # 2. Build Docker images (only when Dockerfile changes)
bash wan_multinode_train.sh "$NODES" build

# # 3. Launch training
bash wan_multinode_train.sh "$NODES" launch
