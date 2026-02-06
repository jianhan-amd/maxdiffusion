#!/bin/bash

# Required environment variables for wan_multinode_train.sh

# core42-4-a08u25:172.29.0.73
export COORDINATOR_IP=172.29.0.73
export IMAGE_TAG=your-name-wan-multinode-train:v1
# Please keep MULTI_NODES_LOG_DIR outside SHARED_CODE_BASE_PATH since we are going to sync the whole SHARED_CODE_BASE_PATH
export MULTI_NODES_LOG_DIR=/home/amd/your_dir/multi_node_log
export SHARE_DOCKERFILE_PATH=/home/amd/your_dir/maxdiffusion/multi_node/docker/jax_maxdiffusion_wan2.1_train_inference.ubuntu.amd.Dockerfile
export SHARED_CODE_BASE_PATH=/home/amd/your_dir/maxdiffusion
export MAXDIFFUSION_DIR_IN_DOCKER=/app/maxdiffusion
# Please also change between pretrained_model_name_or_path: 'Wan-AI/Wan2.1-T2V-14B-Diffusers' or 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers' under src/maxdiffusion/configs/base_wan_14b.yml
export RUN_NAME=WAN_14B_FSDP8
export REMOVE_IMAGES=n
export REGISTRY_USERNAME=""
export REGISTRY_TOKEN=""
export CHMOD_RUN=n

# Define node list
# Please put the JAX COORDINATOR to the first of the list. The JAX COORDINATOR node will be launched before others to make sure all nodes can connect to the JAX COORDINATOR service.
# core42-4-a08u25:172.29.0.73
NODES="core42-4-a08u25,core42-1-a08u07,core42-3-a08u19,core42-5-a08u01"

# 1. Clean and sync codebase
# To remove Docker images during cleanup, uncomment the line below:
bash wan_multinode_train.sh "$NODES" clean

# # 2. Build Docker images (only when Dockerfile changes)
bash wan_multinode_train.sh "$NODES" build

# # 3. Launch training
bash wan_multinode_train.sh "$NODES" launch
