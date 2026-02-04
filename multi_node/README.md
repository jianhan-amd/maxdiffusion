# Multi-Node WAN Training Guide

Distributed WAN model training across multiple nodes with AMD ROCm GPUs.

## Quick Start

### Option 1: Using Helper Script (Recommended)

```bash
cd /home/amd/jianhan/github/maxdiffusion/multi_node

# Edit run_multinode_train.sh to set configuration and enable/disable steps
bash run_multinode_train.sh
```

### Option 2: Manual Execution

```bash
# Set ALL required environment variables (no defaults)
export COORDINATOR_IP="172.29.0.73"
export IMAGE_TAG="maxdiffusion-multinode-train:v1"
export MULTI_NODES_LOG_DIR="/home/amd/jianhan/multi_node_log"
export SHARE_DOCKERFILE_PATH="/home/amd/jianhan/github/maxdiffusion/multi_node/docker/jax_maxdiffusion_wan2.1_train_inference.ubuntu.amd.Dockerfile"
export SHARED_CODE_BASE_PATH="/home/amd/jianhan/github/maxdiffusion"
export MAXDIFFUSION_DIR_IN_DOCKER="/app/maxdiffusion"
export RUN_NAME="WAN_14B_FSDP8"
export REMOVE_IMAGES="n"
export CHMOD_RUN="n"
export REGISTRY_USERNAME="rocmshared"
export REGISTRY_TOKEN="your_token"

# Run commands
bash wan_multinode_train.sh "node1,node2,node3,node4" clean
bash wan_multinode_train.sh "node1,node2,node3,node4" build
bash wan_multinode_train.sh "node1,node2,node3,node4" launch

# Monitor training
tail -f ${MULTI_NODES_LOG_DIR}/slurm_logs/${RUN_NAME}_*/node_*_rank_0.log
```

## Prerequisites

- **Password-less SSH**: Set up SSH keys for all nodes
  ```bash
  ssh-keygen -t ed25519 -C "multinode-training"
  for node in node1 node2 node3; do ssh-copy-id $node; done
  ```
- **Docker 20.10+** on all nodes
- **AMD ROCm 5.7+** with MI250/MI300 GPUs
- **Port 12345 open** between nodes (JAX coordinator)
- **50GB+ disk space** per node

## Environment Variables

**All variables are required (no defaults):**

| Variable | Description | Example |
|----------|-------------|---------|
| `COORDINATOR_IP` | JAX coordinator IP | `172.29.0.73` |
| `IMAGE_TAG` | Docker image name | `maxdiffusion-multinode-train:v1` |
| `MULTI_NODES_LOG_DIR` | Base log directory | `/home/amd/jianhan/multi_node_log` |
| `SHARE_DOCKERFILE_PATH` | Path to Dockerfile | `/home/amd/.../jax_maxdiffusion_wan2.1...Dockerfile` |
| `SHARED_CODE_BASE_PATH` | Codebase path | `/home/amd/jianhan/github/maxdiffusion` |
| `MAXDIFFUSION_DIR_IN_DOCKER` | Docker mount path | `/app/maxdiffusion` |
| `RUN_NAME` | Experiment name | `WAN_14B_FSDP8` or `WAN_1_3B_FSDP8` |
| `REMOVE_IMAGES` | Remove images on clean | `y` or `n` |
| `CHMOD_RUN` | Only fix permissions (skip training) | `y` or `n` (default: `n`) |
| `REGISTRY_USERNAME` | Docker Hub username | `rocmshared` |
| `REGISTRY_TOKEN` | Docker Hub token | Your token |

## Scripts Overview

- **`run_multinode_train.sh`**: Helper script with pre-configured variables. Edit to set config and enable/disable steps
- **`wan_multinode_train.sh`**: Main wrapper for clean/build/launch operations (requires all env vars)
- **`wan_multinode_train_clean.sh`**: Cleans containers and syncs codebase via rsync
- **`wan_multinode_train_build_docker.sh`**: Builds Docker images in parallel (5 retries)
- **`wan_multinode_train_launch.sh`**: Launches distributed training with JAX

## Directory Structure

```
multi_node_log/
├── slurm_logs/
│   ├── CLEAN_*N_*/              # Cleanup logs
│   ├── BUILD_DOCKER_*N_*/       # Build logs
│   └── ${RUN_NAME}_*N_*/        # Training logs (e.g., WAN_14B_FSDP8_4N_20260204-141300)
│       ├── node_*_rank_0.log    # Primary logs
│       └── host_output.{out,err}
└── output/
    └── ${RUN_NAME}_*N_*/        # Checkpoints
```

## Typical Workflow

```bash
# First time: Run all steps
bash run_multinode_train.sh

# Code changes: Skip build (edit run_multinode_train.sh, comment out build line)
# Dockerfile changes: Run build only (comment out clean and launch)
# Quick iteration: Run clean + launch only (comment out build)
```

## Common Commands

```bash
# Change model
export RUN_NAME="WAN_1_3B_FSDP8"  # or WAN_14B_FSDP8

# Remove Docker images (free disk space)
export REMOVE_IMAGES="y"

# Fix permissions only (no training) - useful for permission issues
export CHMOD_RUN="y"
bash wan_multinode_train.sh "node1,node2,node3,node4" launch

# Monitor latest run
LATEST=$(ls -td ${MULTI_NODES_LOG_DIR}/slurm_logs/${RUN_NAME}_* | head -1)
tail -f ${LATEST}/node_*_rank_0.log

# Average step time (exclude warmup)
grep "seconds:" ${LATEST}/node_*_rank_0.log | tail -n +2 | \
    awk -F'seconds: ' '{print $2}' | awk '{sum+=$1; count++} END {printf "Avg: %.2fs\n", sum/count}'

# Check GPU utilization
for node in core42-5-a08u01 core42-1-a08u07 core42-3-a08u19 core42-4-a08u25; do
    ssh $node "rocm-smi --showuse"
done

# Check containers
for node in core42-5-a08u01 core42-1-a08u07; do
    ssh $node "docker ps"
done
```

## Performance (WAN 14B, 4 nodes × 8 GPUs)

- **Batch size/device**: 1
- **Resolution**: 1280×720 × 85 frames
- **Speed**: ~82-83s/step (after warmup)
- **Throughput**: ~255 TFLOP/s/device
- **FPS/device**: ~1.03
- **First step**: ~300s (JIT compilation)

**Single Node**: For testing, omit node list (defaults to single node): `bash wan_multinode_train.sh "" launch`  
Or specify: `bash wan_multinode_train.sh "core42-4-a08u25" launch`

## Troubleshooting

```bash
# SSH issues
ssh -vvv node1  # Test connectivity
eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519

# Docker issues
ssh node1 "docker ps"  # Check Docker
ssh node1 "sudo usermod -aG docker $USER"  # Add to docker group

# JAX timeout
ssh node1 "hostname -I"  # Get coordinator IP
export COORDINATOR_IP="172.29.0.XX"
ssh node2 "nc -zv $COORDINATOR_IP 12345"  # Test port

# GPU not visible
ssh node1 "rocm-smi"  # Check GPUs
ssh node1 "docker run --rm --privileged -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ${IMAGE_TAG} rocm-smi"

# Build failures
cat ${MULTI_NODES_LOG_DIR}/slurm_logs/BUILD_DOCKER_*/build_*.log
ssh node1 "docker system prune -af"  # Clean cache

# Permission issues (codebase not writable)
export CHMOD_RUN="y"
bash wan_multinode_train.sh "node1,node2" launch  # Fix perms only

# Debug mode
bash -x wan_multinode_train.sh "node1,node2" clean  # Verbose
```

## Log Analysis

```bash
# Find latest run
LOG_DIR=$(ls -td ${MULTI_NODES_LOG_DIR}/slurm_logs/${RUN_NAME}_* | head -1)

# View metrics
grep "seconds:\|loss:\|TFLOP/s" ${LOG_DIR}/node_*_rank_0.log

# Calculate stats
grep "seconds:" ${LOG_DIR}/node_*_rank_0.log | tail -n +2 | \
    awk -F'seconds: ' '{print $2}' | awk '{sum+=$1; count++} END {printf "Mean: %.2fs, Total: %d steps\n", sum/count, count}'
```

## Resources

- [JAX Distributed](https://jax.readthedocs.io/en/latest/multi_process.html)
- [AMD ROCm](https://rocmdocs.amd.com/)
- [MaxDiffusion](https://github.com/google/maxdiffusion)
