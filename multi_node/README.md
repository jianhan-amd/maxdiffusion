# Multi-Node WAN Training Guide

Distributed WAN model training across multiple nodes with AMD ROCm GPUs.

## Quick Start

```bash
cd /home/amd/jianhan/github/maxdiffusion/multi_node

# 1. Clean and sync codebase
bash wan_multinode_train.sh "node1,node2,node3,node4" clean

# 2. Build Docker images (only when Dockerfile changes)
bash wan_multinode_train.sh "node1,node2,node3,node4" build

# 3. Launch training
bash wan_multinode_train.sh "node1,node2,node3,node4" launch

# Monitor training
tail -f ${MULTI_NODES_LOG_DIR}/slurm_logs/WAN_14B_FSDP8_*/node_*_rank_0.log
```

## SSH Keys Setup

**Required: Password-less SSH access to all nodes**

```bash
# 1. Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "multinode-training"

# 2. Copy SSH key to all nodes
for node in node1 node2 node3 node4; do
    ssh-copy-id -i ~/.ssh/id_ed25519.pub user@$node
done

# 3. Test password-less access
for node in node1 node2 node3 node4; do
    ssh $node "hostname && date" || echo "FAILED: $node"
done

# 4. Optional: Add to ~/.ssh/config for easier access
cat >> ~/.ssh/config << 'EOF'
Host node1 node2 node3 node4
    User your_username
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
EOF
```

## Prerequisites

- Password-less SSH access (see above)
- Docker 20.10+ on all nodes
- AMD ROCm 5.7+ with MI250/MI300 GPUs
- Port 12345 open between nodes (JAX coordinator)
- 50GB+ disk space per node

## Environment Variables

```bash
export COORDINATOR_IP="172.29.0.73"                    # JAX coordinator IP
export IMAGE_TAG="maxdiffusion-multinode-train:v1"   # Docker image name
export RUN_NAME="WAN_14B_FSDP8"                       # Experiment name
export REMOVE_IMAGES="n"                              # Remove images on clean (y/n)
export MULTI_NODES_LOG_DIR="/home/amd/jianhan/multi_node_log"  # Log dir

# Then run
bash wan_multinode_train.sh "node1,node2" launch
```

## Scripts Overview

### wan_multinode_train.sh (Wrapper)
Main script for all operations: `clean`, `build`, `launch`

### wan_multinode_train_clean.sh
Cleans containers and syncs codebase to all nodes

### wan_multinode_train_build_docker.sh
Builds Docker images in parallel with retry logic (5 attempts)

### wan_multinode_train_launch.sh
Launches distributed training with JAX multi-process

## Directory Structure

```
multi_node_log/
├── slurm_logs/
│   ├── CLEAN_*N_*/           # Cleanup logs
│   ├── BUILD_DOCKER_*N_*/    # Build logs
│   └── WAN_14B_FSDP8_*N_*/   # Training logs
└── output/
    └── WAN_14B_FSDP8_*N_*/   # Checkpoints and outputs
```

## Common Commands

```bash
# Change model
export RUN_NAME="WAN_1_3B_FSDP8"  # or "FLUX_DEV_FSDP8"

# Use default nodes
bash wan_multinode_train.sh "" clean

# Monitor latest run
LATEST=$(ls -td ${MULTI_NODES_LOG_DIR}/slurm_logs/WAN_14B_FSDP8_* | head -1)
tail -f ${LATEST}/node_*_rank_0.log

# Extract metrics
grep "seconds:" ${LATEST}/node_*_rank_0.log | tail -n +2 | \
    awk -F'seconds: ' '{print $2}' | awk '{print $1}' | \
    awk '{sum+=$1; count++} END {printf "Avg: %.2fs\n", sum/count}'

# Check GPU utilization
for node in node1 node2; do
    echo "=== $node ==="
    ssh $node "rocm-smi --showuse"
done
```

## Performance (WAN 14B, 4 nodes × 8 GPUs)

- **Batch size/device**: 1
- **Resolution**: 1280×720 × 85 frames
- **Speed**: ~82-83s/step (after warmup)
- **Throughput**: ~255 TFLOP/s/device
- **FPS/device**: ~1.03
- **First step**: ~300s (JIT compilation)

## Troubleshooting

### SSH Issues
```bash
# Test connectivity
ssh -vvv node1

# Check SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### Docker Issues
```bash
# Check Docker
ssh node1 "docker ps"

# Add user to docker group
ssh node1 "sudo usermod -aG docker $USER"
```

### JAX Initialization Timeout
```bash
# Check firewall (port 12345)
ssh node1 "sudo iptables -L | grep 12345"

# Test connectivity
ssh node2 "nc -zv node1 12345"

# Fix coordinator IP
ssh node1 "hostname -I"  # Get correct IP
export COORDINATOR_IP="172.29.0.XX"
```

### GPU Not Visible
```bash
# Check GPUs
ssh node1 "rocm-smi"

# Check in container
ssh node1 "docker run --rm --privileged \
    -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    maxdiffusion-multinode-train:v1 rocm-smi"
```

### Build Failures
```bash
# Check build logs
cat ${MULTI_NODES_LOG_DIR}/slurm_logs/BUILD_DOCKER_*/build_*.log

# Clean Docker
ssh node1 "docker system prune -af"
```

### Debug Mode
```bash
# Enable verbose output
bash -x wan_multinode_train.sh "node1,node2" clean
```

## Log Analysis

```bash
LOG_DIR=$(ls -td ${MULTI_NODES_LOG_DIR}/slurm_logs/WAN_14B_FSDP8_* | head -1)

# Training metrics
grep "seconds:" ${LOG_DIR}/node_*_rank_0.log
grep "loss:" ${LOG_DIR}/node_*_rank_0.log
grep "TFLOP/s/device:" ${LOG_DIR}/node_*_rank_0.log

# Average step time (exclude warmup)
grep "seconds:" ${LOG_DIR}/node_*_rank_0.log | tail -n +2 | \
    awk -F'seconds: ' '{sum+=$2; count++} END {print sum/count}'
```

## Advanced Configuration

```bash
# Custom Docker image
export IMAGE_TAG="my-custom-wan:v1"

# Custom coordinator IP (use InfiniBand)
export COORDINATOR_IP="172.29.0.73"

# Resume from checkpoint
# Edit: src/maxdiffusion/configs/base_wan_14b.yml
# Set: checkpoint_dir: "/path/to/checkpoint"
```

## File Locations

| Item | Path |
|------|------|
| Scripts | `multi_node/wan_multinode_train*.sh` |
| Logs | `${MULTI_NODES_LOG_DIR}/slurm_logs/${EXP_NAME}/` |
| Outputs | `${MULTI_NODES_LOG_DIR}/output/${EXP_NAME}/` |
| Configs | `src/maxdiffusion/configs/` |

## Resources

- [JAX Distributed](https://jax.readthedocs.io/en/latest/multi_process.html)
- [AMD ROCm](https://rocmdocs.amd.com/)
- [MaxDiffusion](https://github.com/google/maxdiffusion)
