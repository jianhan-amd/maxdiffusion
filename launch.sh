#!/usr/bin/env bash




# Parse LOG_PATH argument from command line
LOG_PATH=""
FILTERED_ARGS=()
for arg in "$@"; do
  if [[ $arg == LOG_PATH=* ]]; then
    LOG_PATH="${arg#*=}"
  else
    FILTERED_ARGS+=("$arg")
  fi
done

# Set default log file if not provided
if [ -z "$LOG_PATH" ]; then
  LOG_PATH="$PWD/output/"
fi

export HF_TOKEN=""
export HF_HOME="/app/hf_home/"

export MIOPEN_CUSTOM_CACHE_DIR="/app/.cache/miopen/"
export JAX_COMPILATION_CACHE_DIR="/app/.cache/jax/"
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="all"

# export TF_CPP_MIN_LOG_LEVEL=0
# export TF_CPP_MAX_VLOG_LEVEL=3
export JAX_TRACEBACK_FILTERING=off

timestamp=$(date +%Y%m%d-%H%M%S)

export LIBTPU_INIT_ARGS=""

export KERAS_BACKEND="jax"
export JAX_SPMD_MODE="allow_all"
export TOKENIZERS_PARALLELISM="1"

# to skip hard-coded GCS calls
export SKIP_GCS=1

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=16384

export NVTE_FUSED_ATTN=1
export NVTE_CK_USES_BWD_V3=1        # activates v3, 0 default
export NVTE_CK_USES_FWD_V3=1        # 0 for fsdp tpu
export NVTE_CK_IS_V3_ATOMIC_FP32=0  # default
export NVTE_CK_HOW_V3_BF16_CVT=1    # default
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

export NCCL_IB_HCA=bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7
export NCCL_SOCKET_IFNAME=enp159s0np0
export NCCL_IB_GID_INDEX=3
export NCCL_PROTO=Simple

export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_MAX_NCHANNELS=16
export RCCL_MSCCL_ENABLE=0
export GPU_MAX_HW_QUEUES=2
export HIP_FORCE_DEV_KERNARG=1
export HSA_NO_SCRATCH_RECLAIM=1
# NCCL flags
export NCCL_DEBUG=WARN  #WARN, INFO
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_PROTO=Simple
export NCCL_IB_TIMEOUT=20
export NCCL_IB_TC=41
export NCCL_IB_SL=0

export GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}
export NCCL_CROSS_NIC=0
export NCCL_CHECKS_DISABLE=1
export NCCL_IB_QPS_PER_CONNECTION=1
##
#OCI said the below env var can improve all-to-all communication:
export NCCL_PXN_DISABLE=0
# export NCCL_P2P_NET_CHUNKSIZE=524288
# export NCCL_MAX_NCHANNELS=16


# UCX flags
export UCX_TLS=tcp,self,sm
export UCX_IB_TRAFFIC_CLASS=41
export UCX_IB_SL=0


HOST_NAME=$(hostname)

export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_cublaslt=True
 --xla_gpu_graph_level=0 --xla_gpu_autotune_level=5 --xla_gpu_enable_reduce_scatter_combine_by_dim=false
 --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_all_gather_combine_threshold_bytes=134217728 
 --xla_gpu_reduce_scatter_combine_threshold_bytes=134217728
 --xla_dump_to=${LOG_PATH}/${HOST_NAME}_xla_dump_${timestamp}"

rm -rf /app/.cache/*
python3 setup.py develop

EXP_NAME="train"
LOG_FILE="$LOG_PATH/output_$HOST_NAME.log"


# python -m src.maxdiffusion.train_flux src/maxdiffusion/configs/base_flux_dev.yml \
python -m src.maxdiffusion.train_wan src/maxdiffusion/configs/base_wan_14b.yml \
        run_name="run_$EXP_NAME" output_dir="$LOG_PATH" \
        hardware=gpu \
        attention=cudnn_flash_te \
        max_train_steps=20 \
        dcn_data_parallelism=1 \
        dcn_fsdp_parallelism=-1 \
        ici_data_parallelism=1 \
        ici_fsdp_parallelism=8 \
        per_device_batch_size=1 \
        "${FILTERED_ARGS[@]}" |& tee -a "$LOG_FILE"



