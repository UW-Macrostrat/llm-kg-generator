#!/bin/bash

# export environmental variables
export $(grep -v '^#' llm.env | xargs)

# create cache directories (needed for CHTC)
mkdir tmp/hf_cache
mkdir tmp/numba_cache
mkdir tmp/vllm_config
mkdir tmp/outlines_cache
mkdir tmp/triton_cache
export HF_HOME=`pwd`/tmp/hf_cache
export NUMBA_CACHE_DIR=`pwd`/tmp/numba_cache
export VLLM_CONFIG_ROOT=`pwd`/tmp/vllm_config
export OUTLINES_CACHE_DIR=`pwd`/tmp/outlines_cache
export TRITON_CACHE_DIR=`pwd`/tmp/triton_cache
export http_proxy=

# debugging
export VLLM_LOGGING_LEVEL=DEBUG
export CUDA_LAUNCH_BLOCKING=1
export VLLM_TRACE_FUNCTION=1

# launch vllm server
python3 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --enable-prefix-caching \
    --max-model-len 8192 \
    > /dev/null &

# python3 -m vllm.entrypoints.openai.api_server --model ${MODEL_NAME}  --dtype float16 --gpu-memory-utilization 0.7  --max-model-len 4096 > /dev/null &

# start worker
cd /vllm-workspace/
python3 -u -m workers.worker