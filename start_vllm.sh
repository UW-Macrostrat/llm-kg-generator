#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

# This launcher is for the AWQ model discussed in the conversation.
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct-AWQ}" \
    --quantization awq \
    --dtype float16 \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-2048}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS:-1}" \
    --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-512}" \
    --enforce-eager
