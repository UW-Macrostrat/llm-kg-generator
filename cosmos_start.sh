#!/bin/bash

python3 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --dtype float16 \
    --gpu-memory-utilization 0.7  \
    --max-model-len 4096 > /dev/null &
python3 -u -m workers.worker