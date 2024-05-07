#!/bin/bash

# python3 -m vllm.entrypoints.openai.api_server --model ${MODEL_NAME} &
python3 -m vllm.entrypoints.openai.api_server --model ${MODEL_NAME} --dtype float16 &

if [ "$1" = "test" ]; then
    python3 arq_worker.py
else
    arq arq_worker.WorkerSettings
fi