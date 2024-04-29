#!/bin/bash

# python3 -m vllm.entrypoints.openai.api_server --model ${MODEL_NAME} &
python3 -m vllm.entrypoints.openai.api_server --model ${MODEL_NAME} --dtype float16 &

arq arq_worker.WorkerSettings