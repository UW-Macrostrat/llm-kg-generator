#!/bin/bash
python3 -m vllm.entrypoints.openai.api_server --model ${MODEL_NAME} --dtype float16 &
sleep 10
arq arq_worker.WorkerSettings
# python3 arq_worker.py