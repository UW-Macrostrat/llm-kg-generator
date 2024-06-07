export $(grep -v '^#' llm.env | xargs)

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

python3 -m vllm.entrypoints.openai.api_server --model ${MODEL_NAME} --enable-prefix-caching > /dev/null 2>&1 &
# python3 -u /vllm-workspace/weaviate_worker.py --worker
python3 -u /vllm-workspace/map_worker.py --worker