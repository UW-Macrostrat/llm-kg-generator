FROM vllm/vllm-openai

# install dependencies
RUN pip install openai arq pydantic "weaviate-client==3.*"

# default environmental variables
ENV MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct 
# ENV MODEL_NAME=astronomer/Llama-3-8B-Instruct-GPTQ-4-Bit
ENV REDIS_HOST=cosmos0001.chtc.wisc.edu

COPY start.sh .
RUN chmod +x start.sh

COPY wrapper_classes wrapper_classes
COPY prompts prompts
COPY arq_worker.py ./

ENTRYPOINT [ "./start.sh" ]