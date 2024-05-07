FROM vllm/vllm-openai

# install dependencies
RUN pip install openai arq pydantic "weaviate-client==3.*" devtools

# default environmental variables

COPY start.sh .
RUN chmod +x start.sh

COPY wrapper_classes wrapper_classes
COPY prompts prompts
COPY arq_worker.py ./

ENTRYPOINT [ "./start.sh" ]