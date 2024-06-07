FROM vllm/vllm-openai

RUN pip install pydantic "weaviate-client==3.*"

COPY wrapper_classes wrapper_classes
COPY prompts prompts
COPY weaviate_worker.py ./

ENTRYPOINT [ ]
