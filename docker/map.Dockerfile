FROM vllm/vllm-openai

RUN pip install pydantic "weaviate-client==3.*" devtools

COPY wrapper_classes wrapper_classes
COPY prompts prompts
COPY map_worker.py .

ENTRYPOINT [ ]
