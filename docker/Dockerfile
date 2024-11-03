FROM vllm/vllm-openai

RUN pip install --upgrade pydantic "weaviate-client==3.*" devtools grpcio-tools protobuf

COPY workers workers
RUN python3 -m grpc_tools.protoc -Iworkers/pb --python_out=workers/pb --grpc_python_out=workers/pb workers/pb/*.proto

ENTRYPOINT [ ]
