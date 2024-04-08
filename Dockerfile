# Use an official Python runtime as a parent image
FROM python:3.10 AS build

# Install dependencies
COPY requirements.txt ./
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY *.py ./

# Generate gRPC stub files
COPY *.proto ./
RUN python3.10 -m grpc_tools.protoc -I=. --python_out=. --grpc_python_out=. ./workerserver.proto

# Run gRPC server
CMD python3.10 kg_generator.py
