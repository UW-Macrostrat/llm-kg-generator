source venv/bin/activate
cd pb
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. job_manager.proto