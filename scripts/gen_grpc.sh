#!/bin/bash

python -m grpc_tools.protoc \
    -I./grpc_service \
    --python_out=./grpc_service \
    --grpc_python_out=./grpc_service \
    ./grpc_service/model_service.proto
