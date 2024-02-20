#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: grpc 服务管理
@Author: Kermit
@Date: 2024-02-19 12:50:04
"""

from concurrent import futures

import grpc
from grpc._server import _Server

import config
from grpc_service import model_service_pb2_grpc
from grpc_service.model_service import ModelServiceServicer

server: _Server


def start_server(max_workers=10, port=50051, wait_for_termination=True):
    global server
    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=config.model_service_server_thread_name_prefix,
        ),
        # options 参见：https://grpc.github.io/grpc/python/glossary.html#term-channel_arguments
        options=(
            ("grpc.max_receive_message_length", 1073741824),  # 默认 4M，调整为 1024M
            ("grpc.max_send_message_length", 1073741824),
        ),
    )
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(
        ModelServiceServicer(), server
    )
    server.add_insecure_port(f"[::]:{port}")
    print(f"Starting grpc server on port {port}")
    server.start()
    if wait_for_termination:
        server.wait_for_termination()


def stop_server():
    global server
    server.stop(0)
    print("Grpc server stopped")
