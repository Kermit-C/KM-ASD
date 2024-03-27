#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-03-27 14:25:36
"""

import threading
import time
from typing import Optional

import GPUtil
import psutil

import config
from manager.metric_manager import MetricCollector, create_collector
from utils.logger_util import sm_logger

cpu_metric_thread: Optional[threading.Thread] = None
cpu_metric_collector: MetricCollector
mem_metric_collector: MetricCollector
swap_mem_metric_collector: MetricCollector
gpu_metric_thread: Optional[threading.Thread] = None
gpu_load_metric_collector_list: list[MetricCollector]
gpu_mem_metric_collector_list: list[MetricCollector]


def init():
    """初始化系统资源监控指标线程"""
    global cpu_metric_thread
    global gpu_metric_thread

    if cpu_metric_thread is not None or gpu_metric_thread is not None:
        sm_logger.warning("System metric manager already initialized.")
        return

    cpu_metric_thread = threading.Thread(target=_get_and_metric_cpu, daemon=True)
    cpu_metric_thread.start()
    gpu_metric_thread = threading.Thread(target=_get_and_metric_gpu, daemon=True)
    gpu_metric_thread.start()

    sm_logger.info("System metric manager initialized.")


def load_metric_collector():
    """加载系统资源监控指标采集器"""
    global cpu_metric_collector
    global mem_metric_collector
    global swap_mem_metric_collector
    global gpu_load_metric_collector_list
    global gpu_mem_metric_collector_list
    cpu_metric_collector = create_collector("system_cpu")
    mem_metric_collector = create_collector("system_mem")
    swap_mem_metric_collector = create_collector("system_swap_mem")
    gpu_load_metric_collector_list = [
        create_collector(f"system_gpu_load_{i}") for i in config.system_metric_gpu_list
    ]
    gpu_mem_metric_collector_list = [
        create_collector(f"system_gpu_mem_{i}") for i in config.system_metric_gpu_list
    ]


def _get_and_metric_cpu():
    """获取并记录CPU、内存、交换内存使用率"""
    # 初始化 cpu_percent 收集
    _ = psutil.cpu_percent(interval=None, percpu=False)
    time.sleep(1)

    while True:
        cpu_percent = psutil.cpu_percent(interval=None, percpu=False)
        mem_percent = psutil.virtual_memory().percent
        swap_mem_percent = psutil.swap_memory().percent

        cpu_metric_collector.collect(cpu_percent)
        mem_metric_collector.collect(mem_percent)
        swap_mem_metric_collector.collect(swap_mem_percent)

        time.sleep(1)


def _get_and_metric_gpu():
    """获取并记录 GPU 使用率"""
    gpus = GPUtil.getGPUs()
    if not gpus:
        return

    while True:
        for i, gpu in enumerate(GPUtil.getGPUs()):
            if i not in config.system_metric_gpu_list:
                continue
            j = config.system_metric_gpu_list.index(i)
            gpu_load_metric_collector_list[j].collect(gpu.load * 100)
            gpu_mem_metric_collector_list[j].collect(gpu.memoryUtil * 100)

        time.sleep(1)
