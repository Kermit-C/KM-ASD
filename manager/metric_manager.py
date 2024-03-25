#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-03-25 14:53:54
"""


import os
import threading
import time
from queue import Queue
from typing import Optional

import config
from utils.logger_util import mt_logger
from utils.metric_util import MetricDumper

metric_reduce_thread: Optional[threading.Thread] = None
metric_dump_thread: Optional[threading.Thread] = None
metric_collector_list: Optional[list["MetricCollector"]] = None


def init():
    """初始化指标管理器"""
    global metric_reduce_thread
    global metric_dump_thread
    global metric_collector_list

    if (
        metric_reduce_thread is not None
        or metric_dump_thread is not None
        or metric_collector_list is not None
    ):
        mt_logger.warning("Metric manager already initialized.")
        return

    metric_collector_list = []

    metric_reduce_thread = threading.Thread(target=_reduce, daemon=True)
    metric_reduce_thread.start()
    metric_dump_thread = threading.Thread(target=_dump, daemon=True)
    metric_dump_thread.start()

    mt_logger.info("Metric manager initialized.")


def create_collector(
    metric_name: str, dir_path: str = config.metric_dump_dir
) -> "MetricCollector":
    """创建指标收集器"""
    if metric_collector_list is None:
        raise Exception("Metric manager not initialized.")
    collector: "MetricCollector" = MetricCollector(metric_name, dir_path)
    metric_collector_list.append(collector)
    return collector


def _reduce():
    assert metric_collector_list is not None
    curr_second = 0
    while True:
        for collector in metric_collector_list:
            collector.reduce_per_second()
            if curr_second == 0:
                collector.reduce_per_minute()
        curr_second += 1
        curr_second %= 60
        time.sleep(1)


def _dump():
    assert metric_collector_list is not None
    while True:
        for collector in metric_collector_list:
            collector.dump()
        time.sleep(config.metric_dump_interval)


class MetricCollector:
    """指标收集器"""

    def __init__(self, metric_name: str, dir_path: str, decimal_places: int = 2):
        self.metric_name = metric_name
        self.dir_path = dir_path
        self.decimal_places = decimal_places

        self.metric_dumper_per_metric = MetricDumper(
            os.path.join(dir_path, f"{metric_name}.csv")
        )
        self.metric_dumper_per_metric.write_headers(["timestamp", "value"])
        self.metric_dumper_per_second = MetricDumper(
            os.path.join(dir_path, f"{metric_name}_second.csv")
        )
        self.metric_dumper_per_second.write_headers(
            ["timestamp", "cnt", "avg", "max", "min", "p90", "p95", "p99"]
        )
        self.metric_dumper_per_minute = MetricDumper(
            os.path.join(dir_path, f"{metric_name}_minute.csv")
        )
        self.metric_dumper_per_minute.write_headers(
            ["timestamp", "cnt", "avg", "max", "min", "p90", "p95", "p99"]
        )

        self.queue_per_metric = Queue()
        self.queue_per_second = Queue()
        self.queue_per_minute = Queue()

        self.cnt_queue_per_second = Queue()
        self.cnt_queue_per_minute = Queue()
        self.avg_queue_per_second = Queue()
        self.avg_queue_per_minute = Queue()
        self.max_queue_per_second = Queue()
        self.max_queue_per_minute = Queue()
        self.min_queue_per_second = Queue()
        self.min_queue_per_minute = Queue()
        self.p90_queue_per_second = Queue()
        self.p90_queue_per_minute = Queue()
        self.p95_queue_per_second = Queue()
        self.p95_queue_per_minute = Queue()
        self.p99_queue_per_second = Queue()
        self.p99_queue_per_minute = Queue()

    def collect(self, value):
        """收集指标"""
        timestamp = time.time()
        self.queue_per_metric.put((timestamp, value))
        self.queue_per_second.put((timestamp, value))
        self.queue_per_minute.put((timestamp, value))

    def dump(self):
        """写入文件"""
        while not self.queue_per_metric.empty():
            timestamp, value = self.queue_per_metric.get()
            value = round(value, self.decimal_places)
            self.metric_dumper_per_metric.write_metrics([timestamp, value])

        while not self.cnt_queue_per_second.empty():
            timestamp, cnt = self.cnt_queue_per_second.get()
            _, avg = self.avg_queue_per_second.get()
            _, max = self.max_queue_per_second.get()
            _, min = self.min_queue_per_second.get()
            _, p90 = self.p90_queue_per_second.get()
            _, p95 = self.p95_queue_per_second.get()
            _, p99 = self.p99_queue_per_second.get()
            avg = round(avg, self.decimal_places)
            max = round(max, self.decimal_places)
            min = round(min, self.decimal_places)
            p90 = round(p90, self.decimal_places)
            p95 = round(p95, self.decimal_places)
            p99 = round(p99, self.decimal_places)
            self.metric_dumper_per_second.write_metrics(
                [timestamp, cnt, avg, max, min, p90, p95, p99]
            )

        while not self.cnt_queue_per_minute.empty():
            timestamp, cnt = self.cnt_queue_per_minute.get()
            _, avg = self.avg_queue_per_minute.get()
            _, max = self.max_queue_per_minute.get()
            _, min = self.min_queue_per_minute.get()
            _, p90 = self.p90_queue_per_minute.get()
            _, p95 = self.p95_queue_per_minute.get()
            _, p99 = self.p99_queue_per_minute.get()
            avg = round(avg, self.decimal_places)
            max = round(max, self.decimal_places)
            min = round(min, self.decimal_places)
            p90 = round(p90, self.decimal_places)
            p95 = round(p95, self.decimal_places)
            p99 = round(p99, self.decimal_places)
            self.metric_dumper_per_minute.write_metrics(
                [timestamp, cnt, avg, max, min, p90, p95, p99]
            )

    def reduce_per_second(self):
        cnt = 0
        avg = 0
        max = 0
        min = 0
        p90 = 0
        p95 = 0
        p99 = 0
        buffer_list = []
        timestamp = time.time()
        while not self.queue_per_second.empty():
            ts, value = self.queue_per_second.get()
            # 如果时间戳超过 1 秒，则不再处理
            if timestamp - ts > 1:
                continue
            # 如果是未来时间戳，则重新放回队列
            if timestamp < ts:
                self.queue_per_second.put((ts, value))
                break
            cnt += 1
            avg += value
            max = max if max > value else value
            min = min if min < value else value
            buffer_list.append(value)
        avg /= cnt if cnt > 0 else 1
        buffer_list.sort()
        p90 = buffer_list[int(len(buffer_list) * 0.9)] if buffer_list else 0
        p95 = buffer_list[int(len(buffer_list) * 0.95)] if buffer_list else 0
        p99 = buffer_list[int(len(buffer_list) * 0.99)] if buffer_list else 0
        self.cnt_queue_per_second.put((timestamp, cnt))
        self.avg_queue_per_second.put((timestamp, avg))
        self.max_queue_per_second.put((timestamp, max))
        self.min_queue_per_second.put((timestamp, min))
        self.p90_queue_per_second.put((timestamp, p90))
        self.p95_queue_per_second.put((timestamp, p95))
        self.p99_queue_per_second.put((timestamp, p99))

    def reduce_per_minute(self):
        cnt = 0
        avg = 0
        max = 0
        min = 0
        p90 = 0
        p95 = 0
        p99 = 0
        buffer_list = []
        timestamp = time.time()
        while not self.queue_per_minute.empty():
            ts, value = self.queue_per_minute.get()
            # 如果时间戳超过 60 秒，则不再处理
            if timestamp - ts > 60:
                continue
            # 如果是未来时间戳，则重新放回队列
            if timestamp < ts:
                self.queue_per_minute.put((ts, value))
                break
            cnt += 1
            avg += value
            max = max if max > value else value
            min = min if min < value else value
            buffer_list.append(value)
        avg /= cnt if cnt > 0 else 1
        buffer_list.sort()
        p90 = buffer_list[int(len(buffer_list) * 0.9)] if buffer_list else 0
        p95 = buffer_list[int(len(buffer_list) * 0.95)] if buffer_list else 0
        p99 = buffer_list[int(len(buffer_list) * 0.99)] if buffer_list else 0
        self.cnt_queue_per_minute.put((timestamp, cnt))
        self.avg_queue_per_minute.put((timestamp, avg))
        self.max_queue_per_minute.put((timestamp, max))
        self.min_queue_per_minute.put((timestamp, min))
        self.p90_queue_per_minute.put((timestamp, p90))
        self.p95_queue_per_minute.put((timestamp, p95))
        self.p99_queue_per_minute.put((timestamp, p99))
