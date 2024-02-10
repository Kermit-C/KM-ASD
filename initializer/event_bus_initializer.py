#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 事件总线初始化
@Author: chenkeming
@Date: 2024-02-10 15:50:27
"""

from config import event_bus
from event_bus.event_bus_factory import create_processor
from event_bus.processor.video_to_image_processor import VideoToImageProcessor


def init_event_bus():
    """初始化事件总线"""
    create_processor(
        VideoToImageProcessor(
            event_bus["processors"][VideoToImageProcessor]["processor_name"]
        ),
        event_bus["publisher"]["name"],
        event_bus["processors"][VideoToImageProcessor]["topic"],
    )
