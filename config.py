#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

from event_bus.processor.video_to_image_processor import VideoToImageProcessor

# 事件总线配置
event_bus = {
    "publisher": {
        "name": "default_event_bus_publisher",
    },
    "processors": {
        VideoToImageProcessor: {
            "processor_name": "video_to_image_processor",
            "topic": "video_to_image_topic",
        },
    },
}
