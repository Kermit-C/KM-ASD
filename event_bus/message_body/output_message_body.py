#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-18 16:12:33
"""

from event_bus.event_message import EventMessageBody


class OutputMessageBody(EventMessageBody):
    def __init__(self, output_image_path: str, output_image_name: str):
        # TODO
        # 输出图像路径
        self.output_image_path = output_image_path
        # 输出图像名称
        self.output_image_name = output_image_name
