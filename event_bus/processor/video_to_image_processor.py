#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:43:36
"""


from event_bus.event_bus_processor import BaseEventBusProcessor
from event_bus.message_body.video_to_image_message_body import VideoToImageMessageBody


class VideoToImageProcessor(BaseEventBusProcessor):
    """视频转图片帧处理器"""

    def __init__(self, processor_name: str):
        super().__init__(processor_name)

    def process(self, event_message: VideoToImageMessageBody):
        # TODO
        video_to_image_event = event_message.event
        self.video_to_image(
            video_to_image_event.video_path, video_to_image_event.image_dir
        )
