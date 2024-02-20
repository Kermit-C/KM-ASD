#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 说话人检测处理器
@Author: Kermit
@Date: 2024-02-18 15:53:31
"""


from event_bus.event_bus_processor import BaseEventBusProcessor
from event_bus.message_body.asd_message_body import AsdMessageBody
from event_bus.message_body.reduce_message_body import ReduceMessageBody


class AsdProcessor(BaseEventBusProcessor):
    """说话人检测处理器"""

    def __init__(self, processor_name: str):
        super().__init__(processor_name)

    def process(self, event_message_body: AsdMessageBody):
        # TODO: 待实现
        if event_message_body.type == "V":
            asd_status = 0
            self.publish_next(
                "reduce_topic",
                ReduceMessageBody(
                    type="ASD",
                    frame_count=event_message_body.frame_count,
                    frame_timestamp=event_message_body.frame_timestamp,
                    frame_face_idx=event_message_body.frame_face_idx,
                    frame_face_count=event_message_body.frame_face_count,
                    frame_face_bbox=event_message_body.frame_face_bbox,
                    frame_asd_status=asd_status,
                ),
            )

    def process_exception(
        self, event_message_body: AsdMessageBody, exception: Exception
    ):
        # TODO: 待实现
        pass
