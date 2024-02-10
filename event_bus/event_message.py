#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 事件总线消息
@Author: chenkeming
@Date: 2024-02-09 17:16:48
"""


class EventMessageBody:
    def __init__(self, event_type: str, data: dict):
        # TODO
        self.event_type = event_type
        self.data = data


class EventMessage:
    def __init__(self, request_id: str, body: EventMessageBody):
        # TODO
        self.request_id = request_id
        self.body = body

    def copy(self):
        return EventMessage(self.request_id, self.body)
