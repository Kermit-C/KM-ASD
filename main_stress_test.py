#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 压力测试入口
@Author: Kermit
@Date: 2024-04-09 18:06:06
"""

import initializer
from service.stress_test_service import start

if __name__ == "__main__":
    initializer.init_event_bus()
    start(wait_for_termination=True)
